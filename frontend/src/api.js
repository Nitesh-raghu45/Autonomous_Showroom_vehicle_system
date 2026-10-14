/**
 * Backend client.
 *
 * The chat endpoint is server-sent events over a POST, which EventSource cannot
 * do (it is GET-only), so the stream is read off fetch's ReadableStream and the
 * SSE frames are parsed here. Frames are separated by a blank line; a partial
 * frame is held in `buffer` until the rest of it arrives.
 */

async function asJson(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}

export function normalizeStreamError(err) {
  const message = String(err?.message || err || "Unknown error");

  if (/abort|cancelled/i.test(message)) return "The request was cancelled.";
  if (/timeout|ERR_CONNECTION_TIMEOUT|timed out/i.test(message)) {
    return "The backend did not respond in time. Please check that the server is running.";
  }
  if (/failed to fetch|network|fetch/i.test(message)) {
    return "Connection failed while contacting the backend. Please check the server status.";
  }

  return message;
}


/**
 * fetch with a connect/headers deadline and automatic retry.
 *
 * Why: the server sends response headers immediately (even when the LLM is
 * slow), so if no headers arrive within `headerMs` the TCP connection almost
 * certainly never opened (Chrome would otherwise wait ~21 s and report
 * ERR_CONNECTION_TIMED_OUT). Retrying on a fresh connection usually succeeds.
 * A caller-initiated abort is never retried. The deadline only covers the time
 * until headers arrive; it is cleared before the body is read, so long LLM
 * streams are unaffected.
 */
async function fetchRetry(url, opts = {}, { attempts = 3, headerMs = 5000 } = {}) {
  const userSignal = opts.signal;
  let lastErr;
  for (let i = 0; i < attempts; i++) {
    if (userSignal?.aborted) throw new DOMException("Aborted", "AbortError");
    const ctl = new AbortController();
    const onUserAbort = () => ctl.abort();
    userSignal?.addEventListener("abort", onUserAbort);
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      ctl.abort();
    }, headerMs);
    try {
      return await fetch(url, { ...opts, signal: ctl.signal });
    } catch (err) {
      if (userSignal?.aborted) throw err;
      lastErr = timedOut ? new Error("timed out waiting for the backend") : err;
    } finally {
      clearTimeout(timer);
      userSignal?.removeEventListener("abort", onUserAbort);
    }
  }
  throw lastErr;
}

export const getHealth = () => fetch("/health").then(asJson);
export const getMetrics = () => fetch("/metrics").then(asJson);
export const listCustomers = () => fetch("/customers").then(asJson);
export const getCustomer = (number) => fetch(`/customers/${encodeURIComponent(number)}`).then(asJson);

export function startSession(customerNumber) {
  return fetchRetry("/session/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_number: customerNumber }),
  }).then(asJson);
}

/** Register a new customer. Throws an Error with a readable message (and `.status`). */
export async function registerCustomer({ name, mobile, email }) {
  const res = await fetch("/customer/register", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ name, mobile_no: mobile, email }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    // 422 validation errors arrive as a list of {loc, msg}; keep the field so the chat can re-ask it
    const first  = Array.isArray(body.detail) ? body.detail[0] : null;
    const err    = new Error(
      first ? String(first.msg || "Invalid value").replace(/^Value error, /, "")
            : body.detail || `${res.status} ${res.statusText}`
    );
    err.status = res.status;
    err.field  = first?.loc?.[first.loc.length - 1] ?? null;
    throw err;
  }
  return body;
}

export function endSession(sessionId) {
  return fetch(`/session/${sessionId}`, { method: "DELETE" }).then(asJson);
}

function parseFrame(frame) {
  let event = "message";
  const dataLines = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  let data = {};
  try {
    data = JSON.parse(dataLines.join("\n") || "{}");
  } catch {
    data = {};
  }
  return { event, data };
}

/**
 * Stream one answer. After /session/start verifies the customer, customerNumber
 * can be empty: the backend resolves the customer from the verified session id.
 * `onEvent({event, data})` fires for session/meta/token/done/error.
 */
export async function streamChat({ sessionId, customerNumber, question, onEvent, signal }) {
  let res;
  try {
    res = await fetchRetry("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        customer_number: customerNumber,
        question,
      }),
      signal,
    });
  } catch (err) {
    throw new Error(normalizeStreamError(err));
  }

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    throw new Error(normalizeStreamError(new Error(detail.slice(0, 300) || `Stream failed: ${res.status}`)));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        if (!frame.trim()) continue;
        onEvent(parseFrame(frame));
      }
    }
    if (buffer.trim()) onEvent(parseFrame(buffer));
  } catch (err) {
    throw new Error(normalizeStreamError(err));
  } finally {
    reader.releaseLock();
  }
}