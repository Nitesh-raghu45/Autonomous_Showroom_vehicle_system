import { useCallback, useEffect, useRef, useState } from "react";

import Conversation    from "./components/Conversation.jsx";
import InfoPanel       from "./components/InfoPanel.jsx";
import InventoryBrowser from "./components/InventoryBrowser.jsx";
import TracePanel      from "./components/TracePanel.jsx";
import { endSession, getHealth, normalizeStreamError, registerCustomer, startSession, streamChat } from "./api.js";

let seq = 0;
const nextId = () => `t${++seq}`;

function extractMobile(text) {
  const digits = text.replace(/\D/g, "");
  if (!digits) return null;
  const last10 = digits.length >= 10 ? digits.slice(-10) : digits;
  return last10.length === 10 ? last10 : null;
}

// ── Conversational registration helpers ────────────────────────────
// Mirror the backend rules (app/models.py) so mistakes are caught before the request.
const NAME_PREFIX = /^(?:hi[,!]?\s+)?(?:my\s+(?:full\s+)?name\s+is|name\s*[:=-]?|i\s*am|i'?m|it'?s|this\s+is)\s+/i;
const EMAIL_RE    = /[^@\s,;<>()]+@[^@\s,;<>()]+\.[^@\s,;<>().]+/;
const CANCEL_RE   = /^(?:cancel|stop|no|nope|never\s?mind|start over|exit|quit)[.!]?$/i;

function extractName(text) {
  const name = text.trim().replace(NAME_PREFIX, "").replace(/[.!]+$/, "").replace(/\s+/g, " ").trim();
  if ((name.match(/[A-Za-z]/g) || []).length < 2) return { error: "Could you share your full name? It needs at least 2 letters." };
  if (!/^[A-Za-z .'-]+$/.test(name)) return { error: "A name can only have letters, spaces, dots, apostrophes and hyphens — what's your full name?" };
  if (name.length > 80) return { error: "That's a bit long for a name — could you share it in under 80 characters?" };
  // "nitesh raghuwanshi" -> "Nitesh Raghuwanshi"; leave names the person already capitalised alone
  const pretty = name === name.toLowerCase() ? name.replace(/\b[a-z]/g, (c) => c.toUpperCase()) : name;
  return { name: pretty };
}

const extractEmail = (text) => (text.match(EMAIL_RE) || [null])[0];

const ASK_FOR_MOBILE =
  "Hi, I'm the Customer Assist bot. To pull up your details, could you share the 10-digit mobile number on your account?";
const RETRY_MOBILE =
  "That doesn't look like a full 10-digit number — could you try again?";
const OFFER_REGISTER = (num) =>
  `I couldn't find an account for ${num}. Let's create one — what's your full name?\n(Or type a different mobile number to try again.)`;
const ASK_EMAIL = (name) =>
  `Thanks, ${name.split(" ")[0]}! What's your email address?`;
const GREETING_HINTS = ["Hi", "Hello", "I need help with my enquiry"];

export default function App() {
  const [health,          setHealth]         = useState(null);
  const [session,         setSession]        = useState(null);
  const [turns,           setTurns]          = useState([]);
  const [draft,           setDraft]          = useState("");
  const [busy,            setBusy]           = useState(false);
  const [error,           setError]          = useState(null);
  const [showTrace,       setShowTrace]      = useState(false);

  // Lifted state: chatbot drives these
  const [autoFilters,     setAutoFilters]    = useState(null);  // → InventoryBrowser
  const [infoRefresh,     setInfoRefresh]    = useState(0);     // → InfoPanel

  // Conversational registration: null, or { mobile, step: "name" | "email", name }
  const [reg,             setReg]            = useState(null);

  const inputRef  = useRef(null);
  const abortRef  = useRef(null);
  const turnsRef  = useRef(turns);
  turnsRef.current = turns;

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "unreachable" }));
  }, []);

  const patchTurn = useCallback((id, patch) =>
    setTurns((prev) =>
      prev.map((t) => (t.id === id ? { ...t, ...(typeof patch === "function" ? patch(t) : patch) } : t))
    ), []);

  const addTurn = useCallback((turn) => {
    const id = nextId();
    setTurns((prev) => [...prev, { id, ...turn }]);
    return id;
  }, []);

  const sayLocally = useCallback((text, extra = {}) => {
    const id = addTurn({ role: "assistant", text: "", pending: true, ...extra });
    window.setTimeout(() => patchTurn(id, { text, pending: false }), 320);
    return id;
  }, [addTurn, patchTurn]);

  // ── Begin session (existing customer) ────────────────────────────
  const beginSession = useCallback(async (mobile) => {
    try {
      const started = await startSession(mobile);
      setSession(started);
      sayLocally(
        `Thanks — I've got ${started.customer_name}'s record up (ID ${started.customer_id}). Your profile is now visible on the right. What would you like to know?`
      );
    } catch {
      // New customer: collect name, then email, right here in the chat.
      setReg({ mobile, step: "name", name: null });
      sayLocally(OFFER_REGISTER(mobile));
    }
  }, [sayLocally]);

  // ── Conversational registration (name → email → create account) ──
  const continueRegistration = useCallback(async (body) => {
    if (CANCEL_RE.test(body)) {
      setReg(null);
      sayLocally("No problem — I haven't created anything. Type your 10-digit mobile number whenever you're ready.");
      return;
    }

    // A different 10-digit number (with no email in it) means "try this number instead".
    const otherMobile = extractEmail(body) ? null : extractMobile(body);
    if (otherMobile && otherMobile !== reg.mobile) {
      setReg(null);
      await beginSession(otherMobile);
      return;
    }

    if (reg.step === "name") {
      const { name, error } = extractName(body);
      if (error) { sayLocally(error); return; }
      setReg({ ...reg, step: "email", name });
      sayLocally(ASK_EMAIL(name));
      return;
    }

    // step === "email"
    const email = extractEmail(body);
    if (!email) {
      sayLocally("That doesn't look like an email address — could you type it like name@example.com?");
      return;
    }
    try {
      const created = await registerCustomer({ name: reg.name, mobile: reg.mobile, email });
      setReg(null);
      sayLocally(`🎉 Account created! Welcome, ${created.customer_name} (ID: ${created.customer_id}). Starting your session…`);
      try {
        const started = await startSession(created.mobile_no);
        setSession(started);
        sayLocally(`You're all set! Ask me anything about your enquiry, or say "show me SUVs" to browse inventory.`);
      } catch {
        sayLocally("Your account is ready, but I couldn't start the session automatically — please type your mobile number again.");
      }
    } catch (err) {
      if (err.status === 409) {                       // number registered in the meantime
        setReg(null);
        sayLocally(`${err.message}`);
      } else if (err.field === "name") {
        setReg({ ...reg, step: "name", name: null });
        sayLocally(`${err.message}. Could you share your full name again?`);
      } else if (err.field === "email") {
        sayLocally(`${err.message}. Could you check the email address and type it again?`);
      } else {
        sayLocally(`Sorry, I couldn't create the account: ${err.message}. Please try your email again, or type "cancel".`);
      }
    }
  }, [reg, beginSession, sayLocally]);

  // ── Stream LLM answer ─────────────────────────────────────────────
  const askModel = useCallback(async (text, activeSession) => {
    const answerId   = addTurn({ role: "assistant", text: "", pending: true });
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat({
        sessionId:      activeSession.session_id,
        // The customer was verified when the session started. The session id
        // is now the authenticated customer context; do not resend the id.
        customerNumber: "",
        question:       text,
        signal:         controller.signal,
        onEvent: ({ event, data }) => {
          if (event === "session" && data.session_id) {
            setSession((s) => (s ? { ...s, session_id: data.session_id } : s));
          } else if (event === "meta") {
            patchTurn(answerId, { meta: data });
          } else if (event === "token") {
            patchTurn(answerId, (t) => ({ text: t.text + data.text }));
          } else if (event === "replace") {
            patchTurn(answerId, { text: data.text });
          } else if (event === "action") {
            patchTurn(answerId, { action: data });
            // Drive sidebar panels based on action type
            if (data.type === "SEARCH_INVENTORY") {
              setAutoFilters({
                make:      data.make      ?? null,
                model:     data.model     ?? null,
                vtype:     data.vtype     ?? "all",
                fuel:      data.fuel      ?? null,
                min_price: data.min_price ?? null,
                max_price: data.max_price ?? null,
              });
            } else if (["UPDATE_INFO", "BOOK_APPOINTMENT", "CANCEL_APPOINTMENT"].includes(data.type)) {
              // Bump counter so InfoPanel re-fetches
              setInfoRefresh((n) => n + 1);
            }
          } else if (event === "done") {
            patchTurn(answerId, { stats: data, pending: false });
          } else if (event === "error") {
            patchTurn(answerId, { error: data.message, pending: false });
          }
        },
      });
    } catch (err) {
      if (err.name !== "AbortError") {
        patchTurn(answerId, { error: normalizeStreamError(err), pending: false });
      }
    } finally {
      patchTurn(answerId, { pending: false });
      abortRef.current = null;
    }
  }, [addTurn, patchTurn]);

  // ── Main send handler ─────────────────────────────────────────────
  const send = useCallback(
    async (text) => {
      const body = (text ?? draft).trim();
      if (!body || busy) return;

      setDraft("");
      setError(null);
      addTurn({ role: "you", text: body });
      setBusy(true);

      try {
        if (session) {
          await askModel(body, session);
          return;
        }

        if (reg) {
          await continueRegistration(body);
          return;
        }

        const mobile = extractMobile(body);
        if (mobile) {
          await beginSession(mobile);
        } else {
          sayLocally(turnsRef.current.length <= 1 ? ASK_FOR_MOBILE : RETRY_MOBILE);
        }
      } finally {
        setBusy(false);
        inputRef.current?.focus();
      }
    },
    [draft, busy, session, reg, askModel, beginSession, continueRegistration, sayLocally, addTurn]
  );

  const startOver = useCallback(async () => {
    abortRef.current?.abort();
    if (session) await endSession(session.session_id).catch(() => {});
    setSession(null);
    setTurns([]);
    setError(null);
    setReg(null);
    setAutoFilters(null);
    setInfoRefresh(0);
  }, [session]);

  const lastAnswer = [...turns].reverse().find((t) => t.role === "assistant" && t.stats);
  const redisUp    = health?.redis === "up";

  return (
    <div className="shell">
      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="masthead">
        <div className="masthead-brand">
          <div className="masthead-logo">🚘</div>
          <div>
            <h1>Customer Assist</h1>
            <p className="masthead-sub">Vehicle Dealership Self-Service Portal</p>
          </div>
        </div>

        <div className="health-group">
          <div className="health">
            <span className={`dot ${health ? (health.status === "ok" ? "up" : "down") : ""}`} />
            {health
              ? health.status === "ok"
                ? `${health.provider} · ${health.customers_loaded} records · redis ${health.redis}`
                : "backend unreachable"
              : "checking backend…"}
          </div>
          {(session || turns.length > 0) && (
            <button className="btn quiet" onClick={startOver} disabled={busy}>
              Start over
            </button>
          )}
        </div>
      </header>

      {/* ── Banners ─────────────────────────────────────────────── */}
      {error && <div className="banner banner--error">{error}</div>}
      {health && health.status === "ok" && !redisUp && (
        <div className="banner">
          Redis is not reachable — caching and appointments disabled. Check REDIS_URL in .env.
        </div>
      )}

      {/* ── Session strip ───────────────────────────────────────── */}
      {session && (
        <div className="loaded">
          <strong>{session.customer_name}</strong>
          <span>ID <em className="num">{session.customer_id}</em></span>
          <span>Session expires in <em className="num">{Math.round(session.ttl_seconds / 60)} min</em></span>
        </div>
      )}

      {/* ── Main Layout ─────────────────────────────────────────── */}
      <div className="app-layout">

        {/* LEFT — Chat */}
        <div className="chat-col">
          <div className="panel chat-panel">
            <h2>💬 Conversation</h2>
            <Conversation
              turns={turns}
              awaitingCustomer={!session}
              hints={GREETING_HINTS}
              onSuggest={send}
            />
            <div className="composer">
              <input
                ref={inputRef}
                value={draft}
                disabled={busy}
                placeholder={
                  session
                    ? "Ask about your enquiry, say 'show me EVs', 'update my email'…"
                    : reg?.step === "name"
                      ? "Type your full name (or a different mobile number)"
                      : reg?.step === "email"
                        ? "Type your email address"
                        : "Enter your 10-digit mobile number"
                }
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && send()}
              />
              <button className="btn" onClick={() => send()} disabled={busy || !draft.trim()}>
                {busy ? "…" : "Send"}
              </button>
            </div>
          </div>

          {/* Request Trace — toggle */}
          <div className="trace-toggle-bar">
            <button
              className="btn quiet btn-sm trace-toggle-btn"
              onClick={() => setShowTrace(v => !v)}
            >
              {showTrace ? "▲ Hide request trace" : "▼ Show request trace"}
            </button>
          </div>
          {showTrace && (
            <div className="trace-drawer">
              <TracePanel turn={lastAnswer} session={session} health={health} />
            </div>
          )}
        </div>

        {/* RIGHT — Sidebar */}
        <aside className="side-panel">

          {/* Customer Info Card */}
          <div className="side-card">
            <div className="side-card__title">👤 My Profile</div>
            <InfoPanel session={session} refreshTrigger={infoRefresh} />
          </div>

          {/* Inventory Browser */}
          <div className="side-card side-card--inventory">
            <div className="side-card__title">🚗 Vehicle Inventory</div>
            <InventoryBrowser autoFilters={autoFilters} />
          </div>

        </aside>
      </div>
    </div>
  );
}