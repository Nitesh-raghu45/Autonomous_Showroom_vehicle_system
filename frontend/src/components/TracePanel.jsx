/**
 * The request trace. This is the point of the demo: for the last question asked,
 * show which of the three caches answered it and what that saved.
 *
 * The gates are numbered because they genuinely run in sequence, cheapest first.
 * The timing breakdown shows exactly where each millisecond was spent.
 */

const IDLE = { state: "idle", verdict: "not run yet" };

function gatesFor(turn) {
  if (!turn || (!turn.meta && !turn.stats)) return [IDLE, IDLE, IDLE];

  const meta = turn.meta ?? {};
  const stats = turn.stats ?? {};
  const viaRouter = stats.source === "router" || stats.source === "semantic";

  const routerGate = stats.source
    ? viaRouter
      ? { state: "hit", verdict: stats.source === "semantic"
          ? `MiniLM matched ${stats.semantic?.intent} (${stats.semantic?.score}), no LLM`
          : `rule matched: ${stats.intent}, no model needed` }
      : { state: "miss", verdict: `no rule matched (${meta.reason ?? "unclear"}), escalated` }
    : IDLE;

  const dataGate = meta.profile_cache
    ? meta.profile_cache === "l1"
      ? { state: "hit", verdict: "in-process cache (L1), no Redis read" }
      : meta.profile_cache === "redis"
        ? { state: "hit", verdict: "read from Redis (L2)" }
        : { state: "miss", verdict: "fell back to the JSON seed" }
    : IDLE;

  let llmGate = IDLE;
  if (viaRouter) {
    llmGate = { state: "idle", verdict: "skipped, deterministic answer" };
  } else if (stats.source === "answer_cache") {
    llmGate = { state: "hit", verdict: "repeat question, answer cache" };
  } else if (stats.source === "llm_timeout" || stats.source === "llm_error") {
    llmGate = { state: "miss", verdict: `fallback used (${stats.source})` };
  } else if (stats.usage) {
    const cached = stats.usage.cached_prompt_tokens ?? 0;
    const total = stats.usage.prompt_tokens ?? 0;
    llmGate = { state: "miss", verdict: `1 call, ${total} prompt tokens (${cached} cached)` };
  }

  return [routerGate, dataGate, llmGate];
}

const GATE_COPY = [
  { name: "Intent router", what: "Rules + entity extraction, microseconds" },
  { name: "Data cache", what: "L1 in-process, then one Redis pipeline" },
  { name: "LLM planner", what: "Only for ambiguous or multi-part requests" },
];

/** Tiny bar that fills proportionally to the value vs. the total. */
function TimingBar({ value, total }) {
  const pct = total > 0 ? Math.min(100, Math.round((value / total) * 100)) : 0;
  return (
    <div className="timing-bar-wrap" title={`${value} ms (${pct}%)`}>
      <div className="timing-bar-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

function TimingRow({ label, ms, total, note }) {
  return (
    <tr className="timing-row">
      <td className="timing-label">{label}</td>
      <td className="timing-ms">{ms != null ? `${ms} ms` : "—"}</td>
      <td className="timing-bar-cell">
        {ms != null && <TimingBar value={ms} total={total} />}
      </td>
      {note && <td className="timing-note">{note}</td>}
    </tr>
  );
}

function TimingBreakdown({ timings, latencyMs }) {
  if (!timings) return null;

  const rows = [
    { label: "⚡ Data fetch (L1 / Redis)", ms: timings.data_ms, note: "one pipelined round trip" },
    { label: "🧭 Intent routing", ms: timings.route_ms },
    { label: "🧾 Deterministic execution", ms: timings.exec_ms ?? null, note: timings.exec_ms == null ? "not used" : undefined },
    { label: "🤖 LLM call", ms: timings.llm_ms > 0 ? timings.llm_ms : null, note: !(timings.llm_ms > 0) ? "not used" : undefined },
  ];

  return (
    <div className="timing-breakdown">
      <h3 className="timing-heading">⏱ Timing Breakdown</h3>
      <table className="timing-table">
        <thead>
          <tr>
            <th>Step</th>
            <th>Time</th>
            <th style={{ minWidth: "80px" }}>Share</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <TimingRow
              key={r.label}
              label={r.label}
              ms={r.ms}
              total={latencyMs}
              note={r.note}
            />
          ))}
          <tr className="timing-total-row">
            <td className="timing-label"><strong>Total</strong></td>
            <td className="timing-ms"><strong>{latencyMs} ms</strong></td>
            <td />
          </tr>
        </tbody>
      </table>
      <p className="timing-note-small">
        Session write-back happens after the reply is sent and is not on the critical path.
      </p>
    </div>
  );
}

export default function TracePanel({ turn, session, health }) {
  const gates = gatesFor(turn);
  const stats = turn?.stats ?? {};
  const meta = turn?.meta ?? {};
  const usage = stats.usage;

  return (
    <section className="panel trace">
      <h2>Request trace</h2>

      <div className="gates">
        {GATE_COPY.map((gate, i) => (
          <div className={`gate ${gates[i].state}`} key={gate.name}>
            <div className="n">{i + 1}</div>
            <div>
              <div className="name">{gate.name}</div>
              <div className="what">{gate.what}</div>
              <div className="verdict">{gates[i].verdict}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="readout">
        <dl>
          <dt>Answered in</dt>
          <dd>{stats.latency_ms != null ? `${stats.latency_ms} ms` : "—"}</dd>

          <dt>Prompt tokens</dt>
          <dd>{usage ? usage.prompt_tokens : "—"}</dd>

          <dt>Output tokens</dt>
          <dd>{usage ? usage.completion_tokens : "—"}</dd>

          <dt>Turns in context</dt>
          <dd>{meta.history_turns ?? "—"}</dd>

          <dt>Route</dt>
          <dd>{stats.route ?? "—"}{stats.sla_met === false ? " ⚠ over 500 ms" : ""}</dd>

          <dt>Model</dt>
          <dd>{meta.model ?? health?.model ?? "—"}</dd>
        </dl>
      </div>

      {/* Per-step timing breakdown — only shown when backend sends timings */}
      <TimingBreakdown timings={stats.timings} latencyMs={stats.latency_ms} />

      <p className="note">
        Factual questions (your email, quotation total, appointments, stock counts) are
        answered by rules and templates in a few milliseconds. Only requests the router
        cannot resolve confidently reach the model.
        {session ? (
          <>
            {" "}
            Session <code>{session.session_id.slice(0, 8)}</code>.
          </>
        ) : null}
      </p>
    </section>
  );
}
