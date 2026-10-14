import { useEffect, useRef } from "react";
import ActionCard from "./ActionCard.jsx";

const QUESTION_HINTS = [
  "Show me used diesel cars under 20 lakh",
  "What electric vehicles do you have?",
  "Book a test drive for next Monday at 10am",
  "What car is this customer interested in?",
  "Update my email to new@email.com",
  "Show me all new SUVs",
];

function Stamp({ turn }) {
  const stats = turn.stats ?? {};
  const meta = turn.meta ?? {};
  if (!stats.source) return null;

  const fromRouter = stats.source === "router" || stats.source === "semantic";
  const fromCache = stats.source === "answer_cache";
  const cachedTokens = stats.prompt_cache_hit_tokens ?? 0;

  return (
    <div className="stamp">
      <span className={fromRouter || fromCache ? "hit" : "miss"}>
        {fromRouter ? "deterministic" : fromCache ? "served from cache" : "generated"}
      </span>
      <span>{stats.latency_ms} ms</span>
      {meta.profile_cache ? (
        <span className={meta.profile_cache === "l1" || meta.profile_cache === "redis" ? "hit" : "miss"}>
          profile {meta.profile_cache}
        </span>
      ) : null}
      {stats.usage ? (
        <>
          <span>
            {stats.usage.prompt_tokens} in / {stats.usage.completion_tokens} out
          </span>
          <span className={cachedTokens ? "hit" : ""}>{cachedTokens} prompt-cached</span>
        </>
      ) : null}
    </div>
  );
}

export default function Conversation({ turns, awaitingCustomer, hints, onSuggest }) {
  const threadRef = useRef(null);

  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const chips = awaitingCustomer ? hints ?? [] : QUESTION_HINTS;

  return (
    <div className="thread" ref={threadRef}>
      {turns.length === 0 ? (
        <div className="empty">
          <p>
            {awaitingCustomer
              ? "Say hello to get started -- the assistant will ask for your mobile number."
              : "Ask anything about your record, or try: book a test drive, update my email."}
          </p>
          {chips.length > 0 && (
            <div className="chips">
              {chips.map((text) => (
                <button className="chip" key={text} onClick={() => onSuggest(text)}>
                  {text}
                </button>
              ))}
            </div>
          )}
        </div>
      ) : (
        turns.map((turn) => (
          <div className={`turn ${turn.role}`} key={turn.id}>
            <div className="speaker">{turn.role === "you" ? "You" : "Assistant"}</div>
            <div className={`text ${turn.error ? "failed" : ""}`}>
              {turn.error || turn.text}
              {turn.pending && !turn.error ? <i className="caret" /> : null}
            </div>
            {turn.role === "assistant" && turn.action ? (
              <ActionCard action={turn.action} />
            ) : null}
            {turn.role === "assistant" ? <Stamp turn={turn} /> : null}
          </div>
        ))
      )}
    </div>
  );
}