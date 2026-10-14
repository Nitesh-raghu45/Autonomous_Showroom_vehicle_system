/**
 * ActionCard — shown in the chat bubble when the LLM executes an action.
 * Handles: BOOK_APPOINTMENT, UPDATE_INFO, SEARCH_INVENTORY
 */

const TYPE_ICONS = {
  TEST_DRIVE:          "🚗",
  CALLBACK:            "📞",
  ADVANCE_PAYMENT:     "💳",
  DOCUMENT_SUBMISSION: "📄",
  OTHER:               "📅",
  UPDATE_INFO:         "✏️",
  BOOK_APPOINTMENT:    "📅",
  SEARCH_INVENTORY:    "🔍",
};

const TYPE_LABELS = {
  TEST_DRIVE:          "Test Drive",
  CALLBACK:            "Callback",
  ADVANCE_PAYMENT:     "Advance Payment",
  DOCUMENT_SUBMISSION: "Document Submission",
  OTHER:               "Appointment",
  UPDATE_INFO:         "Info Updated",
  BOOK_APPOINTMENT:    "Appointment Booked",
  SEARCH_INVENTORY:    "Inventory Search Applied",
};

export default function ActionCard({ action }) {
  if (!action) return null;

  const { type, action_type, result, error } = action;
  const resolvedType = type || action_type;

  if (error) {
    return (
      <div className="action-card action-card--error">
        <span className="action-icon">⚠️</span>
        <div>
          <div className="action-title">Action Failed</div>
          <div className="action-detail">{error}</div>
        </div>
      </div>
    );
  }

  // ── SEARCH_INVENTORY ────────────────────────────────────────────────────
  if (resolvedType === "SEARCH_INVENTORY") {
    const parts = [];
    if (action.vtype && action.vtype !== "all") parts.push(action.vtype);
    if (action.make)  parts.push(action.make);
    if (action.model) parts.push(action.model);
    if (action.fuel)  parts.push(action.fuel);
    if (action.min_price || action.max_price) {
      const lo = action.min_price ? `₹${(action.min_price / 100000).toFixed(0)}L` : "any";
      const hi = action.max_price ? `₹${(action.max_price / 100000).toFixed(0)}L` : "any";
      parts.push(`${lo}–${hi}`);
    }
    return (
      <div className="action-card action-card--search">
        <span className="action-icon">🔍</span>
        <div>
          <div className="action-title">Inventory filtered ✓</div>
          <div className="action-detail">
            {parts.length ? parts.join(" · ") : "Showing all vehicles"}
            {" "}— see panel on the right →
          </div>
        </div>
      </div>
    );
  }

  // ── BOOK_APPOINTMENT ────────────────────────────────────────────────────
  if (resolvedType === "BOOK_APPOINTMENT" && result) {
    const apptType = result.appointment_type ?? "";
    return (
      <div className="action-card action-card--success">
        <span className="action-icon">{TYPE_ICONS[apptType] ?? "📅"}</span>
        <div>
          <div className="action-title">
            {TYPE_LABELS[apptType] ?? "Appointment"} Booked ✓
          </div>
          <div className="action-detail">
            <strong>Date:</strong> {result.date}
            &nbsp;·&nbsp;
            <strong>Time:</strong> {result.time}
          </div>
          {result.notes && (
            <div className="action-detail action-note">{result.notes}</div>
          )}
          <div className="action-id">ID: {result.appointment_id}</div>
        </div>
      </div>
    );
  }

  // ── UPDATE_INFO ─────────────────────────────────────────────────────────
  if (resolvedType === "UPDATE_INFO" && result) {
    const fields = result.updated_fields ?? [];
    return (
      <div className="action-card action-card--success">
        <span className="action-icon">✏️</span>
        <div>
          <div className="action-title">Profile Updated ✓</div>
          <div className="action-detail">
            Changed: {fields.length ? fields.join(", ") : "nothing"}
          </div>
        </div>
      </div>
    );
  }

  // ── Generic fallback ────────────────────────────────────────────────────
  if (!result && !error) return null;

  const icon  = TYPE_ICONS[resolvedType]  ?? "✅";
  const label = TYPE_LABELS[resolvedType] ?? resolvedType;
  return (
    <div className="action-card action-card--success">
      <span className="action-icon">{icon}</span>
      <div>
        <div className="action-title">{label} ✓</div>
      </div>
    </div>
  );
}
