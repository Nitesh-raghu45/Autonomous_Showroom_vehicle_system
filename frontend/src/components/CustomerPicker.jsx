export default function CustomerPicker({
  value,
  onChange,
  onLoad,
  onReset,
  customers,
  busy,
  hasSession,
}) {
  return (
    <div className="picker">
      <div className="field">
        <label htmlFor="number">Customer number</label>
        <input
          id="number"
          value={value}
          placeholder="10001 or a mobile number"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onLoad()}
        />
      </div>

      {customers.length > 0 && (
        <div className="field">
          <label htmlFor="pick">Or pick one</label>
          <select
            id="pick"
            value=""
            onChange={(e) => e.target.value && onChange(e.target.value)}
          >
            <option value="">{customers.length} records loaded</option>
            {customers.map((c) => (
              <option key={c.customer_id} value={c.customer_id}>
                {c.customer_id} — {c.name} — {c.vehicle || "no vehicle"}
              </option>
            ))}
          </select>
        </div>
      )}

      <button className="btn" onClick={onLoad} disabled={busy || !value.trim()}>
        {busy ? "Loading" : "Load customer"}
      </button>

      {hasSession && (
        <button className="btn quiet" onClick={onReset} disabled={busy}>
          Clear session
        </button>
      )}
    </div>
  );
}
