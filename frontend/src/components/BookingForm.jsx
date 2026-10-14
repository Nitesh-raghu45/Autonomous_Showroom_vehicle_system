/**
 * BookingForm — standalone form to schedule any appointment type.
 * Used inside InfoPanel. Calls POST /customer/{id}/schedule.
 */
import { useState } from "react";

const APPT_TYPES = [
  { value: "TEST_DRIVE",          label: "🚗 Test Drive" },
  { value: "CALLBACK",            label: "📞 Callback" },
  { value: "ADVANCE_PAYMENT",     label: "💳 Advance Payment" },
  { value: "DOCUMENT_SUBMISSION", label: "📄 Document Submission" },
  { value: "OTHER",               label: "📅 Other" },
];

const todayISO = () => new Date().toISOString().split("T")[0];

export default function BookingForm({ customerId, onBooked }) {
  const [type,  setType]  = useState("TEST_DRIVE");
  const [date,  setDate]  = useState(todayISO());
  const [hour,  setHour]  = useState(10);
  const [min,   setMin]   = useState(0);
  const [notes, setNotes] = useState("");
  const [busy,  setBusy]  = useState(false);
  const [error, setError] = useState(null);

  const submit = async () => {
    setError(null);
    setBusy(true);
    try {
      const res = await fetch(`/customer/${customerId}/schedule`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          appointment_type: type,
          date,
          hour: parseInt(hour),
          minute: parseInt(min),
          notes: notes.trim() || null,
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || `Error ${res.status}`);
      onBooked?.(body);
      setNotes("");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="booking-form">
      <h4 className="booking-form__title">New Appointment</h4>

      <label className="form-label">Type</label>
      <select className="form-select" value={type} onChange={e => setType(e.target.value)}>
        {APPT_TYPES.map(t => (
          <option key={t.value} value={t.value}>{t.label}</option>
        ))}
      </select>

      <div className="booking-row">
        <div>
          <label className="form-label">Date</label>
          <input
            className="form-input"
            type="date"
            value={date}
            min={todayISO()}
            onChange={e => setDate(e.target.value)}
          />
        </div>
        <div>
          <label className="form-label">Hour</label>
          <select className="form-select" value={hour} onChange={e => setHour(e.target.value)}>
            {Array.from({ length: 13 }, (_, i) => i + 8).map(h => (
              <option key={h} value={h}>{String(h).padStart(2, "0")}:00</option>
            ))}
          </select>
        </div>
        <div>
          <label className="form-label">Min</label>
          <select className="form-select" value={min} onChange={e => setMin(e.target.value)}>
            {[0, 15, 30, 45].map(m => (
              <option key={m} value={m}>{String(m).padStart(2, "0")}</option>
            ))}
          </select>
        </div>
      </div>

      <label className="form-label">Notes (optional)</label>
      <textarea
        className="form-textarea"
        rows={2}
        maxLength={500}
        placeholder="Any special requirements…"
        value={notes}
        onChange={e => setNotes(e.target.value)}
      />

      {error && <p className="form-error">{error}</p>}

      <button className="btn btn-accent" onClick={submit} disabled={busy || !date}>
        {busy ? "Booking…" : "Confirm Booking"}
      </button>
    </div>
  );
}
