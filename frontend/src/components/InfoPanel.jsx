/**
 * InfoPanel — customer info + appointments sidebar card.
 *
 * Props:
 *   session        — active session object (customer_id, customer_name, etc.)
 *   refreshTrigger — increment this number when UPDATE_INFO action fires so
 *                    the panel re-fetches automatically.
 */
import { useCallback, useEffect, useState } from "react";
import BookingForm from "./BookingForm.jsx";

const APPT_ICONS = {
  TEST_DRIVE:          "🚗",
  CALLBACK:            "📞",
  ADVANCE_PAYMENT:     "💳",
  DOCUMENT_SUBMISSION: "📄",
  OTHER:               "📅",
};

function AppointmentBadge({ appt, onCancel }) {
  const icon        = APPT_ICONS[appt.appointment_type] ?? "📅";
  const isCancelled = appt.status === "cancelled";
  return (
    <div className={`appt-card ${isCancelled ? "appt-card--cancelled" : ""}`}>
      <div className="appt-card__header">
        <span className="appt-icon">{icon}</span>
        <strong>{appt.appointment_type.replace(/_/g, " ")}</strong>
        <span className={`appt-status appt-status--${appt.status}`}>{appt.status}</span>
      </div>
      <div className="appt-card__detail">
        {appt.date} at {appt.time}
        {appt.notes && <span className="appt-notes"> · {appt.notes}</span>}
      </div>
      {!isCancelled && (
        <button className="btn btn-ghost btn-sm" onClick={() => onCancel(appt.appointment_id)}>
          Cancel
        </button>
      )}
    </div>
  );
}

export default function InfoPanel({ session, refreshTrigger = 0 }) {
  const [record,       setRecord]       = useState(null);
  const [appointments, setAppointments] = useState([]);
  const [loading,      setLoading]      = useState(false);
  const [saving,       setSaving]       = useState(false);
  const [saveMsg,      setSaveMsg]      = useState(null);
  const [error,        setError]        = useState(null);
  const [showAppts,    setShowAppts]    = useState(false);
  const [showForm,     setShowForm]     = useState(false);

  // Editable fields
  const [name,      setName]      = useState("");
  const [email,     setEmail]     = useState("");
  const [altMobile, setAltMobile] = useState("");

  const customerId = session?.customer_id;

  const fetchData = useCallback(async () => {
    if (!customerId) return;
    setLoading(true);
    try {
      const [recRes, apptRes] = await Promise.all([
        fetch(`/customers/${customerId}`).then(r => r.json()),
        fetch(`/customer/${customerId}/schedules`).then(r => r.json()),
      ]);
      const cust = recRes.record?.customer ?? {};
      setRecord(recRes.record);
      setName(cust.name ?? "");
      setEmail(cust.email ?? "");
      setAltMobile(cust.alt_mobile_no ?? "");
      setAppointments(apptRes.appointments ?? []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [customerId]);

  useEffect(() => { fetchData(); }, [fetchData]);

  // Re-fetch whenever chatbot performs UPDATE_INFO
  useEffect(() => {
    if (refreshTrigger > 0) fetchData();
  }, [refreshTrigger, fetchData]);

  const saveInfo = async () => {
    setSaving(true);
    setSaveMsg(null);
    setError(null);
    try {
      const res = await fetch(`/customer/${customerId}/info`, {
        method:  "PATCH",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          name:          name.trim()      || null,
          email:         email.trim()     || null,
          alt_mobile_no: altMobile.trim() || null,
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || `Error ${res.status}`);
      setSaveMsg(`✅ Saved: ${body.updated_fields.join(", ") || "nothing changed"}`);
      await fetchData();
      setTimeout(() => setSaveMsg(null), 4000);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const cancelAppt = async (apptId) => {
    try {
      const res = await fetch(`/customer/${customerId}/schedule/${apptId}/cancel`, {
        method: "PATCH",
      });
      if (!res.ok) throw new Error("Failed to cancel");
      await fetchData();
    } catch (e) {
      setError(e.message);
    }
  };

  const handleBooked = (appt) => {
    setAppointments(prev => [appt, ...prev]);
    setShowForm(false);
  };

  if (!session) {
    return (
      <div className="info-panel-empty">
        <div className="info-panel-empty__icon">👤</div>
        <p>Identify yourself in the chat to see your profile here.</p>
      </div>
    );
  }

  if (loading && !record) return <div className="info-panel-loading">Loading profile…</div>;

  const cust = record?.customer ?? {};
  const enq  = record?.enquiry  ?? {};
  const activeAppts = appointments.filter(a => a.status !== "cancelled");

  return (
    <div className="info-panel">
      {/* ── Customer Details ──────────────────── */}
      <div className="info-panel__header">
        <div className="info-panel__avatar">
          {(cust.name || session.customer_name || "?")[0].toUpperCase()}
        </div>
        <div>
          <div className="info-panel__name">{cust.name || session.customer_name}</div>
          <div className="info-panel__meta">ID: {cust.customer_id} · 📱 {cust.mobile_no}</div>
        </div>
      </div>

      {/* Editable fields */}
      <div className="info-grid info-grid--compact">
        <div>
          <label className="form-label">Full Name</label>
          <input
            className="form-input"
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Your name"
          />
        </div>
        <div>
          <label className="form-label">Email</label>
          <input
            className="form-input"
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            placeholder="you@example.com"
          />
        </div>
        <div>
          <label className="form-label">Alt. Mobile</label>
          <input
            className="form-input"
            value={altMobile}
            onChange={e => setAltMobile(e.target.value)}
            placeholder="10-digit"
            maxLength={10}
          />
        </div>
        <div>
          <label className="form-label">Interested In</label>
          <input
            className="form-input"
            readOnly
            value={`${enq.interested_make ?? ""} ${enq.interested_model ?? ""}`.trim() || "—"}
          />
        </div>
      </div>

      {error   && <p className="form-error">{error}</p>}
      {saveMsg && <p className="form-success">{saveMsg}</p>}

      <button className="btn btn-accent info-save-btn" onClick={saveInfo} disabled={saving}>
        {saving ? "Saving…" : "Save Changes"}
      </button>

      {/* ── Appointments ──────────────────────── */}
      <div className="info-appts-toggle" onClick={() => setShowAppts(v => !v)}>
        <span>
          📅 Appointments
          {activeAppts.length > 0 && (
            <span className="info-appts-badge">{activeAppts.length}</span>
          )}
        </span>
        <span className="info-toggle-arrow">{showAppts ? "▲" : "▼"}</span>
      </div>

      {showAppts && (
        <div className="info-appts-body">
          <div className="info-appts-actions">
            <button className="btn btn-ghost btn-sm" onClick={() => setShowForm(v => !v)}>
              {showForm ? "Close" : "+ New Appointment"}
            </button>
          </div>

          {showForm && (
            <BookingForm customerId={customerId} onBooked={handleBooked} />
          )}

          {appointments.length === 0 ? (
            <p className="info-empty">No appointments yet.</p>
          ) : (
            <div className="appt-list">
              {appointments.map(a => (
                <AppointmentBadge key={a.appointment_id} appt={a} onCancel={cancelAppt} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
