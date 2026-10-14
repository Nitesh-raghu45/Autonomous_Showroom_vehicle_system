/**
 * InventoryBrowser — live inventory panel.
 *
 * Props:
 *   autoFilters — { make, model, vtype, fuel, min_price, max_price } from the
 *                 chatbot's SEARCH_INVENTORY action. Applied automatically.
 *                 User can still override with manual controls.
 */
import { useCallback, useEffect, useRef, useState } from "react";

const FUEL_TYPES    = ["All", "Petrol", "Diesel", "CNG", "Electric", "Hybrid"];
const VEHICLE_TYPES = ["all", "new", "used"];

function VehicleCard({ v }) {
  const typeColor = v.type === "new" ? "badge--new" : "badge--used";
  return (
    <div className="vehicle-card">
      <div className="vehicle-card__header">
        <span className={`badge ${typeColor}`}>{v.type.toUpperCase()}</span>
        <span className="vehicle-card__year">{v.year}</span>
      </div>
      <div className="vehicle-card__make">{v.make}</div>
      <div className="vehicle-card__model">{v.model}</div>
      <div className="vehicle-card__variant">{v.variant}</div>

      <div className="vehicle-card__pills">
        <span className="pill">{v.fuel}</span>
        <span className="pill">{v.transmission}</span>
        <span className="pill">{v.color}</span>
      </div>

      {v.type === "used" && (
        <div className="vehicle-card__mileage">
          🛣 {v.mileage_km.toLocaleString()} km
        </div>
      )}

      <div className="vehicle-card__price">
        ₹ {(v.price / 100000).toFixed(2)} L
      </div>
      <div className="vehicle-card__id">Stock: {v.stock_id}</div>
    </div>
  );
}

export default function InventoryBrowser({ autoFilters = null }) {
  const [makes,   setMakes]   = useState([]);
  const [models,  setModels]  = useState([]);
  const [results, setResults] = useState(null);
  const [count,   setCount]   = useState(null);

  const [make,    setMake]    = useState("");
  const [model,   setModel]   = useState("");
  const [vtype,   setVtype]   = useState("all");
  const [fuel,    setFuel]    = useState("All");
  const [minP,    setMinP]    = useState("");
  const [maxP,    setMaxP]    = useState("");
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  const [autoTag, setAutoTag] = useState(null); // label shown when bot filtered

  // Track previous autoFilters to avoid re-applying same ones
  const prevAutoRef = useRef(null);

  // Load makes + count on mount
  useEffect(() => {
    Promise.all([
      fetch("/inventory/makes").then(r => r.json()),
      fetch("/inventory/count").then(r => r.json()),
    ]).then(([makesData, countData]) => {
      setMakes(makesData.makes ?? []);
      setCount(countData);
    }).catch(e => setError(e.message));
  }, []);

  // Load models when make changes
  useEffect(() => {
    setModel("");
    setModels([]);
    if (!make) return;
    fetch(`/inventory/${encodeURIComponent(make)}/models`)
      .then(r => r.json())
      .then(data => setModels(data.models ?? []))
      .catch(() => {});
  }, [make]);

  const search = useCallback(async (overrides = {}) => {
    setLoading(true);
    setError(null);
    const effectiveMake  = overrides.make  ?? make;
    const effectiveModel = overrides.model ?? model;
    const effectiveVtype = overrides.vtype ?? vtype;
    const effectiveFuel  = overrides.fuel  ?? fuel;
    const effectiveMinP  = overrides.minP  ?? minP;
    const effectiveMaxP  = overrides.maxP  ?? maxP;

    try {
      const params = new URLSearchParams();
      if (effectiveMake)                    params.set("make",  effectiveMake);
      if (effectiveModel)                   params.set("model", effectiveModel);
      if (effectiveVtype && effectiveVtype !== "all") params.set("type", effectiveVtype);
      if (effectiveFuel  && effectiveFuel  !== "All") params.set("fuel", effectiveFuel);
      if (effectiveMinP)  params.set("min_price", String(parseInt(effectiveMinP) * 100000));
      if (effectiveMaxP)  params.set("max_price", String(parseInt(effectiveMaxP) * 100000));

      const res  = await fetch(`/inventory/search?${params}`);
      const data = await res.json();
      setResults(data.results ?? []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [make, model, vtype, fuel, minP, maxP]);

  // Apply autoFilters from chatbot when they change
  useEffect(() => {
    if (!autoFilters) return;
    const key = JSON.stringify(autoFilters);
    if (prevAutoRef.current === key) return;
    prevAutoRef.current = key;

    // Apply to state
    const af = autoFilters;
    const newMake  = af.make  ?? "";
    const newModel = af.model ?? "";
    const newVtype = af.vtype ?? "all";
    const newFuel  = af.fuel  ?? "All";
    const newMinP  = af.min_price ? String(Math.round(af.min_price / 100000)) : "";
    const newMaxP  = af.max_price ? String(Math.round(af.max_price / 100000)) : "";

    setMake(newMake);
    setModel(newModel);
    setVtype(newVtype);
    setFuel(newFuel);
    setMinP(newMinP);
    setMaxP(newMaxP);

    // Build a readable tag
    const parts = [];
    if (newVtype !== "all") parts.push(newVtype);
    if (newMake)  parts.push(newMake);
    if (newModel) parts.push(newModel);
    if (newFuel !== "All") parts.push(newFuel);
    if (newMinP || newMaxP) parts.push(`₹${newMinP || "0"}–${newMaxP || "∞"}L`);
    setAutoTag(parts.length ? parts.join(" · ") : "all vehicles");

    // Trigger search with the new values directly (state update is async)
    search({
      make:  newMake,
      model: newModel,
      vtype: newVtype,
      fuel:  newFuel,
      minP:  newMinP,
      maxP:  newMaxP,
    });
  }, [autoFilters, search]);

  // Auto-search when manual filters change (but not when autoFilters triggered)
  useEffect(() => { search(); }, [make, model, vtype, fuel]); // eslint-disable-line react-hooks/exhaustive-deps

  const clearAuto = () => {
    setAutoTag(null);
    prevAutoRef.current = null;
    setMake(""); setModel(""); setVtype("all"); setFuel("All");
    setMinP(""); setMaxP("");
    setResults(null);
  };

  return (
    <div className="inv-browser">
      {/* Summary strip */}
      {count && (
        <div className="inv-summary">
          <span>📦 <strong>{count.total}</strong> vehicles</span>
          <span>🆕 {count.new_cars} new</span>
          <span>🔄 {count.used_cars} used</span>
          <span>🏷 {count.makes} makes</span>
        </div>
      )}

      {/* Auto-filter tag from chatbot */}
      {autoTag && (
        <div className="inv-auto-tag">
          <span>🤖 Filtered by chat: <strong>{autoTag}</strong></span>
          <button className="btn-clear" onClick={clearAuto} title="Clear chat filters">✕</button>
        </div>
      )}

      {/* Manual Filters */}
      <div className="inv-filters">
        <div className="filter-group">
          <label className="form-label">Make</label>
          <select className="form-select" value={make} onChange={e => setMake(e.target.value)}>
            <option value="">All Makes</option>
            {makes.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        <div className="filter-group">
          <label className="form-label">Model</label>
          <select className="form-select" value={model} onChange={e => setModel(e.target.value)} disabled={!make}>
            <option value="">All Models</option>
            {models.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>

        <div className="filter-group">
          <label className="form-label">Type</label>
          <div className="toggle-group">
            {VEHICLE_TYPES.map(t => (
              <button
                key={t}
                className={`toggle-btn ${vtype === t ? "active" : ""}`}
                onClick={() => setVtype(t)}
              >
                {t === "all" ? "All" : t === "new" ? "🆕 New" : "🔄 Used"}
              </button>
            ))}
          </div>
        </div>

        <div className="filter-group">
          <label className="form-label">Fuel</label>
          <select className="form-select" value={fuel} onChange={e => setFuel(e.target.value)}>
            {FUEL_TYPES.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>

        <div className="filter-group">
          <label className="form-label">Price (Lakh)</label>
          <div className="price-range">
            <input
              className="form-input"
              type="number"
              placeholder="Min"
              value={minP}
              onChange={e => setMinP(e.target.value)}
              min={0}
            />
            <span>–</span>
            <input
              className="form-input"
              type="number"
              placeholder="Max"
              value={maxP}
              onChange={e => setMaxP(e.target.value)}
              min={0}
            />
          </div>
        </div>

        <button className="btn btn-accent filter-search-btn" onClick={() => search()} disabled={loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </div>

      {error && <p className="form-error">{error}</p>}

      {/* Results */}
      {results === null ? (
        <p className="inv-hint">Select filters above to browse inventory.</p>
      ) : results.length === 0 ? (
        <p className="inv-hint">No vehicles match your filters.</p>
      ) : (
        <>
          <p className="inv-count">{results.length} vehicle{results.length !== 1 ? "s" : ""} found</p>
          <div className="vehicle-grid">
            {results.map(v => <VehicleCard key={v.stock_id} v={v} />)}
          </div>
        </>
      )}
    </div>
  );
}
