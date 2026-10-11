import { useMemo, useState } from "react";
import { useI18n } from "../i18n.jsx";

const W = 800, H = 300, L = 44, R = 12, T = 12, B = 28;

export default function TrendChart({ rows }) {
  const { t, locale } = useI18n();
  const [hover, setHover] = useState(null);
  const pts = useMemo(() => (rows || []).filter((r) => r.actual != null && r.predicted != null), [rows]);

  const geo = useMemo(() => {
    if (pts.length < 2) return null;
    const maxRaw = Math.max(...pts.map((r) => Math.max(r.actual, r.predicted, r.baseline || 0)));
    const maxV = Math.ceil(Math.max(maxRaw, 100) / 50) * 50;
    const x = (i) => L + ((W - L - R) * i) / (pts.length - 1);
    const y = (v) => T + (H - T - B) * (1 - v / maxV);
    const path = (key) => pts.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(r[key]).toFixed(1)}`).join(" ");
    return { maxV, x, y, path };
  }, [pts]);

  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.round(((px - L) / (W - L - R)) * (pts.length - 1));
    setHover(Math.min(pts.length - 1, Math.max(0, i)));
  };

  const ticks = [];
  if (geo) for (let t = 0; t <= geo.maxV; t += 50) ticks.push(t);
  const h = hover != null ? pts[hover] : null;
  const fmt = (iso) => new Date(`${iso}T00:00:00`).toLocaleDateString(locale, { day: "numeric", month: "short", year: "numeric" });

  return (
    <section className="card">
      <h2>{t("chart.title")}</h2>
      <p className="muted">{t("chart.desc")}</p>
      {!geo ? (
        <p className="muted">{t("chart.empty")}</p>
      ) : (
        <div className="chart">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t("chart.aria", { n: pts.length })}
               onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={L} x2={W - R} y1={geo.y(t)} y2={geo.y(t)} stroke="currentColor" strokeOpacity="0.12" />
                <text x={L - 6} y={geo.y(t) + 4} textAnchor="end">{t}</text>
              </g>
            ))}
            <line x1={L} x2={W - R} y1={geo.y(90)} y2={geo.y(90)} stroke="#F28C28" strokeDasharray="5 4" strokeWidth="1.5" />
            <text x={L + 6} y={geo.y(90) - 5}>{t("chart.threshold")}</text>
            {[0, Math.floor(pts.length / 2), pts.length - 1].map((i) => (
              <text key={i} x={geo.x(i)} y={H - 8} textAnchor={i === 0 ? "start" : i === pts.length - 1 ? "end" : "middle"}>
                {new Date(`${pts[i].date}T00:00:00`).toLocaleDateString(locale, { day: "numeric", month: "short" })}
              </text>
            ))}
            {pts[0].baseline != null && <path d={geo.path("baseline")} fill="none" className="ln-base" strokeDasharray="5 4" strokeWidth="1.5" />}
            <path d={geo.path("actual")} fill="none" className="ln-actual" strokeWidth="2" strokeLinejoin="round" />
            <path d={geo.path("predicted")} fill="none" stroke="#1f9d7f" strokeWidth="2.5" strokeLinejoin="round" />
            {h && (
              <g>
                <line x1={geo.x(hover)} x2={geo.x(hover)} y1={T} y2={H - B} stroke="currentColor" strokeOpacity="0.35" />
                <circle cx={geo.x(hover)} cy={geo.y(h.actual)} r="4" className="dot-actual" />
                <circle cx={geo.x(hover)} cy={geo.y(h.predicted)} r="4" fill="#1f9d7f" />
              </g>
            )}
          </svg>
          <div className="readout" aria-live="polite">
            {h ? (
              <>
                <b>{t("chart.read", { date: fmt(h.date), a: Math.round(h.actual), p: Math.round(h.predicted) })}</b>
                {h.baseline != null && t("chart.readBase", { b: Math.round(h.baseline) })}
                {t("chart.unit")}
              </>
            ) : (
              t("chart.hint")
            )}
          </div>
        </div>
      )}
      <div className="legend" aria-hidden="true">
        <span><i className="sw actual" />{t("chart.actual")}</span>
        <span><i className="sw pred" />{t("chart.pred")}</span>
        <span><i className="sw base" />{t("chart.base")}</span>
      </div>
    </section>
  );
}
