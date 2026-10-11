import { COLORS, fmtDate, categoryClass } from "../lib.js";
import { useI18n } from "../i18n.jsx";
import RangeLine from "./RangeLine.jsx";

export default function ForecastCard({ forecast: f }) {
  const { t, locale } = useI18n();
  const r = f.prediction_range;
  return (
    <div className="card forecast">
      <h2>
        {t("fc.title")} <span>{fmtDate(f.target_date, undefined, locale)}</span>
      </h2>
      <div className="aqi-row">
        <div className="aqi-num" style={{ color: COLORS[f.category] }}>{f.aqi}</div>
        <div>
          <span className={`pill ${categoryClass(f.category)}`}>{t(`cat.${f.category}`)}</span>
          {/* with the calibrated range shown below, the older "likely lo-hi" figures are not repeated (two ranges would confuse) */}
          <p className="muted">
            {r ? t("fc.pm", { pm: Math.round(f.pm25) })
               : t("fc.range", { pm: Math.round(f.pm25), lo: Math.round(f.pm25_low), hi: Math.round(f.pm25_high) })}
          </p>
        </div>
      </div>
      {r && <RangeLine range={r} />}
      {f.today_pm25 != null && <p className="muted small">{t("fc.today", { pm: Math.round(f.today_pm25) })}</p>}
    </div>
  );
}
