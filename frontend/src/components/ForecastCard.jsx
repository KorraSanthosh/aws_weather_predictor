import { COLORS, fmtDate, categoryClass } from "../lib.js";
import { useI18n } from "../i18n.jsx";

export default function ForecastCard({ forecast: f }) {
  const { t, locale } = useI18n();
  return (
    <div className="card forecast">
      <h2>
        {t("fc.title")} <span>{fmtDate(f.target_date, undefined, locale)}</span>
      </h2>
      <div className="aqi-row">
        <div className="aqi-num" style={{ color: COLORS[f.category] }}>{f.aqi}</div>
        <div>
          <span className={`pill ${categoryClass(f.category)}`}>{t(`cat.${f.category}`)}</span>
          <p className="muted">{t("fc.range", { pm: Math.round(f.pm25), lo: Math.round(f.pm25_low), hi: Math.round(f.pm25_high) })}</p>
        </div>
      </div>
      {f.today_pm25 != null && <p className="muted small">{t("fc.today", { pm: Math.round(f.today_pm25) })}</p>}
    </div>
  );
}
