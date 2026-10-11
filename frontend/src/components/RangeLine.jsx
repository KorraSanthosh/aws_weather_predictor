import { categoryClass } from "../lib.js";
import { useI18n } from "../i18n.jsx";

// The calibrated 90 % range around the forecast (from the API, or web/data/range.json in the offline demo).
// It shows uncertainty only: the forecast, the AQI category and the school decision are not derived from it.
export default function RangeLine({ range: r }) {
  const { t } = useI18n();
  const single = r.category_low === r.category_high;
  return (
    <div className="range">
      <p className="range-line">
        <span className="range-label">{t("rng.label")}</span>
        <span className={`pill sm ${categoryClass(r.category_low)}`}>{t(`cat.${r.category_low}`)}</span>
        {!single && (
          <>
            <span className="range-to">{t("rng.to")}</span>
            <span className={`pill sm ${categoryClass(r.category_high)}`}>{t(`cat.${r.category_high}`)}</span>
          </>
        )}
        <span className="muted range-pm">{t("rng.pm", { lo: r.pm25_low, hi: r.pm25_high })}</span>
      </p>
      <p className="muted small">{t("rng.note")}</p>
    </div>
  );
}
