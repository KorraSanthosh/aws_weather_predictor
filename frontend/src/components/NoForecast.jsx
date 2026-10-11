import { useI18n } from "../i18n.jsx";

// Shown instead of the forecast and decision cards when the API has no forecast stored yet (HTTP 404).
export default function NoForecast() {
  const { t } = useI18n();
  return (
    <section className="hero">
      <div className="card forecast">
        <h2>{t("nf.title")}</h2>
        <span className="pill">{t("nf.pill")}</span>
        <p className="muted">{t("nf.body")}</p>
      </div>
      <div className="card decision">
        <h2>{t("dec.title")}</h2>
        <div className="decision-word none">–</div>
        <p className="headline">{t("nf.headline")}</p>
        <p>{t("nf.advice")}</p>
      </div>
    </section>
  );
}
