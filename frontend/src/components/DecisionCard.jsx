import { useI18n, guidanceKeys } from "../i18n.jsx";

export default function DecisionCard({ guidance: g, category }) {
  const { t } = useI18n();
  const [h, a] = guidanceKeys(category);
  return (
    <div className="card decision">
      <h2>{t("dec.title")}</h2>
      <div className={`decision-word ${g.decision}`}>{t(`decision.${g.decision}`)}</div>
      <p className="headline">{t(h)}</p>
      <p>{t(a)}</p>
    </div>
  );
}
