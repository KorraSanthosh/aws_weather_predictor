import { useI18n } from "../i18n.jsx";

export default function MetricsCard({ m }) {
  const { t } = useI18n();
  const rows = [
    [t("m.model"), m.model, true],
    [t("m.persist"), m.baseline_persistence],
    [t("m.mean7"), m.baseline_7day_mean],
    ...(m.cams_global_model ? [[t("m.cams"), m.cams_global_model]] : []),
  ];
  const max = Math.max(...rows.map((r) => r[1].mae));
  const where = m.test_description || `${m.test_period?.[0]} to ${m.test_period?.[1]}`;
  const flag = Math.round(m.model.poor_or_worse_recall * 100);
  return (
    <section className="card">
      <h2>{t("m.title")}</h2>
      <p className="muted small">
        {t("m.note1", { n: m.model.n_days, where, acc: Math.round(m.model.category_accuracy * 100) })}
        {m.model.poor_or_worse_share != null
          ? t("m.note2a", { share: Math.round(m.model.poor_or_worse_share * 100), flag })
          : t("m.note2b", { flag })}
      </p>
      <div className="metrics">
        {rows.map(([label, v, ours]) => (
          <div className="mrow" key={label}>
            <div>{label}</div>
            <div className="mbar"><span className={ours ? "ours" : ""} style={{ width: `${(100 * v.mae) / max}%` }} /></div>
            <div>{v.mae.toFixed(1)}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
