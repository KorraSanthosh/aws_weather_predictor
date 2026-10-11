import { COLORS } from "../lib.js";
import { useI18n } from "../i18n.jsx";

export default function StatTiles({ stations }) {
  const { t } = useI18n();
  if (!stations) return null;
  const withData = stations.stations.filter((s) => s.pm25 != null);
  if (!withData.length) return null;
  const sorted = [...withData].sort((a, b) => b.aqi - a.aqi);
  const worst = sorted[0];
  const best = sorted[sorted.length - 1];
  const tiles = [
    { label: t("tiles.reporting"), value: `${withData.length}/${stations.stations.length}`, sub: t("tiles.reportingSub") },
    { label: t("tiles.highest"), value: worst.aqi, sub: worst.short, color: COLORS[worst.category] },
    { label: t("tiles.lowest"), value: best.aqi, sub: best.short, color: COLORS[best.category] },
    { label: t("tiles.poor"), value: withData.filter((s) => s.aqi > 200).length, sub: t("tiles.poorSub") },
  ];
  return (
    <section className="tiles" aria-label={t("tiles.aria")}>
      {tiles.map((x) => (
        <div className="tile" key={x.label}>
          <span className="tile-label">{x.label}</span>
          <span className="tile-value" style={x.color ? { color: x.color } : undefined}>{x.value}</span>
          <span className="tile-sub">{x.sub}</span>
        </div>
      ))}
    </section>
  );
}
