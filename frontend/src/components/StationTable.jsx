import { useMemo, useState } from "react";
import { COLORS, DARK_TEXT } from "../lib.js";
import { valueFor } from "./StationMap.jsx";
import { useI18n } from "../i18n.jsx";

export default function StationTable({ stations, mode, selected, onSelect }) {
  const { t } = useI18n();
  const [q, setQ] = useState("");
  const rows = useMemo(() => {
    const term = q.trim().toLowerCase();
    return stations
      .filter((s) => !term || s.name.toLowerCase().includes(term))
      .map((s) => ({ s, v: valueFor(s, mode) }))
      .sort((a, b) => (b.v?.aqi ?? -1) - (a.v?.aqi ?? -1));
  }, [stations, mode, q]);

  return (
    <div className="stations">
      <div className="stations-head">
        <h3>{t("st.title")}</h3>
        <input type="search" placeholder={t("st.search")} value={q} onChange={(e) => setQ(e.target.value)} aria-label={t("st.search")} />
      </div>
      <ul>
        {rows.map(({ s, v }) => (
          <li key={s.id}>
            <button type="button" className={selected === s.id ? "row sel" : "row"} onClick={() => onSelect(s.id)}>
              <span className="chip" style={v ? { background: COLORS[v.category], color: DARK_TEXT.has(v.category) ? "#1c2321" : "#fff" } : undefined}>
                {v ? v.aqi : t("map.na")}
              </span>
              <span className="row-main">
                <b>{s.short}</b>
                <small>{v ? t("st.row", { cat: t(`cat.${v.category}`), pm: Math.round(v.pm25) }) : t("st.noReading")}</small>
              </span>
            </button>
          </li>
        ))}
        {!rows.length && <li className="muted small pad">{t("st.none")}</li>}
      </ul>
    </div>
  );
}
