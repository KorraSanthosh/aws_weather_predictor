import { useEffect, useState } from "react";
import { COLORS, fetchAreaPhoto } from "../lib.js";
import { useI18n } from "../i18n.jsx";

export default function AreaCard({ station: st, value: v }) {
  const { t } = useI18n();
  const [photo, setPhoto] = useState(undefined);       // undefined = loading, null = none found

  useEffect(() => {
    let live = true;
    setPhoto(undefined);
    if (st) fetchAreaPhoto(st).then((p) => live && setPhoto(p));
    return () => { live = false; };
  }, [st]);

  if (!st) {
    return <div className="area empty muted small">{t("area.empty")}</div>;
  }
  return (
    <div className="area">
      <div className="area-photo" aria-busy={photo === undefined}>
        {photo ? (
          <img src={photo.src} alt={t("area.alt", { name: st.short, title: photo.title })} loading="lazy" />
        ) : (
          <div className="area-ph muted small">{photo === undefined ? t("area.loading") : t("area.none")}</div>
        )}
        {v && <span className="area-badge" style={{ background: COLORS[v.category] }}>{v.aqi}</span>}
      </div>
      <div className="area-body">
        <b>{st.short}</b>
        <span className="muted small">{v ? t("area.pm", { cat: t(`cat.${v.category}`), pm: Math.round(v.pm25) }) : t("st.noReading")}</span>
        {photo && (
          <a className="small" href={photo.url} target="_blank" rel="noreferrer noopener">
            {t("area.credit", { title: photo.title })}
          </a>
        )}
      </div>
    </div>
  );
}
