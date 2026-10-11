import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, Circle, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import { COLORS, DARK_TEXT } from "../lib.js";
import { useI18n } from "../i18n.jsx";

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const valueFor = (st, mode) =>
  mode === "today" ? (st.pm25 != null ? st : null) : st.tomorrow_est || null;

function icon(st, v, active, na) {
  const cls = "aqi-pin" + (v ? (DARK_TEXT.has(v.category) ? " dark" : "") : " na") + (active ? " active" : "");
  const style = v ? ` style="background:${COLORS[v.category]}"` : "";
  return L.divIcon({
    className: "",
    html: `<div class="${cls}"${style}>${v ? v.aqi : esc(na)}</div><span class="pin-name">${esc(st.short)}</span>`,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
  });
}

function View({ stations, selected }) {
  const map = useMap();
  useEffect(() => {
    const b = L.latLngBounds(stations.map((s) => [s.lat, s.lon]));
    map.fitBounds(b.pad(0.25));
  }, [map, stations]);
  useEffect(() => {
    const s = stations.find((x) => x.id === selected);
    if (s) map.flyTo([s.lat, s.lon], Math.max(map.getZoom(), 12), { duration: 0.6 });
  }, [map, stations, selected]);
  return null;
}

export default function StationMap({ stations, mode, selected, onSelect }) {
  const { t } = useI18n();
  const center = useMemo(() => [28.64, 77.2], []);
  const [base, setBase] = useState("street");
  return (
    <div className="map" data-base={base} role="region" aria-label={t("map.region")}>
      <div className="basemap" role="group" aria-label={t("map.bg")}>
        <button type="button" className={base === "street" ? "on" : ""} aria-pressed={base === "street"} onClick={() => setBase("street")}>{t("map.street")}</button>
        <button type="button" className={base === "satellite" ? "on" : ""} aria-pressed={base === "satellite"} onClick={() => setBase("satellite")}>{t("map.satellite")}</button>
      </div>
      <MapContainer center={center} zoom={11} scrollWheelZoom={false} style={{ height: "100%", width: "100%" }}>
        {base === "street" ? (
          <TileLayer key="street" attribution="© OpenStreetMap contributors" url="https://tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={17} />
        ) : (
          <TileLayer key="sat" attribution="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
                     url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" maxZoom={18} />
        )}
        <View stations={stations} selected={selected} />
        {stations.map((st) => {
          const v = valueFor(st, mode);
          const pos = [st.lat, st.lon];
          return (
            <span key={`${st.id}-${mode}`}>
              {v && <Circle center={pos} radius={3800} pathOptions={{ stroke: false, fillColor: COLORS[v.category], fillOpacity: 0.32 }} />}
              {v && <Circle center={pos} radius={1800} pathOptions={{ stroke: false, fillColor: COLORS[v.category], fillOpacity: 0.28 }} />}
              <Marker position={pos} icon={icon(st, v, selected === st.id, t("map.na"))} title={`${st.short}${v ? `: AQI ${v.aqi}` : ""}`}
                      eventHandlers={{ click: () => onSelect(st.id) }}>
                <Popup>
                  <div className="pop">
                    <h3>{st.short}</h3>
                    <p>{v ? t("map.pop", { aqi: v.aqi, cat: t(`cat.${v.category}`), pm: Math.round(v.pm25) }) : t("map.noData")}</p>
                  </div>
                </Popup>
              </Marker>
            </span>
          );
        })}
      </MapContainer>
      <div className="scale" aria-hidden="true">
        {Object.entries(COLORS).map(([k, c]) => (
          <span key={k}><i style={{ background: c }} />{t(`cat.${k}`)}</span>
        ))}
      </div>
    </div>
  );
}
