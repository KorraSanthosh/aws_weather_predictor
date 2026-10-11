import { useEffect, useState } from "react";
import { loadAll, fmtDate } from "./lib.js";
import { useI18n, LangToggle } from "./i18n.jsx";
import ForecastCard from "./components/ForecastCard.jsx";
import DecisionCard from "./components/DecisionCard.jsx";
import StatTiles from "./components/StatTiles.jsx";
import StationMap, { valueFor } from "./components/StationMap.jsx";
import StationTable from "./components/StationTable.jsx";
import AreaCard from "./components/AreaCard.jsx";
import TrendChart from "./components/TrendChart.jsx";
import MetricsCard from "./components/MetricsCard.jsx";
import AirHero from "./components/AirHero.jsx";
import NoForecast from "./components/NoForecast.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";
import { useTheme } from "./theme.js";

export default function App() {
  const { t, locale } = useI18n();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("today");
  const [selected, setSelected] = useState(null);
  const [theme, setTheme] = useTheme();

  useEffect(() => {
    loadAll().then(setData).catch((e) => setError(e.message));
  }, []);

  const f = data?.forecast;
  const list = data?.stations?.stations;
  const sel = list?.find((s) => s.id === selected);
  return (
    <>
      <header className="top">
        <div className="wrap top-inner">
          <div>
            <p className="eyebrow">{t("app.eyebrow")}</p>
            <h1>
              {locale === "hi-IN" ? (
                <>कल की <span className="grad">हवा</span></>
              ) : (
                <>Kal Ki <span className="grad">Hawa</span></>
              )}
            </h1>
            <p className="tag">{t("app.tagline")}</p>
          </div>
          <div className="top-right">
            <div className="top-toggles">
              <LangToggle />
              <ThemeToggle mode={theme} onChange={setTheme} />
            </div>
            {f?.generated_at && <p className="stamp">{t("app.updated", { time: new Date(f.generated_at).toLocaleString(locale) })}</p>}
          </div>
        </div>
      </header>

      {f?.data_source === "synthetic" && <div className="banner">{t("app.demo")}</div>}
      {error && <div className="banner error">{t("app.error", { msg: error })}</div>}
      {!data && !error && <p className="wrap muted loading">{t("app.loading")}</p>}

      {data && (
        <main className="wrap">
          {data.noForecast ? (
            <NoForecast />
          ) : (
            <>
              <AirHero forecast={f} stationsDate={data.stations?.date} />
              <section className="hero">
                <ForecastCard forecast={f} />
                <DecisionCard guidance={f.guidance} category={f.category} />
              </section>
            </>
          )}

          <StatTiles stations={data.stations} />

          {data.stations && (
            <section className="card map-card" aria-labelledby="map-title">
              <div className="map-head">
                <div>
                  <h2 id="map-title">{t("map.title")}</h2>
                  <p className="muted small">{t("map.note", { date: fmtDate(data.stations.date, undefined, locale) })}</p>
                </div>
                <div className="toggle" role="group" aria-label={t("map.view")}>
                  <button type="button" className={mode === "today" ? "on" : ""} aria-pressed={mode === "today"} onClick={() => setMode("today")}>
                    {t("map.today")}
                  </button>
                  {list.some((s) => s.tomorrow_est) && (
                    <button type="button" className={mode === "tomorrow" ? "on" : ""} aria-pressed={mode === "tomorrow"} onClick={() => setMode("tomorrow")}>
                      {t("map.tomorrow")}
                    </button>
                  )}
                </div>
              </div>
              <div className="map-grid">
                <StationMap stations={list} mode={mode} selected={selected} onSelect={setSelected} />
                <div className="side">
                  <StationTable stations={list} mode={mode} selected={selected} onSelect={setSelected} />
                  <AreaCard station={sel} value={sel ? valueFor(sel, mode) : null} />
                </div>
              </div>
              <p className="muted small">{t("map.foot")}</p>
            </section>
          )}

          <TrendChart rows={data.history} />
          {data.metrics && <MetricsCard m={data.metrics} />}
        </main>
      )}

      <footer className="wrap foot">
        <p>{t("foot.data")}</p>
        <p>{t("foot.note")}</p>
      </footer>
    </>
  );
}
