import { COLORS, CATEGORIES, fmtDate } from "../lib.js";
import { useI18n } from "../i18n.jsx";

// Indian AQI bands (upper bounds) -> one gauge segment each.
const BANDS = [50, 100, 200, 300, 400, 500];
const CX = 270, CY = 138, R = 80, START = 150, SEG = 40;      // 240-degree gauge, open at the bottom

const pt = (deg, r = R) => [CX + r * Math.cos((deg * Math.PI) / 180), CY + r * Math.sin((deg * Math.PI) / 180)];
const arc = (a, b) => {
  const [x1, y1] = pt(a), [x2, y2] = pt(b);
  return `M${x1.toFixed(1)} ${y1.toFixed(1)} A${R} ${R} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`;
};

function angleFor(aqi) {
  const v = Math.max(0, Math.min(500, aqi));
  let i = BANDS.findIndex((b) => v <= b);
  if (i < 0) i = BANDS.length - 1;
  const lo = i === 0 ? 0 : BANDS[i - 1];
  return START + i * SEG + ((v - lo) / (BANDS[i] - lo)) * SEG;
}

export default function AirHero({ forecast: f, stationsDate }) {
  const { t, locale } = useI18n();
  const catName = t(`cat.${f.category}`);
  const [mx, my] = pt(angleFor(f.aqi));
  const color = COLORS[f.category];
  return (
    <section className="air-hero" aria-label={t("hero.title")}>
      <div className="hero-bokeh" aria-hidden="true" />
      <div className="hero-copy">
        <p className="eyebrow">{t("hero.delhi", { date: fmtDate(f.target_date, undefined, locale) })}</p>
        <h2 className="hero-title">{t("hero.title")}<br /><span style={{ color }}>{catName}</span></h2>
        <p className="hero-sub">
          {t("hero.sub", { aqi: f.aqi, pm: Math.round(f.pm25), lo: Math.round(f.pm25_low), hi: Math.round(f.pm25_high) })}
          {stationsDate ? t("hero.stationsDay", { date: fmtDate(stationsDate, { day: "numeric", month: "short" }, locale) }) : ""}
        </p>
        <p className={`hero-decision ${f.guidance.decision}`}>{t("hero.decision")} <b>{t(`decision.${f.guidance.decision}`)}</b></p>
      </div>

      <svg className="hero-art" viewBox="0 0 560 390" role="img"
           aria-label={t("hero.alt", { aqi: f.aqi, cat: catName })}>
        <defs>
          <linearGradient id="skin" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#f2c2a2" /><stop offset="1" stopColor="#d9976f" />
          </linearGradient>
          <radialGradient id="glass" cx="35%" cy="28%" r="80%">
            <stop offset="0" stopColor="#ffffff" stopOpacity=".55" />
            <stop offset=".55" stopColor="#ffffff" stopOpacity=".14" />
            <stop offset="1" stopColor="#bfe3d4" stopOpacity=".28" />
          </radialGradient>
          <filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6" /></filter>
        </defs>

        {/* hand: stacked round-capped strokes, palm up, fingers curling toward the sphere */}
        <g fill="none" strokeLinecap="round" strokeLinejoin="round">
          <ellipse cx="300" cy="372" rx="190" ry="10" fill="#000" opacity=".12" filter="url(#soft)" stroke="none" />
          <path d="M620 332 C520 340 424 330 352 304" stroke="url(#skin)" strokeWidth="78" />
          <path d="M352 296 C292 270 214 256 128 262" stroke="#cf8f68" strokeWidth="26" />
          <path d="M356 310 C292 294 206 286 112 292" stroke="url(#skin)" strokeWidth="26" />
          <path d="M360 324 C300 318 220 318 132 324" stroke="#cf8f68" strokeWidth="24" />
          <path d="M366 338 C312 340 252 346 186 352" stroke="url(#skin)" strokeWidth="21" />
          <path d="M410 282 C392 258 360 240 324 238" stroke="url(#skin)" strokeWidth="30" />
          <path d="M350 300 C292 284 214 276 128 280" stroke="#fff" strokeOpacity=".2" strokeWidth="5" />
          <path d="M356 314 C294 302 206 300 118 306" stroke="#fff" strokeOpacity=".22" strokeWidth="5" />
        </g>

        {/* glass sphere */}
        <circle cx={CX} cy={CY} r="124" fill="url(#glass)" stroke="#fff" strokeOpacity=".7" strokeWidth="2" />
        <circle cx={CX} cy={CY} r="104" fill="#0b1f1a" fillOpacity=".34" stroke="#fff" strokeOpacity=".35" />
        <ellipse cx="205" cy="62" rx="26" ry="12" fill="#fff" opacity=".55" transform="rotate(-35 205 62)" />
        <circle cx="187" cy="95" r="3" fill="#fff" opacity=".8" />

        {/* gauge */}
        {CATEGORIES.map((c, i) => (
          <path key={c} d={arc(START + i * SEG + 1.5, START + (i + 1) * SEG - 1.5)} stroke={COLORS[c]} strokeWidth="14"
                strokeLinecap="round" fill="none" opacity={c === f.category ? 1 : 0.8} />
        ))}
        <circle cx={mx} cy={my} r="9" fill="#fff" stroke={color} strokeWidth="5" />
        <text x={CX} y={CY + 6} textAnchor="middle" className="g-num">{f.aqi}</text>
        <text x={CX} y={CY + 30} textAnchor="middle" className="g-cat">{catName}</text>
        <text x={CX} y={CY - 38} textAnchor="middle" className="g-lbl">{t("hero.gaugeLabel")}</text>
      </svg>
    </section>
  );
}
