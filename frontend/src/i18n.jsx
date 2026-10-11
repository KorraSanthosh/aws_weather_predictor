import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

const KEY = "kkh-lang";

const STRINGS = {
  en: {
    "app.title": "Kal Ki Hawa — Tomorrow's air for Delhi schools",
    "app.eyebrow": "Air quality forecasting for schools",
    "app.tagline": "Tomorrow's air, today — a PM2.5 and AQI forecast for Delhi",
    "app.updated": "Updated {time}",
    "app.demo": "DEMO DATA — these numbers are synthetic test data, not real air-quality measurements.",
    "app.loading": "Loading forecast…",
    "app.error": "Could not load the forecast. {msg}",
    "hero.delhi": "Delhi · {date}",
    "hero.title": "Air quality tomorrow:",
    "hero.sub": "AQI {aqi} · PM2.5 about {pm} µg/m³ (likely {lo}–{hi}).",
    "hero.stationsDay": " Station map: latest complete day {date}.",
    "hero.decision": "School decision:",
    "hero.gaugeLabel": "AQI FORECAST",
    "hero.alt": "A hand holding a glass sphere with an air quality gauge reading AQI {aqi}, {cat}",
    "fc.title": "Forecast for",
    "fc.range": "PM2.5 about {pm} µg/m³ (likely {lo}–{hi})",
    "fc.today": "Latest full day measured: {pm} µg/m³ PM2.5.",
    "dec.title": "School decision",
    "decision.GO": "GO",
    "decision.CAUTION": "CAUTION",
    "decision.INDOOR": "INDOOR",
    "g.go.h": "Outdoor activities are fine",
    "g.go.a": "Normal assembly, sports and recess can go ahead.",
    "g.mod.h": "Keep outdoor time short",
    "g.mod.a": "Shorten outdoor sports and avoid strenuous exercise. Students with asthma or breathing problems should take it easy.",
    "g.poor.h": "Move activities indoors",
    "g.poor.a": "Hold PE, assembly and recess indoors. Limit time outside, especially for younger children and students with breathing problems.",
    "g.bad.h": "No outdoor activities",
    "g.bad.a": "Keep all activities indoors with windows closed where possible. Follow advisories from school management and the local authority on any closure.",
    "cat.Good": "Good", "cat.Satisfactory": "Satisfactory", "cat.Moderate": "Moderate",
    "cat.Poor": "Poor", "cat.Very Poor": "Very Poor", "cat.Severe": "Severe",
    "tiles.aria": "Station summary",
    "tiles.reporting": "Stations reporting", "tiles.reportingSub": "complete day",
    "tiles.highest": "Highest AQI", "tiles.lowest": "Lowest AQI",
    "tiles.poor": "Poor or worse", "tiles.poorSub": "stations right now",
    "map.title": "Air quality across Delhi",
    "map.note": "Coloured areas show the AQI around each monitoring station. Latest complete day: {date}.",
    "map.today": "Latest measured", "map.tomorrow": "Tomorrow (estimate)", "map.view": "Map view",
    "map.foot": "Station values are measured PM2.5 converted to the Indian AQI. The model forecasts the city average, so the tomorrow view scales each station by the city forecast; it is an estimate, not a per-station prediction. Station positions are approximate. Area photos are looked up from Wikipedia and may not show the exact monitor site.",
    "map.region": "Map of Delhi air quality stations",
    "map.bg": "Map background", "map.street": "Street", "map.satellite": "Satellite",
    "map.noData": "No complete reading for this day.",
    "map.pop": "AQI {aqi} · {cat} · PM2.5 {pm} µg/m³",
    "map.na": "n/a",
    "st.title": "Stations", "st.search": "Search station", "st.none": "No station matches.",
    "st.noReading": "No complete reading", "st.row": "{cat} · PM2.5 {pm}",
    "area.empty": "Pick a station on the map or in the list to see its area.",
    "area.loading": "Loading area photo…", "area.none": "No area photo found",
    "area.credit": "Photo: Wikipedia — {title}", "area.alt": "{name}, Delhi (photo from Wikipedia: {title})",
    "area.pm": "{cat} · PM2.5 {pm} µg/m³",
    "chart.title": "How good is the forecast?",
    "chart.desc": "Predicted vs actual daily PM2.5 (µg/m³) on days the model had never seen. Hover the chart to read a day.",
    "chart.empty": "Not enough history yet — the chart fills in as days pass.",
    "chart.aria": "Line chart of actual versus predicted daily PM2.5 over {n} days",
    "chart.threshold": "Poor and above (>90)",
    "chart.hint": "Move over the chart for exact values.",
    "chart.actual": "Actual", "chart.pred": "Our forecast", "chart.base": "Tomorrow = today",
    "chart.read": "{date} · Actual {a} · Our forecast {p}",
    "chart.readBase": " · Tomorrow = today {b}", "chart.unit": " µg/m³",
    "m.title": "Accuracy on unseen days",
    "m.note1": "Average error in µg/m³ (lower is better) over {n} days the model had not seen: {where}. It picks the right AQI category {acc}% of the time. ",
    "m.note2a": "Poor-or-worse days were {share}% of these days; the model flagged {flag}% of them.",
    "m.note2b": "It flagged {flag}% of Poor-or-worse days.",
    "m.model": "Our model", "m.persist": "Tomorrow = today", "m.mean7": "7-day average", "m.cams": "CAMS global model",
    "foot.data": "Data: ground-station PM2.5 from OpenAQ (open data on AWS), weather from Open-Meteo, map tiles © OpenStreetMap contributors, CAMS model benchmark. Built on AWS: S3, Lambda, EventBridge, DynamoDB, SNS, API Gateway, Amplify.",
    "foot.note": "Guidance is general and not medical advice. School closure decisions rest with school management and the local authority.",
    "theme.aria": "Colour theme", "theme.light": "Light", "theme.dark": "Dark", "theme.system": "System",
    "theme.title": "{name} theme", "lang.aria": "Language",
  },
  hi: {
    "app.title": "कल की हवा — दिल्ली के स्कूलों के लिए कल की वायु गुणवत्ता",
    "app.eyebrow": "स्कूलों के लिए वायु गुणवत्ता पूर्वानुमान",
    "app.tagline": "कल की हवा, आज ही — दिल्ली के लिए PM2.5 और AQI का पूर्वानुमान",
    "app.updated": "अपडेट: {time}",
    "app.demo": "डेमो डेटा — ये संख्याएँ नकली परीक्षण डेटा हैं, वास्तविक वायु गुणवत्ता माप नहीं।",
    "app.loading": "पूर्वानुमान लोड हो रहा है…",
    "app.error": "पूर्वानुमान लोड नहीं हो सका। {msg}",
    "hero.delhi": "दिल्ली · {date}",
    "hero.title": "कल की वायु गुणवत्ता:",
    "hero.sub": "AQI {aqi} · PM2.5 लगभग {pm} µg/m³ (संभावित {lo}–{hi})।",
    "hero.stationsDay": " स्टेशन मैप: नवीनतम पूरा दिन {date}।",
    "hero.decision": "स्कूल का निर्णय:",
    "hero.gaugeLabel": "AQI पूर्वानुमान",
    "hero.alt": "एक हाथ में काँच का गोला, जिसमें वायु गुणवत्ता गेज AQI {aqi}, {cat} दिखा रहा है",
    "fc.title": "पूर्वानुमान:",
    "fc.range": "PM2.5 लगभग {pm} µg/m³ (संभावित {lo}–{hi})",
    "fc.today": "अंतिम पूरे दिन का माप: {pm} µg/m³ PM2.5।",
    "dec.title": "स्कूल का निर्णय",
    "decision.GO": "बाहर ठीक",
    "decision.CAUTION": "सावधानी",
    "decision.INDOOR": "अंदर रहें",
    "g.go.h": "बाहर की गतिविधियाँ ठीक हैं",
    "g.go.a": "सामान्य प्रार्थना सभा, खेल और अवकाश हो सकते हैं।",
    "g.mod.h": "बाहर कम समय बिताएँ",
    "g.mod.a": "बाहरी खेल छोटे रखें और भारी व्यायाम से बचें। अस्थमा या साँस की समस्या वाले बच्चे आराम से रहें।",
    "g.poor.h": "गतिविधियाँ अंदर करें",
    "g.poor.a": "पीटी, प्रार्थना सभा और अवकाश अंदर कराएँ। बाहर समय सीमित रखें, खासकर छोटे बच्चों और साँस की समस्या वाले विद्यार्थियों के लिए।",
    "g.bad.h": "बाहर कोई गतिविधि नहीं",
    "g.bad.a": "सभी गतिविधियाँ अंदर रखें, संभव हो तो खिड़कियाँ बंद रखें। स्कूल बंद करने के बारे में स्कूल प्रबंधन और स्थानीय प्राधिकरण की सलाह मानें।",
    "cat.Good": "अच्छा", "cat.Satisfactory": "संतोषजनक", "cat.Moderate": "मध्यम",
    "cat.Poor": "खराब", "cat.Very Poor": "बहुत खराब", "cat.Severe": "गंभीर",
    "tiles.aria": "स्टेशनों का सारांश",
    "tiles.reporting": "रिपोर्ट करने वाले स्टेशन", "tiles.reportingSub": "पूरा दिन",
    "tiles.highest": "सबसे अधिक AQI", "tiles.lowest": "सबसे कम AQI",
    "tiles.poor": "खराब या उससे बदतर", "tiles.poorSub": "स्टेशन अभी",
    "map.title": "दिल्ली भर में वायु गुणवत्ता",
    "map.note": "रंगीन क्षेत्र हर निगरानी स्टेशन के आसपास का AQI दिखाते हैं। नवीनतम पूरा दिन: {date}।",
    "map.today": "नवीनतम मापा गया", "map.tomorrow": "कल (अनुमान)", "map.view": "मैप दृश्य",
    "map.foot": "स्टेशन के मान मापे गए PM2.5 से भारतीय AQI में बदले गए हैं। मॉडल पूरे शहर के औसत का पूर्वानुमान लगाता है, इसलिए \"कल\" दृश्य हर स्टेशन को शहर के पूर्वानुमान के अनुपात में बदलता है; यह अनुमान है, स्टेशन-स्तर का पूर्वानुमान नहीं। स्टेशनों की स्थिति लगभग सही है। क्षेत्र की तस्वीरें विकिपीडिया से ली जाती हैं और ज़रूरी नहीं कि वास्तविक मॉनिटर स्थल दिखाएँ।",
    "map.region": "दिल्ली के वायु गुणवत्ता स्टेशनों का मैप",
    "map.bg": "मैप की पृष्ठभूमि", "map.street": "सड़क", "map.satellite": "सैटेलाइट",
    "map.noData": "इस दिन का कोई पूरा माप नहीं।",
    "map.pop": "AQI {aqi} · {cat} · PM2.5 {pm} µg/m³",
    "map.na": "—",
    "st.title": "स्टेशन", "st.search": "स्टेशन खोजें", "st.none": "कोई स्टेशन नहीं मिला।",
    "st.noReading": "कोई पूरा माप नहीं", "st.row": "{cat} · PM2.5 {pm}",
    "area.empty": "इलाका देखने के लिए मैप या सूची में कोई स्टेशन चुनें।",
    "area.loading": "इलाके की तस्वीर लोड हो रही है…", "area.none": "इलाके की तस्वीर नहीं मिली",
    "area.credit": "तस्वीर: विकिपीडिया — {title}", "area.alt": "{name}, दिल्ली (विकिपीडिया की तस्वीर: {title})",
    "area.pm": "{cat} · PM2.5 {pm} µg/m³",
    "chart.title": "पूर्वानुमान कितना सटीक है?",
    "chart.desc": "उन दिनों के लिए अनुमानित बनाम वास्तविक दैनिक PM2.5 (µg/m³), जो मॉडल ने पहले कभी नहीं देखे थे। किसी दिन का मान पढ़ने के लिए चार्ट पर माउस ले जाएँ।",
    "chart.empty": "अभी पर्याप्त इतिहास नहीं है — जैसे-जैसे दिन बीतेंगे चार्ट भरता जाएगा।",
    "chart.aria": "{n} दिनों के वास्तविक और अनुमानित दैनिक PM2.5 का लाइन चार्ट",
    "chart.threshold": "खराब और ऊपर (>90)",
    "chart.hint": "सटीक मान देखने के लिए चार्ट पर माउस ले जाएँ।",
    "chart.actual": "वास्तविक", "chart.pred": "हमारा पूर्वानुमान", "chart.base": "कल = आज",
    "chart.read": "{date} · वास्तविक {a} · हमारा पूर्वानुमान {p}",
    "chart.readBase": " · कल = आज {b}", "chart.unit": " µg/m³",
    "m.title": "अनदेखे दिनों पर सटीकता",
    "m.note1": "औसत त्रुटि µg/m³ में (कम बेहतर है), {n} ऐसे दिनों पर जो मॉडल ने नहीं देखे थे। परीक्षण अवधि: {where}। यह {acc}% बार सही AQI श्रेणी चुनता है। ",
    "m.note2a": "खराब-या-बदतर दिन इन दिनों का {share}% थे; मॉडल ने उनमें से {flag}% पहचाने।",
    "m.note2b": "मॉडल ने खराब-या-बदतर दिनों में से {flag}% पहचाने।",
    "m.model": "हमारा मॉडल", "m.persist": "कल = आज", "m.mean7": "7 दिन का औसत", "m.cams": "CAMS वैश्विक मॉडल",
    "foot.data": "डेटा: OpenAQ (AWS पर खुला डेटा) से ज़मीनी स्टेशनों का PM2.5, Open-Meteo से मौसम, मैप टाइल © OpenStreetMap योगदानकर्ता, CAMS मॉडल बेंचमार्क। AWS पर बना: S3, Lambda, EventBridge, DynamoDB, SNS, API Gateway, Amplify।",
    "foot.note": "मार्गदर्शन सामान्य है, चिकित्सा सलाह नहीं। स्कूल बंद करने का निर्णय स्कूल प्रबंधन और स्थानीय प्राधिकरण का है।",
    "theme.aria": "रंग थीम", "theme.light": "हल्का", "theme.dark": "गहरा", "theme.system": "सिस्टम",
    "theme.title": "{name} थीम", "lang.aria": "भाषा",
  },
};

const readLang = () => {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "en" || v === "hi") return v;
  } catch { /* storage blocked */ }
  return (navigator.language || "").toLowerCase().startsWith("hi") ? "hi" : "en";
};

const Ctx = createContext(null);

export function I18nProvider({ children }) {
  const [lang, setLang] = useState(readLang);
  useEffect(() => {
    document.documentElement.lang = lang;
    document.title = STRINGS[lang]["app.title"];
    try { localStorage.setItem(KEY, lang); } catch { /* ignore */ }
  }, [lang]);

  const value = useMemo(() => {
    const dict = STRINGS[lang];
    const t = (key, vars) =>
      (dict[key] ?? STRINGS.en[key] ?? key).replace(/\{(\w+)\}/g, (_, k) => (vars && vars[k] != null ? vars[k] : `{${k}}`));
    const locale = lang === "hi" ? "hi-IN" : "en-IN";
    return { lang, setLang, t, locale };
  }, [lang]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useI18n = () => useContext(Ctx);

// Guidance text comes from the API in English; map it to translation keys by category.
export function guidanceKeys(category) {
  if (category === "Good" || category === "Satisfactory") return ["g.go.h", "g.go.a"];
  if (category === "Moderate") return ["g.mod.h", "g.mod.a"];
  if (category === "Poor") return ["g.poor.h", "g.poor.a"];
  return ["g.bad.h", "g.bad.a"];
}

export function LangToggle() {
  const { lang, setLang, t } = useI18n();
  return (
    <div className="theme" role="group" aria-label={t("lang.aria")}>
      <button type="button" className={lang === "en" ? "on" : ""} aria-pressed={lang === "en"} onClick={() => setLang("en")}>English</button>
      <button type="button" className={lang === "hi" ? "on" : ""} aria-pressed={lang === "hi"} onClick={() => setLang("hi")} lang="hi">हिन्दी</button>
    </div>
  );
}
