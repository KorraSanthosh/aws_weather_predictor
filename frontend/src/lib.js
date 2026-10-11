// Indian AQI categories and colours (CPCB) used across the dashboard.
export const CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"];
export const COLORS = {
  Good: "#009865",
  Satisfactory: "#7FBF3F",
  Moderate: "#E8C100",
  Poor: "#F28C28",
  "Very Poor": "#D7263D",
  Severe: "#7E0023",
};
export const DARK_TEXT = new Set(["Satisfactory", "Moderate", "Poor"]);

const API = (import.meta.env.VITE_KKH_API || "").replace(/\/$/, "");

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url} returned ${r.status}`);
  return r.json();
}

const local = (name) => `${import.meta.env.BASE_URL}data/${name}.json`;

export async function loadAll() {
  const metrics = getJSON(local("metrics")).catch(() => null);
  const stations = getJSON(local("stations")).catch(() => null);
  let forecast, history;
  if (API) {
    const d = await getJSON(`${API}/forecast`);
    forecast = d.forecast;
    history = d.history;
    if (!history || history.filter((h) => h.actual != null).length < 5) {
      history = await getJSON(local("history")).catch(() => history || []);
    }
  } else {
    [forecast, history] = await Promise.all([getJSON(local("forecast")), getJSON(local("history"))]);
  }
  return { forecast, history, metrics: await metrics, stations: await stations };
}

export function fmtDate(iso, opts, locale = "en-IN") {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(
    locale,
    opts || { weekday: "long", day: "numeric", month: "long" }
  );
}

export const categoryClass = (c) => (c || "").replace(/\s+/g, "");

// ---- local-area photo: looked up live from Wikipedia (CC-licensed thumbnails), cached per session ----
const photoCache = new Map();
const WIKI = "https://en.wikipedia.org/w/api.php";

export function fetchAreaPhoto(st) {
  const key = st.id;
  if (photoCache.has(key)) return photoCache.get(key);
  const params = new URLSearchParams({
    action: "query", format: "json", origin: "*", prop: "pageimages|info", piprop: "thumbnail",
    pithumbsize: "520", inprop: "url", redirects: "1",
  });
  if (st.wiki) params.set("titles", st.wiki);          // optional exact page set in station_coords.json
  else { params.set("generator", "search"); params.set("gsrlimit", "5"); params.set("gsrsearch", `${st.short} Delhi`); }
  const p = fetch(`${WIKI}?${params}`)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
    .then((d) => {
      const pages = Object.values(d?.query?.pages || {})
        .filter((x) => x.thumbnail?.source)
        .sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
      const best = pages[0];
      return best ? { src: best.thumbnail.source, title: best.title, url: best.fullurl } : null;
    })
    .catch(() => null);
  photoCache.set(key, p);
  return p;
}
