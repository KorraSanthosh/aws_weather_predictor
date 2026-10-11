(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var API = (window.KKH_API || "").replace(/\/$/, "");

  function getJSON(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error(url + " returned " + r.status);
      return r.json();
    });
  }

  function showError(msg) { var e = $("error"); e.textContent = msg; e.hidden = false; }

  function fmtDate(iso) {
    var d = new Date(iso + "T00:00:00");
    return d.toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "long" });
  }

  function render(forecast, history, metrics) {
    $("target-date").textContent = fmtDate(forecast.target_date);
    $("aqi").textContent = forecast.aqi;
    var cat = $("category");
    cat.textContent = forecast.category;
    cat.className = "pill " + forecast.category.split(" ")[0];
    var pr = forecast.prediction_range;
    if (pr) {
      // the calibrated range is shown below; the older "likely lo-hi" figures are not repeated (two ranges would confuse)
      $("pm-range").textContent = "PM2.5 about " + Math.round(forecast.pm25) + " µg/m³";
      drawRange(pr);
    } else {
      $("pm-range").textContent = "PM2.5 about " + Math.round(forecast.pm25) + " µg/m³ (likely " +
        Math.round(forecast.pm25_low) + "–" + Math.round(forecast.pm25_high) + ")";
    }
    if (forecast.today_pm25 != null) {
      $("today-line").textContent = "Latest full day measured: " + Math.round(forecast.today_pm25) + " µg/m³ PM2.5.";
    }
    var g = forecast.guidance;
    var dec = $("decision");
    dec.textContent = g.decision; dec.className = "decision-word " + g.decision;
    $("headline").textContent = g.headline;
    $("advice").textContent = g.advice;
    if (forecast.data_source === "synthetic") $("demo-banner").hidden = false;
    if (forecast.generated_at) {
      $("stamp").textContent = "Updated " + new Date(forecast.generated_at).toLocaleString("en-IN");
    }
    drawChart(history || []);
    if (metrics) drawMetrics(metrics);
  }

  // Calibrated 90 % range around the forecast: uncertainty only; the category and school decision do not come from it.
  function drawRange(pr) {
    var host = $("range-line");
    host.textContent = "";
    function add(text, cls) {
      var s = document.createElement("span");
      s.textContent = text; if (cls) s.className = cls;
      host.appendChild(s); return s;
    }
    function pill(cat) { add(cat, "pill sm " + cat.split(" ")[0]); }
    add("Likely range:", "range-label");
    pill(pr.category_low);
    if (pr.category_high !== pr.category_low) { add("to", "range-to"); pill(pr.category_high); }
    add("(PM2.5 " + pr.pm25_low + "–" + pr.pm25_high + " µg/m³)", "muted range-pm");
    $("range").hidden = false;
  }

  function drawChart(rows) {
    var host = $("chart");
    host.textContent = "";
    var pts = rows.filter(function (r) { return r.actual != null && r.predicted != null; });
    if (pts.length < 2) { host.textContent = "Not enough history yet — the chart fills in as days pass."; return; }

    var W = 800, H = 300, L = 44, R = 12, T = 12, B = 28;
    var maxV = Math.max.apply(null, pts.map(function (r) { return Math.max(r.actual, r.predicted, r.baseline || 0); }));
    maxV = Math.ceil(Math.max(maxV, 100) / 50) * 50;
    var x = function (i) { return L + (W - L - R) * i / (pts.length - 1); };
    var y = function (v) { return T + (H - T - B) * (1 - v / maxV); };
    var NS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Line chart of actual versus predicted daily PM2.5 over " + pts.length + " days");

    function el(name, attrs, text) {
      var n = document.createElementNS(NS, name);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      if (text != null) n.textContent = text;
      svg.appendChild(n); return n;
    }
    for (var t = 0; t <= maxV; t += 50) {
      el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: "currentColor", "stroke-opacity": 0.12 });
      el("text", { x: L - 6, y: y(t) + 4, "text-anchor": "end" }, t);
    }
    el("line", { x1: L, x2: W - R, y1: y(90), y2: y(90), stroke: "#F28C28", "stroke-dasharray": "5 4", "stroke-width": 1.5 });
    el("text", { x: W - R - 4, y: y(90) - 5, "text-anchor": "end" }, "Poor and above (>90)");
    [0, Math.floor(pts.length / 2), pts.length - 1].forEach(function (i) {
      el("text", { x: x(i), y: H - 8, "text-anchor": i === 0 ? "start" : (i === pts.length - 1 ? "end" : "middle") },
        new Date(pts[i].date + "T00:00:00").toLocaleDateString("en-IN", { day: "numeric", month: "short" }));
    });
    function path(key, color, dash, width) {
      var d = pts.map(function (r, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(r[key]).toFixed(1); }).join(" ");
      el("path", { d: d, fill: "none", stroke: color, "stroke-width": width, "stroke-dasharray": dash || "none",
                   "stroke-linejoin": "round", "stroke-linecap": "round" });
    }
    if (pts[0].baseline != null) path("baseline", "#8a948f", "5 4", 1.5);
    path("actual", getComputedStyle(document.body).color, null, 2);
    path("predicted", "#1f9d7f", null, 2.5);
    host.appendChild(svg);
  }

  function drawMetrics(m) {
    var rows = [
      ["Our model", m.model, true],
      ["Tomorrow = today", m.baseline_persistence, false],
      ["7-day average", m.baseline_7day_mean, false]
    ];
    if (m.cams_global_model) rows.push(["CAMS global model", m.cams_global_model, false]);
    var max = Math.max.apply(null, rows.map(function (r) { return r[1].mae; }));
    var host = $("metrics"); host.textContent = "";
    rows.forEach(function (r) {
      var row = document.createElement("div"); row.className = "mrow";
      var a = document.createElement("div"); a.textContent = r[0];
      var bar = document.createElement("div"); bar.className = "mbar";
      var s = document.createElement("span"); s.style.width = (100 * r[1].mae / max) + "%";
      if (r[2]) s.className = "ours";
      bar.appendChild(s);
      var v = document.createElement("div"); v.textContent = r[1].mae.toFixed(1);
      row.appendChild(a); row.appendChild(bar); row.appendChild(v); host.appendChild(row);
    });
    var where = m.test_description || (m.test_period[0] + " to " + m.test_period[1]);
    $("metrics-note").textContent = "Average error in µg/m³ (lower is better) over " + m.model.n_days +
      " days the model had not seen: " + where + ". It picks the right AQI category " +
      Math.round(m.model.category_accuracy * 100) + "% of the time. " +
      (m.model.poor_or_worse_share != null
        ? "Poor-or-worse days were " + Math.round(m.model.poor_or_worse_share * 100) + "% of these days; the model flagged " +
          Math.round(m.model.poor_or_worse_recall * 100) + "% of them."
        : "It flagged " + Math.round(m.model.poor_or_worse_recall * 100) + "% of Poor-or-worse days.");
    $("metrics-card").hidden = false;
  }


  // ---------- Delhi map: station AQI markers + coloured zones ----------
  var AQI_COLORS = { "Good": "#009865", "Satisfactory": "#7FBF3F", "Moderate": "#E8C100",
                     "Poor": "#F28C28", "Very Poor": "#D7263D", "Severe": "#7E0023" };
  var DARK_TEXT = { "Satisfactory": 1, "Moderate": 1, "Poor": 1 };

  function drawScale() {
    var host = $("scale"); host.textContent = "";
    Object.keys(AQI_COLORS).forEach(function (k) {
      var s = document.createElement("span"), i = document.createElement("i");
      i.style.background = AQI_COLORS[k]; s.appendChild(i); s.appendChild(document.createTextNode(k)); host.appendChild(s);
    });
  }

  function initMap(data) {
    if (!window.L || !data || !data.stations || !data.stations.length) {
      $("map-card").hidden = true; return;
    }
    drawScale();
    var map = L.map("map", { scrollWheelZoom: false }).setView([28.64, 77.2], 11);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      { maxZoom: 17, attribution: "© OpenStreetMap contributors" }).addTo(map);
    var layer = L.layerGroup().addTo(map);
    var mode = "today";

    function valueOf(st) { return mode === "today" ? (st.pm25 != null ? st : null) : (st.tomorrow_est || null); }

    function popup(st) {
      var v = valueOf(st), box = document.createElement("div"); box.className = "pop";
      var h = document.createElement("h3"); h.textContent = st.short; box.appendChild(h);
      var p = document.createElement("p");
      p.textContent = v ? "AQI " + v.aqi + " · " + v.category + " · PM2.5 " + Math.round(v.pm25) + " µg/m³"
                        : "No complete reading for this day.";
      box.appendChild(p);
      var q = document.createElement("p"); q.className = "muted";
      q.textContent = (mode === "today" ? "Measured on " : "Estimate for the day after ") + data.date;
      box.appendChild(q);
      return box;
    }

    function paint() {
      layer.clearLayers();
      data.stations.forEach(function (st) {
        var v = valueOf(st);
        var pos = [st.lat, st.lon];
        if (v) {
          L.circle(pos, { radius: 3800, stroke: false, fillColor: AQI_COLORS[v.category], fillOpacity: 0.32 }).addTo(layer);
          L.circle(pos, { radius: 1800, stroke: false, fillColor: AQI_COLORS[v.category], fillOpacity: 0.28 }).addTo(layer);
        }
        var wrap = document.createElement("div");
        var pin = document.createElement("div");
        pin.className = "aqi-pin" + (v ? (DARK_TEXT[v.category] ? " dark" : "") : " na");
        if (v) pin.style.background = AQI_COLORS[v.category];
        pin.textContent = v ? v.aqi : "n/a";
        var lab = document.createElement("span"); lab.className = "pin-name"; lab.textContent = st.short;
        wrap.appendChild(pin); wrap.appendChild(lab);
        var icon = L.divIcon({ className: "", html: wrap, iconSize: [40, 40], iconAnchor: [20, 20] });
        var m = L.marker(pos, { icon: icon, title: st.short + (v ? ": AQI " + v.aqi : ": no data") }).addTo(layer);
        m.bindPopup(popup(st));
      });
    }

    function setMode(m) {
      mode = m;
      $("view-today").className = m === "today" ? "on" : ""; $("view-today").setAttribute("aria-pressed", m === "today");
      $("view-tomorrow").className = m === "tomorrow" ? "on" : ""; $("view-tomorrow").setAttribute("aria-pressed", m === "tomorrow");
      paint();
    }
    $("view-today").onclick = function () { setMode("today"); };
    $("view-tomorrow").onclick = function () { setMode("tomorrow"); };
    if (!data.stations.some(function (s) { return s.tomorrow_est; })) $("view-tomorrow").hidden = true;
    $("map-note").textContent = "Coloured areas show the AQI around each monitoring station. Latest complete day: " + fmtDate(data.date) + ".";
    paint();
    map.fitBounds(L.latLngBounds(data.stations.map(function (s) { return [s.lat, s.lon]; })).pad(0.25));
  }
  getJSON("data/stations.json").then(initMap).catch(function () { $("map-card").hidden = true; });

  var metricsP = getJSON("data/metrics.json").catch(function () { return null; });
  var main;
  if (API) {
    main = getJSON(API + "/forecast").then(function (d) { return [d.forecast, d.history]; });
  } else {
    var rangeP = getJSON("data/range.json").catch(function () { return null; });   // optional (scripts/make_range_json.py)
    main = Promise.all([getJSON("data/forecast.json"), getJSON("data/history.json"), rangeP]).then(function (r) {
      var fc = r[0], rg = r[2];
      // attach only when it was made for exactly this forecast, so a stale file is never shown against a newer one
      if (rg && rg.prediction_range && rg.target_date === fc.target_date && rg.pm25 === fc.pm25) fc.prediction_range = rg.prediction_range;
      return [fc, r[1]];
    });
  }
  Promise.all([main, metricsP]).then(function (res) {
    var fc = res[0][0], hist = res[0][1], metrics = res[1];
    // when the live API has little history yet, fall back to the evaluation chart
    if (API && (!hist || hist.filter(function (h) { return h.actual != null; }).length < 5)) {
      return getJSON("data/history.json").then(function (h) { render(fc, h, metrics); }).catch(function () { render(fc, hist, metrics); });
    }
    render(fc, hist, metrics);
  }).catch(function (e) {
    showError("Could not load the forecast. " + e.message);
  });
})();
