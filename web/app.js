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
    $("pm-range").textContent = "PM2.5 about " + Math.round(forecast.pm25) + " µg/m³ (likely " +
      Math.round(forecast.pm25_low) + "–" + Math.round(forecast.pm25_high) + ")";
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

  var metricsP = getJSON("data/metrics.json").catch(function () { return null; });
  var main;
  if (API) {
    main = getJSON(API + "/forecast").then(function (d) { return [d.forecast, d.history]; });
  } else {
    main = Promise.all([getJSON("data/forecast.json"), getJSON("data/history.json")]);
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
