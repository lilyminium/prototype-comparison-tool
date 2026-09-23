// Minimal SVG charts: histogram (1–2 series), line (profile), scatter-line (scan). One y-axis each.
// Colors come from CSS custom properties (--series-1, --series-2, ink/grid tokens) so light/dark swap.
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9 * step; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

const fmt = (v, d = 3) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(Math.min(d, 3)) : v.toPrecision(3));

function tickFormatter(ticks) {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
  const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  return (v) => v.toFixed(decimals);
}

/** Render at the container's pixel width (min 320) so text stays legible on phones. */
function fitWidth(container, width) {
  const w = container.clientWidth || width;
  return Math.max(320, Math.min(width, w));
}

function frame(container, { width, height, xDomain, yDomain, xLabel, yLabel, xTicks }) {
  container.innerHTML = "";
  const m = { top: 12, right: 16, bottom: 42, left: 56 };
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart", role: "img" }, container);
  const iw = width - m.left - m.right, ih = height - m.top - m.bottom;
  const x = (v) => m.left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * iw;
  const y = (v) => m.top + ih - ((v - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * ih;
  const grid = el("g", { class: "grid" }, svg);
  const yTicks = niceTicks(yDomain[0], yDomain[1], 4);
  const fy = tickFormatter(yTicks);
  for (const t of yTicks) {
    el("line", { x1: m.left, x2: m.left + iw, y1: y(t), y2: y(t) }, grid);
    el("text", { x: m.left - 8, y: y(t) + 4, "text-anchor": "end", class: "tick" }, svg).textContent = fy(t);
  }
  const xt = xTicks || niceTicks(xDomain[0], xDomain[1], Math.max(3, Math.floor(iw / 90)));
  const fx = tickFormatter(xt);
  for (const t of xt) {
    el("text", { x: x(t), y: m.top + ih + 18, "text-anchor": "middle", class: "tick" }, svg).textContent = fx(t);
  }
  el("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih, class: "axis" }, svg);
  el("text", { x: m.left + iw / 2, y: height - 6, "text-anchor": "middle", class: "axis-label" }, svg).textContent = xLabel;
  el("text", { x: 14, y: m.top + ih / 2, "text-anchor": "middle", class: "axis-label", transform: `rotate(-90 14 ${m.top + ih / 2})` }, svg).textContent = yLabel;
  return { svg, x, y, m, iw, ih };
}

function tooltip(container) {
  let tip = container.querySelector(".tooltip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tooltip";
    tip.hidden = true;
    container.appendChild(tip);
  }
  return tip;
}

function placeTip(tip, container, evt, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = container.getBoundingClientRect();
  const left = Math.min(evt.clientX - r.left + 12, r.width - tip.offsetWidth - 4);
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${Math.max(4, evt.clientY - r.top - tip.offsetHeight - 10)}px`;
}

/**
 * Histogram. series: [{name, slot, hist: {lo, hi, width, counts}}] (same bins; slot = color slot 1/2).
 * refLines: [{x, label}] drawn in primary ink.
 */
export function histogramChart(container, { series, xLabel, yLabel, refLines = [], xTicks, width = 720, height = 280 }) {
  width = fitWidth(container, width);
  const bins = series[0].hist;
  const nb = bins.counts.length;
  const yMax = Math.max(1e-12, ...series.flatMap((s) => s.hist.counts)) * 1.08;
  const f = frame(container, { width, height, xDomain: [bins.lo, bins.hi], yDomain: [0, yMax], xLabel, yLabel, xTicks });
  const bw = f.iw / nb;
  const gap = series.length > 1 ? 1 : 0;
  series.forEach((s, si) => {
    // s.slot keeps a series' color tied to its identity when other series are toggled off
    const g = el("g", { class: `series series-${s.slot || si + 1}` }, f.svg);
    const sub = series.length > 1 ? (bw - 2) / series.length : bw - 1;
    s.hist.counts.forEach((c, b) => {
      if (c <= 0) return;
      const x0 = f.x(bins.lo + b * bins.width) + 1 + si * (sub + gap);
      const h = f.y(0) - f.y(c);
      el("rect", { x: x0, y: f.y(c), width: Math.max(0.5, sub - gap), height: Math.max(0.5, h), rx: Math.min(2, sub / 2) }, g);
    });
  });
  for (const r of refLines) {
    if (r.x < bins.lo || r.x > bins.hi) continue;
    el("line", { x1: f.x(r.x), x2: f.x(r.x), y1: f.m.top, y2: f.m.top + f.ih, class: "ref" }, f.svg);
    el("text", { x: f.x(r.x) + 4, y: f.m.top + 12, class: "ref-label" }, f.svg).textContent = r.label;
  }
  const hover = el("rect", { x: f.m.left, y: f.m.top, width: f.iw, height: f.ih, class: "hover-target" }, f.svg);
  const cross = el("line", { y1: f.m.top, y2: f.m.top + f.ih, class: "crosshair", visibility: "hidden" }, f.svg);
  const tip = tooltip(container);
  hover.addEventListener("pointermove", (evt) => {
    const pt = f.svg.getBoundingClientRect();
    const sx = ((evt.clientX - pt.left) / pt.width) * width;
    const b = Math.min(nb - 1, Math.max(0, Math.floor((sx - f.m.left) / bw)));
    const lo = bins.lo + b * bins.width;
    cross.setAttribute("x1", f.x(lo + bins.width / 2));
    cross.setAttribute("x2", f.x(lo + bins.width / 2));
    cross.setAttribute("visibility", "visible");
    const rows = series
      .map((s, si) => `<div><span class="swatch s${s.slot || si + 1}"></span>${s.name}: <b>${fmt(s.hist.counts[b])}</b></div>`)
      .join("");
    placeTip(tip, container, evt, `<div class="tip-title">${fmt(lo)} – ${fmt(lo + bins.width)}</div>${rows}`);
  });
  hover.addEventListener("pointerleave", () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
  });
}

/** Line chart (single series) with optional marker points; crosshair tooltip. */
export function lineChart(container, { x, y, xLabel, yLabel, markers = [], xTicks, width = 720, height = 220, yUnit = "" }) {
  width = fitWidth(container, width);
  const yLo = Math.min(0, ...y), yHi = Math.max(...y) * 1.08 || 1;
  const f = frame(container, { width, height, xDomain: [x[0], x[x.length - 1]], yDomain: [yLo, yHi], xLabel, yLabel, xTicks });
  el("path", { d: x.map((v, i) => `${i ? "L" : "M"}${f.x(v)},${f.y(y[i])}`).join(""), class: "line" }, f.svg);
  for (const mk of markers) el("circle", { cx: f.x(mk.x), cy: f.y(mk.y), r: 4, class: "marker" }, f.svg);
  const hover = el("rect", { x: f.m.left, y: f.m.top, width: f.iw, height: f.ih, class: "hover-target" }, f.svg);
  const cross = el("line", { y1: f.m.top, y2: f.m.top + f.ih, class: "crosshair", visibility: "hidden" }, f.svg);
  const dot = el("circle", { r: 4, class: "hover-dot", visibility: "hidden" }, f.svg);
  const tip = tooltip(container);
  hover.addEventListener("pointermove", (evt) => {
    const pt = f.svg.getBoundingClientRect();
    const sx = ((evt.clientX - pt.left) / pt.width) * width;
    const xv = x[0] + ((sx - f.m.left) / f.iw) * (x[x.length - 1] - x[0]);
    let i = 0;
    for (let k = 1; k < x.length; k++) if (Math.abs(x[k] - xv) < Math.abs(x[i] - xv)) i = k;
    cross.setAttribute("x1", f.x(x[i]));
    cross.setAttribute("x2", f.x(x[i]));
    cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", f.x(x[i]));
    dot.setAttribute("cy", f.y(y[i]));
    dot.setAttribute("visibility", "visible");
    placeTip(tip, container, evt, `<div class="tip-title">${fmt(x[i])}°</div><div>${fmt(y[i])} ${yUnit}</div>`);
  });
  hover.addEventListener("pointerleave", () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
  });
}

/**
 * Scan chart: points joined in x order (one series). Returns a handle:
 *   setCurrent(i)  mark point i (index into `points`) as the current frame
 *   onSelect       assign a callback(i) to be told when the user clicks a point
 */
export function scanChart(container, { points, xLabel, yLabel, width = 720, height = 220 }) {
  width = fitWidth(container, width);
  const pts = points.map((p, i) => ({ ...p, i })).sort((a, b) => a.x - b.x);
  const ys = pts.map((p) => p.y);
  const f = frame(container, { width, height, xDomain: [-180, 180], yDomain: [Math.min(0, ...ys), Math.max(...ys) * 1.08 || 1], xLabel, yLabel, xTicks: [-180, -120, -60, 0, 60, 120, 180] });
  el("path", { d: pts.map((p, k) => `${k ? "L" : "M"}${f.x(p.x)},${f.y(p.y)}`).join(""), class: "line" }, f.svg);
  const tip = tooltip(container);
  const current = el("circle", { r: 8, class: "current", visibility: "hidden" }, f.svg);
  const handle = {
    onSelect: null,
    setCurrent(i) {
      const p = points[i];
      current.setAttribute("cx", f.x(p.x));
      current.setAttribute("cy", f.y(p.y));
      current.setAttribute("visibility", "visible");
    },
  };
  for (const p of pts) {
    const c = el("circle", { cx: f.x(p.x), cy: f.y(p.y), r: 4, class: "marker" }, f.svg);
    const hit = el("circle", { cx: f.x(p.x), cy: f.y(p.y), r: 10, class: "hit" }, f.svg);
    hit.addEventListener("pointermove", (evt) => placeTip(tip, container, evt, `<div class="tip-title">${p.x}°</div><div>${fmt(p.y)} kcal/mol</div>${p.note || ""}`));
    hit.addEventListener("pointerleave", () => (tip.hidden = true));
    hit.addEventListener("click", () => handle.onSelect?.(p.i));
    c.setAttribute("aria-label", `${p.x}°: ${fmt(p.y)}`);
  }
  return handle;
}
