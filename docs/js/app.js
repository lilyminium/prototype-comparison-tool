// Sage 2.3.0 training-set explorer: UI. Data layout: docs/data/SCHEMA.md. Everything runs client-side.
import { histogramChart, lineChart, scanChart } from "./charts.js";
import { openViewer } from "./viewer3d.js";
import { depiction, fetchBuffer, fetchJSON, moleculeSvg, parseAssignments, parseGeometry } from "./data.js";
import { classifyMeasurement, extent, histogram, tanimotoAll, universeKey, weightedCircularStats, weightedLinearStats } from "./stats.js";

const DATA = "data/";
const HANDLER_LABEL = { Bonds: "Bond", Angles: "Angle", ProperTorsions: "Proper torsion", ImproperTorsions: "Improper torsion" };
const MEASURE_HANDLER = { bond: "Bonds", angle: "Angles", proper: "ProperTorsions", improper: "ImproperTorsions" };
const UNIT = { Bonds: "Å", Angles: "°", ProperTorsions: "°", ImproperTorsions: "°" };
const main = document.getElementById("main");
const state = { meta: null, params: null, molecules: null, topologies: null, paramIndex: null };

// ---------------------------------------------------------------- utilities
const h = (tag, attrs = {}, ...children) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
};
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const mb = (bytes) => `${(bytes / 1e6).toFixed(1)} MB`;
const stat = (k, v) => h("div", { class: "stat" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v));

// ---------------------------------------------------------------- worker (RDKit.js)
let worker = null;
let workerReady = null;
let msgId = 0;
const pending = new Map();
function startWorker() {
  worker = new Worker("js/classic/worker.js");
  worker.onmessage = (e) => {
    const { id, ok, result, error, progress } = e.data;
    if (progress) {
      for (const p of pending.values()) p.onProgress?.(progress);
      return;
    }
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    ok ? p.resolve(result) : p.reject(new Error(error));
  };
  workerReady = call("init", { topologies: { mdl_smiles: state.topologies.mdl_smiles, mdl_order: state.topologies.mdl_order }, molecules: { display_smiles: state.molecules.display_smiles } });
  return workerReady;
}
function call(type, args = {}, onProgress) {
  const id = ++msgId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ id, type, ...args });
  });
}
async function rdkit(type, args, onProgress) {
  if (!worker) startWorker();
  await workerReady;
  return call(type, args, onProgress);
}
function cancelWorker() {
  worker?.terminate();
  for (const p of pending.values()) p.reject(new Error("cancelled"));
  pending.clear();
  worker = null;
}

// ---------------------------------------------------------------- depictions
/**
 * A molecule card: 2D RDKit depiction plus, when `view3d` = {t, highlight, label} is given, a 3D button.
 * Optimization records open the QM minimum; TorsionDrive records open the animated scan with its
 * energy profile.
 */
function molCard(caption, svgPromise, view3d = null) {
  const box = h("div", { class: "mol" }, h("div", { class: "muted" }, "drawing…"));
  const actions = view3d
    ? h(
        "div",
        { class: "mol-actions" },
        h(
          "button",
          { type: "button", onclick: () => openViewer({ ...view3d, state, rdkit }).catch((e) => alert(e.message)) },
          state.topologies.source[view3d.t] === "td" ? "3D scan + energy" : "3D structure",
        ),
      )
    : null;
  svgPromise
    .then((svg) => {
      box.innerHTML = svg;
      box.append(caption);
      if (actions) box.append(actions);
    })
    .catch((e) => (box.textContent = e.message));
  return box;
}
/**
 * Render a record's PRECOMPUTED depiction with highlights. Highlight indices come from precomputed maps:
 * the heavy-only drawing (via heavy_index) unless a highlighted atom is a hydrogen, in which case the
 * explicit-H drawing (atom order = topology order). `atoms`: one interaction (bonds between consecutive
 * atoms, or the improper star); `matches`: SMARTS matches (all their atoms and every bond between them).
 */
async function renderTopology(t, { atoms = [], improper = false, matches = null }) {
  const dep = await depiction(DATA, t);
  const all = matches ? [...new Set(matches.flat())] : atoms;
  const heavy = all.every((a) => dep.heavy_index[a] >= 0);
  const idx = heavy ? (a) => dep.heavy_index[a] : (a) => a;
  const bondList = heavy ? dep.bonds_heavy : dep.bonds_h;
  const pairKey = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
  const bondIndex = new Map(bondList.map(([a, b], i) => [pairKey(a, b), i]));
  let bonds;
  if (matches) {
    const inSet = new Set(all.map(idx));
    bonds = bondList.map(([a, b], i) => (inSet.has(a) && inSet.has(b) ? i : -1)).filter((i) => i >= 0);
  } else {
    const d = atoms.map(idx);
    const pairs = improper && d.length === 4 ? [[d[1], d[0]], [d[1], d[2]], [d[1], d[3]]] : d.slice(1).map((b, i) => [d[i], b]);
    bonds = pairs.map(([a, b]) => bondIndex.get(pairKey(a, b))).filter((i) => i !== undefined);
  }
  return rdkit("render", { molblock: heavy ? dep.molblock_heavy : dep.molblock_h, atoms: all.map(idx), bonds });
}

/** Topology to show in 3D for a molecule: its first optimization record, else its first TorsionDrive. */
function representativeTopology(mol_idx) {
  const tops = state.molecules.topologies[mol_idx];
  return tops.find((t) => state.topologies.source[t] === "opt") ?? tops[0];
}
function topologyCaption(t, extra = "") {
  const T = state.topologies;
  return h(
    "div",
    { class: "cap" },
    `${T.source[t] === "opt" ? "Optimization" : "TorsionDrive"} record ${T.record_id[t]}`,
    h("br"),
    h("span", { class: "mono" }, state.molecules.display_smiles[T.mol_idx[t]]),
    extra ? h("div", {}, extra) : null,
  );
}

// ---------------------------------------------------------------- routing
const routes = { params: renderParams, param: renderParam, smarts: renderSmarts, molecule: renderMolecule, about: renderAbout };
function route() {
  const [name, ...rest] = (location.hash.slice(1) || "params").split("/");
  const tab = name === "param" ? "params" : name;
  document.querySelectorAll("nav.tabs a").forEach((a) => (a.dataset.tab === tab ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  main.innerHTML = "";
  (routes[name] || renderParams)(...rest.map(decodeURIComponent));
  window.scrollTo(0, 0);
}

// ---------------------------------------------------------------- Parameters list
// One table per handler. "FF order" is the parameter's position within its handler in the OFFXML
// (the SMIRNOFF hierarchy: later parameters override earlier ones). Every column is sortable.
const listState = { handler: "Bonds", query: "", sort: "order", dir: 1 };
const naturalCompare = (a, b) => String(a).localeCompare(String(b), undefined, { numeric: true });

function termsFor(p, n) {
  // A parameter can repeat a periodicity (t25, t26); keep every term, in OFFXML order
  return p.periodicity.map((m, i) => (m === n ? i : -1)).filter((i) => i >= 0);
}

function columnsFor(handler, rows) {
  const num = (label, get, digits, title) => ({ label, get, cell: (p) => fmt(get(p), digits), num: true, title });
  const cols = [
    { label: "FF order", get: (p) => p.hierarchy_index + 1, cell: (p) => p.hierarchy_index + 1, num: true, title: "Position within this handler in openff_unconstrained-2.3.0.offxml (later overrides earlier)" },
    { label: "ID", get: (p) => p.param_id, cell: (p) => h("b", {}, p.param_id), natural: true },
    { label: "SMIRKS", get: (p) => p.smirks, cell: (p) => h("span", { class: "smirks" }, p.smirks) },
  ];
  if (handler === "Bonds") {
    cols.push(num("r₀ (Å)", (p) => p.length_angstrom, 4, "equilibrium length"), num("k (kcal/mol/Å²)", (p) => p.k, 2));
  } else if (handler === "Angles") {
    cols.push(num("θ₀ (°)", (p) => p.angle_deg, 3, "equilibrium angle"), num("k (kcal/mol/rad²)", (p) => p.k, 2));
  } else {
    const present = [1, 2, 3, 4, 5, 6].filter((n) => rows.some((p) => p.periodicity.includes(n)));
    for (const n of present) {
      const pick = (p, key) => termsFor(p, n).map((i) => p[key][i]);
      cols.push(
        { label: `k n=${n} (kcal/mol)`, get: (p) => pick(p, "k_raw")[0] ?? null, cell: (p) => pick(p, "k_raw").map((v) => fmt(v, 4)).join(" / "), num: true, title: "raw k before division by idivf; repeated periodicities shown in OFFXML order" },
        { label: `phase n=${n} (°)`, get: (p) => pick(p, "phase_deg")[0] ?? null, cell: (p) => pick(p, "phase_deg").map((v) => fmt(v, 0)).join(" / "), num: true },
      );
    }
    cols.push({ label: "idivf", get: (p) => p.idivf[0], cell: (p) => [...new Set(p.idivf)].map((v) => fmt(v, 1)).join(" / "), num: true, title: "resolved idivf (ImproperTorsions default_idivf=auto → 3)" });
  }
  cols.push(num("Molecules", (p) => p.n_molecules, 0), num("Opt obs.", (p) => p.n_obs_opt, 0), num("TD obs.", (p) => p.n_obs_td, 0));
  if (handler === "Bonds" || handler === "Angles") {
    const d = handler === "Bonds" ? 4 : 2;
    const u = handler === "Bonds" ? "Å" : "°";
    cols.push(
      num(`Opt mean (${u})`, (p) => p.opt_stats?.mean ?? null, d),
      num(`Opt SD (${u})`, (p) => p.opt_stats?.std ?? null, d),
      num(`Opt mean − ${handler === "Bonds" ? "r₀" : "θ₀"} (${u})`, (p) => p.opt_stats?.mean_minus_centre ?? null, d, "descriptive offset, not a force-field error (see About)"),
    );
  } else {
    cols.push(
      num("Opt circ. mean (°)", (p) => p.opt_stats?.circ_mean ?? null, 1),
      num("Opt circ. SD (°)", (p) => p.opt_stats?.circ_std ?? null, 1, "large for multimodal distributions"),
      num("Undefined", (p) => p.n_invalid ?? 0, 0, "near-linear dihedrals excluded from statistics"),
    );
  }
  return cols;
}

function renderParams() {
  const hs = ["Bonds", "Angles", "ProperTorsions", "ImproperTorsions"];
  const chips = h(
    "div",
    { class: "chips" },
    ...hs.map((x) =>
      h("button", { type: "button", "aria-pressed": String(listState.handler === x), onclick: () => ((listState.handler = x), (listState.sort = "order"), (listState.dir = 1), rerender()) }, `${HANDLER_LABEL[x]}s (${state.params.filter((p) => p.handler === x).length})`),
    ),
  );
  const search = h("input", { type: "search", placeholder: "Filter by id or SMIRKS text", value: listState.query });
  search.addEventListener("input", () => ((listState.query = search.value), drawTable()));
  const tableBox = h("div", { class: "table-wrap" });
  main.append(
    h(
      "section",
      { class: "card" },
      h("h2", {}, "Valence parameters"),
      h("p", { class: "secondary" }, "Parameters of openff_unconstrained-2.3.0 with their values and training-set coverage. Summary statistics use the optimization geometries only. Click a column header to sort, a row to see the distribution."),
      h("div", { class: "row" }, chips, h("div", { class: "grow" }, search)),
      tableBox,
    ),
  );
  function rerender() {
    main.innerHTML = "";
    renderParams();
  }
  function drawTable() {
    const q = listState.query.toLowerCase();
    const inHandler = state.params.filter((p) => p.handler === listState.handler);
    const cols = columnsFor(listState.handler, inHandler);
    let rows = inHandler.filter((p) => !q || p.param_id.toLowerCase().includes(q) || p.smirks.toLowerCase().includes(q));
    const col = cols.find((c) => c.label === listState.sort) || cols[0];
    rows = rows.slice().sort((a, b) => {
      const va = col.get(a), vb = col.get(b);
      if (va === null || va === undefined) return 1; // blanks last in both directions
      if (vb === null || vb === undefined) return -1;
      const c = col.num ? va - vb : naturalCompare(va, vb);
      return c * listState.dir || a.hierarchy_index - b.hierarchy_index;
    });
    const head = h(
      "tr",
      {},
      ...cols.map((c) => {
        const active = c === col;
        return h(
          "th",
          {
            class: c.num ? "num" : "",
            title: c.title,
            "aria-sort": active ? (listState.dir > 0 ? "ascending" : "descending") : "none",
            onclick: () => {
              listState.dir = active ? -listState.dir : 1;
              listState.sort = c.label;
              drawTable();
            },
          },
          c.label + (active ? (listState.dir > 0 ? " ▲" : " ▼") : ""),
        );
      }),
    );
    const body = rows.map((p) => h("tr", { class: "clickable", onclick: () => (location.hash = `param/${encodeURIComponent(p.param_id)}`) }, ...cols.map((c) => h("td", { class: c.num ? "num" : "" }, c.cell(p)))));
    tableBox.innerHTML = "";
    tableBox.append(h("p", { class: "muted" }, `${rows.length} of ${inHandler.length} ${HANDLER_LABEL[listState.handler].toLowerCase()} parameters`), h("table", {}, h("thead", {}, head), h("tbody", {}, ...body)));
  }
  drawTable();
}

// ---------------------------------------------------------------- Parameter detail
// Everything on this page is precomputed by scripts/09_build_site_data.py (param/<id>.json): histograms and
// statistics for both weightings, the torsion energy profile and its minima, driven scans, examples.
async function renderParam(id) {
  const p = state.params.find((x) => x.param_id === id);
  if (!p) {
    main.append(h("p", {}, `Unknown parameter ${id}`));
    return;
  }
  const isTorsion = p.handler === "ProperTorsions" || p.handler === "ImproperTorsions";
  const unit = UNIT[p.handler];
  main.append(
    h(
      "section",
      { class: "card" },
      h("p", {}, h("a", { href: "#params" }, "← All parameters")),
      h("h2", {}, `${p.param_id} · ${HANDLER_LABEL[p.handler]}`),
      h("p", {}, h("code", {}, p.smirks)),
      h(
        "div",
        { class: "stats" },
        p.handler === "Bonds" ? [stat("Sage 2.3.0 r₀", `${fmt(p.length_angstrom, 4)} Å`), stat("Sage 2.3.0 k", `${fmt(p.k, 2)} kcal/mol/Å²`)] : null,
        p.handler === "Angles" ? [stat("Sage 2.3.0 θ₀", `${fmt(p.angle_deg, 3)}°`), stat("Sage 2.3.0 k", `${fmt(p.k, 2)} kcal/mol/rad²`)] : null,
        isTorsion ? p.periodicity.map((n, i) => stat(`term ${i + 1}: n=${n}, phase ${p.phase_deg[i]}°`, `k ${fmt(p.k_raw[i], 4)} / idivf ${p.idivf[i]}`)) : null,
        stat("molecules", p.n_molecules),
        stat("assignments", p.n_assignments),
        stat("opt / TD observations", `${p.n_obs_opt} / ${p.n_obs_td}`),
        isTorsion ? stat("undefined (near-linear)", p.n_invalid) : null,
      ),
      h("p", {}, h("a", { href: `#smarts/${encodeURIComponent(p.smirks)}` }, "Search the training set with this SMIRKS →")),
    ),
  );
  if (!p.detail) {
    main.append(h("section", { class: "card" }, h("p", { class: "note" }, "This parameter is not assigned to any atom in the training set (zero coverage), so there is no QM distribution to show.")));
    return;
  }

  // The QM distribution uses optimization minima only. TorsionDrive grid points are constrained
  // geometries; they appear in the scan panel and in "Training molecules with this parameter".
  const weighting = h("select", { class: "auto" }, h("option", { value: "instances" }, "Weight: every observation"), h("option", { value: "molecule" }, "Weight: each molecule sums to 1"));
  const statsBox = h("div", { class: "stats" });
  const chartBox = h("div", { class: "chart-box" });
  const extraBox = h("div", { class: "chart-box" });
  main.append(
    h(
      "section",
      { class: "card" },
      h("h3", {}, isTorsion ? "QM dihedral distribution (optimization minima)" : `QM ${p.handler === "Bonds" ? "bond length" : "angle"} distribution (optimization minima)`),
      h("div", { class: "row" }, weighting),
      h(
        "p",
        { class: "muted" },
        isTorsion
          ? "Dihedrals are measured in the atom order label_molecules assigns (IUPAC sign convention, as OpenMM). Undefined (near-linear) dihedrals and torsions about the frozen bonds of the 13 constrained optimizations are excluded. "
          : `The dashed line is the Sage 2.3.0 equilibrium value (${p.handler === "Bonds" ? "r₀" : "θ₀"}). QM − ${p.handler === "Bonds" ? "r₀" : "θ₀"} is descriptive, not a force-field error: the MM minimum also depends on the other bonded and nonbonded terms. `,
        "TorsionDrive grid points are not included here.",
      ),
      chartBox,
      statsBox,
      extraBox,
    ),
  );
  const detail = await fetchJSON(DATA + p.detail);

  function draw() {
    const w = detail.weightings[weighting.value];
    statsBox.innerHTML = "";
    if (!w.n_obs) {
      chartBox.innerHTML = `<p class="note">No optimization-minimum observations for this parameter${p.n_obs_td ? "; it occurs only in TorsionDrive records (see the training molecules below)" : ""}.</p>`;
      return;
    }
    const refLines = isTorsion
      ? []
      : [p.handler === "Bonds" ? { x: p.length_angstrom, label: `Sage 2.3.0 r₀ = ${fmt(p.length_angstrom, 4)} Å` } : { x: p.angle_deg, label: `Sage 2.3.0 θ₀ = ${fmt(p.angle_deg, 2)}°` }];
    histogramChart(chartBox, {
      series: [{ name: "Optimization minima", slot: 1, hist: { lo: w.lo, hi: w.hi, width: (w.hi - w.lo) / w.nb, counts: w.counts } }],
      xLabel: isTorsion ? "dihedral (°)" : p.handler === "Bonds" ? "bond length (Å)" : "angle (°)",
      yLabel: weighting.value === "molecule" ? "molecule-weighted count" : p.handler === "ImproperTorsions" ? "count (each improper = 1)" : "count",
      refLines,
      xTicks: isTorsion ? [-180, -120, -60, 0, 60, 120, 180] : undefined,
    });
    statsBox.append(stat("molecules", w.n_mol), stat("observations", p.handler === "ImproperTorsions" ? `${w.n_obs / 3} impropers` : w.n_obs));
    const st = w.stats;
    if (isTorsion) {
      // For multimodal torsions the circular mean is not a typical value; R shows how concentrated it is
      statsBox.append(stat("circular mean", `${fmt(st.circMean, 1)}°`), stat("mean resultant length R", fmt(st.resultantLength, 3)), stat("circular SD", st.circStd === null ? "–" : `${fmt(st.circStd, 1)}°`));
    } else {
      const centre = p.handler === "Bonds" ? p.length_angstrom : p.angle_deg;
      const d = p.handler === "Bonds" ? 4 : 2;
      statsBox.append(
        stat("mean", `${fmt(st.mean, d)} ${unit}`),
        stat("SD", `${fmt(st.std, d)} ${unit}`),
        stat("median", `${fmt(st.median, d)} ${unit}`),
        stat(`mean − Sage ${p.handler === "Bonds" ? "r₀" : "θ₀"}`, `${fmt(st.mean - centre, d)} ${unit}`),
        stat("range", `${fmt(st.min, d)} – ${fmt(st.max, d)}`),
      );
    }
  }
  weighting.addEventListener("change", draw);
  draw();

  if (isTorsion) {
    const prof = detail.profile;
    const x = prof.y.map((_, i) => prof.x0 + i * prof.step);
    const profBox = h("div", { class: "chart-box" });
    extraBox.append(
      h("h3", {}, p.handler === "ImproperTorsions" ? "Single-term functional form (illustration)" : "Force-field torsion energy profile"),
      h(
        "p",
        { class: "muted" },
        p.handler === "ImproperTorsions"
          ? "One trefoil term, k/idivf·(1+cos(nφ−phase)). An improper's energy is the sum over its three terms, which do not move together."
          : `Σ k/idivf·(1+cos(nφ−phase)) for this parameter alone. Minima: ${prof.minima.map((m) => `${m.phi}°`).join(", ") || "none"}. Plotted on its own axis, aligned with the histogram above.`,
      ),
      profBox,
    );
    lineChart(profBox, { x, y: prof.y, xLabel: "dihedral (°)", yLabel: "energy (kcal/mol)", markers: prof.minima.map((m) => ({ x: m.phi, y: m.energy })), xTicks: [-180, -120, -60, 0, 60, 120, 180], yUnit: "kcal/mol" });
    if (detail.scans.length) renderScans(extraBox, p, detail.scans);
  }
  renderExamples(p, detail.examples);
}

function renderScans(box, p, scans) {
  const T = state.topologies;
  const select = h("select", {}, ...scans.map((s, i) => h("option", { value: i }, `${state.molecules.display_smiles[T.mol_idx[s.t]]} (TD ${T.record_id[s.t]})`)));
  const chart = h("div", { class: "chart-box" });
  const pic = h("div", { class: "mols" });
  box.append(h("h3", {}, `TorsionDrive scans driving this torsion (${scans.length})`), h("p", { class: "muted" }, "QM relative energy of each constrained optimization along the driven dihedral. Use “3D scan + energy” to animate the scan in 3D alongside this profile."), select, chart, pic);
  const draw = () => {
    const s = scans[select.value];
    scanChart(chart, { points: s.frames.map(([grid, energy]) => ({ x: grid, y: energy })), xLabel: "driven dihedral, grid angle (°)", yLabel: "relative QM energy (kcal/mol)" });
    pic.innerHTML = "";
    pic.append(molCard(topologyCaption(s.t), renderTopology(s.t, { atoms: s.atoms }), { t: s.t, highlight: s.atoms, label: `${p.param_id} atoms ${s.atoms.join("-")}` }));
  };
  select.addEventListener("change", draw);
  draw();
}

function renderExamples(p, examples) {
  // One example assignment per record, precomputed; optimization and TorsionDrive records filtered separately
  const bySource = { opt: examples.filter((e) => e[2] === "opt"), td: examples.filter((e) => e[2] === "td") };
  const optBox = h("input", { type: "checkbox", checked: bySource.opt.length > 0 });
  const tdBox = h("input", { type: "checkbox", checked: bySource.opt.length === 0 });
  const grid = h("div", { class: "mols" });
  const pager = h("div", { class: "pager" });
  main.append(
    h(
      "section",
      { class: "card" },
      h("h3", {}, `Training molecules with this parameter (${p.n_molecules} molecules)`),
      h(
        "div",
        { class: "row" },
        h("label", { class: "inline" }, optBox, `Optimizations (${bySource.opt.length} records)`),
        h("label", { class: "inline" }, tdBox, `TorsionDrives (${bySource.td.length} records)`),
      ),
      h("p", { class: "muted" }, "One example assignment per record, atoms highlighted. Optimization records open the QM minimum in 3D; TorsionDrive records open the animated scan with its energy profile."),
      grid,
      pager,
    ),
  );
  let page = 0;
  const per = 12;
  const selected = () => [...(optBox.checked ? bySource.opt : []), ...(tdBox.checked ? bySource.td : [])];
  const draw = () => {
    const list = selected();
    grid.innerHTML = "";
    if (!list.length) grid.append(h("p", { class: "muted" }, "Select optimizations and/or TorsionDrives."));
    for (const [t, atoms] of list.slice(page * per, page * per + per)) {
      grid.append(molCard(topologyCaption(t, `atoms ${atoms.join("-")}`), renderTopology(t, { atoms, improper: p.handler === "ImproperTorsions" }), { t, highlight: atoms, label: `${p.param_id} atoms ${atoms.join("-")}` }));
    }
    pager.innerHTML = "";
    const pages = Math.max(1, Math.ceil(list.length / per));
    pager.append(
      h("button", { type: "button", disabled: page === 0, onclick: () => (page--, draw()) }, "Previous"),
      h("span", { class: "muted" }, `page ${page + 1} of ${pages} · ${list.length} records`),
      h("button", { type: "button", disabled: page + 1 >= pages, onclick: () => (page++, draw()) }, "Next"),
    );
  };
  [optBox, tdBox].forEach((c) => c.addEventListener("change", () => ((page = 0), draw())));
  draw();
}

// ---------------------------------------------------------------- SMARTS search
let smartsGeneration = 0; // incremented by each new search and by Cancel; stale work checks it and stops
const yieldToBrowser = () => new Promise((r) => setTimeout(r, 0));

async function renderSmarts(initial) {
  const input = h("textarea", { spellcheck: "false", "aria-label": "SMARTS pattern" }, initial || "[#6X3:1]=[#8X1:2]");
  const run = h("button", { type: "button", class: "primary" }, "Search");
  const cancel = h("button", { type: "button", disabled: true }, "Cancel");
  const progress = h("div", { class: "progress" }, h("div", {}));
  const status = h("p", { class: "muted" }, "Matches use OpenFF's settings (all tagged permutations, chirality-aware) on the training molecules' MDL-aromatic forms, reproducing ForceField.label_molecules matching exactly.");
  const results = h("div");
  main.append(
    h(
      "section",
      { class: "card" },
      h("h2", {}, "SMARTS search"),
      h("p", { class: "secondary" }, "Find training-set records matching a SMARTS pattern. Map 2–4 atoms (:1…:4) as a bond, angle, proper (:1-:2-:3-:4) or improper (:2 central) to get the QM distribution of that internal coordinate in the optimization minima, and the Sage parameters assigned to the matched atoms."),
      input,
      h("div", { class: "row" }, run, cancel),
      h(
        "p",
        { class: "muted" },
        "Examples: ",
        ...[["[#6X4:1]-[#6X4:2]", "C–C single bond"], ["[#6X3:1]:[#6X3:2]", "aromatic C:C"], ["[#1:1]-[#8X2:2]-[#6:3]", "H–O–C angle"], ["[*:1]-[#6X4:2]-[#6X4:3]-[*:4]", "sp3–sp3 torsion"], ["[*:1]~[#7X3:2](~[*:3])~[*:4]", "N improper"]].flatMap(([s, l]) => [h("a", { href: `#smarts/${encodeURIComponent(s)}` }, l), " · "]),
      ),
      progress,
      status,
    ),
    results,
  );
  const go = async () => {
    const smarts = input.value.trim();
    if (!smarts) return;
    const gen = ++smartsGeneration;
    const current = () => gen === smartsGeneration;
    run.disabled = true;
    cancel.disabled = false;
    results.innerHTML = "";
    progress.firstChild.style.width = "0%";
    status.textContent = "Searching…";
    const t0 = performance.now();
    try {
      const r = await rdkit("smarts", { smarts }, (pr) => current() && (progress.firstChild.style.width = `${(100 * pr.done) / pr.total}%`));
      if (!current()) return;
      progress.firstChild.style.width = "100%";
      status.textContent = `Searched ${state.topologies.source.length} records in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
      await showSmartsResults(results, r, current);
    } catch (e) {
      if (current()) status.innerHTML = `<span class="error">${esc(e.message)}</span>`;
    } finally {
      if (current()) {
        run.disabled = false;
        cancel.disabled = true;
      }
    }
  };
  run.addEventListener("click", go);
  cancel.addEventListener("click", () => {
    smartsGeneration++; // stops geometry work and discards any late results
    cancelWorker();
    results.innerHTML = "";
    status.textContent = "Cancelled.";
    run.disabled = false;
    cancel.disabled = true;
  });
  if (initial) go();
}

async function showSmartsResults(box, r, current) {
  const T = state.topologies;
  const hits = Array.from(r.hitTopologies);
  const width = r.width;
  const tuplesOf = (k) => {
    const out = [];
    for (let n = r.offsets[k]; n < r.offsets[k + 1]; n++) out.push(Array.from(r.tuples.slice(n * width, n * width + width)));
    return out;
  };
  const mols = new Set(hits.map((t) => T.mol_idx[t]));
  const nOpt = hits.filter((t) => T.source[t] === "opt").length;
  box.append(
    h(
      "section",
      { class: "card" },
      h("h3", {}, "Matches"),
      h(
        "div",
        { class: "stats" },
        stat("molecules", `${mols.size} / ${state.molecules.display_smiles.length}`),
        stat("optimization records", nOpt),
        stat("TorsionDrive records", hits.length - nOpt),
        stat(r.info.nTags ? "ordered tagged matches" : "unique matches", r.offsets[r.offsets.length - 1]),
      ),
    ),
  );
  if (r.info.nTags >= 2) {
    const cls = classifyMeasurement(r.info.nTags, r.info.bondedTagPairs);
    const card = h("section", { class: "card" });
    box.append(card);
    if (cls.error) card.append(h("p", { class: "note" }, `No geometry: ${cls.error}.`));
    else await smartsGeometry(card, cls.kind, hits, tuplesOf, current);
    if (!current()) return;
  }
  renderSmartsRecords(box, r, hits, tuplesOf);
}

function renderSmartsRecords(box, r, hits, tuplesOf) {
  const T = state.topologies;
  const optBox = h("input", { type: "checkbox", checked: true });
  const tdBox = h("input", { type: "checkbox", checked: true });
  const grid = h("div", { class: "mols" });
  const pager = h("div", { class: "pager" });
  const improper = r.info.nTags === 4 && classifyMeasurement(4, r.info.bondedTagPairs).kind === "improper";
  box.append(
    h(
      "section",
      { class: "card" },
      h("h3", {}, "Matching records"),
      h("div", { class: "row" }, h("label", { class: "inline" }, optBox, "Optimizations"), h("label", { class: "inline" }, tdBox, "TorsionDrives")),
      h("p", { class: "muted" }, "All matched atoms are highlighted."),
      grid,
      pager,
    ),
  );
  let page = 0;
  const per = 12;
  const draw = () => {
    const ks = hits.map((t, k) => k).filter((k) => (T.source[hits[k]] === "opt" ? optBox.checked : tdBox.checked));
    grid.innerHTML = "";
    for (const k of ks.slice(page * per, page * per + per)) {
      const t = hits[k];
      const matches = tuplesOf(k);
      const all = [...new Set(matches.flat())];
      grid.append(
        molCard(topologyCaption(t, `${matches.length} match(es)`), renderTopology(t, { matches, improper }), { t, highlight: all, label: `${matches.length} match(es)` }),
      );
    }
    pager.innerHTML = "";
    const pages = Math.max(1, Math.ceil(ks.length / per));
    pager.append(
      h("button", { type: "button", disabled: page === 0, onclick: () => (page--, draw()) }, "Previous"),
      h("span", { class: "muted" }, `page ${page + 1} of ${pages} · ${ks.length} records`),
      h("button", { type: "button", disabled: page + 1 >= pages, onclick: () => (page++, draw()) }, "Next"),
    );
  };
  [optBox, tdBox].forEach((c) => c.addEventListener("change", () => ((page = 0), draw())));
  draw();
}

async function smartsGeometry(card, kind, hits, tuplesOf, current) {
  // Every bond, angle, proper chain and improper star of every optimization record, with its value(s),
  // validity, frozen-bond flag and Sage parameter, is precomputed (scripts/05b_geometry_universe.py).
  // Here matched tuples are only turned into canonical keys and looked up; no geometry is computed.
  const T = state.topologies;
  const handler = MEASURE_HANDLER[kind];
  card.append(h("h3", {}, `QM ${kind === "bond" ? "bond length" : kind === "angle" ? "angle" : kind + " dihedral"} of the mapped atoms (optimization minima)`));
  const note = h("p", { class: "muted" }, "Loading precomputed geometry…");
  card.append(note);
  const geometry = parseGeometry(await fetchBuffer(DATA + "geom_opt.bin"));
  if (!current()) return;
  const values = [], weights = [];
  const paramCounts = new Map();
  let nInvalid = 0, nConstrained = 0;
  const optHits = hits.map((t, k) => [t, k]).filter(([t]) => T.source[t] === "opt");
  for (let n = 0; n < optHits.length; n++) {
    if (n % 500 === 0) {
      note.textContent = `Looking up… ${n} / ${optHits.length} records`;
      await yieldToBrowser();
      if (!current()) return;
    }
    const [t, k] = optHits[n];
    const table = geometry.lookup(kind, t);
    const seen = new Set();
    for (const tup of tuplesOf(k)) {
      const key = universeKey(kind, tup);
      if (seen.has(key)) continue; // reverse / permuted matches of the same interaction
      seen.add(key);
      const row = table.get(key);
      if (!row) throw new Error(`no precomputed ${kind} ${key} in record ${T.record_id[t]}`);
      if (row.frozen) {
        nConstrained++;
        continue;
      }
      const pid = row.param >= 0 ? state.params[row.param].param_id : "(no Sage parameter)";
      paramCounts.set(pid, (paramCounts.get(pid) || 0) + 1);
      if (!row.valid) {
        nInvalid++;
        continue;
      }
      for (const v of row.values) (values.push(v), weights.push(1 / row.values.length));
    }
  }
  note.remove();
  if (!values.length) {
    card.append(h("p", { class: "muted" }, "No measurable observations in the optimization minima."));
    return;
  }
  const isDihedral = kind === "proper" || kind === "improper";
  let lo = -180, hi = 180;
  if (!isDihedral) {
    [lo, hi] = extent(values);
    const pad = (hi - lo) * 0.05 || (kind === "bond" ? 0.01 : 1);
    [lo, hi] = [lo - pad, hi + pad];
  }
  const chart = h("div", { class: "chart-box" });
  card.append(chart);
  histogramChart(chart, {
    series: [{ name: "Optimization minima", slot: 1, hist: histogram(values, weights, lo, hi, isDihedral ? 72 : 60) }],
    xLabel: kind === "bond" ? "bond length (Å)" : kind === "angle" ? "angle (°)" : "dihedral (°)",
    yLabel: kind === "improper" ? "count (each improper = 1)" : "count",
    xTicks: isDihedral ? [-180, -120, -60, 0, 60, 120, 180] : undefined,
  });
  const st = h("div", { class: "stats" });
  if (isDihedral) {
    const c = weightedCircularStats(values, weights);
    st.append(stat("circular mean", `${fmt(c.circMean, 1)}°`), stat("mean resultant length R", fmt(c.resultantLength, 3)), stat("circular SD", c.circStd === null ? "–" : `${fmt(c.circStd, 1)}°`));
  } else {
    const s = weightedLinearStats(values, weights);
    const d = kind === "bond" ? 4 : 2;
    st.append(stat("mean", fmt(s.mean, d)), stat("SD", fmt(s.std, d)), stat("median", fmt(s.median, d)), stat("range", `${fmt(s.min, d)} – ${fmt(s.max, d)}`));
  }
  if (nInvalid) st.append(stat("undefined (near-linear)", nInvalid));
  if (nConstrained) st.append(stat("excluded: about a frozen bond", nConstrained));
  card.append(st);
  const rows = [...paramCounts.entries()].sort((a, b) => b[1] - a[1]);
  card.append(
    h("h3", {}, `Sage ${HANDLER_LABEL[handler].toLowerCase()} parameters on these atoms`),
    h("p", { class: "muted" }, "From the precomputed label_molecules assignments (exact), counted per matched interaction in the optimization records. Improper stars with no Sage improper are listed as “(no Sage parameter)”."),
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, h("th", {}, "Parameter"), h("th", {}, "SMIRKS"), h("th", { class: "num" }, "Matched interactions"))),
        h(
          "tbody",
          {},
          ...rows.map(([pid, n]) => {
            const p = state.params.find((x) => x.param_id === pid);
            return h("tr", {}, h("td", {}, p ? h("a", { href: `#param/${pid}` }, pid) : pid), h("td", {}, h("span", { class: "smirks" }, p?.smirks || "")), h("td", { class: "num" }, n));
          }),
        ),
      ),
    ),
  );
}

// ---------------------------------------------------------------- Molecule search
async function renderMolecule(initial) {
  const input = h("input", { type: "text", spellcheck: "false", value: initial || "c1ccc2c(c1)ccc(=O)o2", "aria-label": "SMILES" });
  const run = h("button", { type: "button", class: "primary" }, "Compare");
  const out = h("div");
  main.append(
    h(
      "section",
      { class: "card" },
      h("h2", {}, "Molecule search"),
      h("p", { class: "secondary" }, "Compare a molecule (SMILES) with the training set: nearest neighbours by Morgan fingerprint (radius 2, 2048 bits, Tanimoto), training molecules that contain it as a substructure (heavy-atom graph: hydrogen counts of the query are not constrained, so -OH also matches -OMe), and exact matches."),
      h("p", { class: "note" }, "Force-field parameters are shown only for molecules that are in the training set, from the precomputed label_molecules assignments. Assigning Sage parameters to new molecules in the browser is not offered: stock RDKit.js cannot apply the MDL aromaticity model OpenFF uses, so its assignments could differ from the toolkit's."),
      h("div", { class: "row" }, h("div", { class: "grow" }, input), run),
    ),
    out,
  );
  const go = async () => {
    const target = `#molecule/${encodeURIComponent(input.value.trim())}`;
    if (location.hash !== target) {
      location.hash = target; // the router re-renders and runs the search
      return;
    }
    out.innerHTML = "<p class='muted'>Comparing… (the first search parses all training molecules, a few seconds)</p>";
    try {
      const [r, fpBuf, popBuf, canonical] = await Promise.all([
        rdkit("molecule", { smiles: input.value.trim() }),
        fetchBuffer(DATA + "fp_morgan.bin"),
        fetchBuffer(DATA + "fp_popcount.bin"),
        fetchJSON(DATA + "canonical_smiles.json"),
      ]);
      // exact match: the query's RDKit.js canonical SMILES vs the precomputed dataset canonical SMILES
      r.exact = canonical.flatMap((c, m) => (c === r.canonical ? [m] : []));
      out.innerHTML = "";
      await showMoleculeResults(out, r, new Uint8Array(fpBuf), new Uint16Array(popBuf));
    } catch (e) {
      out.innerHTML = `<p class="error">${esc(e.message)}</p>`;
    }
  };
  run.addEventListener("click", go);
  input.addEventListener("keydown", (e) => e.key === "Enter" && go());
  if (initial) go();
}

async function showMoleculeResults(out, r, fps, popcounts) {
  const M = state.molecules, T = state.topologies;
  const query = h("div", { class: "mol", style: "max-width:380px" });
  query.innerHTML = r.svg;
  query.append(h("div", { class: "cap mono" }, r.canonical));
  out.append(h("section", { class: "card" }, h("h3", {}, "Query"), query));

  if (r.exact.length) {
    const card = h("section", { class: "card" }, h("h3", {}, "In the training set"));
    out.append(card);
    const [asgBuf, conformers] = await Promise.all([fetchBuffer(DATA + "assignments.bin"), fetchJSON(DATA + "conformers.json")]);
    const asg = parseAssignments(asgBuf);
    for (const m of r.exact) {
      for (const t of M.topologies[m]) {
        const counts = new Map();
        for (const a of asg.forTopology(t)) {
          const p = state.params[a.paramIndex];
          counts.set(p.param_id, (counts.get(p.param_id) || 0) + 1);
        }
        const unas = T.unassigned_propers[t];
        const isTd = T.source[t] === "td";
        // (DOM append() would print a null child as the text "null", so build the list explicitly)
        const parts = [
          h(
            "p",
            {},
            `${isTd ? "TorsionDrive" : "Optimization"} record ${T.record_id[t]} (${T.entry_name[t]}), ${T.n_conf[t]} ${isTd ? "grid points" : "conformer"}${isTd ? "" : `, final QM energy ${conformers.energy_hartree[T.conf_start[t]].toFixed(6)} Eh${conformers.opt_group_rel_kcal[T.conf_start[t]] !== null ? ` (ΔE ${conformers.opt_group_rel_kcal[T.conf_start[t]].toFixed(2)} kcal/mol vs lowest optimization conformer)` : ""}`} `,
            h("button", { type: "button", onclick: () => openViewer({ t, state, rdkit }).catch((e) => alert(e.message)) }, isTd ? "3D scan + energy" : "3D structure"),
          ),
          h("div", { class: "chips" }, ...[...counts.entries()].map(([pid, n]) => h("a", { href: `#param/${pid}` }, h("button", { type: "button" }, `${pid}${n > 1 ? ` ×${n}` : ""}`)))),
        ];
        if (unas.length) parts.push(h("p", { class: "note" }, `${unas.length} proper torsion(s) have no Sage 2.3.0 parameter: ${unas.map((a) => a.join("-")).join(", ")}`));
        card.append(...parts);
      }
    }
  } else {
    out.append(h("section", { class: "card" }, h("p", {}, "Not in the training set (exact match on canonical isomeric SMILES).")));
  }

  const sim = tanimotoAll(r.fp, fps, popcounts);
  const order = Array.from(sim.keys()).sort((a, b) => sim[b] - sim[a]).slice(0, 24);
  const simGrid = h("div", { class: "mols" });
  out.append(h("section", { class: "card" }, h("h3", {}, "Most similar training molecules"), simGrid));
  for (const m of order) simGrid.append(molCard(h("div", { class: "cap" }, `Tanimoto ${sim[m].toFixed(3)}`, h("br"), h("span", { class: "mono" }, M.display_smiles[m])), moleculeSvg(DATA, m), { t: representativeTopology(m) }));

  const sub = r.substructure;
  const subGrid = h("div", { class: "mols" });
  out.append(h("section", { class: "card" }, h("h3", {}, `Training molecules containing the query as a substructure (${sub.length})`), sub.length > 24 ? h("p", { class: "muted" }, "Showing 24.") : null, subGrid));
  for (const m of sub.slice(0, 24)) subGrid.append(molCard(h("div", { class: "cap mono" }, M.display_smiles[m]), moleculeSvg(DATA, m), { t: representativeTopology(m) }));
}

// ---------------------------------------------------------------- About
function renderAbout() {
  const meta = state.meta;
  const v = meta.validation;
  const src = meta.sources;
  const zero = v.label_summary.zero_coverage_params;
  const unas = state.topologies.unassigned_propers.map((u, t) => [u, t]).filter(([u]) => u.length);
  main.append(
    h(
      "section",
      { class: "card" },
      h("h2", {}, "About this explorer"),
      h("p", {}, "This page compares SMARTS patterns and molecules with the training set of the OpenFF Sage 2.3.0 force field, and each of its valence parameters with the QM geometries of the atoms it is assigned to. All data were preprocessed with the scripts in this repository (OpenFF Toolkit ForceField.label_molecules); the browser only reads the resulting files and runs RDKit.js."),
      h("p", { class: "muted" }, "Claude Opus 5.5 (Anthropic) was used in creating this page and its preprocessing scripts."),
      h("h3", {}, "Data"),
      h(
        "ul",
        {},
        h("li", {}, "Optimizations: ", h("a", { href: `https://doi.org/${src.optimization.doi}` }, src.optimization.title), ` (Zenodo ${src.optimization.zenodo_record}, ${src.optimization.license?.id}): ${meta.counts.topologies.opt} records.`),
        h("li", {}, `TorsionDrives: QCArchive dataset “${src.torsiondrive.dataset_name}” (id ${src.torsiondrive.dataset_id}, fetched ${src.torsiondrive.fetched_utc.slice(0, 10)} with qcportal ${src.torsiondrive.qcportal_version}), the minimum-energy optimization at each grid point: ${meta.counts.topologies.td} scans, ${meta.counts.conformers.td} grid points.`),
        h("li", {}, "Force field: ", h("a", { href: src.force_field.url }, "openff_unconstrained-2.3.0.offxml"), ` (openff-forcefields ${src.force_field.ref}, sha256 ${src.force_field.sha256.slice(0, 12)}…).`),
        h("li", {}, `${meta.counts.molecules} unique molecules; ${meta.counts.exclusions} records excluded.`),
        h("li", {}, `RDKit.js ${src.rdkit_js.version} (vendored; the same bytes computed the fingerprints).`),
      ),
      h("h3", {}, "Validation"),
      h(
        "ul",
        {},
        h("li", {}, `Atom order: every record's QC symbols and connectivity match its mapped SMILES (${v.extract_summary.n_conformers} conformers, 0 exclusions). Driven dihedrals match their grid angle within ${fmt(v.extract_summary.grid_deviation_deg.max, 3)}°.`),
        h("li", {}, `Assignments: label_molecules and OpenFF Interchange agree for all ${Object.values(v.label_summary.n_assignments).reduce((a, b) => a + b)} interactions.`),
        h("li", {}, `Torsion energies computed here match OpenMM (Reference platform) to ${v.energy_validation.proper_max_abs_diff_kj.toExponential(1)} kJ/mol (propers) and ${v.energy_validation.improper_max_abs_diff_kj.toExponential(1)} kJ/mol (impropers) across ${v.energy_validation.n_conformers} conformers.`),
        h("li", {}, `SMARTS matching in RDKit.js reproduces OpenFF's matching for all ${v.parity_summary.n_smirks} Sage SMIRKS on all ${v.parity_summary.n_topologies} records (${v.parity_summary.n_pairs_with_matches_python} matching pairs, ${v.parity_summary.n_mismatched_pairs} mismatches).`),
      ),
      h("h3", {}, "How to read the distributions"),
      h(
        "ul",
        {},
        h("li", {}, "Dihedrals use the IUPAC/OpenMM sign convention and circular statistics. A dihedral is undefined when either bond angle is within ~1.15° of linear; these are counted but excluded."),
        h("li", {}, "Each improper counts once: its three trefoil terms are each weighted 1/3."),
      ),
      zero.length ? [h("h3", {}, `Parameters with no training-set coverage (${zero.length})`), h("p", {}, ...zero.flatMap((z) => [h("a", { href: `#param/${z}` }, z), " "]))] : null,
      unas.length ? [h("h3", {}, "Interactions without a Sage parameter"), ...unas.map(([u, t]) => h("p", {}, `${state.topologies.source[t]} record ${state.topologies.record_id[t]}: `, h("span", { class: "mono" }, state.molecules.display_smiles[state.topologies.mol_idx[t]]), ` — ${u.length} proper torsion(s)`))] : null,
      h("h3", {}, "QM specifications"),
      h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Program"), h("th", {}, "Method / basis"), h("th", {}, "Optimizer keywords"))), h("tbody", {}, ...meta.specs.map((s) => h("tr", {}, h("td", {}, `${s.qc_program} / ${s.optimizer}`), h("td", {}, `${s.method} / ${s.basis}`), h("td", { class: "mono" }, JSON.stringify(s.optimizer_keywords))))))),
      h("h3", {}, "Licences and citation"),
      h("p", {}, "Training data: CC-BY-4.0 (Zenodo record above). Force field: openff-forcefields (CC-BY-4.0). RDKit and RDKit.js: BSD-3-Clause. Please cite the Sage 2.3.0 dataset and force field when using this material."),
    ),
  );
}

// ---------------------------------------------------------------- boot
document.getElementById("theme-toggle").addEventListener("click", () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const cur = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("theme", next);
  } catch (e) {}
});

(async function boot() {
  try {
    const [meta, params, molecules, topologies] = await Promise.all(["meta.json", "params.json", "molecules.json", "topologies.json"].map((f) => fetchJSON(DATA + f)));
    Object.assign(state, { meta, params, molecules, topologies });
    document.getElementById("subtitle").textContent = `${meta.counts.molecules} molecules · ${params.length} valence parameters · openff_unconstrained-2.3.0`;
    window.addEventListener("hashchange", route);
    route();
  } catch (e) {
    main.innerHTML = `<p class="error">Failed to load data: ${esc(e.message)}</p>`;
  }
})();
