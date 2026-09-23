// 3D viewer (NGL Viewer, the library nglview wraps). One shared viewer in a <dialog>, so at most one
// extra WebGL context. Optimization records: the QM minimum, with the molecule's other optimization
// conformers superposed (alignments and RMSDs precomputed by scripts/08d_align_conformers.py).
// TorsionDrive records: an animation over the grid points synchronized with the QM energy profile
// (frame order precomputed). Structures come from the precomputed explicit-H depiction molblock, which is
// in topology atom order; only its coordinates are replaced with QM coordinates.
import { scanChart } from "./charts.js";
import { conformerCoords, depiction, fetchBuffer, fetchJSON } from "./data.js";
import { withCoordinates } from "./molblock.js";

const HIGHLIGHT = "#eb6834"; // series slot 2; a highlight, not a data series
// Other conformers of the same molecule (categorical slots 3 and 5; the clicked record keeps element colors)
const CONFORMER_COLORS = ["#1baf7a", "#e87ba4", "#4a3aa7"];
let nglPromise = null;
function loadNGL() {
  if (!nglPromise) {
    nglPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "vendor/ngl/ngl.js";
      s.onload = () => resolve(window.NGL);
      s.onerror = () => reject(new Error("could not load NGL"));
      document.head.appendChild(s);
    });
  }
  return nglPromise;
}

let dialog = null;
let stage = null;
let timer = null;

function ensureDialog() {
  if (dialog) return dialog;
  dialog = document.createElement("dialog");
  dialog.className = "viewer3d";
  dialog.innerHTML = `
    <div class="viewer-head"><h3 class="viewer-title"></h3><button type="button" class="viewer-close" aria-label="Close">Close</button></div>
    <p class="viewer-sub muted"></p>
    <div class="viewer-body">
      <div class="viewer-stage"></div>
      <div class="viewer-side">
        <div class="viewer-conformers" hidden></div>
        <div class="viewer-controls" hidden>
          <button type="button" class="viewer-play">Pause</button>
          <input type="range" class="viewer-slider" min="0" value="0" aria-label="Grid point" />
          <div class="viewer-readout"></div>
        </div>
        <div class="viewer-chart chart-box"></div>
        <div class="viewer-note muted"></div>
      </div>
    </div>`;
  document.body.appendChild(dialog);
  dialog.querySelector(".viewer-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => {
    clearInterval(timer);
    timer = null;
    stage?.removeAllComponents();
  });
  dialog.addEventListener("click", (e) => e.target === dialog && dialog.close());
  return dialog;
}

/**
 * Open the viewer for topology t. `highlight`: topology atom indices to emphasise (e.g. the matched
 * atoms). `state` has topologies/molecules. All structure data is precomputed.
 */
export async function openViewer({ t, highlight = [], label = "", state }) {
  const T = state.topologies;
  const d = ensureDialog();
  const isTd = T.source[t] === "td";
  d.querySelector(".viewer-title").textContent = `${isTd ? "TorsionDrive" : "Optimization"} record ${T.record_id[t]}`;
  d.querySelector(".viewer-sub").textContent = `${state.molecules.display_smiles[T.mol_idx[t]]}${label ? ` · ${label}` : ""}`;
  d.querySelector(".viewer-chart").innerHTML = "";
  d.querySelector(".viewer-readout").textContent = "";
  d.querySelector(".viewer-note").textContent = "Loading…";
  d.querySelector(".viewer-controls").hidden = !isTd; // optimization records are snapshots: no playback
  d.querySelector(".viewer-conformers").hidden = true;
  d.querySelector(".viewer-conformers").innerHTML = "";
  d.showModal();

  const [NGL, coordsBuf, dep, conformers, groups] = await Promise.all([
    loadNGL(),
    fetchBuffer(`data/coords_${isTd ? "td" : "opt"}.bin`),
    depiction("data/", t),
    fetchJSON("data/conformers.json"),
    isTd ? null : fetchJSON("data/conformer_groups.json"),
  ]);
  const coords = new Float32Array(coordsBuf);
  const stageEl = d.querySelector(".viewer-stage");
  if (!stage) {
    stage = new NGL.Stage(stageEl, { backgroundColor: getComputedStyle(document.body).getPropertyValue("--surface").trim() || "white" });
    window.addEventListener("resize", () => stage.handleResize());
  }
  stage.setParameters({ backgroundColor: getComputedStyle(document.body).getPropertyValue("--surface").trim() || "white" });
  stage.removeAllComponents();
  stage.handleResize();

  // Frames: optimization = 1; TorsionDrive = grid points in the precomputed grid-angle order
  const frames = isTd
    ? T.frame_order[t].map((j) => ({ j, grid: conformers.grid_deg[T.conf_start[t] + j], energy: conformers.rel_energy_kcal[T.conf_start[t] + j] }))
    : [{ j: 0, grid: null, energy: null }];
  const xyzOf = (f) => conformerCoords(coords, T, t, f.j);

  const sdf = withCoordinates(dep.molblock_h, xyzOf(frames[0]));
  const comp = await stage.loadFile(new Blob([sdf], { type: "text/plain" }), { ext: "sdf", name: `record-${T.record_id[t]}` });
  comp.addRepresentation("ball+stick", { multipleBond: "symmetric", aspectRatio: 1.8, radiusScale: 0.9 });
  const driven = isTd ? T.driven[t] : null;
  const emph = [...new Set([...(highlight || []), ...(driven || [])])];
  if (emph.length) {
    comp.addRepresentation("ball+stick", { sele: emph.map((i) => `@${i}`).join(" or "), color: HIGHLIGHT, radiusScale: 1.35, aspectRatio: 1.4 });
  }
  comp.autoView();

  const note = d.querySelector(".viewer-note");
  note.textContent = isTd
    ? `Driven dihedral ${driven.join("-")} highlighted. Each frame is the constrained QM minimum at that grid angle.`
    : `QM-optimized geometry (B3LYP-D3BJ/DZVP); final QM energy ${conformers.energy_hartree[T.conf_start[t]].toFixed(6)} Eh.${emph.length ? " Highlighted atoms: " + emph.join(", ") + "." : ""}`;
  if (!isTd) {
    await addOtherConformers({ d, t, comp, dep, state, conformers, groups });
    return;
  }

  // TorsionDrive: animation + synchronized energy profile
  const slider = d.querySelector(".viewer-slider");
  const play = d.querySelector(".viewer-play");
  const readout = d.querySelector(".viewer-readout");
  slider.max = String(frames.length - 1);
  const chart = scanChart(d.querySelector(".viewer-chart"), {
    points: frames.map((f) => ({ x: f.grid, y: f.energy })),
    xLabel: "driven dihedral, grid angle (°)",
    yLabel: "relative QM energy (kcal/mol)",
    width: 520,
    height: 240,
  });
  const structure = comp.structure;
  const show = (k) => {
    const f = frames[k];
    structure.updatePosition(Float32Array.from(xyzOf(f)));
    comp.updateRepresentations({ position: true });
    slider.value = String(k);
    readout.textContent = `${f.grid}° · ${f.energy.toFixed(2)} kcal/mol · frame ${k + 1}/${frames.length}`;
    chart.setCurrent(k);
  };
  let k = 0;
  const start = () => {
    clearInterval(timer);
    timer = setInterval(() => show((k = (k + 1) % frames.length)), 350);
    play.textContent = "Pause";
  };
  const stop = () => {
    clearInterval(timer);
    timer = null;
    play.textContent = "Play";
  };
  play.onclick = () => (timer ? stop() : start());
  slider.oninput = () => {
    stop();
    show((k = Number(slider.value)));
  };
  chart.onSelect = (i) => {
    stop();
    show((k = i));
  };
  show(0);
  start();
}

/**
 * Optimization records of the same molecule are separate conformers: each gets a checkbox; all are shown
 * superposed on the clicked record. Superpositions, RMSDs and relative QM energies are precomputed.
 */
async function addOtherConformers({ d, t, comp, dep, state, conformers, groups }) {
  const T = state.topologies;
  const others = groups[String(t)] || [];
  const box = d.querySelector(".viewer-conformers");
  if (!others.length) return;
  box.hidden = false;
  const row = (label, color, onChange) => {
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.addEventListener("change", () => onChange(cb.checked));
    const lab = document.createElement("label");
    lab.className = "inline";
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = color;
    lab.append(cb, sw, label);
    box.append(lab);
  };
  const dE = (u) => {
    const e = conformers.opt_group_rel_kcal[T.conf_start[u]];
    return e === null ? "" : `, ΔE ${e.toFixed(2)} kcal/mol`;
  };
  const title = document.createElement("div");
  title.className = "muted";
  title.textContent = `${others.length + 1} optimization conformers of this molecule (superposed on heavy atoms):`;
  box.append(title);
  if (conformers.opt_group_same_level[T.conf_start[t]] === false) {
    const warn = document.createElement("div");
    warn.className = "muted";
    warn.textContent = "Relative QM energies not shown: these records were computed at different QC levels.";
    box.append(warn);
  }
  row(`record ${T.record_id[t]} (this record, element colours${dE(t)})`, "linear-gradient(90deg,#909090,#ff0d0d,#3050f8)", (on) => comp.setVisibility(on));
  for (const [n, o] of others.entries()) {
    const other = await depiction("data/", o.t);
    const color = CONFORMER_COLORS[n % CONFORMER_COLORS.length];
    const c = await stage.loadFile(new Blob([withCoordinates(other.molblock_h, o.coords)], { type: "text/plain" }), { ext: "sdf" });
    c.addRepresentation("ball+stick", { color, multipleBond: "symmetric", aspectRatio: 1.8, radiusScale: 0.7, opacity: 0.85 });
    row(`record ${T.record_id[o.t]} (heavy-atom RMSD ${o.rmsd.toFixed(2)} Å${dE(o.t)})`, color, (on) => c.setVisibility(on));
  }
}
