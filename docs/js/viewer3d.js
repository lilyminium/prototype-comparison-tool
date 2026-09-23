// 3D viewer (NGL Viewer, the library nglview wraps). One shared viewer in a <dialog>, so at most one
// extra WebGL context. Optimization records: the QM minimum. TorsionDrive records: an animation over
// the grid points (ordered by grid angle) synchronized with the QM relative-energy profile.
import { scanChart } from "./charts.js";
import { conformerCoords, fetchBuffer, fetchJSON } from "./data.js";
import { superpose } from "./align.js";
import { topologyMolblock } from "./molblock.js";

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
 * atoms). `rdkit(type, args)` calls the RDKit worker; `state` has topologies/molecules.
 */
export async function openViewer({ t, highlight = [], label = "", state, rdkit }) {
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

  const [NGL, coordsBuf, mb, conformers] = await Promise.all([
    loadNGL(),
    fetchBuffer(`data/coords_${isTd ? "td" : "opt"}.bin`),
    rdkit("molblock", { t }),
    fetchJSON("data/conformers.json"),
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

  // Frames: optimization = 1; TorsionDrive = grid points sorted by grid angle
  const frames = [];
  for (let j = 0; j < T.n_conf[t]; j++) {
    const c = T.conf_start[t] + j;
    frames.push({ j, grid: isTd ? conformers.grid_deg[c] : null, energy: isTd ? conformers.rel_energy_kcal[c] : null });
  }
  if (isTd) frames.sort((a, b) => a.grid - b.grid);
  const xyzOf = (f) => conformerCoords(coords, T, t, f.j);

  const sdf = topologyMolblock(mb.molblock, mb.order, xyzOf(frames[0]));
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
    await addOtherConformers({ d, t, comp, coords, mb, state, rdkit, conformers });
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
 * Optimization records of the same molecule are separate conformers. Show each with a checkbox,
 * superposed on the clicked record (Kabsch on heavy atoms). Records of one molecule share a canonical
 * MDL SMILES, so SMILES atom n corresponds across records.
 */
async function addOtherConformers({ d, t, comp, coords, mb, state, rdkit, conformers }) {
  const T = state.topologies;
  const others = state.molecules.topologies[T.mol_idx[t]].filter((u) => u !== t && T.source[u] === "opt");
  const box = d.querySelector(".viewer-conformers");
  if (!others.length) return;
  box.hidden = false;
  const heavy = mb.molblock.split("\n").slice(4, 4 + mb.order.length).map((l) => l.slice(31, 34).trim() !== "H");
  const target = conformerCoords(coords, T, t, 0);
  const row = (label, color, checked, onChange) => {
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checked;
    cb.addEventListener("change", () => onChange(cb.checked));
    const lab = document.createElement("label");
    lab.className = "inline";
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = color;
    lab.append(cb, sw, label);
    box.append(lab);
  };
  const title = document.createElement("div");
  title.className = "muted";
  title.textContent = `${others.length + 1} optimization conformers of this molecule (superposed on heavy atoms):`;
  box.append(title);
  // Relative QM energies are precomputed (09_build_site_data.py) and only present when all records share
  // the same QC level
  const dE = (u) => {
    const c = T.conf_start[u];
    return conformers.opt_group_rel_kcal[c] === null ? "" : `, ΔE ${conformers.opt_group_rel_kcal[c].toFixed(2)} kcal/mol`;
  };
  if (conformers.opt_group_same_level[T.conf_start[t]] === false) {
    const warn = document.createElement("div");
    warn.className = "muted";
    warn.textContent = "Relative QM energies not shown: these records were computed at different QC levels.";
    box.append(warn);
  }
  row(`record ${T.record_id[t]} (this record, element colours${dE(t)})`, "linear-gradient(90deg,#909090,#ff0d0d,#3050f8)", true, (on) => comp.setVisibility(on));
  for (const [n, u] of others.entries()) {
    if (T.mdl_smiles[u] !== T.mdl_smiles[t]) continue; // atom correspondence requires identical canonical SMILES
    const mbU = await rdkit("molblock", { t: u });
    const pairs = [];
    mbU.order.forEach((topU, k) => heavy[k] && pairs.push([topU, mb.order[k]]));
    const { coords: aligned, rmsd } = superpose(conformerCoords(coords, T, u, 0), target, pairs);
    const color = CONFORMER_COLORS[n % CONFORMER_COLORS.length];
    const c = await stage.loadFile(new Blob([topologyMolblock(mbU.molblock, mbU.order, aligned)], { type: "text/plain" }), { ext: "sdf" });
    c.addRepresentation("ball+stick", { color, multipleBond: "symmetric", aspectRatio: 1.8, radiusScale: 0.7, opacity: 0.85 });
    row(`record ${T.record_id[u]} (heavy-atom RMSD ${rmsd.toFixed(2)} Å${dE(u)})`, color, true, (on) => c.setVisibility(on));
  }
}
