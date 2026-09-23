// Pure helpers used by the RDKit.js Web Worker (classic script) and by the node tests (require()).
(function (root) {
  "use strict";

  // Loading training molecules for SMARTS/SMIRKS matching: keep OpenFF's MDL aromaticity and
  // explicit hydrogens exactly as supplied (validated against OpenFF: scripts/08a/08b).
  const MDL_MOL_OPTIONS = JSON.stringify({ setAromaticity: false, kekulize: false, removeHs: false });
  // OpenFF's RDKitToolkitWrapper._find_smarts_matches settings
  const MATCH_OPTIONS = JSON.stringify({ uniquify: false, maxMatches: 4294967295, useChirality: true });
  const MORGAN_OPTIONS = JSON.stringify({ radius: 2, nBits: 2048 });

  /** Atom-map numbers of query atoms from a V3000 molblock (the last positional atom field). */
  function queryMapInfo(v3000) {
    const lines = v3000.split("\n");
    const atomMaps = [];
    const bonds = [];
    let block = null;
    for (const raw of lines) {
      const line = raw.trim();
      if (line === "M  V30 BEGIN ATOM") block = "atom";
      else if (line === "M  V30 BEGIN BOND") block = "bond";
      else if (line.startsWith("M  V30 END")) block = null;
      else if (block === "atom" && line.startsWith("M  V30 ")) {
        // M  V30 index type x y z aamap [KEY=VALUE ...]
        const tok = line.slice(7).trim().split(/\s+/);
        atomMaps.push(Number(tok[5]));
      } else if (block === "bond" && line.startsWith("M  V30 ")) {
        const tok = line.slice(7).trim().split(/\s+/);
        bonds.push([Number(tok[2]) - 1, Number(tok[3]) - 1]);
      }
    }
    // Map numbers must be unique and run 1..n without gaps; anything else is rejected (contiguous=false)
    const positive = atomMaps.filter((m) => m > 0);
    const nTags = positive.length ? Math.max(...positive) : 0;
    const unique = new Set(positive).size === positive.length;
    const tagIndex = new Array(nTags).fill(-1);
    atomMaps.forEach((m, i) => {
      if (m > 0) tagIndex[m - 1] = i;
    });
    let noGaps = true;
    for (let k = 0; k < nTags; k++) if (tagIndex[k] < 0) noGaps = false;
    const contiguous = unique && noGaps;
    const tagOf = new Map(tagIndex.map((q, t) => [q, t + 1]));
    const bondedTagPairs = bonds.filter(([a, b]) => tagOf.has(a) && tagOf.has(b)).map(([a, b]) => [tagOf.get(a), tagOf.get(b)]);
    return { nAtoms: atomMaps.length, tagIndex, nTags, contiguous, bondedTagPairs };
  }

  /**
   * Unique tagged tuples (in topology atom indices) of `qmol` in `mol`.
   * With tags: ordered tagged tuples (OpenFF semantics, all permutations kept, duplicates removed).
   * Without tags: one tuple of all query atoms per unique match.
   */
  function matchTuples(mol, qmol, tagIndex, mdlOrder, options) {
    const res = JSON.parse(mol.get_substruct_matches(qmol, options || MATCH_OPTIONS));
    if (!Array.isArray(res)) return [];
    const seen = new Map();
    for (const { atoms } of res) {
      const idx = tagIndex.length ? tagIndex.map((q) => atoms[q]) : atoms;
      const tuple = idx.map((a) => mdlOrder[a]);
      const key = tagIndex.length ? tuple.join(",") : [...tuple].sort((a, b) => a - b).join(",");
      if (!seen.has(key)) seen.set(key, tuple);
    }
    return [...seen.values()];
  }

  /**
   * Render a PRECOMPUTED dataset depiction (molblock with 2D coordinates from scripts/08e_depictions.py)
   * with highlights. Refuses molecules without coordinates, so RDKit.js never lays out a dataset molecule.
   * Output is identical to Python RDKit for the same molblock/highlights/size (tests/js/render_oracle.test.mjs).
   */
  function renderDepiction(RDKit, molblock, atoms, bonds, width, height) {
    const mol = RDKit.get_mol(molblock, JSON.stringify({ removeHs: false }));
    if (!mol) throw new Error("could not load a precomputed depiction");
    try {
      if (!mol.has_coords()) throw new Error("precomputed depiction has no 2D coordinates");
      // a conformer that exists but is degenerate (all atoms at one point) is not a usable layout either
      const conf = (JSON.parse(mol.get_json()).molecules[0].conformers || [])[0];
      const pts = conf ? conf.coords : [];
      const spread = pts.length > 1 && pts.some((c) => c[0] !== pts[0][0] || c[1] !== pts[0][1]);
      if (pts.length > 1 && !spread) throw new Error("precomputed depiction has no 2D coordinates (degenerate)");
      return mol.get_svg_with_highlights(JSON.stringify({ atoms, bonds, width, height, clearBackground: false }));
    } finally {
      mol.delete();
    }
  }

  const api = { MDL_MOL_OPTIONS, MATCH_OPTIONS, MORGAN_OPTIONS, queryMapInfo, matchTuples, renderDepiction };
  root.WorkerLib = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
