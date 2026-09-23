// Rewrite an RDKit V2000 molblock (atoms in SMILES order) into topology atom order with given
// coordinates, so that viewer atom index == topology atom index. No DOM; tested in tests/js/.

/**
 * @param {string} v2000   molblock from RDKit (atoms in mdl_smiles order)
 * @param {number[]} order order[n] = topology atom of molblock atom n
 * @param {ArrayLike<number>} xyz topology-ordered coordinates (Å), length 3 * nAtoms
 */
export function topologyMolblock(v2000, order, xyz) {
  const lines = v2000.split("\n");
  const counts = lines[3];
  const nAtoms = parseInt(counts.slice(0, 3), 10);
  const nBonds = parseInt(counts.slice(3, 6), 10);
  if (nAtoms !== order.length || 3 * nAtoms !== xyz.length) throw new Error("molblock/topology atom count mismatch");
  const atomLines = lines.slice(4, 4 + nAtoms);
  const bondLines = lines.slice(4 + nAtoms, 4 + nAtoms + nBonds);
  const f = (v) => v.toFixed(4).padStart(10);
  const newAtoms = new Array(nAtoms);
  atomLines.forEach((line, n) => {
    const t = order[n];
    newAtoms[t] = f(xyz[3 * t]) + f(xyz[3 * t + 1]) + f(xyz[3 * t + 2]) + line.slice(30);
  });
  const pad3 = (v) => String(v).padStart(3);
  const newBonds = bondLines.map((line) => {
    const a = order[parseInt(line.slice(0, 3), 10) - 1] + 1;
    const b = order[parseInt(line.slice(3, 6), 10) - 1] + 1;
    return pad3(a) + pad3(b) + line.slice(6);
  });
  // Charges are carried in the atom lines' charge field; drop property lines that index atoms
  return [...lines.slice(0, 4), ...newAtoms, ...newBonds, "M  END", ""].join("\n");
}
