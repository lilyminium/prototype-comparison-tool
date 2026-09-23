// Put coordinates into a precomputed molblock (text formatting only). The explicit-H depiction molblock is
// already in topology atom order (scripts/08e_depictions.py), so no reordering or bond perception happens.
export function withCoordinates(molblock, xyz) {
  const lines = molblock.split("\n");
  const n = parseInt(lines[3].slice(0, 3), 10);
  if (3 * n !== xyz.length) throw new Error("molblock/coordinate atom count mismatch");
  const f = (v) => v.toFixed(4).padStart(10);
  for (let i = 0; i < n; i++) lines[4 + i] = f(xyz[3 * i]) + f(xyz[3 * i + 1]) + f(xyz[3 * i + 2]) + lines[4 + i].slice(30);
  return lines.join("\n");
}
