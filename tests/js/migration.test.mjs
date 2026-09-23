// Migration oracle (PLAN_precompute.md S4): the Python-precomputed parameter pages must show what the old
// browser code showed (tests/js/snapshot_before.json, captured before the JS paths were removed).
// Accepted differences: none for counts, bins, examples and scans; statistics within 1e-9 relative.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const DATA = new URL("../../docs/data/", import.meta.url);
const snapPath = new URL("snapshot_before.json", import.meta.url);
const params = JSON.parse(readFileSync(new URL("params.json", DATA), "utf8"));

const close = (a, b, rel = 1e-9) => (a === null && b === null) || Math.abs(a - b) <= rel * Math.max(1, Math.abs(a), Math.abs(b));

test("precomputed parameter pages equal the old browser output for every parameter", { skip: !existsSync(snapPath) }, () => {
  const snap = JSON.parse(readFileSync(snapPath, "utf8"));
  let compared = 0;
  for (const p of params) {
    const old = snap[p.param_id];
    if (old.empty) {
      assert.equal(p.detail, null, p.param_id);
      continue;
    }
    const d = JSON.parse(readFileSync(new URL(p.detail, DATA), "utf8"));
    for (const w of ["instances", "molecule"]) {
      const o = old[w], n = d.weightings[w];
      assert.equal(n.n_obs, o.n_obs, `${p.param_id} ${w} n_obs`);
      if (!o.n_obs) continue;
      assert.equal(n.n_mol, o.n_mol, `${p.param_id} ${w} n_mol`);
      assert.equal(n.nb, o.nb);
      assert.ok(close(n.lo, o.lo, 0) && close(n.hi, o.hi, 0), `${p.param_id} ${w} range`);
      assert.equal(n.counts.length, o.counts.length);
      n.counts.forEach((c, i) => assert.ok(close(c, o.counts[i], 1e-12), `${p.param_id} ${w} bin ${i}: ${c} vs ${o.counts[i]}`));
      for (const [k, v] of Object.entries(o.stats)) assert.ok(close(n.stats[k], v), `${p.param_id} ${w} ${k}: ${n.stats[k]} vs ${v}`);
    }
    if (old.profile) {
      old.profile.y.forEach((y, i) => assert.ok(close(d.profile.y[i], y, 1e-12), `${p.param_id} profile`));
      assert.deepEqual(d.profile.minima.map((m) => m.phi), old.profile.minima.map((m) => m.phi), `${p.param_id} minima`);
    }
    assert.deepEqual(d.examples, old.examples, `${p.param_id} examples`);
    if (old.scans) {
      assert.equal(d.scans.length, old.scans.length, `${p.param_id} scans`);
      d.scans.forEach((s, i) => {
        assert.equal(s.t, old.scans[i].t);
        assert.deepEqual(s.atoms, old.scans[i].atoms);
        assert.deepEqual(s.frames.map((f) => [f[0], f[1]]), old.scans[i].frames);
      });
    }
    compared++;
  }
  assert.ok(compared > 380, `compared ${compared}`);
});
