# Sage 2.3.0 training-set explorer

A static GitHub Pages site (`docs/`) for comparing SMARTS patterns and molecules with the OpenFF Sage 2.3.0 training set, and each valence parameter of `openff_unconstrained-2.3.0.offxml` with the QM geometries it is assigned to. Everything runs client-side with RDKit.js. All preprocessing is in `scripts/`.

## What the site does

- **Parameters**: every Bonds, Angles, ProperTorsions and ImproperTorsions parameter. For each one:
  - its assignment counts;
  - the QM distribution of its internal coordinate, with a molecule-balanced weighting option;
  - the Sage 2.3.0 values (r₀/θ₀ and k; torsion k, phase, periodicity, idivf) drawn on the distribution;
  - for torsions, the Fourier energy profile and its minima, plus the TorsionDrive scans that drive it;
  - example molecules with the assigned atoms highlighted.
- **SMARTS search**: matches a pattern against all 6,067 QC records with OpenFF's matching semantics. Map 2–4 atoms to get the QM bond, angle, proper or improper distribution, and the Sage parameters `label_molecules` assigned to those atoms.
- **Molecule search**:
  - Morgan-fingerprint nearest neighbours;
  - training molecules that contain the query as a substructure;
  - an exact-match lookup, with the query's assigned parameters when it is in the training set.
- **About & methods**: provenance, validation results and caveats.

Assigning Sage parameters to new molecules in the browser is deliberately not offered. Stock RDKit.js cannot apply the MDL aromaticity model that OpenFF uses (checked against RDKit.js 2026.3.6), so its assignments could differ from the toolkit's.

## Data

| Source | What |
|---|---|
| [Zenodo 18436107](https://zenodo.org/records/18436107) (CC-BY-4.0) | optimization dataset view: 4,696 B3LYP-D3BJ/DZVP optimizations |
| QCArchive dataset "OpenFF SMIRNOFF Sage 2.3.0" (torsiondrive, id 470) | 1,371 TorsionDrives; the minimum-energy optimization at each of 30,365 grid points |
| openff-forcefields 2026.01.0 | `openff_unconstrained-2.3.0.offxml` (sha256 recorded) |

The TorsionDrive data are fetched from QCArchive with `include=["minimum_optimizations", ...]` rather than from the 14.7 GB Zenodo view, which stores every child optimization in full; the fetch takes about 5 minutes. `scripts/01_download.py --include-td-view` fetches the view if you need it.

## Reproduce

```bash
pixi install && pixi run npm install
pixi run npx playwright install --with-deps chromium firefox webkit   # browsers for the e2e tests (--with-deps installs Linux system libraries)
pixi run download     # Zenodo optimization view + pinned OFFXML (md5/sha256 verified)
pixi run ledger       # closed-world inventory of the optimization view
pixi run fetch-td     # TorsionDrive records + minimum optimizations from QCArchive
pixi run process      # extract -> label -> geometry -> validate-energy -> fingerprints -> parity -> site-data
pixi run test         # pytest, node tests of the site code, Playwright in chromium/firefox/webkit
pixi run serve        # http://127.0.0.1:8000/
```

`process` is deterministic: a clean rebuild reproduces every file in `docs/data/` byte for byte, and `docs/data/meta.json` lists their sha256.

## Pipeline and checks

| Script | Output | Check |
|---|---|---|
| `01_download.py` | raw views, OFFXML | size + md5 against Zenodo before rename; OFFXML sha256 |
| `02_ledger.py`, `02b_fetch_torsiondrive.py` | inventories | 4,696 / 1,371 / 30,365 records, all complete |
| `03_extract.py` | topologies, conformers, constraints | each record's QC symbols and connectivity equal its mapped SMILES (fails closed; 0 exclusions); driven dihedral vs grid key ≤ 0.01° |
| `04_label.py` | assignments | intact force field, RDKit-only registry; `label_molecules` = Interchange for all 791,702 interactions; bonds/angles/propers universe checked (1 record has 4 unassigned propers) |
| `05_geometry.py` | values | IUPAC dihedral sign tested against RDKit; near-linear dihedrals flagged |
| `06_validate_energy.py` | — | torsion energies vs OpenMM Reference: ≤ 7e-13 kJ/mol on every conformer; synthetic grids incl. asymmetric phases |
| `07_fingerprints.mjs` | Morgan fingerprints | vendored RDKit.js = Python RDKit bit for bit (5,444/5,444) |
| `08a/08b_smarts_parity` | — | RDKit.js on the shipped MDL SMILES reproduces OpenFF matching for all 466 SMIRKS × 6,067 records (280,372 pairs, 0 mismatches) |
| `09_build_site_data.py` | `docs/data/` | closed-world parameter roster; size gates (27 MB compressed total, 0.4 MB initial) |

`STATUS.md` logs each step and `PLAN.md` records the design decisions. `docs/data/SCHEMA.md` documents the file formats.

## Caveats

- "QM value − Sage 2.3.0 r₀/θ₀" is descriptive, not a force-field error: the MM minimum depends on all terms together. Comparing QM with MM-minimized geometries would be the evaluative test and is not done here.
- TorsionDrive grid points and the 13 constrained optimization records are constrained geometries. Their values are off by default, and torsions about driven or frozen bonds are always excluded from distributions.
- Dihedral statistics are circular. For multimodal torsions, the circular mean is not a typical value; check R.
- 26 valence parameters have no training-set coverage.

## Licences

Training data CC-BY-4.0; force field CC-BY-4.0 (openff-forcefields); RDKit / RDKit.js BSD-3-Clause (vendored in `docs/vendor/rdkit/`); NGL Viewer MIT (vendored in `docs/vendor/ngl/`). Code in this repository: add a licence before publishing.
