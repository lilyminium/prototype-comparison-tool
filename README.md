# Sage 2.3.0 training-set explorer

A static website for browsing the QM training data behind the OpenFF Sage 2.3.0 force field
(`openff_unconstrained-2.3.0.offxml`).

**Live site:** https://lilyminium.github.io/prototype-comparison-tool/

- **Parameters**: pick any bond, angle, proper or improper torsion parameter to see which training molecules use it, the QM distribution of its internal coordinate against the Sage 2.3.0 value, and (for torsions) the energy profile and the TorsionDrive scans behind it, with a 3D viewer.
- **SMARTS search**: find every training record that matches a pattern; map 2–4 atoms to get the QM bond, angle or torsion distribution and the Sage parameters assigned there.
- **Molecule search**: similar molecules (Morgan fingerprints), training molecules that contain yours, and an exact-match lookup.

The page does not assign Sage parameters to new molecules: stock RDKit.js cannot reproduce the MDL aromaticity model that OpenFF uses.

**Preview locally:**

```bash
pixi install && pixi run npm install
pixi run serve        # http://127.0.0.1:8000/
```

Claude Opus 5.5 (Anthropic) was used to write the site and its preprocessing scripts.

## Details

Everything that does not depend on user input is precomputed in Python and served from `docs/data/`: parameter histograms and statistics, torsion energy profiles, the geometry of every bond, angle, torsion and improper, conformer superpositions and 2D depictions. In the browser, RDKit.js only parses and matches queries and draws the precomputed depictions; NGL shows 3D structures. `docs/data/SCHEMA.md` documents the file formats.

### Data

| Source | What |
|---|---|
| [Zenodo 18436107](https://zenodo.org/records/18436107) (CC-BY-4.0) | optimization dataset view: 4,696 B3LYP-D3BJ/DZVP optimizations |
| QCArchive dataset "OpenFF SMIRNOFF Sage 2.3.0" (torsiondrive, id 470) | 1,371 TorsionDrives; the minimum-energy optimization at each of 30,365 grid points |
| openff-forcefields 2026.01.0 | `openff_unconstrained-2.3.0.offxml` (sha256 recorded) |

The TorsionDrive data are fetched from QCArchive with `include=["minimum_optimizations", ...]` rather than from the 14.7 GB Zenodo view, which stores every child optimization in full; the fetch takes about 5 minutes. `scripts/01_download.py --include-td-view` fetches the view if you need it.

### Reproduce

```bash
pixi install && pixi run npm install
pixi run npx playwright install --with-deps chromium firefox webkit   # browsers for the e2e tests (--with-deps installs Linux system libraries)
pixi run download     # Zenodo optimization view + pinned OFFXML (md5/sha256 verified)
pixi run ledger       # closed-world inventory of the optimization view
pixi run fetch-td     # TorsionDrive records + minimum optimizations from QCArchive
pixi run process      # extract -> label -> geometry -> validate-energy -> geometry-universe -> fingerprints
                      #   -> parity -> align-conformers -> depictions -> site-data
pixi run test         # pytest, node tests of the site code, Playwright in chromium/firefox/webkit
pixi run serve        # http://127.0.0.1:8000/
```

`docs/data/meta.json` lists the sha256 of every file in `docs/data/`. A clean rebuild reproduces them byte for byte. (2D depictions use the RDKit native depictor: CoordGen gave run-to-run different layouts for some strained polycyclic molecules.)

### Pipeline and checks

| Script | Output | Check |
|---|---|---|
| `01_download.py` | raw views, OFFXML | size + md5 against Zenodo before rename; OFFXML sha256 |
| `02_ledger.py`, `02b_fetch_torsiondrive.py` | inventories | 4,696 / 1,371 / 30,365 records, all complete |
| `03_extract.py` | topologies, conformers, constraints | each record's QC symbols and connectivity equal its mapped SMILES (fails closed; 0 exclusions); driven dihedral vs grid key ≤ 0.01° |
| `04_label.py` | assignments | intact force field, RDKit-only registry; `label_molecules` = Interchange for all 791,702 interactions; bonds/angles/propers universe checked (1 record has 4 unassigned propers) |
| `05_geometry.py` | values of assigned interactions | measured with RDKit; IUPAC dihedral sign; near-linear dihedrals flagged |
| `05b_geometry_universe.py` | every bond, angle, proper and improper star of every optimization record (720,776) | same RDKit measurement on the served Float32 coordinates; parameter index or −1 |
| `06_validate_energy.py` | — | Interchange systems with the force field's NAGL charges (model hash checked): torsion energies vs OpenMM Reference ≤ 7e-13 kJ/mol on every conformer; synthetic grids incl. asymmetric phases |
| `07_fingerprints.mjs` | Morgan fingerprints, popcounts, canonical SMILES | vendored RDKit.js = Python RDKit bit for bit (5,444/5,444) |
| `08a/08b_smarts_parity` | — | RDKit.js on the shipped MDL SMILES reproduces OpenFF matching for all 466 SMIRKS × 6,067 records (280,372 pairs, 0 mismatches) |
| `08d_align_conformers.py` | superposed optimization conformers (48 molecules, 108 ordered pairs) | Kabsch without reflection; tested against scipy (`tests/test_alignment.py`) |
| `08e_depictions.py` | 2D molblocks per record, molecule SVGs | vendored RDKit.js renders the same SVG as Python RDKit (`10b_render_oracle.py`, `tests/js/render_oracle.test.mjs`) |
| `09_build_site_data.py` | `docs/data/` | closed-world parameter roster; size gates (26.9 MB compressed total, 0.41 MB initial load) |

### Caveats

- "QM value − Sage 2.3.0 r₀/θ₀" is descriptive, not a force-field error: the MM minimum depends on all terms together. Comparing QM with MM-minimized geometries would be the evaluative test and is not done here.
- TorsionDrive grid points and the 13 constrained optimization records are constrained geometries. Their values are off by default, and torsions about driven or frozen bonds are always excluded from distributions.
- Dihedral statistics are circular. For multimodal torsions, the circular mean is not a typical value; check R.
- 26 valence parameters have no training-set coverage.
- 17 optimization records are near-duplicates of another record of the same molecule (RMSD < 0.01 Å after superposition).

### Licences

Code in this repository: MIT (`LICENSE`). Training data CC-BY-4.0; force field CC-BY-4.0 (openff-forcefields); RDKit / RDKit.js BSD-3-Clause (vendored in `docs/vendor/rdkit/`); NGL Viewer MIT (vendored in `docs/vendor/ngl/`).
