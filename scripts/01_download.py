"""Download the Sage 2.3.0 training-set dataset views and the force field.

Sources
-------
- Zenodo record 18436107 ("QC Fitting Datasets for OpenFF SMIRNOFF Sage 2.3.0"), CC-BY-4.0.
  The optimization view (40 MB) is fetched. The TorsionDrive view (14.7 GB, ~6 h from Zenodo) is
  only fetched with --include-td-view; the TorsionDrive data are instead pulled from QCArchive by
  02b_fetch_torsiondrive.py.
- openff_unconstrained-2.3.0.offxml from openff-forcefields at a pinned git ref.

Every file is verified against the checksum published by Zenodo (md5) and a sha256 is recorded
in data/raw/download_manifest.json. Existing files that already verify are not re-downloaded.
"""

import argparse
import hashlib
import json
import pathlib
import urllib.request

ZENODO_RECORD = "18436107"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
OPT_VIEW = "OpenFF-SMIRNOFF-Sage-2.3.0_optimization_view.sqlite"
TD_VIEW = "OpenFF-SMIRNOFF-Sage-2.3.0_torsiondrive_view.sqlite"
OFFXML_NAME = "openff_unconstrained-2.3.0.offxml"
# openff-forcefields release "Sage 2.3.0"; the file is byte-identical in 2026.09.0
OFFXML_REF = "2026.01.0"
OFFXML_URL = (
    "https://raw.githubusercontent.com/openforcefield/openff-forcefields/"
    "{ref}/openforcefields/offxml/" + OFFXML_NAME
)

CHUNK = 1 << 22


def file_hashes(path: pathlib.Path) -> dict[str, str]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def download(url: str, dest: pathlib.Path, size: int | None = None, md5: str | None = None) -> None:
    """Download to a .part file; only rename to ``dest`` once size and md5 verify."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
    if size is not None and tmp.stat().st_size != size:
        raise RuntimeError(f"truncated download of {dest.name}: {tmp.stat().st_size} != {size} bytes")
    if md5 is not None and file_hashes(tmp)["md5"] != md5:
        raise RuntimeError(f"md5 mismatch for {dest.name}")
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("data/raw"))
    parser.add_argument("--offxml-ref", default=OFFXML_REF)
    parser.add_argument("--include-td-view", action="store_true", help="also fetch the 14.7 GB TorsionDrive view")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(ZENODO_API) as response:
        record = json.load(response)
    zenodo_files = {f["key"]: f for f in record["files"]}

    manifest = {
        "zenodo": {
            "record": ZENODO_RECORD,
            "doi": record.get("doi"),
            "title": record["metadata"]["title"],
            "license": record["metadata"].get("license"),
            "files": {},
        },
        "offxml": {},
    }

    for name in [OPT_VIEW, TD_VIEW] if args.include_td_view else [OPT_VIEW]:
        entry = zenodo_files[name]
        algo, expected = entry["checksum"].split(":")
        assert algo == "md5", entry["checksum"]
        dest = args.outdir / name
        if not (dest.exists() and dest.stat().st_size == entry["size"]):
            print(f"downloading {name} ({entry['size'] / 1e9:.2f} GB)", flush=True)
            download(entry["links"]["self"], dest, size=entry["size"], md5=expected)
        hashes = file_hashes(dest)
        if hashes["md5"] != expected:
            raise RuntimeError(
                f"md5 mismatch for {name}: {hashes['md5']} != {expected}"
            )
        print(f"verified {name}", flush=True)
        manifest["zenodo"]["files"][name] = {"size": entry["size"], **hashes}

    dest = args.outdir / OFFXML_NAME
    url = OFFXML_URL.format(ref=args.offxml_ref)
    download(url, dest)
    manifest["offxml"] = {"url": url, "ref": args.offxml_ref, **file_hashes(dest)}

    out = args.outdir / "download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
