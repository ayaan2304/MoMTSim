"""
00_download_data.py — Download the datasets into data/ automatically.
USE: So the project is fully reproducible from code alone — run this first
on a fresh clone and it fetches MoMTSim (both files) directly from
Mendeley Data and verifies them against the official SHA-256 hashes
published by Mendeley, then fetches PaySim. PaySim's canonical source is
Kaggle (needs a free Kaggle account — see README Step 3); this script uses
a public mirror and then VERIFIES the file by its published statistics
(6,362,620 rows / 8,213 fraud) in 01_inspect_data.py, so a wrong or partial
file cannot slip through silently.
Run:  python src/00_download_data.py
"""

# --- Imports ---
# hashlib: SHA-256 verification of the MoMTSim files (USE: proves the
# downloaded bytes are exactly the publisher's bytes, not a corrupt copy).
import hashlib
# sys/pathlib: locate the data folder relative to this script.
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Project config (USE: data folder path from the single source of truth).
from config import DATA_DIR

# --- Download catalogue (USE: filename -> (url, expected size, sha256|None).
# MoMTSim entries come from Mendeley's official public API metadata for
# dataset zhj366m53p v2, captured in data/mendeley_meta.json during setup.
FILES = {
    "MoMTSim_20240722202413_1000_dataset.csv": (
        "https://data.mendeley.com/public-files/datasets/zhj366m53p/files/"
        "8a68b93f-5121-4510-bb6d-7862cd0776df/file_downloaded",
        366_397_921,
        "99fd07c3a9d3c4bd6d3462240058ca19d0d9e9284683f78bf77542ff7fcc05e7",
    ),
    "synthetic_mobile_money_transaction_dataset.csv": (
        "https://data.mendeley.com/public-files/datasets/zhj366m53p/files/"
        "57c2ed32-a9b6-4af8-9510-9b064625d5d7/file_downloaded",
        156_564_413,
        "da951eb95735da96271740a3e66b676b342d3831ce3111cd19dbfa020d3bd0a7",
    ),
    # PaySim mirror (USE: same 493,534,783-byte CSV as the Kaggle original;
    # correctness is verified by row/fraud counts in 01_inspect_data.py).
    "paysim.csv": (
        "https://huggingface.co/datasets/theman10/paysim/resolve/main/paysim.csv",
        493_534_783,
        None,
    ),
}


def sha256_of(path):
    """USE: Stream a file through SHA-256 in chunks (files are 150-500 MB,
    so they are never loaded into memory whole)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    """USE: Download one file with a simple progress printout (percent of the
    expected size), so long downloads don't look frozen."""
    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0 and blocks % 200 == 0:
            print(f"\r    {100 * done / total:.0f}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print("\r    100%")


def main():
    # Make sure data/ exists (USE: fresh-clone safety).
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, (url, size, sha) in FILES.items():
        dest = DATA_DIR / name
        # Skip files that are already present with the exact expected size
        # (USE: re-running the script after a partial setup costs no time).
        if dest.exists() and dest.stat().st_size == size:
            print(f"{name}: already present ({size:,} bytes) — skipping")
        else:
            print(f"{name}: downloading {size:,} bytes...")
            download(url, dest)
        # Verify size exactly (USE: catches truncated downloads).
        actual = dest.stat().st_size
        assert actual == size, f"{name}: size mismatch {actual} != {size}"
        # Verify SHA-256 where the publisher provides one (USE: MoMTSim only;
        # PaySim is verified by content statistics in the next script).
        if sha:
            got = sha256_of(dest)
            assert got == sha, f"{name}: SHA-256 mismatch!"
            print(f"{name}: size OK, SHA-256 OK")
        else:
            print(f"{name}: size OK (content verified by 01_inspect_data.py)")

    print("\nAll datasets ready in data/ — next: python src/01_inspect_data.py")


# Standard entry point.
if __name__ == "__main__":
    main()
