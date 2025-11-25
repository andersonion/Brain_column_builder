#!/usr/bin/env python

"""
build_pairs_from_freesurfer.py

Builds "pair" files for cortical thickness based on FreeSurfer
white and pial surfaces.

For each hemisphere, we assume:

    - lh.white and lh.pial share the same vertex count and ordering
    - rh.white and rh.pial share the same vertex count and ordering

We construct:

    lh_pair: (N_lh, 6) array
        [white_x, white_y, white_z,  pial_x, pial_y, pial_z]

    rh_pair: (N_rh, 6) array
        [white_x, white_y, white_z,  pial_x, pial_y, pial_z]

and save them as:

    <output_dir>/<ID>/columns_1mm/<ID>_pair_lh.mat   (var: lh_pair)
    <output_dir>/<ID>/columns_1mm/<ID>_pair_rh.mat   (var: rh_pair)

These are then used by get_thickness.py, which computes:

    thickness = ||pial - white|| (per vertex)

Usage:

    python build_pairs_from_freesurfer.py \
        --ID S00775 \
        --output-dir /mnt/newStor/paros/paros_WORK/column_code_tester/

This version includes:

    - Check-for-previous-work using a checksum file per subject
    - --force flag to recompute even if outputs + checksum exist
"""

import argparse
from pathlib import Path
import hashlib

import numpy as np
from scipy.io import savemat
import nibabel as nib

PIPELINE_VERSION = "build_pairs_v1.0_20251125"


def _subject_checksum(ID: str, surf_dir: Path) -> str:
    """Compute a checksum for this subject's pair-build configuration.

    Currently based on:
        - PIPELINE_VERSION (bump to invalidate all old results)
        - subject ID
        - surf_dir path

    We intentionally *do not* hash the surf file contents to keep this fast.
    If you change surfaces without changing ID / surf_dir, use --force.
    """
    h = hashlib.sha256()
    h.update(PIPELINE_VERSION.encode("utf-8"))
    h.update(ID.encode("utf-8"))
    h.update(str(surf_dir).encode("utf-8"))
    return h.hexdigest()


def _load_fs_surface(surf_path: Path) -> np.ndarray:
    """
    Load a FreeSurfer surface and return vertex coordinates (N, 3).

    surf_path: path to lh.white, lh.pial, etc.
    """
    if not surf_path.is_file():
        raise FileNotFoundError(f"Surface file not found: {surf_path}")

    coords, faces = nib.freesurfer.io.read_geometry(str(surf_path))
    print(f"[SURF] Loaded {surf_path.name}: {coords.shape[0]} vertices, {faces.shape[0]} faces")
    return coords


def _build_pair(white_coords: np.ndarray, pial_coords: np.ndarray) -> np.ndarray:
    """Given white and pial vertex coordinates, build pair array:

        [white_x, white_y, white_z,  pial_x, pial_y, pial_z]

    Both white_coords and pial_coords should be shape (N, 3).
    """
    if white_coords.shape != pial_coords.shape:
        raise ValueError(
            f"White and pial shapes differ: {white_coords.shape} vs {pial_coords.shape}"
        )

    pair = np.hstack([white_coords, pial_coords])  # (N, 6)
    return pair


def build_pairs_from_freesurfer(ID: str, output_dir, force: bool = False):
    """Build pair files for subject ID using FreeSurfer surfaces stored under:

        <output_dir>/<ID>/<ID>/surf/

    Writes (into <output_dir>/<ID>/columns_1mm/):

        <ID>_pair_lh.mat  (lh_pair)
        <ID>_pair_rh.mat  (rh_pair)
        <ID>_pair_lh.csv  (debug)
        <ID>_pair_rh.csv  (debug)

    A checksum file <ID>_pairs.sha256 is used to avoid redoing work
    when outputs already exist and are considered up to date.
    """
    output_dir = Path(output_dir)

    subj_root = output_dir / ID / ID
    surf_dir = subj_root / "surf"

    lh_white_path = surf_dir / "lh.white"
    lh_pial_path = surf_dir / "lh.pial"
    rh_white_path = surf_dir / "rh.white"
    rh_pial_path = surf_dir / "rh.pial"

    # Output directory for pair files
    pair_dir = output_dir / ID / "columns_1mm"
    pair_dir.mkdir(parents=True, exist_ok=True)

    lh_out_mat = pair_dir / f"{ID}_pair_lh.mat"
    rh_out_mat = pair_dir / f"{ID}_pair_rh.mat"
    lh_csv = pair_dir / f"{ID}_pair_lh.csv"
    rh_csv = pair_dir / f"{ID}_pair_rh.csv"
    checksum_path = pair_dir / f"{ID}_pairs.sha256"

    outputs_exist = (
        lh_out_mat.is_file()
        and rh_out_mat.is_file()
        and lh_csv.is_file()
        and rh_csv.is_file()
    )

    checksum_ok = False
    if checksum_path.is_file():
        try:
            stored = checksum_path.read_text().strip()
            expected = _subject_checksum(ID, surf_dir)
            if stored == expected:
                checksum_ok = True
        except Exception as e:
            print(f"[WARN] could not read checksum for {ID}: {e}")

    print(f"[INFO] Subject:        {ID}")
    print(f"[INFO] Surf dir:       {surf_dir}")
    print(f"[INFO] Output dir:     {pair_dir}")
    print(f"[INFO] Pipeline ver:   {PIPELINE_VERSION}")
    print(f"[INFO] FORCE mode:     {force}")
    print(f"[INFO] Outputs exist:  {outputs_exist}")
    print(f"[INFO] Checksum OK:    {checksum_ok}")

    if (not force) and outputs_exist and checksum_ok:
        print("[SKIP] Pair files + checksum present and up to date; skipping rebuild.")
        return
    elif force:
        print("[INFO] FORCE recompute: rebuilding pairs regardless of existing outputs.")
    else:
        if outputs_exist and not checksum_ok:
            print("[INFO] Outputs exist but checksum missing/mismatch; recomputing.")
        else:
            print("[INFO] Outputs missing; computing pairs from surfaces.")

    # Load surfaces
    lh_white = _load_fs_surface(lh_white_path)
    lh_pial = _load_fs_surface(lh_pial_path)
    rh_white = _load_fs_surface(rh_white_path)
    rh_pial = _load_fs_surface(rh_pial_path)

    # Build pairs
    lh_pair = _build_pair(lh_white, lh_pial)
    rh_pair = _build_pair(rh_white, rh_pial)

    print(f"[PAIR] LH pair shape: {lh_pair.shape}")
    print(f"[PAIR] RH pair shape: {rh_pair.shape}")

    # Save MAT files
    savemat(str(lh_out_mat), {"lh_pair": lh_pair})
    savemat(str(rh_out_mat), {"rh_pair": rh_pair})

    print(f"[SAVE] Wrote LH pair MAT: {lh_out_mat}")
    print(f"[SAVE] Wrote RH pair MAT: {rh_out_mat}")

    # Also drop quick CSVs for debugging (one row per vertex)
    np.savetxt(lh_csv, lh_pair, delimiter=",")
    np.savetxt(rh_csv, rh_pair, delimiter=",")

    print(f"[SAVE] Wrote LH pair CSV: {lh_csv}")
    print(f"[SAVE] Wrote RH pair CSV: {rh_csv}")

    # Write checksum marker
    try:
        checksum = _subject_checksum(ID, surf_dir)
        checksum_path.write_text(checksum + "\n")
    except Exception as e:
        print(f"[WARN] could not write checksum for {ID}: {e}")

    print("\n[DONE] Pair construction complete.")


def _cli():
    parser = argparse.ArgumentParser(
        description="Build cortical white–pial vertex pairs from FreeSurfer surfaces."
    )
    parser.add_argument("--ID", required=True, help="Subject ID, e.g. S00775")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root dir where <ID>/<ID>/surf/ lives and where columns_1mm will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute pairs even if outputs + checksum exist.",
    )
    args = parser.parse_args()

    build_pairs_from_freesurfer(args.ID, args.output_dir, force=args.force)


if __name__ == "__main__":
    _cli()
