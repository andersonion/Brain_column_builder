#!/usr/bin/env python

"""
run_column_pipeline.py

Wrapper to process ONE subject/ID through the full column/thickness pipeline:

    0) vertices_connect.py                    (build columns + *_column_*_dwi.mat)
    1) coordinates_in_regions_oneMM_DD.py     (run ONCE; contrast-agnostic)
    2a) generate_columns_only (per contrast)  (sample intensities, write cols CSVs)
    2b) clean_bad_columns_across_contrasts    (remove any col with 0/NaN across all contrasts)
    2c) summarize_from_existing_columns       (per contrast; summary + QA from CLEANED CSVs)
    3) build_pairs_from_freesurfer.py
    4) get_thickness.py

Conventions
-----------

- `output_dir` is the root that contains:

      <output_dir>/<ID>/DWI2T1_dti_*.dat  (or other transform .dat)
      <output_dir>/<ID>/columns/...
      <output_dir>/<ID>/<ID>/surf/...
      <output_dir>/<ID>/<ID>/label/...

- `input_dir` is where the MRI contrast images live, e.g.:

      <input_dir>/<ID>/<ID>_<contrast>_masked.nii.gz
      <input_dir>/<ID>_<contrast>_masked.nii.gz

- Column cp_dwi & region coords are contrast-independent and live under:

      <output_dir>/<ID>/columns/
      <output_dir>/<ID>/columns/label_coord_1mm/

- Per-contrast intensity samples & QA live under:

      <output_dir>/<ID>/<contrast>/...
"""

import argparse
import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from vertices_connect import vertices_connect
from coordinates_in_regions_oneMM_DD import coordinates_in_regions_oneMM_DD
from get_columns_in_regions_oneMM_DD import (
    generate_columns_only,
    summarize_from_existing_columns,
)
from build_pairs_from_freesurfer import build_pairs_from_freesurfer
from get_thickness import get_thickness


# ============================================================
#  Cleaning logic: remove columns with any 0 or NaN/Inf
#  consistently across ALL contrasts and coordinate CSVs.
# ============================================================


def _clog(msg: str) -> None:
    print(msg, flush=True)


def _find_region_from_filename(subject: str, contrast: str, path: Path) -> str:
    """
    Given something like:
        /.../D0007/fa/fa_cols_by_column/D0007_lh_superiorfrontal_cols_fa.csv
    or:
        D0007_rh_superiorfrontal_cols_fa.csv

    extract "lh_superiorfrontal" or "rh_superiorfrontal" as the 'region' key
    used for label_coord_1mm ({region}.csv).

    Pattern assumed:
        <subject>_<region>_cols_<contrast>.csv
    where <region> may contain underscores and the hemi prefix.
    """
    base = path.name
    prefix = f"{subject}_"
    suffix = f"_cols_{contrast}.csv"
    if not base.startswith(prefix) or not base.endswith(suffix):
        raise ValueError(f"Unexpected filename pattern for per-column CSV: {base}")
    region = base[len(prefix) : -len(suffix)]
    return region


def _collect_bad_columns_for_contrast(
    subject: str,
    contrast: str,
    out_root: Path,
) -> Dict[str, Set[int]]:
    """
    For a single contrast, scan all per-region per-column CSVs and identify
    which column indices (0-based) contain any 0 or NaN/Inf.

    Returns: dict[region] -> set(bad column indices)
    """
    bad_cols: Dict[str, Set[int]] = {}

    cols_dir = out_root / subject / contrast / f"{contrast}_cols_by_column"
    if not cols_dir.is_dir():
        _clog(f"[CLEAN] WARN: no cols_by_column dir for contrast={contrast}: {cols_dir}")
        return bad_cols

    pattern = cols_dir / f"{subject}_*_cols_{contrast}.csv"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        _clog(f"[CLEAN] WARN: no per-column CSVs for contrast={contrast} in {cols_dir}")
        return bad_cols

    _clog(f"[CLEAN] Scanning contrast={contrast}: {len(files)} region CSV(s)")

    for p in files:
        path = Path(p)
        try:
            region = _find_region_from_filename(subject, contrast, path)
        except ValueError as e:
            _clog(f"[CLEAN] WARN: {e}")
            continue

        df = pd.read_csv(path)
        arr = df.to_numpy()

        # Invalid if NaN/Inf OR exactly zero
        invalid = (~np.isfinite(arr)) | (arr == 0)

        bad_indices = np.where(invalid.any(axis=0))[0]
        if bad_indices.size == 0:
            continue

        if region not in bad_cols:
            bad_cols[region] = set()
        bad_cols[region].update(bad_indices.tolist())

        _clog(
            f"[CLEAN]   {path.name} → "
            f"{bad_indices.size} bad columns "
            f"(idx 0-based, first few: {sorted(bad_indices.tolist())[:10]}"
            f"{'...' if bad_indices.size > 10 else ''})"
        )

    return bad_cols


def _union_bad_columns(
    per_contrast_bad: List[Dict[str, Set[int]]]
) -> Dict[str, Set[int]]:
    """
    Merge region→bad index sets from multiple contrasts into a master union.
    """
    master: Dict[str, Set[int]] = {}
    for d in per_contrast_bad:
        for region, idxs in d.items():
            if region not in master:
                master[region] = set()
            master[region].update(idxs)
    return master


def _save_master_bad_list(
    subject: str,
    out_root: Path,
    master_bad: Dict[str, Set[int]],
) -> Path:
    """
    Save master bad-column list as JSON for record-keeping.
    Indices are stored as sorted 0-based lists.
    """
    columns_dir = out_root / subject / "columns"
    columns_dir.mkdir(parents=True, exist_ok=True)
    out_path = columns_dir / "bad_columns_master.json"

    serializable = {
        region: sorted(list(idxs)) for region, idxs in master_bad.items()
    }

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)

    _clog(f"[CLEAN] Master bad-column list → {out_path}")
    return out_path


def _drop_columns_from_csv(path: Path, bad_indices: Set[int]) -> None:
    if not bad_indices:
        return

    df = pd.read_csv(path)
    n_cols = df.shape[1]

    valid_bad = sorted(i for i in bad_indices if 0 <= i < n_cols)
    if not valid_bad:
        return

    _clog(
        f"[CLEAN]   {path.name}: dropping {len(valid_bad)} / {n_cols} columns "
        f"(idx 0-based, first few: {valid_bad[:10]}{'...' if len(valid_bad) > 10 else ''})"
    )

    cols = df.columns
    keep_mask = np.ones(n_cols, dtype=bool)
    keep_mask[valid_bad] = False
    keep_cols = cols[keep_mask]

    df_clean = df[keep_cols]
    df_clean.to_csv(path, index=False)


def _clean_coords_for_region(
    subject: str,
    out_root: Path,
    region: str,
    bad_indices: Set[int],
) -> None:
    """
    Clean the source coordinate file:
      output/{subject}/columns/label_coord_1mm/{region}.csv
    by removing the same columns.
    """
    coords_path = out_root / subject / "columns" / "label_coord_1mm" / f"{region}.csv"
    if not coords_path.is_file():
        _clog(f"[CLEAN] WARN: coord file missing for region={region}: {coords_path}")
        return

    _clog(f"[CLEAN] Coords: region={region}")
    _drop_columns_from_csv(coords_path, bad_indices)


def _clean_all_contrast_csvs(
    subject: str,
    contrasts: List[str],
    out_root: Path,
    master_bad: Dict[str, Set[int]],
) -> None:
    """
    Apply master bad-column removals to each contrast's per-region CSVs.
    """
    for contrast in contrasts:
        cols_dir = out_root / subject / contrast / f"{contrast}_cols_by_column"
        if not cols_dir.is_dir():
            _clog(f"[CLEAN] WARN: no cols_by_column dir for contrast={contrast}: {cols_dir}")
            continue

        pattern = cols_dir / f"{subject}_*_cols_{contrast}.csv"
        files = sorted(glob.glob(str(pattern)))
        if not files:
            _clog(f"[CLEAN] WARN: no per-column CSVs for contrast={contrast} in {cols_dir}")
            continue

        _clog(f"[CLEAN] Contrast={contrast}: cleaning {len(files)} region CSV(s)")
        for p in files:
            path = Path(p)
            try:
                region = _find_region_from_filename(subject, contrast, path)
            except ValueError as e:
                _clog(f"[CLEAN] WARN: {e}")
                continue

            bad_indices = master_bad.get(region, set())
            if not bad_indices:
                continue

            _drop_columns_from_csv(path, bad_indices)


def clean_bad_columns_across_contrasts(
    subject: str,
    out_root: Path,
    contrasts: List[str],
) -> Dict[str, Set[int]]:
    """
    High-level entry point called from run_subject_pipeline.

    1) For each contrast, find per-region columns that contain any 0 or NaN/Inf.
    2) Union across contrasts to get master bad-column set per region.
    3) Remove those columns from:
         - columns/label_coord_1mm/{region}.csv
         - {contrast}/{contrast}_cols_by_column/{subject}_{region}_cols_{contrast}.csv
    4) Save bad_columns_master.json and return master dict.
    """
    out_root = Path(out_root)

    _clog("")
    _clog("------------------------------------------------------------")
    _clog("[CLEAN] Removing columns with 0/NaN across all contrasts")
    _clog("------------------------------------------------------------")

    per_contrast_bad: List[Dict[str, Set[int]]] = []
    for c in contrasts:
        _clog(f"[CLEAN] Scan contrast: {c}")
        bad_for_c = _collect_bad_columns_for_contrast(subject, c, out_root)
        per_contrast_bad.append(bad_for_c)

    master_bad = _union_bad_columns(per_contrast_bad)

    if not master_bad:
        _clog("[CLEAN] No bad columns detected in any region/contrast. Nothing to remove.")
        return {}

    _clog("[CLEAN] Master bad-column counts by region:")
    for region, idxs in sorted(master_bad.items()):
        _clog(f"  - {region}: {len(idxs)} column(s) to drop")

    _save_master_bad_list(subject, out_root, master_bad)

    # Clean coordinate files
    _clog("[CLEAN] Cleaning coordinate files (columns/label_coord_1mm)")
    for region, idxs in sorted(master_bad.items()):
        _clean_coords_for_region(subject, out_root, region, idxs)

    # Clean per-contrast per-region CSVs
    _clog("[CLEAN] Cleaning per-contrast per-region column CSVs")
    _clean_all_contrast_csvs(subject, contrasts, out_root, master_bad)

    _clog("[CLEAN] Done removing bad columns across all contrasts.")
    return master_bad


# ============================================================
#  Main pipeline
# ============================================================


def run_subject_pipeline(
    ID: str,
    input_dir,
    output_dir,
    contrasts,
    transform_file: str | None = None,
    force_all: bool = False,
    nproc_coords: int = 1,  # reserved; currently unused
):
    """
    Run the full pipeline for one subject.

    Parameters
    ----------
    ID : str
        Subject ID (e.g., D0007).
    input_dir : str or Path
        Root for contrast images (QSM, MD, FA, etc.).
    output_dir : str or Path
        Root for FreeSurfer/columns/output products.
    contrasts : list of str
        List of contrast names to sample (e.g., ["adc", "ad", "fa", "rd"]).
    transform_file : str or None
        Optional transform .dat filename inside <output_dir>/<ID>.
        If None, vertices_connect will auto-detect as described in its doc.
    force_all : bool
        If True, passed through to downstream steps that support --force.
    nproc_coords : int
        Reserved; currently unused.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    contrasts = list(contrasts)

    print("\n================ COLUMN / THICKNESS PIPELINE ================")
    print(f"[SUBJECT]        {ID}")
    print(f"[INPUT DIR]      {input_dir}")
    print(f"[OUTPUT DIR]     {output_dir}")
    print(f"[TRANSFORM FILE] {transform_file if transform_file else '(auto-detect)'}")
    print(f"[CONTRASTS]      {', '.join(contrasts)}")
    print(f"[FORCE ALL]      {force_all}")
    print(f"[NPROC COORDS]   {nproc_coords} (reserved)")
    print("=============================================================\n")

    # ------------------------------------------------------------
    # 0) vertices_connect.py  (once; geometric + DWI transform)
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[STEP 0] vertices_connect  (build columns + *_column_*_dwi.mat)")
    print("------------------------------------------------------------")

    vertices_connect(
        ID=ID,
        root_dir=output_dir,
        transform_file=transform_file,
        # voldim/voxres use defaults inside vertices_connect
    )

    # ------------------------------------------------------------
    # 1) coordinates_in_regions_oneMM_DD.py  (once; contrast-agnostic)
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[STEP 1] coordinates_in_regions_oneMM_DD  (contrast-agnostic)")
    print("------------------------------------------------------------")

    coordinates_in_regions_oneMM_DD(
        ID=ID,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------
    # 2a) generate per-contrast columns (values along columns)
    #     → writes *_cols_{contrast}.csv only
    # ------------------------------------------------------------
    for contrast in contrasts:
        print("\n------------------------------------------------------------")
        print(f"[STEP 2a] generate_columns_only for contrast: {contrast}")
        print("------------------------------------------------------------")

        generate_columns_only(
            ID=ID,
            input_dir=input_dir,
            output_dir=output_dir,
            contrast=contrast,
            force=force_all,
        )

    # ------------------------------------------------------------
    # 2b) clean columns across contrasts (0 or NaN/Inf → drop column)
    #     → updates label_coord_1mm/*.csv and *_cols_{contrast}.csv
    # ------------------------------------------------------------
    clean_bad_columns_across_contrasts(
        subject=ID,
        out_root=output_dir,
        contrasts=contrasts,
    )

    # ------------------------------------------------------------
    # 2c) summarize from cleaned columns: per-region means, QA plots
    # ------------------------------------------------------------
    for contrast in contrasts:
        print("\n------------------------------------------------------------")
        print(f"[STEP 2c] summarize_from_existing_columns for contrast: {contrast}")
        print("------------------------------------------------------------")

        summarize_from_existing_columns(
            ID=ID,
            output_dir=output_dir,
            contrast=contrast,
        )

    # ------------------------------------------------------------
    # 3) build_pairs_from_freesurfer.py  (once, contrast-independent)
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[STEP 3] build_pairs_from_freesurfer")
    print("------------------------------------------------------------")

    build_pairs_from_freesurfer(
        ID=ID,
        output_dir=output_dir,
        force=force_all,
    )

    # ------------------------------------------------------------
    # 4) get_thickness.py  (once, contrast-independent)
    # ------------------------------------------------------------
    print("\n------------------------------------------------------------")
    print("[STEP 4] get_thickness")
    print("------------------------------------------------------------")

    get_thickness(
        ID=ID,
        output_dir=output_dir,
        force=force_all,
    )

    print("\n==================== PIPELINE COMPLETE ======================")
    print(f"[DONE] Subject {ID}")
    print("=============================================================\n")


def _cli():
    parser = argparse.ArgumentParser(
        description="Wrapper to run full cortical column / thickness pipeline for one subject."
    )
    parser.add_argument("--ID", required=True, help="Subject ID, e.g. D0007")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root for contrast images (where <ID>_<contrast>[_masked].nii.gz live).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Root for FreeSurfer/columns/outputs "
            "(contains <ID>/, <ID>/<ID>/surf, and transform .dat)."
        ),
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        required=True,
        help="One or more contrast names to process (e.g. adc ad fa rd qsm cbf).",
    )
    parser.add_argument(
        "--transform-file",
        default=None,
        help=(
            "Optional transform .dat filename inside <output-dir>/<ID>. "
            "If omitted, vertices_connect will auto-detect."
        ),
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force recompute for ALL downstream steps that support it.",
    )
    parser.add_argument(
        "--nproc-coords",
        type=int,
        default=1,
        help="Currently unused; reserved for future parallelization in the coords step.",
    )

    args = parser.parse_args()

    run_subject_pipeline(
        ID=args.ID,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        contrasts=args.contrasts,
        transform_file=args.transform_file,
        force_all=args.force_all,
        nproc_coords=args.nproc_coords,
    )


if __name__ == "__main__":
    _cli()
