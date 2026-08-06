#!/usr/bin/env python3
"""
run_column_pipeline.py

Run the cortical-column and thickness pipeline for one subject.

Pipeline order
--------------
0. vertices_connect.py
   Build white-to-pial columns and transform them to DWI voxel coordinates.

1. coordinates_in_regions_oneMM_DD.py
   Split the shared, contrast-independent coordinates into cortical regions.

2a. get_columns_in_regions_oneMM_DD.generate_columns_only()
    Sample every requested contrast along the regional columns.

2b. Cross-contrast cleaning
    For each hemisphere/region, remove a cortical column everywhere when any
    depth sample is zero or non-finite in any requested contrast. The same
    cortical-column rows are removed from every contrast CSV, and the matching
    21-point coordinate blocks are removed from both the shared MAT and CSV.

2c. get_columns_in_regions_oneMM_DD.summarize_from_existing_columns()
    Rebuild regional means, QA plots, and summary CSVs from the cleaned data.

3. get_thickness.py
   Calculate regional cortical thickness from the pair files generated in
   step 0. build_pairs_from_freesurfer.py is intentionally not called because
   vertices_connect.py already writes the required pair files.

Directory conventions
---------------------
Input contrast images may use any of these layouts, with masked files preferred:

    <input_dir>/<ID>/<ID>_<contrast>_masked.nii.gz
    <input_dir>/<ID>_<contrast>_masked.nii.gz
    <input_dir>/<ID>/<ID>_<contrast>.nii.gz
    <input_dir>/<ID>_<contrast>.nii.gz

FreeSurfer and transform inputs are expected under:

    <output_dir>/<ID>/<ID>/surf/
    <output_dir>/<ID>/<ID>/label/
    <output_dir>/<ID>/*.dat

Shared coordinate products are written under:

    <output_dir>/<ID>/columns/

Per-contrast products are written under:

    <output_dir>/<ID>/<contrast>/
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from scipy.io import loadmat, savemat

from vertices_connect import vertices_connect
from coordinates_in_regions_oneMM_DD import coordinates_in_regions_oneMM_DD
from get_columns_in_regions_oneMM_DD import (
    generate_columns_only,
    summarize_from_existing_columns,
)
from get_thickness import get_thickness


DEPTH_SAMPLES = 21
PAIR_FILES = ("pair_lh", "pair_rh")
COLUMN_DWI_FILES = ("column_lh_dwi", "column_rh_dwi")


# ============================================================
# Logging and basic file helpers
# ============================================================


def _log(message: str = "") -> None:
    print(message, flush=True)


def _load_csv_2d(path: Path) -> np.ndarray:
    """Load a headerless numeric CSV while preserving a two-dimensional shape."""
    try:
        array = np.loadtxt(path, delimiter=",")
    except ValueError as exc:
        raise ValueError(f"Could not read numeric CSV {path}: {exc}") from exc

    array = np.asarray(array, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2:
        raise ValueError(f"Expected a 2-D CSV at {path}, found shape {array.shape}")
    return array


def _atomic_savetxt(path: Path, array: np.ndarray) -> None:
    """Write a numeric CSV through a temporary file and atomically replace it."""
    temporary = path.with_name(f".{path.name}.tmp")
    np.savetxt(temporary, array, delimiter=",")
    os.replace(temporary, path)


def _atomic_savemat(path: Path, variables: dict) -> None:
    """Write a MAT file through a temporary file and atomically replace it."""
    temporary = path.with_name(f".{path.stem}.tmp.mat")
    savemat(str(temporary), variables)
    os.replace(temporary, path)


def _find_region_from_filename(subject: str, contrast: str, path: Path) -> str:
    """Extract ``lh_<region>`` or ``rh_<region>`` from a per-column filename."""
    prefix = f"{subject}_"
    suffix = f"_cols_{contrast}.csv"
    filename = path.name
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise ValueError(f"Unexpected per-column filename: {filename}")
    return filename[len(prefix) : -len(suffix)]


def _per_column_files(subject: str, contrast: str, out_root: Path) -> List[Path]:
    directory = out_root / subject / contrast / f"{contrast}_cols_by_column"
    return sorted(directory.glob(f"{subject}_*_cols_{contrast}.csv"))


# ============================================================
# Preflight checks
# ============================================================


def _resolve_contrast_image(input_dir: Path, subject: str, contrast: str) -> Path:
    """Resolve a contrast image using the same priority as the sampling script."""
    candidates = [
        input_dir / subject / f"{subject}_{contrast}_masked.nii.gz",
        input_dir / f"{subject}_{contrast}_masked.nii.gz",
        input_dir / subject / f"{subject}_{contrast}.nii.gz",
        input_dir / f"{subject}_{contrast}.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    attempted = "\n".join(f"    {path}" for path in candidates)
    raise FileNotFoundError(
        f"No image found for subject={subject}, contrast={contrast}. Tried:\n{attempted}"
    )


def _preflight(
    subject: str,
    input_dir: Path,
    output_dir: Path,
    contrasts: Sequence[str],
    transform_file: str | None,
) -> Dict[str, Path]:
    """Fail early when required inputs are absent or ambiguous."""
    errors: List[str] = []
    resolved_images: Dict[str, Path] = {}

    if not input_dir.is_dir():
        errors.append(f"Input directory does not exist: {input_dir}")
    if not output_dir.is_dir():
        errors.append(f"Output root does not exist: {output_dir}")

    fs_subject_dir = output_dir / subject / subject
    for relative in (
        "surf/lh.white",
        "surf/lh.pial",
        "surf/rh.white",
        "surf/rh.pial",
        "label/lh.aparc.annot",
        "label/rh.aparc.annot",
    ):
        path = fs_subject_dir / relative
        if not path.is_file():
            errors.append(f"Missing FreeSurfer input: {path}")

    subject_output = output_dir / subject
    if transform_file:
        transform_path = Path(transform_file)
        if not transform_path.is_absolute():
            transform_path = subject_output / transform_path
        if not transform_path.is_file():
            errors.append(f"Requested transform file does not exist: {transform_path}")
    else:
        dat_files = sorted(subject_output.glob("*.dat"))
        if not dat_files:
            errors.append(f"No transform .dat file found in: {subject_output}")

    for contrast in contrasts:
        try:
            resolved_images[contrast] = _resolve_contrast_image(
                input_dir, subject, contrast
            )
        except FileNotFoundError as exc:
            errors.append(str(exc))

    if errors:
        formatted = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(f"Preflight failed:\n{formatted}")

    _log("[PREFLIGHT] Required inputs found.")
    for contrast, path in resolved_images.items():
        _log(f"[PREFLIGHT] {contrast}: {path}")
    return resolved_images


# ============================================================
# Geometry restart checks
# ============================================================


def _geometry_outputs_complete(subject: str, output_dir: Path) -> bool:
    """Return True only when shared column and regional coordinate outputs exist."""
    columns_dir = output_dir / subject / "columns"
    required = [
        columns_dir / f"{subject}_pair_lh.mat",
        columns_dir / f"{subject}_pair_rh.mat",
        columns_dir / f"{subject}_column_lh.mat",
        columns_dir / f"{subject}_column_rh.mat",
        columns_dir / f"{subject}_column_lh_dwi.mat",
        columns_dir / f"{subject}_column_rh_dwi.mat",
    ]
    if not all(path.is_file() for path in required):
        return False

    regional_dir = columns_dir / "label_coord_1mm"
    lh_mats = list(regional_dir.glob("lh_*.mat"))
    rh_mats = list(regional_dir.glob("rh_*.mat"))
    return regional_dir.is_dir() and bool(lh_mats) and bool(rh_mats)


# ============================================================
# Cross-contrast bad-column detection
# ============================================================


def _collect_bad_columns_for_contrast(
    subject: str,
    contrast: str,
    out_root: Path,
) -> Dict[str, Set[int]]:
    """
    Identify cortical-column rows containing zero, NaN, or Inf.

    Per-column CSV shape is ``n_cortical_columns × n_depth_samples``. Therefore
    bad cortical columns are found with ``invalid.any(axis=1)``. The previous
    implementation incorrectly used axis 0, which selected depth positions.
    """
    files = _per_column_files(subject, contrast, out_root)
    if not files:
        raise RuntimeError(
            f"No per-column CSV files found for contrast={contrast} under "
            f"{out_root / subject / contrast}"
        )

    _log(f"[CLEAN] Scanning contrast={contrast}: {len(files)} regional CSV(s)")
    result: Dict[str, Set[int]] = {}

    for path in files:
        region = _find_region_from_filename(subject, contrast, path)
        values = _load_csv_2d(path)
        if values.shape[1] != DEPTH_SAMPLES:
            raise ValueError(
                f"Unexpected depth count in {path}: shape={values.shape}; "
                f"expected {DEPTH_SAMPLES} columns"
            )

        invalid = (~np.isfinite(values)) | (values == 0)
        bad_indices = np.flatnonzero(invalid.any(axis=1))
        if bad_indices.size:
            result[region] = set(int(index) for index in bad_indices)
            preview = bad_indices[:10].tolist()
            suffix = "..." if bad_indices.size > 10 else ""
            _log(
                f"[CLEAN]   {path.name}: {bad_indices.size}/{values.shape[0]} "
                f"bad cortical column(s); first indices={preview}{suffix}"
            )

    return result


def _union_bad_columns(
    per_contrast_bad: Iterable[Dict[str, Set[int]]],
) -> Dict[str, Set[int]]:
    master: Dict[str, Set[int]] = {}
    for mapping in per_contrast_bad:
        for region, indices in mapping.items():
            master.setdefault(region, set()).update(indices)
    return master


def _save_master_bad_list(
    subject: str,
    out_root: Path,
    contrasts: Sequence[str],
    master_bad: Dict[str, Set[int]],
) -> Path:
    """Record exactly which current cortical-column rows were removed."""
    path = out_root / subject / "columns" / "bad_columns_master.json"
    payload = {
        "subject": subject,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contrasts": list(contrasts),
        "depth_samples_per_column": DEPTH_SAMPLES,
        "indexing": "0-based cortical-column row indices at cleaning time",
        "bad_columns": {
            region: sorted(indices) for region, indices in sorted(master_bad.items())
        },
    }
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    _log(f"[CLEAN] Removal record: {path}")
    return path


# ============================================================
# Cross-contrast cleaning
# ============================================================


def _drop_cortical_rows(path: Path, bad_indices: Set[int]) -> Tuple[int, int]:
    """Delete bad cortical-column rows from one headerless per-column CSV."""
    values = _load_csv_2d(path)
    n_before = values.shape[0]
    invalid_indices = sorted(index for index in bad_indices if not 0 <= index < n_before)
    if invalid_indices:
        raise IndexError(
            f"Bad-column indices exceed row count for {path}: "
            f"n_rows={n_before}, invalid indices={invalid_indices[:10]}"
        )

    keep = np.ones(n_before, dtype=bool)
    if bad_indices:
        keep[np.asarray(sorted(bad_indices), dtype=int)] = False
    cleaned = values[keep, :]
    _atomic_savetxt(path, cleaned)
    return n_before, cleaned.shape[0]


def _clean_coordinate_region(
    subject: str,
    out_root: Path,
    region: str,
    bad_indices: Set[int],
) -> Tuple[int, int]:
    """
    Remove complete 21-sample coordinate blocks from the regional MAT and CSV.

    Regional coordinate matrices have shape ``3-or-4 × (n_columns * 21)``.
    Cortical-column row ``i`` therefore maps to coordinate columns
    ``i*21 : (i+1)*21``.
    """
    coordinate_dir = out_root / subject / "columns" / "label_coord_1mm"
    mat_path = coordinate_dir / f"{region}.mat"
    csv_path = coordinate_dir / f"{region}.csv"

    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing regional coordinate MAT: {mat_path}")

    hemi = region.split("_", 1)[0]
    variable_name = f"{hemi}_cp_dwi"
    mat_data = loadmat(mat_path)
    if variable_name not in mat_data:
        raise KeyError(
            f"{mat_path} does not contain {variable_name!r}; "
            f"available keys={sorted(k for k in mat_data if not k.startswith('__'))}"
        )

    coordinates = np.asarray(mat_data[variable_name], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] not in (3, 4):
        raise ValueError(
            f"Unexpected coordinate shape in {mat_path}: {coordinates.shape}"
        )
    if coordinates.shape[1] % DEPTH_SAMPLES:
        raise ValueError(
            f"Coordinate sample count is not divisible by {DEPTH_SAMPLES}: "
            f"{mat_path}, shape={coordinates.shape}"
        )

    n_columns = coordinates.shape[1] // DEPTH_SAMPLES
    invalid_indices = sorted(index for index in bad_indices if not 0 <= index < n_columns)
    if invalid_indices:
        raise IndexError(
            f"Bad-column indices exceed coordinate column count for {mat_path}: "
            f"n_columns={n_columns}, invalid indices={invalid_indices[:10]}"
        )

    keep_columns = np.ones(n_columns, dtype=bool)
    if bad_indices:
        keep_columns[np.asarray(sorted(bad_indices), dtype=int)] = False
    keep_samples = np.repeat(keep_columns, DEPTH_SAMPLES)
    cleaned = coordinates[:, keep_samples]

    _atomic_savemat(mat_path, {variable_name: cleaned})
    _atomic_savetxt(csv_path, cleaned)
    return n_columns, cleaned.shape[1] // DEPTH_SAMPLES


def clean_bad_columns_across_contrasts(
    subject: str,
    out_root: Path,
    contrasts: Sequence[str],
) -> Dict[str, Set[int]]:
    """Apply the union of bad cortical-column rows consistently everywhere."""
    out_root = Path(out_root)
    _log("\n------------------------------------------------------------")
    _log("[STEP 2b] Cross-contrast bad-column cleaning")
    _log("------------------------------------------------------------")

    per_contrast = [
        _collect_bad_columns_for_contrast(subject, contrast, out_root)
        for contrast in contrasts
    ]
    master_bad = _union_bad_columns(per_contrast)
    _save_master_bad_list(subject, out_root, contrasts, master_bad)

    if not master_bad:
        _log("[CLEAN] No zero or non-finite cortical columns were found.")
        return {}

    _log("[CLEAN] Union counts by hemisphere/region:")
    for region, indices in sorted(master_bad.items()):
        _log(f"[CLEAN]   {region}: {len(indices)}")

    # Validate all affected files before modifying any of them.
    for region, indices in master_bad.items():
        coordinate_mat = (
            out_root / subject / "columns" / "label_coord_1mm" / f"{region}.mat"
        )
        if not coordinate_mat.is_file():
            raise FileNotFoundError(f"Missing coordinate MAT before cleaning: {coordinate_mat}")
        for contrast in contrasts:
            path = (
                out_root
                / subject
                / contrast
                / f"{contrast}_cols_by_column"
                / f"{subject}_{region}_cols_{contrast}.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Region {region} exists in the bad-column union but the matching "
                    f"contrast CSV is missing: {path}"
                )
            n_rows = _load_csv_2d(path).shape[0]
            if any(index < 0 or index >= n_rows for index in indices):
                raise IndexError(
                    f"Bad-column index mismatch for {path}: n_rows={n_rows}, "
                    f"max_bad={max(indices)}"
                )

    # Clean every contrast first.
    for contrast in contrasts:
        _log(f"[CLEAN] Contrast={contrast}")
        for path in _per_column_files(subject, contrast, out_root):
            region = _find_region_from_filename(subject, contrast, path)
            indices = master_bad.get(region)
            if not indices:
                continue
            before, after = _drop_cortical_rows(path, indices)
            _log(f"[CLEAN]   {path.name}: {before} -> {after} cortical columns")

    # Then clean the shared MAT and CSV coordinate products.
    _log("[CLEAN] Shared regional coordinates")
    for region, indices in sorted(master_bad.items()):
        before, after = _clean_coordinate_region(
            subject, out_root, region, indices
        )
        _log(f"[CLEAN]   {region}: {before} -> {after} cortical columns")

    _log("[CLEAN] Cross-contrast cleaning complete.")
    return master_bad


# ============================================================
# Alignment and final-output validation
# ============================================================


def _validate_cleaned_alignment(
    subject: str,
    output_dir: Path,
    contrasts: Sequence[str],
) -> None:
    """Verify matching cortical-column counts across coords and all contrasts."""
    coordinate_dir = output_dir / subject / "columns" / "label_coord_1mm"
    errors: List[str] = []

    for mat_path in sorted(coordinate_dir.glob("[lr]h_*.mat")):
        region = mat_path.stem
        hemi = region.split("_", 1)[0]
        key = f"{hemi}_cp_dwi"
        data = loadmat(mat_path)
        if key not in data:
            errors.append(f"{mat_path}: missing variable {key}")
            continue
        coordinates = np.asarray(data[key])
        if coordinates.ndim != 2 or coordinates.shape[1] % DEPTH_SAMPLES:
            errors.append(f"{mat_path}: invalid coordinate shape {coordinates.shape}")
            continue
        expected_rows = coordinates.shape[1] // DEPTH_SAMPLES

        csv_path = mat_path.with_suffix(".csv")
        if not csv_path.is_file():
            errors.append(f"Missing coordinate CSV: {csv_path}")
        else:
            csv_coordinates = _load_csv_2d(csv_path)
            if csv_coordinates.shape != coordinates.shape:
                errors.append(
                    f"Coordinate MAT/CSV mismatch for {region}: "
                    f"MAT={coordinates.shape}, CSV={csv_coordinates.shape}"
                )

        for contrast in contrasts:
            values_path = (
                output_dir
                / subject
                / contrast
                / f"{contrast}_cols_by_column"
                / f"{subject}_{region}_cols_{contrast}.csv"
            )
            if not values_path.is_file():
                errors.append(f"Missing per-column CSV: {values_path}")
                continue
            values = _load_csv_2d(values_path)
            if values.shape != (expected_rows, DEPTH_SAMPLES):
                errors.append(
                    f"Count/depth mismatch for {values_path}: shape={values.shape}, "
                    f"expected=({expected_rows}, {DEPTH_SAMPLES})"
                )

    if errors:
        formatted = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(f"Cleaned-output alignment validation failed:\n{formatted}")
    _log("[VALIDATE] Coordinate and cross-contrast column counts agree.")


def _validate_final_outputs(
    subject: str,
    output_dir: Path,
    contrasts: Sequence[str],
) -> None:
    """Require the major deliverables expected from a successful run."""
    errors: List[str] = []
    subject_dir = output_dir / subject
    columns_dir = subject_dir / "columns"

    for suffix in (*PAIR_FILES, *COLUMN_DWI_FILES):
        path = columns_dir / f"{subject}_{suffix}.mat"
        if not path.is_file():
            errors.append(f"Missing shared output: {path}")

    regional_dir = columns_dir / "label_coord_1mm"
    if not list(regional_dir.glob("lh_*.mat")):
        errors.append(f"No LH regional coordinate MAT files in {regional_dir}")
    if not list(regional_dir.glob("rh_*.mat")):
        errors.append(f"No RH regional coordinate MAT files in {regional_dir}")

    for contrast in contrasts:
        contrast_dir = subject_dir / contrast
        per_column_dir = contrast_dir / f"{contrast}_cols_by_column"
        mean_dir = contrast_dir / f"{contrast}_cols_region_mean"
        qa_dir = contrast_dir / "plots_QA"
        summary = per_column_dir / f"{subject}_cols_{contrast}_summary.csv"
        qa_csv = qa_dir / f"{subject}_profiles_QA_{contrast}.csv"

        if not list(per_column_dir.glob(f"{subject}_lh_*_cols_{contrast}.csv")):
            errors.append(f"No LH per-column files for contrast={contrast}: {per_column_dir}")
        if not list(per_column_dir.glob(f"{subject}_rh_*_cols_{contrast}.csv")):
            errors.append(f"No RH per-column files for contrast={contrast}: {per_column_dir}")
        if not list(mean_dir.glob(f"{subject}_*_cols_{contrast}_mean.csv")):
            errors.append(f"No regional means for contrast={contrast}: {mean_dir}")
        if not list(qa_dir.glob(f"{subject}_*_profile_{contrast}.png")):
            errors.append(f"No QA plots for contrast={contrast}: {qa_dir}")
        if not summary.is_file():
            errors.append(f"Missing contrast summary: {summary}")
        if not qa_csv.is_file():
            errors.append(f"Missing QA profile CSV: {qa_csv}")

    thickness_summary = subject_dir / "thickness" / f"{subject}_thickness_region_means.csv"
    if not thickness_summary.is_file():
        errors.append(f"Missing thickness summary: {thickness_summary}")

    if errors:
        formatted = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(f"Final output validation failed:\n{formatted}")
    _log("[VALIDATE] All required final outputs are present.")


# ============================================================
# Main pipeline
# ============================================================


def run_subject_pipeline(
    ID: str,
    input_dir,
    output_dir,
    contrasts,
    transform_file: str | None = None,
    force_all: bool = False,
) -> None:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    contrasts = list(dict.fromkeys(str(contrast) for contrast in contrasts))

    if not contrasts:
        raise ValueError("At least one contrast is required.")

    _log("\n================ COLUMN / THICKNESS PIPELINE ================")
    _log(f"[SUBJECT]        {ID}")
    _log(f"[INPUT DIR]      {input_dir}")
    _log(f"[OUTPUT DIR]     {output_dir}")
    _log(f"[TRANSFORM FILE] {transform_file or '(auto-detect)'}")
    _log(f"[CONTRASTS]      {', '.join(contrasts)}")
    _log(f"[FORCE ALL]      {force_all}")
    _log("=============================================================\n")

    _preflight(ID, input_dir, output_dir, contrasts, transform_file)

    # Reusing complete shared geometry is important: coordinates may already
    # have been cross-contrast cleaned. Re-running geometry without forcing
    # would restore uncleaned coordinate MATs while existing contrast CSVs
    # remained cleaned, creating a silent row-count mismatch.
    geometry_complete = _geometry_outputs_complete(ID, output_dir)
    if geometry_complete and not force_all:
        _log("[STEPS 0-1] Shared geometry outputs are complete; reusing them.")
    else:
        _log("\n------------------------------------------------------------")
        _log("[STEP 0] vertices_connect")
        _log("------------------------------------------------------------")
        vertices_connect(
            ID=ID,
            root_dir=output_dir,
            transform_file=transform_file,
        )

        _log("\n------------------------------------------------------------")
        _log("[STEP 1] coordinates_in_regions_oneMM_DD")
        _log("------------------------------------------------------------")
        coordinates_in_regions_oneMM_DD(ID=ID, output_dir=output_dir)

    for contrast in contrasts:
        _log("\n------------------------------------------------------------")
        _log(f"[STEP 2a] Sample contrast: {contrast}")
        _log("------------------------------------------------------------")
        generate_columns_only(
            ID=ID,
            input_dir=input_dir,
            output_dir=output_dir,
            contrast=contrast,
            force=force_all,
        )

    clean_bad_columns_across_contrasts(
        subject=ID,
        out_root=output_dir,
        contrasts=contrasts,
    )
    _validate_cleaned_alignment(ID, output_dir, contrasts)

    for contrast in contrasts:
        _log("\n------------------------------------------------------------")
        _log(f"[STEP 2c] Summarize cleaned contrast: {contrast}")
        _log("------------------------------------------------------------")
        summarize_from_existing_columns(
            ID=ID,
            output_dir=output_dir,
            contrast=contrast,
        )

    _log("\n------------------------------------------------------------")
    _log("[STEP 3] get_thickness")
    _log("------------------------------------------------------------")
    get_thickness(ID=ID, output_dir=output_dir, force=force_all)

    _validate_final_outputs(ID, output_dir, contrasts)

    _log("\n==================== PIPELINE COMPLETE ======================")
    _log(f"[DONE] Subject {ID}")
    _log("=============================================================\n")


# ============================================================
# CLI
# ============================================================


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run the cortical-column and thickness pipeline for one subject."
    )
    parser.add_argument("--ID", required=True, help="Subject ID, for example D0007")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root containing the subject's contrast NIfTI files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root containing <ID>/<ID>/surf, <ID>/<ID>/label, and transform files.",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        required=True,
        help="One or more contrast names, for example: adc ad fa rd",
    )
    parser.add_argument(
        "--transform-file",
        default=None,
        help=(
            "Optional transform filename inside <output-dir>/<ID>, or an absolute "
            "path. When omitted, vertices_connect auto-detects the transform."
        ),
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Regenerate shared geometry, resample every contrast, and recompute thickness.",
    )
    args = parser.parse_args()

    run_subject_pipeline(
        ID=args.ID,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        contrasts=args.contrasts,
        transform_file=args.transform_file,
        force_all=args.force_all,
    )


if __name__ == "__main__":
    _cli()