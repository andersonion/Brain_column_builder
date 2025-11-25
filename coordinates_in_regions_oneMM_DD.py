#!/usr/bin/env python

"""
Verbose QA version of coordinates_in_regions_oneMM_DD.

Usage:
    python coordinates_in_regions_oneMM_DD.py \
        --ID S00775 \
        --input-dir  /mnt/newStor/paros/paros_WORK/hanwen/ad_decode_test/output/ \
        --output-dir /mnt/newStor/paros/paros_WORK/column_code_tester/ \
        [--contrast QSM]
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
import nibabel.freesurfer as fsio


# ------------------ LOW-LEVEL LOADERS ------------------ #

def _load_cp_dwi_mats(contrast_dir: Path, ID: str):
    """Load *_column_* DWI coordinate matrices and print shapes."""
    lh_mat = contrast_dir / f"{ID}_column_lh.mat"
    rh_mat = contrast_dir / f"{ID}_column_rh.mat"
    lh_dwi_mat = contrast_dir / f"{ID}_column_lh_dwi.mat"
    rh_dwi_mat = contrast_dir / f"{ID}_column_rh_dwi.mat"

    print("\n[LOAD] column dir:", contrast_dir)
    print("[LOAD] Expecting:")
    print(f"  {lh_mat}")
    print(f"  {rh_mat}")
    print(f"  {lh_dwi_mat}")
    print(f"  {rh_dwi_mat}")

    if not lh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required: {lh_dwi_mat}")
    if not rh_dwi_mat.is_file():
        raise FileNotFoundError(f"Missing required: {rh_dwi_mat}")

    # These two are loaded in MATLAB; mirror that behavior.
    if lh_mat.is_file():
        _ = loadmat(lh_mat)
    if rh_mat.is_file():
        _ = loadmat(rh_mat)

    lh_dwi = loadmat(lh_dwi_mat)
    rh_dwi = loadmat(rh_dwi_mat)

    if "lh_cp_dwi" not in lh_dwi:
        raise KeyError(f"{lh_dwi_mat} missing 'lh_cp_dwi'")
    if "rh_cp_dwi" not in rh_dwi:
        raise KeyError(f"{rh_dwi_mat} missing 'rh_cp_dwi'")

    ori_lh_cp_dwi = np.asarray(lh_dwi["lh_cp_dwi"])
    ori_rh_cp_dwi = np.asarray(rh_dwi["rh_cp_dwi"])

    print(f"[LOAD] LH cp_dwi shape: {ori_lh_cp_dwi.shape}")
    print(f"[LOAD] RH cp_dwi shape: {ori_rh_cp_dwi.shape}")

    return ori_lh_cp_dwi, ori_rh_cp_dwi


def _load_aparc_annot(subject_dir: Path, hemi: str):
    """
    Load FreeSurfer aparc annotation for one hemisphere and print basic stats.

    IMPORTANT: In nibabel, 'labels' are *indices* into ctab/names,
    not the original RGB-coded IDs (ctab[:,4]).
    """
    annot_fname = subject_dir / "label" / f"{hemi}.aparc.annot"
    print(f"\n[ANNOT] Loading {hemi}.aparc.annot from:")
    print("        ", annot_fname)

    if not annot_fname.is_file():
        raise FileNotFoundError(f"Missing annotation: {annot_fname}")

    labels, ctab, names = fsio.read_annot(str(annot_fname))

    struct_names = [
        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
        for n in names
    ]

    unique_labels = np.unique(labels)
    print(f"[ANNOT] {hemi}: labels shape = {labels.shape}, unique labels = {len(unique_labels)}")
    print(f"[ANNOT] {hemi}: struct_names = {len(struct_names)}, ctab rows = {ctab.shape[0]}")
    print(f"[ANNOT] {hemi}: first 10 unique label indices: {unique_labels[:10]}")

    return labels, ctab, struct_names


def _region_indices_for_vertices(vertex_indices: np.ndarray) -> np.ndarray:
    """
    Given vertex indices (0-based), produce 1-based "depth indices" into cp_dwi columns.

    Mirrors the MATLAB logic:
        index = double(sum(repmat([0:47]',1,size(vertex_index,2)) ...
                 .* (vertex_index(ones(48,1),:)-1== ...
                      repmat([0:47]',1,size(vertex_index,2)))));
        index = index + 48;

    Here we simply do:
        - collect all integer depths for each vertex (0..47)
        - add 48, then flatten to 1D.
    """
    if vertex_indices.size == 0:
        return np.array([], dtype=int)

    depths = np.arange(48, dtype=int)     # 0..47
    depth_indices = depths + 48           # 48..95

    # For each vertex_index, we have 48 possible row entries (depths).
    # Flatten them out as a single vector of depth indices.
    repeated = np.tile(depth_indices[:, None], (1, vertex_indices.size))
    index_1b = repeated.flatten(order="F")  # column-major flatten

    return index_1b


# ------------------ CORE PER-HEMI PROCESSING ------------------ #

def _process_hemi(
    hemi: str,
    labels: np.ndarray,
    ctab: np.ndarray,
    struct_names: list[str],
    ori_cp_dwi: np.ndarray,
    out_dir: Path,
):
    """
    Process one hemisphere, logging per-region QA info.

    NOTE: In nibabel:
        - 'labels' are indices into 'ctab' / 'struct_names'.
        - So for row_idx, the corresponding label is exactly 'row_idx'.
    """
    print(f"\n===== PROCESSING {hemi.upper()} HEMISPHERE =====")
    max_index = ori_cp_dwi.shape[1]
    print(f"[{hemi}] cp_dwi columns available: {max_index}")
    print(f"[{hemi}] struct_names length: {len(struct_names)}, ctab rows: {ctab.shape[0]}")

    unique_label_indices = np.unique(labels)
    print(f"[{hemi}] unique label indices (first 20): {unique_label_indices[:20]}")

    n_written = 0
    n_regions_with_vertices = 0
    skipped_regions: list[str] = []

    # MATLAB loop: for i = 2:36, i ~= 5
    for i in range(2, 37):
        row_idx = i - 1  # 0-based row index into struct_names/ctab

        region_name = None
        if 0 <= row_idx < len(struct_names):
            region_name = struct_names[row_idx]

        # MATLAB-mandated skip
        if i == 5:
            msg = (
                f"{hemi} i={i}, row_idx={row_idx}, region={region_name or '<unknown>'}: "
                "MATLAB-specified skip (i=5)"
            )
            print(f"[{hemi}] SKIP MATLAB i=5 -> {msg}")
            skipped_regions.append(msg)
            continue

        if row_idx >= len(struct_names) or row_idx >= ctab.shape[0]:
            msg = (
                f"{hemi} i={i}, row_idx={row_idx}: out of bounds for struct_names/ctab, "
                "skipping region"
            )
            print(f"[{hemi}] {msg}")
            skipped_regions.append(msg)
            continue

        region_id_orig = int(ctab[row_idx, 4])   # original FS ID (informational)
        region_index = row_idx                   # this is what 'labels' stores in nibabel

        # 🔑 KEY FIX: match on 'region_index' (row_idx), not 'region_id_orig'
        vertex_indices = np.where(labels == region_index)[0]
        n_vertices = vertex_indices.size

        print(f"\n[{hemi}] REGION i={i}, row_idx={row_idx}")
        print(f"    name          = '{region_name}'")
        print(f"    region_index  = {region_index} (matches 'labels')")
        print(f"    region_id_orig= {region_id_orig} (ctab[:,4])")
        print(f"    n_vertices    = {n_vertices}")

        if n_vertices == 0:
            msg = (
                f"{hemi} region '{region_name}' (i={i}, index={region_index}): "
                "no vertices with this label index, skipping"
            )
            print(f"    -> {msg}")
            skipped_regions.append(msg)
            continue

        n_regions_with_vertices += 1

        index_1b = _region_indices_for_vertices(vertex_indices)
        valid_mask = index_1b <= max_index
        index_1b = index_1b[valid_mask]

        print(f"    depth indices kept = {index_1b.size} (after clipping > max_index)")

        if index_1b.size == 0:
            msg = (
                f"{hemi} region '{region_name}' (i={i}, index={region_index}): "
                "no valid depth indices after clipping, skipping"
            )
            print(f"    -> {msg}")
            skipped_regions.append(msg)
            continue

        idx_0b = index_1b - 1
        cp_dwi = ori_cp_dwi[:, idx_0b]

        output_name = f"{hemi}_{region_name}"
        mat_path = out_dir / f"{output_name}.mat"
        csv_path = out_dir / f"{output_name}.csv"

        savemat(mat_path, {f"{hemi}_cp_dwi": cp_dwi})
        np.savetxt(csv_path, cp_dwi, delimiter=",")

        print(f"    -> WROTE {mat_path.name}, {csv_path.name}")
        n_written += 1

    print(f"\n>>> SUMMARY {hemi.upper()} <<<")
    print(f"    Regions with vertices: {n_regions_with_vertices}")
    print(f"    Files written:         {n_written}")
    print(f"    Output dir:            {out_dir}")

    if skipped_regions:
        print(f"\n>>> {hemi.upper()} REGIONS SKIPPED ({len(skipped_regions)}) <<<")
        for msg in skipped_regions:
            print(f"    - {msg}")
    else:
        print(f"\n>>> {hemi.upper()} REGIONS SKIPPED <<<")
        print("    None.")


# ------------------ TOP-LEVEL FUNCTION ------------------ #

def coordinates_in_regions_oneMM_DD(ID: str, input_dir, output_dir, contrast: str = "QSM"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    contrast_dir = input_dir / ID / contrast
    subject_dir = input_dir / ID / ID
    out_dir = output_dir / ID / contrast / "label_coord_1mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PATH] -------")
    print("[PATH] Subject ID:", ID)
    print("[PATH] Input dir: ", input_dir)
    print("[PATH] Contrast:   ", contrast)
    print("[PATH] Column dir: ", contrast_dir)
    print("[PATH] Subject dir (labels):", subject_dir)
    print("[PATH] Output dir:", out_dir)

    # Column coordinate matrices
    ori_lh_cp_dwi, ori_rh_cp_dwi = _load_cp_dwi_mats(contrast_dir, ID)

    # Annotation files
    labels_lh, ctab_lh, names_lh = _load_aparc_annot(subject_dir, "lh")
    labels_rh, ctab_rh, names_rh = _load_aparc_annot(subject_dir, "rh")

    # Process hemispheres
    _process_hemi("lh", labels_lh, ctab_lh, names_lh, ori_lh_cp_dwi, out_dir)
    _process_hemi("rh", labels_rh, ctab_rh, names_rh, ori_rh_cp_dwi, out_dir)


# ------------------ CLI ENTRY ------------------ #

def _cli():
    parser = argparse.ArgumentParser(
        description="Verbose QA cortical column region extraction."
    )
    parser.add_argument("--ID", required=True, help="Subject ID (e.g. S00775)")
    parser.add_argument("--input-dir", required=True, help="Input root dir")
    parser.add_argument("--output-dir", required=True, help="Output root dir")
    parser.add_argument(
        "--contrast",
        default="QSM",
        help="Contrast subfolder under ID containing *_column_* files (e.g. QSM, MD, FA).",
    )
    args = parser.parse_args()

    coordinates_in_regions_oneMM_DD(
        args.ID,
        args.input_dir,
        args.output_dir,
        contrast=args.contrast,
    )


if __name__ == "__main__":
    _cli()
