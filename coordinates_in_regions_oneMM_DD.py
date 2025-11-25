#!/usr/bin/env python

"""
Verbose QA version of coordinates_in_regions_oneMM_DD.

Usage:
    python coordinates_in_regions_oneMM_DD.py \
        --ID S00775 \
        --input-dir  /mnt/newStor/paros/paros_WORK/hanwen/ad_decode_test/output/ \
        --output-dir /mnt/newStor/paros/paros_WORK/column_code_tester/
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
import nibabel.freesurfer as fsio


# ------------------ LOW-LEVEL LOADERS ------------------ #

def _load_cp_dwi_mats(qsm_dir: Path, ID: str):
    """Load *_column_* DWI coordinate matrices and print shapes."""
    lh_mat = qsm_dir / f"{ID}_column_lh.mat"
    rh_mat = qsm_dir / f"{ID}_column_rh.mat"
    lh_dwi_mat = qsm_dir / f"{ID}_column_lh_dwi.mat"
    rh_dwi_mat = qsm_dir / f"{ID}_column_rh_dwi.mat"

    print("\n[LOAD] QSM dir:", qsm_dir)
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
    print(f"[ANNOT] {hemi}: first 10 unique label IDs: {unique_labels[:10]}")

    # Also print the first few region names and IDs from the colortable
    print(f"[ANNOT] {hemi}: first 10 colortable entries (name, ID):")
    for idx in range(min(10, len(struct_names), ctab.shape[0])):
        print(f"    idx {idx}: name='{struct_names[idx]}', ID={int(ctab[idx, 4])}")

    return labels, ctab, struct_names


# ------------------ INDEXING UTIL ------------------ #

def _region_indices_for_vertices(vertex_indices_0b: np.ndarray, depth_samples: int = 21):
    """
    From 0-based vertex indices, compute 1-based depth-sample indices,
    mirroring the MATLAB math.
    """
    vertx_1b = vertex_indices_0b.astype(np.int64) + 1  # 1-based
    offsets = np.arange(-20, 1, dtype=np.int64)        # -20..0
    indices = []

    for v in vertx_1b:
        base = v * depth_samples  # v*21
        indices.extend(base + offsets)

    return np.array(indices, dtype=np.int64)  # 1-based indices


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
    """
    print(f"\n===== PROCESSING {hemi.upper()} HEMISPHERE =====")
    max_index = ori_cp_dwi.shape[1]
    print(f"[{hemi}] cp_dwi columns available: {max_index}")
    print(f"[{hemi}] struct_names length: {len(struct_names)}, ctab rows: {ctab.shape[0]}")

    all_label_ids = set(int(x) for x in np.unique(labels))
    all_ctab_ids = set(int(x) for x in ctab[:, 4])
    inter_ids = all_label_ids.intersection(all_ctab_ids)

    print(f"[{hemi}] Unique label IDs in labels: {len(all_label_ids)}")
    print(f"[{hemi}] Unique IDs in ctab:        {len(all_ctab_ids)}")
    print(f"[{hemi}] Intersection size:         {len(inter_ids)}")

    n_written = 0
    n_regions_with_vertices = 0

    # MATLAB loop: for i = 2:36, i ~= 5
    for i in range(2, 37):
        if i == 5:
            print(f"[{hemi}] SKIP MATLAB i=5")
            continue

        row_idx = i - 1  # 0-based row index into struct_names/ctab

        if row_idx >= len(struct_names) or row_idx >= ctab.shape[0]:
            print(f"[{hemi}] i={i}: row_idx={row_idx} out of bounds for struct_names/ctab, skipping.")
            continue

        region_name = struct_names[row_idx]
        region_num = int(ctab[row_idx, 4])

        vertex_indices = np.where(labels == region_num)[0]
        n_vertices = vertex_indices.size
        present_in_labels = region_num in all_label_ids

        print(f"\n[{hemi}] REGION i={i}, row_idx={row_idx}")
        print(f"    name        = '{region_name}'")
        print(f"    region_num  = {region_num}")
        print(f"    in_labels   = {present_in_labels}")
        print(f"    n_vertices  = {n_vertices}")

        if n_vertices == 0:
            print(f"    -> No vertices with label {region_num}, skipping region.")
            continue

        n_regions_with_vertices += 1

        index_1b = _region_indices_for_vertices(vertex_indices)
        valid_mask = index_1b <= max_index
        index_1b = index_1b[valid_mask]

        print(f"    depth indices kept = {index_1b.size} (after clipping > max_index)")

        if index_1b.size == 0:
            print("    -> No valid depth indices after clipping, skipping.")
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


# ------------------ TOP-LEVEL FUNCTION ------------------ #

def coordinates_in_regions_oneMM_DD(ID: str, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    qsm_dir = input_dir / ID / "QSM"
    subject_dir = input_dir / ID / ID
    out_dir = output_dir / ID / "QSM" / "label_coord_1mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PATH] -------")
    print("[PATH] Subject ID:", ID)
    print("[PATH] Input dir: ", input_dir)
    print("[PATH] QSM dir:   ", qsm_dir)
    print("[PATH] Subject dir (labels):", subject_dir)
    print("[PATH] Output dir:", out_dir)

    # Column coordinate matrices
    ori_lh_cp_dwi, ori_rh_cp_dwi = _load_cp_dwi_mats(qsm_dir, ID)

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
    args = parser.parse_args()

    coordinates_in_regions_oneMM_DD(args.ID, args.input_dir, args.output_dir)


if __name__ == "__main__":
    _cli()
