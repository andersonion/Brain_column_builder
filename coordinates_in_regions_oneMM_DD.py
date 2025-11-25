#!/usr/bin/env python

"""
Verbose QA version of coordinates_in_regions_oneMM_DD.

Features:
- Contrast as CLI arg (default: QSM)
- Skips regions whose outputs already exist (unless --force)
- Checksum-based skip: if inputs/code change, existing outputs are treated as stale
- Thread-based parallel execution per hemisphere via --nproc

Usage:
    python coordinates_in_regions_oneMM_DD.py \
        --ID S00775 \
        --input-dir  /mnt/newStor/paros/paros_WORK/hanwen/ad_decode_test/output/ \
        --output-dir /mnt/newStor/paros/paros_WORK/column_code_tester/ \
        --contrast QSM \
        --nproc 4
"""

import argparse
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.io import loadmat, savemat
import nibabel.freesurfer as fsio


PIPELINE_VERSION = "column_coords_v1.1_20251125"


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
    """Load FreeSurfer aparc annotation and print stats."""
    annot_fname = subject_dir / "label" / f"{hemi}.aparc.annot"
    print(f"\n[ANNOT] Loading {hemi}.aparc.annot from:\n        {annot_fname}")

    if not annot_fname.is_file():
        raise FileNotFoundError(f"Missing annotation: {annot_fname}")

    labels, ctab, names = fsio.read_annot(str(annot_fname))

    struct_names = [
        n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)
        for n in names
    ]

    print(f"[ANNOT] {hemi}: labels={labels.shape}, unique={len(np.unique(labels))}")
    print(f"[ANNOT] {hemi}: struct_names={len(struct_names)}, ctab rows={ctab.shape[0]}")
    print(f"[ANNOT] {hemi}: first names: {struct_names[:5]}")

    return labels, ctab, struct_names


def _region_indices_for_vertices(vertex_indices: np.ndarray) -> np.ndarray:
    """
    Given vertex indices (0-based), produce 1-based "depth indices" into cp_dwi columns.
    """
    if vertex_indices.size == 0:
        return np.array([], dtype=int)

    depths = np.arange(48, dtype=int)   # 0..47
    depth_indices = depths + 48         # 48..95

    repeated = np.tile(depth_indices[:, None], (1, vertex_indices.size))
    index_1b = repeated.flatten(order="F")

    return index_1b


# ------------------ CHECKSUM UTILITIES ------------------ #

def _compute_hemi_checksum(
    hemi: str,
    labels: np.ndarray,
    ctab: np.ndarray,
    struct_names: list[str],
    ori_cp_dwi: np.ndarray,
) -> str:
    """Compute a SHA256 checksum for all inputs relevant to one hemisphere."""
    h = hashlib.sha256()
    h.update(PIPELINE_VERSION.encode("utf-8"))
    h.update(hemi.encode("utf-8"))
    h.update(labels.tobytes())
    h.update(ctab.tobytes())
    joined_names = "\n".join(struct_names).encode("utf-8")
    h.update(joined_names)
    h.update(ori_cp_dwi.tobytes())
    return h.hexdigest()


# ------------------ PER-REGION PROCESSING (POSSIBLY PARALLEL) ------------------ #

def _process_single_region(
    i: int,
    hemi: str,
    labels: np.ndarray,
    ctab: np.ndarray,
    struct_names: list[str],
    ori_cp_dwi: np.ndarray,
    out_dir: Path,
    max_index: int,
    skip_existing_ok: bool,
    force: bool,
):
    """
    Process a single region index i.
    Returns:
        (files_written, regions_with_vertices, skipped_messages_list)
    """
    skipped_msgs = []
    files_written = 0
    regions_with_vertices = 0

    row_idx = i - 1
    region_name = struct_names[row_idx] if 0 <= row_idx < len(struct_names) else None

    # MATLAB hard-coded skip
    if i == 5:
        msg = f"{hemi} i={i}, row_idx={row_idx}, region={region_name or '<unknown>'}: MATLAB skip (i=5)"
        print(f"[{hemi}] {msg}")
        skipped_msgs.append(msg)
        return files_written, regions_with_vertices, skipped_msgs

    if row_idx >= len(struct_names) or row_idx >= ctab.shape[0]:
        msg = f"{hemi} i={i}, row_idx={row_idx}: out of bounds for struct_names/ctab → skip"
        print(f"[{hemi}] {msg}")
        skipped_msgs.append(msg)
        return files_written, regions_with_vertices, skipped_msgs

    region_index = row_idx
    region_id_orig = int(ctab[row_idx, 4])

    vertex_indices = np.where(labels == region_index)[0]
    n_vertices = vertex_indices.size

    print(f"\n[{hemi}] REGION i={i}, row_idx={row_idx}")
    print(f"    name          = '{region_name}'")
    print(f"    region_index  = {region_index} (labels index)")
    print(f"    region_id_orig= {region_id_orig} (ctab[:,4])")
    print(f"    n_vertices    = {n_vertices}")

    if n_vertices == 0:
        msg = (
            f"{hemi} region '{region_name}' (i={i}, index={region_index}): "
            "no vertices with this label index → skip"
        )
        print(f"    -> {msg}")
        skipped_msgs.append(msg)
        return files_written, regions_with_vertices, skipped_msgs

    regions_with_vertices = 1

    output_name = f"{hemi}_{region_name}"
    mat_path = out_dir / f"{output_name}.mat"
    csv_path = out_dir / f"{output_name}.csv"

    # Skip if outputs exist and we trust them (skip_existing_ok == True)
    if skip_existing_ok and mat_path.is_file() and csv_path.is_file():
        msg = f"{hemi} region '{region_name}' (i={i}): output exists and checksum OK → skipping"
        print(f"    -> {msg}")
        skipped_msgs.append(msg)
        return files_written, regions_with_vertices, skipped_msgs

    if force:
        print("    -> FORCE recompute")

    # Compute depth indices
    index_1b = _region_indices_for_vertices(vertex_indices)
    valid_mask = index_1b <= max_index
    index_1b = index_1b[valid_mask]

    print(f"    depth indices kept = {index_1b.size} (after clipping > max_index)")

    if index_1b.size == 0:
        msg = (
            f"{hemi} region '{region_name}' (i={i}, index={region_index}): "
            "no valid depth indices after clipping → skip"
        )
        print(f"    -> {msg}")
        skipped_msgs.append(msg)
        return files_written, regions_with_vertices, skipped_msgs

    idx_0b = index_1b - 1
    cp_dwi = ori_cp_dwi[:, idx_0b]

    savemat(mat_path, {f"{hemi}_cp_dwi": cp_dwi})
    np.savetxt(csv_path, cp_dwi, delimiter=",")

    print(f"    -> WROTE {mat_path.name}, {csv_path.name}")
    files_written = 1

    return files_written, regions_with_vertices, skipped_msgs


# ------------------ CORE PER-HEMI PROCESSING ------------------ #

def _process_hemi(
    hemi: str,
    labels: np.ndarray,
    ctab: np.ndarray,
    struct_names: list[str],
    ori_cp_dwi: np.ndarray,
    out_dir: Path,
    force: bool = False,
    nproc: int = 1,
):
    print(f"\n===== PROCESSING {hemi.upper()} HEMISPHERE =====")
    max_index = ori_cp_dwi.shape[1]

    # ---- Checksum logic ----
    checksum_path = out_dir / f"{hemi}_inputs.sha256"
    current_checksum = _compute_hemi_checksum(hemi, labels, ctab, struct_names, ori_cp_dwi)
    previous_checksum = None
    inputs_changed = False

    if checksum_path.is_file():
        try:
            previous_checksum = checksum_path.read_text().strip()
            if previous_checksum != current_checksum:
                inputs_changed = True
        except Exception as e:
            print(f"[{hemi}] WARNING: Failed to read previous checksum: {e}")
            inputs_changed = True
    else:
        inputs_changed = True  # no checksum yet → treat as "new"

    if inputs_changed:
        print(f"[{hemi}] INPUTS OR CODE CHANGED (checksum mismatch or missing).")
        print(f"[{hemi}] Existing outputs will NOT be trusted; recomputing regions as needed.")
    else:
        print(f"[{hemi}] Checksum matches. Existing outputs considered up-to-date.")

    # Write (or overwrite) checksum file with current value
    try:
        checksum_path.write_text(current_checksum + "\n")
    except Exception as e:
        print(f"[{hemi}] WARNING: could not write checksum file: {e}")

    # skip_existing_ok means we are allowed to skip based on existing files.
    skip_existing_ok = (not force) and (not inputs_changed)

    # ---- Parallel region processing ----
    region_indices = [i for i in range(2, 37)]  # MATLAB: 2..36
    total_files_written = 0
    total_regions_with_vertices = 0
    all_skipped_msgs = []

    def _task(i):
        return _process_single_region(
            i=i,
            hemi=hemi,
            labels=labels,
            ctab=ctab,
            struct_names=struct_names,
            ori_cp_dwi=ori_cp_dwi,
            out_dir=out_dir,
            max_index=max_index,
            skip_existing_ok=skip_existing_ok,
            force=force,
        )

    if nproc is None or nproc < 1:
        nproc = 1

    if nproc == 1:
        for i in region_indices:
            files_written, regions_with_vertices, skipped_msgs = _task(i)
            total_files_written += files_written
            total_regions_with_vertices += regions_with_vertices
            all_skipped_msgs.extend(skipped_msgs)
    else:
        print(f"[{hemi}] Running with nproc={nproc} (thread-based parallelism)")
        with ThreadPoolExecutor(max_workers=nproc) as ex:
            for files_written, regions_with_vertices, skipped_msgs in ex.map(_task, region_indices):
                total_files_written += files_written
                total_regions_with_vertices += regions_with_vertices
                all_skipped_msgs.extend(skipped_msgs)

    # ---- Summary ----
    print(f"\n>>> SUMMARY {hemi.upper()} <<<")
    print(f"    Files written this run:   {total_files_written}")
    print(f"    Regions with vertices:    {total_regions_with_vertices}")
    print(f"    Output dir:               {out_dir}")

    print(f"\n>>> {hemi.upper()} REGIONS SKIPPED / NOT WRITTEN <<<")
    if all_skipped_msgs:
        for msg in all_skipped_msgs:
            print("    -", msg)
    else:
        print("    None.")


# ------------------ TOP-LEVEL FUNCTION ------------------ #

def coordinates_in_regions_oneMM_DD(
    ID: str,
    input_dir,
    output_dir,
    contrast: str = "QSM",
    force: bool = False,
    nproc: int = 1,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    contrast_dir = input_dir / ID / contrast
    subject_dir = input_dir / ID / ID
    out_dir = output_dir / ID / contrast / "label_coord_1mm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[PATH] -------")
    print("[PATH] Subject ID:         ", ID)
    print("[PATH] Input dir:          ", input_dir)
    print("[PATH] Contrast:           ", contrast)
    print("[PATH] Column dir:         ", contrast_dir)
    print("[PATH] Subject dir (labels)", subject_dir)
    print("[PATH] Output dir:         ", out_dir)
    print("[PATH] FORCE mode:         ", force)
    print("[PATH] nproc (per hemi):   ", nproc)
    print("[PATH] Pipeline version:   ", PIPELINE_VERSION)

    ori_lh_cp_dwi, ori_rh_cp_dwi = _load_cp_dwi_mats(contrast_dir, ID)

    labels_lh, ctab_lh, names_lh = _load_aparc_annot(subject_dir, "lh")
    labels_rh, ctab_rh, names_rh = _load_aparc_annot(subject_dir, "rh")

    _process_hemi("lh", labels_lh, ctab_lh, names_lh, ori_lh_cp_dwi, out_dir, force, nproc)
    _process_hemi("rh", labels_rh, ctab_rh, names_rh, ori_rh_cp_dwi, out_dir, force, nproc)


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all regions even when outputs + checksum exist.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of parallel worker threads per hemisphere.",
    )
    args = parser.parse_args()

    coordinates_in_regions_oneMM_DD(
        ID=args.ID,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        contrast=args.contrast,
        force=args.force,
        nproc=args.nproc,
    )


if __name__ == "__main__":
    _cli()
