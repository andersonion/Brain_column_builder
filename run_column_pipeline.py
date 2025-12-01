#!/usr/bin/env python

"""
run_column_pipeline.py

Wrapper to process ONE subject/ID through the full column/thickness pipeline:

    0) vertices_connect.py                    (build columns + *_column_*_dwi.mat)
    1) coordinates_in_regions_oneMM_DD.py     (run ONCE; contrast-agnostic)
    2) get_columns_in_regions_oneMM_DD.py     (looped over real contrasts)
    3) build_pairs_from_freesurfer.py
    4) get_thickness.py

Conventions
-----------

- `output_dir` is the root that contains:

      <output_dir>/<ID>/DWI2T1_dti_upsampled.dat
      <output_dir>/<ID>/columns/...
      <output_dir>/<ID>/<ID>/label/...

- FreeSurfer surfaces come from a possibly separate FS subjects dir:

      <fs_subjects_dir>/<ID>/surf/lh.white, etc.

  which can be:
      * provided via --fs-subjects-dir
      * or read from $SUBJECTS_DIR
      * or, if neither is set, defaulted to output_dir (legacy behavior)

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
from pathlib import Path

from vertices_connect import vertices_connect
from coordinates_in_regions_oneMM_DD import coordinates_in_regions_oneMM_DD
from get_columns_in_regions_oneMM_DD import get_columns_in_regions_oneMM_DD
from build_pairs_from_freesurfer import build_pairs_from_freesurfer
from get_thickness import get_thickness


def run_subject_pipeline(
    ID: str,
    input_dir,
    output_dir,
    contrasts,
    fs_subjects_dir=None,
    force_all: bool = False,
    nproc_coords: int = 1,  # reserved; currently unused
):
    """
    Run the full pipeline for one subject.

    Parameters
    ----------
    ID : str
        Subject ID (e.g., S00775).
    input_dir : str or Path
        Root for contrast images (QSM, MD, FA, etc.).
    output_dir : str or Path
        Root for FreeSurfer/columns/output products (DWI2T1 transform, columns, labels, etc.).
    contrasts : list of str
        List of contrast names to sample (e.g., ["QSM", "MD", "FA"]).
    fs_subjects_dir : str or Path, optional
        FreeSurfer subjects root containing <ID>/surf.
        If None, vertices_connect will use $SUBJECTS_DIR or fallback to output_dir.
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
    print(f"[FS SUBJECTS DIR]{' ' if fs_subjects_dir is None else ''}{fs_subjects_dir}")
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
        fs_subjects_dir=fs_subjects_dir,
        # voldim/voxres are defaulted inside vertices_connect
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
    # 2) get_columns_in_regions_oneMM_DD.py  (per contrast)
    # ------------------------------------------------------------
    for contrast in contrasts:
        print("\n------------------------------------------------------------")
        print(f"[STEP 2] get_columns_in_regions_oneMM_DD for contrast: {contrast}")
        print("------------------------------------------------------------")

        get_columns_in_regions_oneMM_DD(
            ID=ID,
            input_dir=input_dir,
            output_dir=output_dir,
            contrast=contrast,
            force=force_all,
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
    parser.add_argument("--ID", required=True, help="Subject ID, e.g. S00775")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root for contrast images (where <ID>_<contrast>[_masked].nii.gz live).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root for FreeSurfer/columns/outputs (contains <ID>/DWI2T1_dti_upsampled.dat, <ID>/columns, <ID>/<ID>/label).",
    )
    parser.add_argument(
        "--fs-subjects-dir",
        default=None,
        help="FreeSurfer SUBJECTS_DIR-style root containing <ID>/surf. "
             "If omitted, vertices_connect uses $SUBJECTS_DIR or falls back to --output-dir.",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        required=True,
        help="One or more contrast names to process (e.g. adc ad fa rd).",
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
        fs_subjects_dir=args.fs_subjects_dir,
        force_all=args.force_all,
        nproc_coords=args.nproc_coords,
    )


if __name__ == "__main__":
    _cli()
