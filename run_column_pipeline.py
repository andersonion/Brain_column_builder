#!/usr/bin/env python

"""
run_column_pipeline.py

Wrapper to process ONE subject/ID through the full column/thickness pipeline:

    1) coordinates_in_regions_oneMM_DD.py
    2) get_columns_in_regions_oneMM_DD.py   (looped over contrasts)
    3) build_pairs_from_freesurfer.py
    4) get_thickness.py

Assumed function signatures (from your updated scripts):

    coordinates_in_regions_oneMM_DD.coordinates_in_regions_oneMM_DD(
        ID: str,
        input_dir,
        output_dir,
        contrast: str = "QSM",
        force: bool = False,
        nproc: int = 1,
    )

    get_columns_in_regions_oneMM_DD.get_columns_in_regions_oneMM_DD(
        ID: str,
        input_dir,
        output_dir,
        contrast: str,
        force: bool = False,
    )

    build_pairs_from_freesurfer.build_pairs_from_freesurfer(
        ID: str,
        output_dir,
        force: bool = False,
    )

    get_thickness.get_thickness(
        ID: str,
        output_dir,
        force: bool = False,
    )

Usage examples
--------------

# Single contrast (QSM), default behavior (will use checksum/skip logic inside scripts)
python run_column_pipeline.py \
    --ID S00775 \
    --input-dir /path/to/input_root \
    --output-dir /path/to/output_root \
    --contrasts QSM

# Multiple contrasts
python run_column_pipeline.py \
    --ID S00775 \
    --input-dir /path/to/input_root \
    --output-dir /path/to/output_root \
    --contrasts QSM MD FA

# Force all steps to recompute (ignoring prior outputs)
python run_column_pipeline.py \
    --ID S00775 \
    --input-dir /path/to/input_root \
    --output-dir /path/to/output_root \
    --contrasts QSM MD \
    --force-all

# Use some parallelism in coordinates step
python run_column_pipeline.py \
    --ID S00775 \
    --input-dir /path/to/input_root \
    --output-dir /path/to/output_root \
    --contrasts QSM \
    --nproc-coords 8
"""

import argparse
from pathlib import Path

from coordinates_in_regions_oneMM_DD import coordinates_in_regions_oneMM_DD
from get_columns_in_regions_oneMM_DD import get_columns_in_regions_oneMM_DD
from build_pairs_from_freesurfer import build_pairs_from_freesurfer
from get_thickness import get_thickness


def run_subject_pipeline(
    ID: str,
    input_dir,
    output_dir,
    contrasts,
    force_all: bool = False,
    nproc_coords: int = 1,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    contrasts = list(contrasts)

    print("\n================ COLUMN / THICKNESS PIPELINE ================")
    print(f"[SUBJECT]   {ID}")
    print(f"[INPUT DIR] {input_dir}")
    print(f"[OUTPUT DIR]{output_dir}")
    print(f"[CONTRASTS] {', '.join(contrasts)}")
    print(f"[FORCE ALL] {force_all}")
    print(f"[NPROC COORDS] {nproc_coords}")
    print("=============================================================\n")

    # ------------------------------------------------------------
    # 1) coordinates_in_regions_oneMM_DD.py  (per contrast)
    # ------------------------------------------------------------
    for contrast in contrasts:
        print("\n------------------------------------------------------------")
        print(f"[STEP 1] coordinates_in_regions_oneMM_DD for contrast: {contrast}")
        print("------------------------------------------------------------")

        coordinates_in_regions_oneMM_DD(
            ID=ID,
            input_dir=input_dir,
            output_dir=output_dir,
            contrast=contrast,
            force=force_all,
            nproc=nproc_coords,
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
        help="Root input dir (where <ID>/<contrast>/... and <ID>/<ID>/ live).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output dir (same root used by the component scripts).",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        required=True,
        help="One or more contrast names to process (e.g. QSM MD FA).",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force recompute for ALL steps (overrides checksum/skip behavior).",
    )
    parser.add_argument(
        "--nproc-coords",
        type=int,
        default=1,
        help="Number of worker threads per hemisphere for coordinates_in_regions_oneMM_DD.",
    )

    args = parser.parse_args()

    run_subject_pipeline(
        ID=args.ID,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        contrasts=args.contrasts,
        force_all=args.force_all,
        nproc_coords=args.nproc_coords,
    )


if __name__ == "__main__":
    _cli()
