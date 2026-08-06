# Brain Column Builder

## Overview

Brain Column Builder samples quantitative MRI contrasts along cortical
columns generated from FreeSurfer surfaces. It produces regional depth
profiles, QA figures, and cortical thickness measurements for each
subject.

The pipeline is restart-safe and can resume partially completed
subjects.

------------------------------------------------------------------------

## Requirements

Software:

-   Python 3
-   FreeSurfer
-   FSL
-   ANTs

Python packages:

-   numpy
-   scipy
-   nibabel
-   pandas
-   matplotlib

------------------------------------------------------------------------

## Expected Inputs

For each subject, the pipeline expects:

``` text
<output_dir>/<ID>/
    DWI2T1_dti.dat
    <ID>/
        surf/
        label/
```

Scalar maps may be stored in either layout:

``` text
<input_dir>/<ID>/<ID>_<contrast>.nii.gz
<input_dir>/<ID>/<ID>_<contrast>_masked.nii.gz
```

or

``` text
<input_dir>/<ID>_<contrast>.nii.gz
<input_dir>/<ID>_<contrast>_masked.nii.gz
```

If both masked and unmasked versions exist, the masked image is used
automatically.

------------------------------------------------------------------------

## Running the Pipeline

### Single Subject

``` bash
python run_column_pipeline.py \
    --ID D0007 \
    --input-dir <input_dir> \
    --output-dir <output_dir> \
    --contrasts adc ad fa rd
```

Force regeneration of all outputs:

``` bash
python run_column_pipeline.py \
    --ID D0007 \
    --input-dir <input_dir> \
    --output-dir <output_dir> \
    --contrasts adc ad fa rd \
    --force-all
```

### Master Script

The recommended workflow is through:

``` text
ADRC_MUSE_master_script.sh
```

By default the master script automatically detects available scalar
contrasts.

Override automatic detection:

``` bash
COLUMN_CONTRASTS="adc ad fa rd"
```

Force recomputation:

``` bash
COLUMN_FORCE_ALL=1
```

------------------------------------------------------------------------

## Pipeline Steps

1.  **vertices_connect.py**\
    Generates cortical columns and transforms them into DWI space.

2.  **coordinates_in_regions_oneMM_DD.py**\
    Determines which cortical columns belong to each anatomical region.

3.  **get_columns_in_regions_oneMM_DD.py**\
    Samples each scalar map along every cortical column.

4.  **Cross-contrast cleaning**\
    Removes invalid cortical columns consistently across all contrasts.

5.  **QA generation**\
    Produces regional depth profiles and quality-control plots.

6.  **get_thickness.py**\
    Computes cortical thickness statistics for each region.

------------------------------------------------------------------------

## Output Structure

``` text
<output_dir>/<ID>/

    columns/
        shared geometry

    adc/
    ad/
    fa/
    rd/
        sampled profiles
        QA figures
        regional summaries

    thickness/
        cortical thickness results
```

------------------------------------------------------------------------

## Restart Behavior

The pipeline checks for existing outputs and skips completed work
whenever possible.

Use `--force-all` (or `COLUMN_FORCE_ALL=1` from the master script) to
regenerate all outputs.

------------------------------------------------------------------------

## Quality Assurance

Review:

-   QA profile PNGs
-   Regional profile CSVs
-   Thickness summary CSV
-   Final log for warnings or skipped regions

Empty cortical regions are expected for some subjects and do not
necessarily indicate a pipeline failure.

------------------------------------------------------------------------

## Contact

Questions or issues should be directed to the current maintainer of this
repository.
