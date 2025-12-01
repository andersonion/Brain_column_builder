#!/bin/bash

###SBATCH --job-name=hanwen_diffusion_and_T1_preprocessing
project='ADRC'

runno="$1"
subject_id="${runno}"

# -------------------------------
# Locate this script directory (for run_column_pipeline.py)
# -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------------
# Base paths (env fallbacks)
# -------------------------------
PAROS="${PAROS:-/mnt/newStor/paros}"
WORK="${WORK:-/mnt/newStor/paros/paros_WORK}"
PAROS="${PAROS%/}"
WORK="${WORK%/}"

# Absolute logs dir so later cd's don't break it
BASE_DIR="$(pwd)"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

NFS_DIR="$PAROS/paros_WORK/hanwen/${project}/output"
mkdir -p "$NFS_DIR"

# Set the input and output directories
input_dir="$PAROS/paros_WORK/hanwen/${project}/input"
output_dir="${NFS_DIR}"
mkdir -p "${input_dir}" "${output_dir}"

# -------------------------------
# Scratch setup
# -------------------------------
LOCAL_SCRATCH="/tmp/$USER/${SLURM_JOB_ID:-manual}"
if [[ -d "/scratch" && -w "/scratch" ]]; then
    LOCAL_SCRATCH="/scratch/$USER/${SLURM_JOB_ID:-manual}"
else
    echo "WARNING: /scratch not available! Using /tmp instead."
    LOCAL_SCRATCH="/tmp/$USER/${SLURM_JOB_ID:-manual}"
fi
mkdir -p "$LOCAL_SCRATCH"
if [[ ! -d "$LOCAL_SCRATCH" ]]; then
    echo " ERROR: Failed to create $LOCAL_SCRATCH!"
    exit 1
else
    echo " Successfully created: $LOCAL_SCRATCH"
fi

# -------------------------------
# Logs
# -------------------------------
TIMING_LOG="${LOG_DIR}/timings_${SLURM_ARRAY_TASK_ID:-0}.log"
MEM_LOG="${LOG_DIR}/memory_${SLURM_ARRAY_TASK_ID:-0}.log"

script_start=$(date +%s)
echo "Job ${SLURM_ARRAY_TASK_ID:-0} started on node $(hostname) at $(date)" | tee -a "$TIMING_LOG"

# Background vmstat snapshot (simple; ok if short-lived)
vmstat -s > "$MEM_LOG" &
VMSTAT_PID=$!

# -------------------------------
# Tiny debug helper
# -------------------------------
_debug_exist() { local p="$1"; [[ -e "$p" ]] && echo "EXISTS: $p" || echo "MISSING: $p"; }

# -------------------------------
# Run-specific dirs/files
# -------------------------------
diff_prep_dir="${WORK}/human/diffusion_prep_MRtrix_${runno}"

# per-run backup directory (do NOT create it; presence indicates real data)
backup_inputs_dir="${WORK}/tmp_ADRC_MUSE/${runno}"
if [[ -d "${backup_inputs_dir}" ]]; then
    backup_inputs_available=1
else
    echo "No per-run backup dir found: ${backup_inputs_dir}"
    backup_inputs_available=0
fi

mkdir -p "$diff_prep_dir"

INPUT_MASK="${diff_prep_dir}/${runno}_mask.nii.gz"
MASK_FILE="$NFS_DIR/${subject_id}/${subject_id}_subjspace_mask.nii.gz"
FINAL_DWI="$NFS_DIR/${subject_id}/${subject_id}_dwi_masked.nii.gz"
mkdir -p "$NFS_DIR/${subject_id}"

# Copy mask to NFS if needed
if [[ -f "${INPUT_MASK}" ]]; then
    [[ -f "${MASK_FILE}" ]] || cp "${INPUT_MASK}" "${MASK_FILE}"
else
    if [[ ! -f "${MASK_FILE}" ]]; then
        echo "WARNING: No good mask file detected! (Looked in ${INPUT_MASK} and ${MASK_FILE})"
    fi
fi

# -------------------------------------------------------------------
# EXACT FINAL_DWI PRECEDENCE
# 1) Use FINAL_DWI if present
# 2) Else copy ${runno}_dwi_masked.nii.gz from diff_prep_dir
# 3) Else mask ${runno}_dwi.nii.gz from diff_prep_dir -> FINAL_DWI
# 4) Else fall back to original scratch/backup block (largest *.nii.gz in per-run backup dir)
# -------------------------------------------------------------------
masked_in_prep="${diff_prep_dir}/${runno}_dwi_masked.nii.gz"
unmasked_in_prep="${diff_prep_dir}/${runno}_dwi.nii.gz"

if [[ -f "${FINAL_DWI}" ]]; then
    echo "FINAL_DWI exists: ${FINAL_DWI}"

elif [[ -f "${masked_in_prep}" ]]; then
    echo "Using masked DWI from diff_prep_dir → ${masked_in_prep}"
    mkdir -p "$(dirname "${FINAL_DWI}")"
    cp -v "${masked_in_prep}" "${FINAL_DWI}"

elif [[ -f "${unmasked_in_prep}" ]]; then
    echo "Masking unmasked DWI from diff_prep_dir → ${unmasked_in_prep}"
    mkdir -p "$(dirname "${FINAL_DWI}")"
    if [[ -f "${MASK_FILE}" ]]; then
        fslmaths "${unmasked_in_prep}" -mul "${MASK_FILE}" "${FINAL_DWI}"
    else
        echo "WARNING: MASK_FILE missing (${MASK_FILE}). Copying unmasked as fallback."
        cp -v "${unmasked_in_prep}" "${FINAL_DWI}"
    fi

else
    echo "Falling back to backup workflow in: ${backup_inputs_dir}"

    INPUT_FILE=""
    if [[ "${backup_inputs_available}" -eq 1 ]]; then
        INPUT_FILE="$(ls -aSh "${backup_inputs_dir}"/*.nii.gz 2>/dev/null | head -1 || true)"
    fi
    if [[ -z "${INPUT_FILE}" ]]; then
        echo "ERROR: No *.nii.gz found in per-run backup dir: ${backup_inputs_dir}"
        exit 3
    fi

    SCRATCH_DWI="$LOCAL_SCRATCH/${subject_id}_dwi4D.nii.gz"
    SCRATCH_SUM="$LOCAL_SCRATCH/${subject_id}_dwi.nii.gz"
    SCRATCH_MASKED="$LOCAL_SCRATCH/${subject_id}_dwi_masked.nii.gz"

    if [[ ! -f ${FINAL_DWI} ]]; then
        if [[ ! -f $SCRATCH_MASKED ]]; then
            if [[ ! -f ${SCRATCH_SUM} ]]; then
                if [[ ! -f ${SCRATCH_DWI} ]]; then
                    # -------------------------------
                    # Step 1: Convert the 4D image and store in local scratch
                    # -------------------------------
                    step_start=$(date +%s)
                    mrconvert "$INPUT_FILE" -coord 3 1:end "$SCRATCH_DWI" -force
                    step_end=$(date +%s)
                    echo "Step 1: Image conversion completed in $((step_end - step_start)) seconds" | tee -a "$TIMING_LOG"
                fi
                # -------------------------------
                # Step 2: Sum up the DWI data and store in local scratch
                # -------------------------------
                step_start=$(date +%s)
                mrmath "$SCRATCH_DWI" sum "$SCRATCH_SUM" -axis 3 -force
                step_end=$(date +%s)
                echo "Step 2: DWI summation completed in $((step_end - step_start)) seconds" | tee -a "$TIMING_LOG"
            fi
            # -------------------------------
            # Step 3: Apply mask and store in local scratch
            # -------------------------------
            step_start=$(date +%s)
            if [[ -f "${MASK_FILE}" ]]; then
                mrcalc "$SCRATCH_SUM" "$MASK_FILE" -mult "$SCRATCH_MASKED" -force
            else
                echo "WARNING: MASK_FILE missing (${MASK_FILE}); copying unmasked sum as fallback."
                cp -v "$SCRATCH_SUM" "$SCRATCH_MASKED"
            fi
            step_end=$(date +%s)
            echo "Step 3: Mask application completed in $((step_end - step_start)) seconds" | tee -a "$TIMING_LOG"
        fi

        # -------------------------------
        # Step 4: Move final processed file to NFS
        # -------------------------------
        step_start=$(date +%s)
        mkdir -p "$(dirname "$FINAL_DWI")"
        mv "$SCRATCH_MASKED" "$FINAL_DWI"
        step_end=$(date +%s)
        echo "Step 4: Moved final processed file to NFS in $((step_end - step_start)) seconds" | tee -a "$TIMING_LOG"

        # Cleanup local scratch space
        rm -rf "$LOCAL_SCRATCH"
    fi
fi

# -------------------------------------------------------------------
# Downstream artifacts (guarded)
# -------------------------------------------------------------------
dwi_masked="${output_dir}/${runno}/${runno}_dwi_masked.nii.gz"
mkdir -p "${output_dir}/${runno}"

if [[ ! -s "${dwi_masked}" ]]; then
    echo ">> Creating masked DWI: ${dwi_masked}"
    if [[ -f "${MASK_FILE}" ]]; then
        fslmaths "${FINAL_DWI}" -mul "${MASK_FILE}" "${dwi_masked}"
    else
        echo "WARNING: MASK_FILE missing; copying FINAL_DWI as masked output."
        cp -v "${FINAL_DWI}" "${dwi_masked}"
    fi
    if [[ ! -s "${dwi_masked}" ]]; then
        echo "ERROR: Masked DWI failed to generate: ${dwi_masked}" >&2
        exit 1
    fi
else
    echo ">> Masked DWI exists. Skipping: ${dwi_masked}"
fi

# -------------------------------
# Inputs for tensor fit (explicit & fail-early toggle)
# -------------------------------
# Set STRICT_GRADS=0 to allow continuing without dwi2tensor.
STRICT_GRADS="${STRICT_GRADS:-1}"

nii4D="${diff_prep_dir}/${runno}_05_dwi_nii4D_biascorrected.mif"
[[ -f "${nii4D}" ]] || nii4D="${FINAL_DWI}"

find_first() {
  # usage: find_first <glob1> <glob2> ...
  for g in "$@"; do
    if compgen -G "$g" >/dev/null; then
      ls -S $g 2>/dev/null | head -1
      return 0
    fi
  done
  return 1
}

# Primary search in MRtrix dir
bvals="$(find_first \
  "${diff_prep_dir}/${runno}_bvals.txt" \
  "${diff_prep_dir}/${runno}.bval" \
  "${diff_prep_dir}/${runno}.bvals" \
  "${diff_prep_dir}/"*"_bvals.txt" \
  "${diff_prep_dir}/"*.bval \
  "${diff_prep_dir}/"*.bvals)" || true

bvecs="$(find_first \
  "${diff_prep_dir}/${runno}_bvecs.txt" \
  "${diff_prep_dir}/${runno}.bvec" \
  "${diff_prep_dir}/${runno}.bvecs" \
  "${diff_prep_dir}/"*"_bvecs.txt" \
  "${diff_prep_dir}/"*.bvec \
  "${diff_prep_dir}/"*.bvecs)" || true

# Backup only if not found and backup dir exists
if [[ -z "${bvals}" && "${backup_inputs_available}" -eq 1 ]]; then
  bvals="$(find_first \
    "${backup_inputs_dir}/"*"_bvals.txt" \
    "${backup_inputs_dir}/"*.bval \
    "${backup_inputs_dir}/"*.bvals)" || true
fi
if [[ -z "${bvecs}" && "${backup_inputs_available}" -eq 1 ]]; then
  bvecs="$(find_first \
    "${backup_inputs_dir}/"*"_bvecs.txt" \
    "${backup_inputs_dir}/"*.bvec \
    "${backup_inputs_dir}/"*.bvecs)" || true
fi

echo "GRADS: bvals=${bvals:-<missing>}  bvecs=${bvecs:-<missing>}" | tee -a "$TIMING_LOG"

# Early decision: bail out (default) if missing grads
if [[ -z "${bvals}" || -z "${bvecs}" ]]; then
  if [[ "${STRICT_GRADS}" -eq 1 ]]; then
    echo "ERROR: Missing gradients; set STRICT_GRADS=0 to skip tensor fit/maps." >&2
    echo "Searched MRtrix: ${diff_prep_dir}" >&2
    [[ "${backup_inputs_available}" -eq 1 ]] && echo "Searched backup: ${backup_inputs_dir}" >&2
    _debug_exist "${diff_prep_dir}"
    [[ "${backup_inputs_available}" -eq 1 ]] && _debug_exist "${backup_inputs_dir}"
    # FINAL_DWI was already generated above; exit to signal upstream.
    exit 4
  else
    echo "WARNING: Missing gradients; proceeding without tensor fit or maps."
  fi
fi

# Tensor fit (only if grads present)
dt="${WORK}/hanwen/${project}/output/${runno}/${runno}_dt.mif"
preexisting_dt="${diff_prep_dir}/${runno}_07_dt.mif"
mkdir -p "$(dirname "${dt}")"

if [[ -n "${bvals}" && -n "${bvecs}" ]]; then
  if [[ ! -f "${dt}" ]]; then
    if [[ -f "${preexisting_dt}" ]]; then
      cp "${preexisting_dt}" "${dt}"
    else
      dwi2tensor "${nii4D}" "${dt}" -fslgrad "${bvecs}" "${bvals}" -mask "${MASK_FILE}" -force
    fi
  fi

  # Scalar maps
  for contrast in fa adc ad rd cl cp cs value vector; do
    output="${WORK}/hanwen/${project}/output/${runno}/${runno}_${contrast}.nii.gz"
    preexisting="${diff_prep_dir}/${runno}_${contrast}.nii.gz"
    if [[ ! -f "${output}" ]]; then
      if [[ -f "${preexisting}" ]]; then
        cp "${preexisting}" "${output}"
      else
        if [[ -f "${dt}" ]]; then
          tensor2metric "${dt}" -${contrast} "${output}" -mask "${MASK_FILE}" -force
        else
          echo "NOTICE: ${contrast} map skipped (no tensor found)."
        fi
      fi
    fi
  done
fi

# -------------------------------------------------------------------
# FreeSurfer nested layout:
#   subjects_root = ${output_dir}/${runno}
#   subject_dir   = ${subjects_root}/${runno}
# -------------------------------------------------------------------
subjects_root="${output_dir}/${runno}"
subject_dir="${subjects_root}/${runno}"
brainmask="${subject_dir}/mri/brainmask.mgz"
wmparc="${subject_dir}/mri/wmparc.mgz"
orig001="${subject_dir}/mri/orig/001.mgz"

# T1 (prefer run output; then per-run backup)
T1_nii="${output_dir}/${runno}/${runno}_T1.nii.gz"
if [[ ! -f "${T1_nii}" && "${backup_inputs_available}" -eq 1 ]]; then
  T1_source="$(ls -S "${backup_inputs_dir}"/*T1*.nii.gz 2>/dev/null | head -1 || true)"
  if [[ -n "${T1_source}" ]]; then
    mkdir -p "$(dirname "${T1_nii}")"
    cp -v "${T1_source}" "${T1_nii}"
  fi
fi

# Optional nuke-and-rebuild (explicit)
if [[ "${RECON_FORCE_NEW:-0}" -eq 1 && -d "${subject_dir}" ]]; then
  echo "RECON_FORCE_NEW=1 → deleting existing subject dir: ${subject_dir}"
  rm -rf "${subject_dir}"
fi

# Subject existence
subject_exists=0
if [[ -d "${subject_dir}" ]]; then
  subject_exists=1
elif compgen -G "${subject_dir}/*" >/dev/null 2>&1; then
  subject_exists=1
fi

# If subject exists but lacks orig/001.mgz and we have a T1, SEED it
if [[ "${subject_exists}" -eq 1 && ! -f "${orig001}" && -f "${T1_nii}" ]]; then
  echo "Subject exists but missing orig/001.mgz; seeding from T1."
  mkdir -p "$(dirname "${orig001}")"
  mri_convert "${T1_nii}" "${orig001}"
fi

# Decide recon-all action (NOTE: -sd must point to subjects_root)
export SUBJECTS_DIR="${subjects_root}"

if [[ -f "${wmparc}" ]]; then
  echo "recon-all outputs already present: ${wmparc} (skipping)"

elif [[ "${subject_exists}" -eq 1 ]]; then
  echo "Subject dir exists; resuming recon-all WITHOUT -i"
  recon-all -s "${runno}" -sd "${subjects_root}" -all

else
  if [[ -f "${T1_nii}" ]]; then
    echo "Starting recon-all fresh with -i ${T1_nii}"
    recon-all -s "${runno}" -i "${T1_nii}" -sd "${subjects_root}" -all
  else
    echo "WARNING: No T1 found for ${runno}; skipping recon-all and bbregister."
  fi
fi

# bbregister only if brainmask exists and is non-empty
test_file="$PAROS/paros_WORK/hanwen/${project}/output/${runno}/DWI2T1_dti.dat"

if [[ -s "${brainmask}" ]]; then
  if [[ ! -f "${test_file}" ]]; then
    _debug_exist "${brainmask}"
    bbregister \
      --s "${subject_id}" \
      --mov "${dwi_masked}" \
      --reg "${test_file}" \
      --sd "${subjects_root}" \
      --dti --init-fsl --12
  else
    echo "bbregister output exists: ${test_file} (skipping)"
  fi
else
  echo "NOTICE: ${brainmask} not present; bbregister skipped."
fi

# -------------------------------------------------------------------
# Column / thickness pipeline (run_column_pipeline.py)
# -------------------------------------------------------------------
# We expect:
#   - Transform file at: ${output_dir}/${runno}/DWI2T1_dti.dat
#   - Scalar maps at:    ${output_dir}/${runno}/${runno}_<contrast>.nii.gz
#   - FreeSurfer at:     ${output_dir}/${runno}/${runno}
#
# Contrasts behavior:
#   - If COLUMN_CONTRASTS is set (space-separated), use exactly that list.
#       e.g. export COLUMN_CONTRASTS="fa adc ad rd qsm cbf"
#   - Otherwise, auto-detect among: fa adc ad rd qsm cbf

column_transform="${output_dir}/${runno}/DWI2T1_dti.dat"
column_input_root="${output_dir}"
subject_map_dir="${column_input_root}/${runno}"

if [[ ! -f "${column_transform}" ]]; then
  echo "NOTICE: Column pipeline skipped: transform file missing: ${column_transform}"
else
  declare -a detected_contrasts=()

  if [[ -n "${COLUMN_CONTRASTS:-}" ]]; then
    # Use user-specified contrasts verbatim
    # Example: COLUMN_CONTRASTS="fa adc ad rd qsm cbf"
    read -r -a detected_contrasts <<< "${COLUMN_CONTRASTS}"
    echo "COLUMN_CONTRASTS provided externally: ${detected_contrasts[*]}"
  else
    # Auto-detect from maps actually present for this subject
    candidate_contrasts=(fa adc ad rd qsm cbf)
    for c in "${candidate_contrasts[@]}"; do
      if [[ -s "${subject_map_dir}/${runno}_${c}.nii.gz" ]]; then
        detected_contrasts+=("$c")
      fi
    done
    echo "Auto-detected contrasts for column pipeline: ${detected_contrasts[*]:-(none)}"
  fi

  if (( ${#detected_contrasts[@]} == 0 )); then
    echo "NOTICE: Column pipeline skipped: no scalar maps found / specified for ${subject_map_dir}"
  else
    echo "Running column pipeline for ${runno}"
    echo "  Input root: ${column_input_root}"
    echo "  Output dir: ${output_dir}"
    echo "  Contrasts:  ${detected_contrasts[*]}"

    python "${SCRIPT_DIR}/run_column_pipeline.py" \
      --ID "${subject_id}" \
      --input-dir "${column_input_root}" \
      --output-dir "${output_dir}" \
      --contrasts "${detected_contrasts[@]}"
  fi
fi

# Finalizing: Log total runtime
# -------------------------------
script_end=$(date +%s)
total_runtime=$((script_end - script_start))
echo "Job ${SLURM_ARRAY_TASK_ID:-0} completed in $total_runtime seconds" | tee -a "$TIMING_LOG"

# Stop Memory Logging
kill "$VMSTAT_PID" 2>/dev/null || true
echo "Final Memory Usage for Job ${SLURM_ARRAY_TASK_ID:-0}:" >> "$MEM_LOG"
#sstat --format=JobID,MaxRSS,AveRSS,Elapsed $SLURM_JOB_ID >> "$MEM_LOG"
