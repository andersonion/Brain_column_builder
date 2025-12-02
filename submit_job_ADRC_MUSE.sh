#!/bin/bash

# SLURM job settings for the master job that will submit other jobs.
#SBATCH -N 1                       # Only need 1 node for this script
#SBATCH --ntasks=1                 # 1 task for this script
#SBATCH --cpus-per-task=1          # 1 CPU per task for this script
#SBATCH -o /home/alex/job.%j.out   # Output log file
#SBATCH -p normal                  # Partition to submit to (adjust as needed)

project=ADRC
# Set the input and output directories
input_dir="${WORK}/tmp_ADRC_MUSE/"
output_dir="$PAROS/paros_WORK/hanwen/${project}/output/"
#code_path="$PAROS/paros_WORK/hanwen/"
code_path="${PWD}/"
mkdir -p ${code_path} ${output_dir};

# List of subjects (or run numbers) to process
cd ${input_dir};

# Scrape runnos from folder names in input directory:
for runno in $(ls -d  D*/ | cut -d '/' -f 1);do
# Looping over the runnos array and submit jobs for each
    # Build the command for this subject
    echo "Processing: ${runno}"

    cmd="${code_path}ADRC_MUSE_master_script.sh ${runno}"
    # Submit the job using your submit_cluster_job.bash script
    bash ${GUNNIES}/submit_cluster_job.bash "$cmd"
done
