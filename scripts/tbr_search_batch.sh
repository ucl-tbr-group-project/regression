#!/bin/bash
#SBATCH -p RCIF
#SBATCH -N 1
#SBATCH -c 40
#SBATCH --time=23:59:59

export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE

srun -n1 ~/fusion/tbr_search_run.sh XX_job_tag
