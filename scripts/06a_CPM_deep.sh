#!/bin/bash
#SBATCH --job-name=CPM_deep_training
#SBATCH --time=00:10:00
#SBATCH --array=0-9999
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/CPM_deep_training/DEEP_v1-0_REPEAT_01_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 06a_CPM_deep.py $SLURM_ARRAY_TASK_ID