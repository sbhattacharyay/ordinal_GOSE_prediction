#!/bin/bash
#SBATCH --job-name=calib_metrics_APM_deep
#SBATCH --time=01:00:00
#SBATCH --array=0-1999
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/APM_deep_performance/recalulate_calibration_APM_deep_performance_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 11a_APM_deep_performance.py $SLURM_ARRAY_TASK_ID