#!/bin/bash
#SBATCH --job-name=SHAP_APM_deep
#SBATCH --time=02:00:00
#SBATCH --array=0-399
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/APM_deep_SHAP/calculate_SHAP_APM_deep_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 12a_APM_deep_SHAP.py $SLURM_ARRAY_TASK_ID