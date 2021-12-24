#!/bin/bash
#SBATCH --job-name=compile_SHAP_APM
#SBATCH --time=00:10:00
#SBATCH --array=0-6199
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/APM_deep_SHAP/compile_SHAP_APM_deep_trial_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 12b_APM_compile_SHAP.py $SLURM_ARRAY_TASK_ID