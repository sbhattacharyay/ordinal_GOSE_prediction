#!/bin/bash
#SBATCH --job-name=ORC_dropout
#SBATCH --time=00:20:00
#SBATCH --array=0-1999
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=MENON-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --mail-type=ALL
#SBATCH --output=./hpc_logs/CPM_deep_interrepeat_dropout/POSTREPEAT01_bs_ORC_%a.out
#SBATCH --mail-user=sb2406@cam.ac.uk

. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load rhel8/default-icl
module load python/3.8

source ~/python_venv/bin/activate

srun python 06c_CPM_deep_hyperparameter_testing $SLURM_ARRAY_TASK_ID