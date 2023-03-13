#!/bin/bash
#$ -l tmem=11.9G
#$ -l h_rt=23:29:30
#$ -S /bin/bash
#$ -N lung
#$ -t 1-95
#$ -o /SAN/orengolab/nsp13/BetaVAEMImputation/logs/
#$ -wd /SAN/orengolab/nsp13/BetaVAEMImputation/
# qsub /SAN/orengolab/nsp13/BetaVAEMImputation/conf/lung_cv.qsub.sh
#These are optional flags but you probably want them in all jobs
#$ -j y


source /SAN/orengolab/nsp13/BetaVAEMImputation/cluster_conda/bin/activate
conda activate vae_imp_test
export PATH=/SAN/orengolab/nsp13/BetaVAEMImputation/:$PATH
export PYTHONPATH=/SAN/orengolab/nsp13/BetaVAEMImputation/:$PYTHONPATH

cd /SAN/orengolab/nsp13/BetaVAEMImputation
python cross_validation/run_cross_validation.py ${SGE_TASK_ID}