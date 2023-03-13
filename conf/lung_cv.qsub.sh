#!/bin/bash
#$ -l tmem=3.9G
#$ -l h_rt=23:29:30
#$ -S /bin/bash
#$ -N lung
#$ -t 1-95
#$ -o /SAN/orengolab/PPI/BetaVAEMImputation/logs/
#$ -wd /SAN/orengolab/PPI/BetaVAEMImputation/
# qsub /SAN/orengolab/PPI/BetaVAEMImputation/conf/lung_cv.qsub.sh
#These are optional flags but you probably want them in all jobs
#$ -j y

source /share/apps/source_files/python/python-3.9.5.source
cd /SAN/orengolab/PPI/BetaVAEMImputation
export PATH=/share/apps/python-3.8.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.8.5-shared/lib:$LD_LIBRARY_PATH
python3 cross_validation/run_cross_validation.py ${SGE_TASK_ID}