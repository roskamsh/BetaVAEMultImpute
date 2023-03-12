#!/bin/bash
#$ -l tmem=3.9G
#$ -l h_vmem=3.9G
#$ -l h_rt=23:29:30
#$ -S /bin/bash
#$ -N lung
#$ -t 1-95
#$ -o /SAN/orengolab/PPI/BetaVAEMImputation/logs/
#$ -wd /SAN/orengolab/PPI/BetaVAEMImputation/
# qsub /SAN/orengolab/PPI/BetaVAEMImputation/conf/luadlusc_cv.qsub.sh
#These are optional flags but you probably want them in all jobs
#$ -j y
cd /SAN/orengolab/PPI/BetaVAEMImputation
python cross_validation/run_cross_validation.py