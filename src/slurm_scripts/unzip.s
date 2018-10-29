#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=10:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=unzip_features
#SBATCH --output=slurm_%j.out

#nvidia-smi

RUNDIR=$SCRATCH/imageCaptioning_run/dataset_check/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

cd $RUNDIR
#cd /home/ovd208/repos/traffic-sign-detection-homework

unzip $SCRATCH/COCO_features/data/trainval_36.zip
