#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=10:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=download_annotations
#SBATCH --output=slurm_%j.out

#nvidia-smi

RUNDIR=$SCRATCH/imageCaptioning_run/dataset_check/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

cd $RUNDIR
#cd /home/ovd208/repos/traffic-sign-detection-homework

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P $SCRATCH/COCO_features/data/
