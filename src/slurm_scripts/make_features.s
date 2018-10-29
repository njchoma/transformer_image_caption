#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=10:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=make_features
#SBATCH --output=slurm_%j.out

#nvidia-smi

RUNDIR=$SCRATCH/imageCaptioning_run/dataset_check/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

cd $RUNDIR
#cd /home/ovd208/repos/traffic-sign-detection-homework
module purge
module load anaconda3/5.3.0
source activate imageCaptioning

$HOME/.conda/envs/imageCaptioning/bin/python $HOME/repos/bottom-up-attention-vqa/tools/detection_features_converter.py
