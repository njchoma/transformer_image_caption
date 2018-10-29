#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=10:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=dataset_check
#SBATCH --output=slurm_%j.out

#nvidia-smi

SRCDIR=$HOME/repos/transformer_image_caption/src
RUNDIR=$SCRATCH/imageCaptioning_run/dataset_check/run-${SLURM_JOB_ID/.*}
mkdir -p $RUNDIR

cd $RUNDIR
#cd /home/ovd208/repos/traffic-sign-detection-homework

module purge
module load anaconda3/5.3.0
source activate imageCaptioning

$HOME/.conda/envs/imageCaptioning/bin/python $SRCDIR/data_helpers/vocab.py --caption_path /scratch/ovd208/COCO_features/data/captions/annotations/captions_train2014.json --vocab_path /scratch/ovd208/COCO_features/data/vocab.pkl
