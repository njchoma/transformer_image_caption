#!/bin/bash
#SBATCH --
#SBATCH

TRAIN_DIR="$SCRATCH/data/image_captioning"
ARTIFACTS_DIR="$SCRATCH/artifacts/image_captioning"
NAME='test_run'


PYARGS="--train_dir $TRAIN_DIR --artifacts_dir $ARTIFACTS_DIR --name $NAME"

source ~/pyenv/openpose/bin/activate
python src/main.py $PYARGS
