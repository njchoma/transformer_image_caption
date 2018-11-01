#!/bin/bash
#SBATCH --
#SBATCH

ROOT_DIR="/scratch/ovd208/COCO_features/data"
ARTIFACTS_DIR="$SCRATCH/artifacts/image_captioning"
NAME='test_run'


PYARGS="--root_dir $ROOT_DIR --artifacts_dir $ARTIFACTS_DIR --name $NAME"

source $SCRATCH/pyenv/img_caption/bin/activate
python src/main.py $PYARGS
