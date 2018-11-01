#!/bin/bash
#SBATCH --
#SBATCH

ROOT_DIR="/scratch/ovd208/COCO_features/data"
ARTIFACTS_DIR="$SCRATCH/artifacts/image_captioning"
NAME='test_run'

echo "Starting $NAME"

OPTIONS="--debug"

PYARGS="--root_dir $ROOT_DIR --artifacts_dir $ARTIFACTS_DIR --name $NAME $OPTIONS"

source $SCRATCH/pyenv/img_caption/bin/activate
python src/main.py $PYARGS
