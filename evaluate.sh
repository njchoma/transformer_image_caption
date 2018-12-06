#!/bin/bash

#SBATCH --output=slurm_out/caption_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00

ROOT_DIR="/scratch/ovd208/COCO_features/data"
ARTIFACTS_DIR="$SCRATCH/artifacts/image_captioning"
NAME='ba_full_test_run'
RUN_NB="$SLURM_ARRAY_TASK_ID"

BATCH_SIZE=1
MAX_NB_EPOCHS=40

echo "Starting $NAME"

OPTIONS="--resume_epoch -1"

PYARGS="--root_dir $ROOT_DIR --artifacts_dir $ARTIFACTS_DIR --name $NAME --max_nb_epochs $MAX_NB_EPOCHS --batch_size $BATCH_SIZE --run_nb $RUN_NB $OPTIONS"

source $SCRATCH/pyenv/img_caption/bin/activate
python src/evaluate_test.py $PYARGS
