#!/bin/bash


# custom config
DATA="/path/to/dataset/folder"
TRAINER=SPAC

DATASET=$1
SEED=$2
DEV=$3

CFG=vit_b16_cross_datasets
SHOTS=16



DIR=output/xd/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${DEV} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
# fi