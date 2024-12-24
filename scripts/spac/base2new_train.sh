#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=SPAC

DATASET=$1
SEED=$2
DEV=$3
CFG=vit_b16_batch4_deep9+4ctx
SHOTS=16


DIR=output/b2n/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    CUDA_VISIBLE_DEVICES=${DEV} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=${DEV} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi