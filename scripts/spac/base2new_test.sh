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
LOADEP=5
SUB=new
 

MODEL_DIR=output/b2n/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
DIR=output/b2n/${TRAINER}/${CFG}_${SHOTS}shots/test_new/${DATASET}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    CUDA_VISIBLE_DEVICES=${DEV} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    CUDA_VISIBLE_DEVICES=${DEV} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi