#!/usr/bin/env bash

# check the enviroment info
nvidia-smi
PYTHON="python -u"

WORK_DIR=$(cd $(dirname $0)/../../../;pwd)
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
cd ${WORK_DIR}

DATA_DIR=$3

BACKBONE="mobilenetv2_fpn128"
MODEL_NAME="nonlocalnowd"
CHECKPOINTS_NAME="mbv2_msra10k_nonlocalnowd_fpn128_gc345_dsnohemce_iters60000_lr2e-2"$2
PRETRAINED_MODEL=$4

CONFIG_FILE='configs/seg/msra10k/NLnowd_fcn_msra10k_seg.conf'
MAX_ITERS=60000
LOSS_TYPE="dsnohemce_loss"
BASE_LR=0.02

LOG_DIR="./log/seg/msra10k/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi

export NCCL_LL_THRESHOLD=0
#export NCCL_TREE_THRESHOLD=0

NGPUS=4
DIST_PYTHON="${PYTHON} -m torch.distributed.launch --nproc_per_node=${NGPUS}"
GC_BLOCK="--gc_index 6 13 17 --gc_ratio 16"

if [[ "$1"x == "train"x ]]; then
  ${DIST_PYTHON} main.py --config_file ${CONFIG_FILE} --phase train --train_batch_size 2 --val_batch_size 1 \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 --drop_last y --syncbn y --dist y \
                         --data_dir ${DATA_DIR} --base_lr ${BASE_LR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                         --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} ${GC_BLOCK} 2>&1 | tee ${LOG_FILE}

elif [[ "$1"x == "resume"x ]]; then
  ${DIST_PYTHON} main.py --config_file ${CONFIG_FILE} --phase train --train_batch_size 2 --val_batch_size 1 \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 --drop_last y --syncbn y --dist y \
                         --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                         --resume_continue y --resume_val y --resume ./checkpoints/seg/msra10k/${CHECKPOINTS_NAME}_latest.pth \
                         --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} ${GC_BLOCK} 2>&1 | tee -a ${LOG_FILE}

elif [[ "$1"x == "val"x ]]; then
  ${PYTHON} main.py --config_file ${CONFIG_FILE} --phase test --gpu 0 1 2 3 --gather n \
                    --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                    --resume ${PRETRAINED_MODEL} --resume_strict y\
                    --test_dir ${DATA_DIR}/val/image --out_dir val ${GC_BLOCK} 2>&1 | tee -a ${LOG_FILE}
  cd metric/seg/
  ${PYTHON} seg_evaluator.py --config_file "../../"${CONFIG_FILE} \
                             --pred_dir ../../results/seg/msra10k/${CHECKPOINTS_NAME}/val/label \
                             --gt_dir ${DATA_DIR}/val/label  2>&1 | tee -a "../../"${LOG_FILE}

elif [[ "$1"x == "test"x ]]; then
  ${PYTHON} main.py --config_file ${CONFIG_FILE} --phase test --gpu 0 1 2 3 --gather n \
                    --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                    --resume ./checkpoints/seg/msra10k/${CHECKPOINTS_NAME}_latest.pth \
                    --test_dir ${DATA_DIR}/test --out_dir test ${GC_BLOCK} 2>&1 | tee -a ${LOG_FILE}

else
  echo "$1"x" is invalid..."
fi
