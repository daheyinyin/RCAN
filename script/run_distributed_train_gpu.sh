#!/bin/bash

# ============================================================================

if [ $# != 3 ]; then
  echo "Usage: bash run_distributed_train_gpu.sh [TRAIN_DATA_DIR] [DEVICE_NUM] [CUDA_VISIBLE_DEVICES]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
export DEVICE_NUM=$2
export RANK_SIZE=$2
export CUDA_VISIBLE_DEVICES=$3

if [ ! -d $PATH1 ]; then
  echo "error: TRAIN_DATA_DIR=$PATH1 is not a directory"
  exit 1
fi


if [ -d "train_dis" ]; then
    rm -rf ./train_dis
fi
mkdir ./train_dis
cp ../*.py ./train_dis
cp -r ../src ./train_dis
cd ./train_dis || exit

env >env.log
echo "train_dis begin."

nohup mpirun -n $DEVICE_NUM --allow-run-as-root \
      python train.py \
      --run_distribute True \
      --device_target GPU \
      --batch_size 16 \
      --lr 1e-4 \
      --scale 2 \
      --dir_data $PATH1 \
      --epochs 500 \
      --test_every 4000 \
      --patch_size 48 > train.dis_log 2>&1 &

