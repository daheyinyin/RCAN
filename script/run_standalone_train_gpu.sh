#!/bin/bash

# ============================================================================

if [ $# != 2 ]; then
  echo "Usage: bash run_standalone_train_gpu.sh [TRAIN_DATA_DIR] [DEVICE_ID]"
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
export DEVICE_ID=$2

if [ ! -d $PATH1 ]; then
  echo "error: TRAIN_DATA_DIR=$PATH1 is not a directory"
  exit 1
fi


if [ -d "train" ]; then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp -r ../src ./train
cd ./train || exit

env >env.log
echo "train begin."
nohup python train.py \
      --device_target GPU \
      --device_id $DEVICE_ID \
      --batch_size 16 \
      --lr 1e-4 \
      --scale 2 \
      --dir_data $PATH1 \
      --epochs 500 \
      --test_every 4000 \
      --patch_size 48 > train.log 2>&1 &

