#!/bin/bash

# ============================================================================

if [ $# != 2 ]; then
  echo "Usage: bash run_distributed_train_ascend.sh [RANK_TABLE_FILE] [TRAIN_DATA_DIR]"
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
PATH2=$(get_real_path $2)

if [ ! -f $PATH1 ]; then
  echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
  exit 1
fi

if [ ! -d $PATH2 ]; then
  echo "error: TRAIN_DATA_DIR=$PATH2 is not a directory"
  exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  rm -rf ./train_parallel$i
  mkdir ./train_parallel$i
  cp ../*.py ./train_parallel$i
  cp *.sh ./train_parallel$i
  cp -r ../src ./train_parallel$i
  cd ./train_parallel$i || exit
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  env >env.log

  nohup python train.py \
        --run_distribute True \
        --batch_size 16 \
        --lr 1e-4 \
        --scale 2 \
        --dir_data $PATH2 \
        --epochs 500 \
        --test_every 4000 \
        --patch_size 48 > train.log 2>&1 &
  cd ..
done
