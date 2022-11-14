#!/bin/bash

# ============================================================================

if [ $# != 4 ]; then
  echo "Usage: bash run_standalone_eval_gpu.sh [TEST_DATA_DIR] [CHECKPOINT_PATH] [DATASET_TYPE] [DEVICE_ID]"
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
DATASET_TYPE=$3
export DEVICE_ID=$4

if [ ! -d $PATH1 ]; then
  echo "error: TEST_DATA_DIR=$PATH1 is not a directory"
  exit 1
fi

if [ ! -f $PATH2 ]; then
  echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
  exit 1
fi

if [ -d "eval" ]; then
  rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp -r ../src ./eval
cd ./eval || exit
env >env.log
echo "start evaluation ..."

python eval.py \
    --device_target GPU \
    --device_id $DEVICE_ID \
    --dir_data=${PATH1} \
    --batch_size 1 \
    --test_only \
    --ext "img" \
    --data_test=${DATASET_TYPE} \
    --ckpt_path=${PATH2} \
    --scale 2 > eval.log 2>&1 &
