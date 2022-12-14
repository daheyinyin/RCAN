#!/bin/bash

# ============================================================================
if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DATASET_TYPE] [SCALE] [DEVICE_ID]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

model=$(get_real_path $1)
data_path=$(get_real_path $2)
dataset_type=$3
scale=$4


if [[ $scale -ne "2" &&  $scale -ne "3" &&  $scale -ne "4" ]]; then
    echo "[SCALE] should be in [2,3,4]"
exit 1
fi

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

log_file="./run_infer.log"
log_file=$(get_real_path $log_file)

echo "***************** param *****************"
echo "mindir name: "$model
echo "dataset path: "$data_path
echo "scale: "$scale
echo "log file: "$log_file
echo "***************** param *****************"

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

export PYTHONPATH=$PWD:$PYTHONPATH

function compile_app()
{
    echo "begin to compile app..."
    cd ../ascend310_infer || exit
    bash build.sh >> $log_file  2>&1
    cd -
    echo "finish compile app"
}

function preprocess()
{
    echo "begin to preprocess..."
    export DEVICE_ID=$device_id
    export RANK_SIZE=1
    rm -rf ../LR
    python ../preprocess.py --dataset_path=$data_path --dataset_type=$dataset_type --scale=$scale --save_path=../LR/ >> $log_file 2>&1
    echo "finish preprocess"
}

function infer()
{
    echo "begin to infer..."
    save_data_path=$data_path"/SR_bin/X"$scale
    if [ -d $save_data_path ]; then
        rm -rf $save_data_path
    fi
    mkdir -p $save_data_path
    ../ascend310_infer/out/main --mindir_path=$model --dataset_path=../LR/ --device_id=0 --save_dir=$save_data_path >> $log_file 2>&1
    echo "finish infer"
}

function postprocess()
{
    echo "begin to postprocess..."
    export DEVICE_ID=$device_id
    export RANK_SIZE=1
    python ../postprocess.py --dataset_path=$data_path --dataset_type=$dataset_type --bin_path=$data_path"/SR_bin/X"$scale --scale=$scale  >> $log_file 2>&1
    echo "finish postprocess"
}

echo "" > $log_file
echo "read the log command: "
echo "    tail -f $log_file"

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed, check $log_file"
    exit 1
fi

preprocess
if [ $? -ne 0 ]; then
    echo "preprocess code failed, check $log_file"
    exit 1
fi

infer
if [ $? -ne 0 ]; then
    echo " execute inference failed, check $log_file"
    exit 1
fi

postprocess
if [ $? -ne 0 ]; then
    echo "postprocess failed, check $log_file"
    exit 1
fi

cat $log_file | tail -n 3 | head -n 1
