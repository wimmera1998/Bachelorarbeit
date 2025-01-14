#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
    # $(dirname "$0")/train.py $CONFIG --resume-from /MapTR_mounted/work_dirs/maptr_tiny_r50_24e/epoch_2.pth  --launcher pytorch ${@:3} --deterministic
