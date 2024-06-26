#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='eval'

python eval.py \
    --config = configs/eval.yaml \
    --base_size 2048 \
    --scales 1.0 \
    --model_path= "your model path"
    --save_folder= "output path"\
    # 2>&1 | tee log/val_best_$now.txt