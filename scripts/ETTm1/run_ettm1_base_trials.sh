#!/bin/bash
echo 'this is ettm1 amp trials'

START_EXP_NUM=4999

for model in Dlinear Informer Transformer; do
    for pred_len in 96 672; do 
        for batch_size in 32 128; do 
            for seed in 1 2 3; do
                START_EXP_NUM=$(( START_EXP_NUM + 1 ))
                echo "Running experiment $START_EXP_NUM"
                python main.py  data=ETTm1 \
                                model=$model \
                                pred_len=$pred_len \
                                optimization.batch_size=$batch_size \
                                seed=$seed \
                                exp_num=$START_EXP_NUM \
                                use_amp=false &
                wait 
                echo "Experiment $START_EXP_NUM done"
            done 
        done 
    done
done
echo "All done"