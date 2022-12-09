
#!/bin/bash

START_EXP_NUM=999

for model in Dlinear Informer Transformer; do
    for seq_len in 96 336; do 
        for batch_size in 32 128; do 
            for seed in 1 2 3; do
                START_EXP_NUM=$(( START_EXP_NUM + 1 ))
                echo "Running experiment $START_EXP_NUM"
                python main.py  data=ETTh1 \
                                model=$model \
                                seq_len=$seq_len \
                                optimization.batch_size=$batch_size \
                                seed=$seed \
                                exp_num=$START_EXP_NUM \
                                use_amp=true &
                wait 
                echo "Experiment $START_EXP_NUM done"
            done 
        done 
    done
done
echo "All done"