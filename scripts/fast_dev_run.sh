#!/bin/bash
echo 'this is a fast dev run to see if amp, torch lightning is working'



for model in Dlinear Informer Transformer; do
    echo "Running experiment"
    python main.py  data=ETTh1 \
                    model=$model \
                    exp_num=9999 \
                    fast_dev_run=true \
                    optimization.batch_size=128 \
                    use_amp=true &
    wait 
    python main.py  data=ETTm1 \
                    model=$model \
                    exp_num=9999 \
                    fast_dev_run=true \
                    optimization.batch_size=128 \
                    use_amp=true &
    wait
done
## The below should result in nearly MSE=0.375 MAE=0.399
python main.py  data=ETTh1 \
                model=Dlinear \
                exp_num=9998 \
                optimization.batch_size=32 \
                optimization.optimizer.learning_rate=0.005 \
                seq_len=336 \
                seed=2021 \
                pred_len=96&
echo "Fast dev run works"