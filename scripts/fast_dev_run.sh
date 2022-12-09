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
echo "Fast dev run works"