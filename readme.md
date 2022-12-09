# Instruction
1. We only need ETT-small. Place your data on `data/ETT-small`.
2. Install `pytorch`, `pytorch-lightning==1.6.5` and `hydra`.  
3. First Install `pytorch` and then do the next installations. 
4. Save `ETT-small` data on `data/` directory

# 1 - CUDA 11.6
```
pip install torch==1.10.0+cu111
pip install pytorch-lightning==1.6.5
```

# 2 - CUDA 10.1
```
pip install torch==1.8.1
pip install torchvision==0.9.1
pip install pytorch-lightning==1.6.5
pip install hydra-core==1.2.0
pip install hydra-colorlog==1.2.0
pip install setuptools==59.5.0
```

# RUN EXPERIMENTS
PLEASE RUN THEM one by one!! \\
**see if this dev file first works. You should see the `Fast dev run works` printed out, this may take a few minutes**
```
sh scripts/fast_dev_run.sh 
```
**run experiments**
The outputs are stored in the `outputs` directory
```
sh run_all_exp.sh
```

# After EXPERIMENTS
run the jupyter file `notebook/combine_all_files.ipynb`, send me the `all_df`.

