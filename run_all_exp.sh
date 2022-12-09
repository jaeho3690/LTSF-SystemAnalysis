#!/bin/bash
chmod +x scripts/ETTh1/run_etth1_amp_trials.sh
chmod +x scripts/ETTh1/run_etth1_base_trials.sh
chmod +x scripts/ETTm1/run_ettm1_amp_trials.sh
chmod +x scripts/ETTm1/run_ettm1_base_trials.sh

wait
./scripts/ETTh1/run_etth1_amp_trials.sh
wait 
./scripts/ETTh1/run_etth1_base_trials.sh
wait 
./scripts/ETTm1/run_ettm1_amp_trials.sh
wait 
./scripts/ETTm1/run_ettm1_base_trials.sh
wait
 