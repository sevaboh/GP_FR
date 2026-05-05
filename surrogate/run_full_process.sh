#!/bin/bash
# do 5 runs of training on small dataset and select bests
./run_all_l0.sh 0
./run_all_l0.sh 1
./run_all_l0.sh 2
./run_all_l0.sh 3
./run_all_l0.sh 4
./select_best.sh "best" "0/ 1/ 2/ 3/ 4/"
# do 1 run on large dataset for sets with R2<0.8 and select bests
./run_big.sh
./select_best.sh "best2" "best/ 5_big/"
# generate larger dataset and run training for sets still with R2<0.8
./run_gen_large.sh 10000 best2
./run_all_l1.sh
./select_best.sh "best3" "best2/ 1_gen/"
# train for inverse problem with restricted dataset
./run_inv_test.sh 0
./run_inv_test.sh 1
