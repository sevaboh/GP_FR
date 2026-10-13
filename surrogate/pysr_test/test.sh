#!/usr/bin/bash
./pysr_baselines.py timedep   --data ../log3d.txt       --outdir out_td --test-frac 0.01 > results_td.txt
./pysr_baselines.py pointwise --data ../res_const_0.txt --outdir out_pw_const_es --unary exp sin > result_pw_const_es.txt
./pysr_baselines.py pointwise --data ../res_sin_0.txt --outdir out_pw_sin_es --unary exp sin > result_pw_sin_es.txt
