#!/bin/bash
echo > best_noise_$1.txt
./convert_txt.sh res_sin_1.txt $1

n_mes=`awk -F"," 'NR==1 { print NF }' res_pr.txt`
n_mes=`expr $n_mes - 12`
n_mes=`expr $n_mes / 3`
n_mes=`expr $n_mes + 1`
echo $n_mes
n_mes=`expr $n_mes - 1`

#inverse for C
for i in 1 2 3 4 5 6
do
    ./gen_prog_run 0 $i 1 2 'best3/'
    ./nn2.py 50 res_pr.txt 0 0 $i 8 best3/mlp_torch0_${i}_50_8.pt `expr $n_mes + 1`
    echo 0 $i >> best_noise_$1.txt
    ./stat.py out_gprog_0_${i}.txt >> best_noise_$1.txt
    ./stat.py out_nn0_${i}_50_8.txt >> best_noise_$1.txt
done
#inverse for P
rm res_pr.txt
ln -s res_pr_P.txt res_pr.txt
for i in 0 7 8
do
    ./gen_prog_run 0 $i 1 2 'best3/'
    ./nn2.py 50 res_pr.txt 0 0 $i 8 best3/mlp_torch0_${i}_50_8.pt `expr $n_mes + 1`
    echo 0 $i >> best_noise_$1.txt
    ./stat.py out_gprog_0_${i}.txt >> best_noise_$1.txt
    ./stat.py out_nn0_${i}_50_8.txt >> best_noise_$1.txt
done
rm res_pr.txt
ln -s res_pr_C.txt res_pr.txt

# cleanup
rm res_pr_gprog*.txt
