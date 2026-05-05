#!/bin/bash
di='5_big'
echo > best_on_big.txt

./convert_txt.sh res1.txt

n_mes=`awk -F"," 'NR==1 { print NF }' res_pr.txt`
n_mes=`expr $n_mes - 12`
n_mes=`expr $n_mes / 3`
n_mes=`expr $n_mes + 1`
echo $n_mes
n_mes=`expr $n_mes - 1`

# test on big / train on big
for i in $(seq 0 $n_mes)
do
    ./gen_prog_run 1 $i 1 2 'best/'
    ./nn2.py 50 res_pr.txt 1 0 $i 8 best/mlp_torch1_${i}_50_8.pt `expr $n_mes + 1`
    echo 1 $i >> best_on_big.txt
    ./stat.py out_gprog_1_${i}.txt >> best_on_big.txt
    ./stat.py out_nn1_${i}_50_8.txt >> best_on_big.txt
    gR2=`./stat.py best/out_gprog_1_${i}.txt | tail -n 1 | cut -d" " -f2`
    nnR2=`./stat.py best/out_nn1_${i}_50_8.txt | tail -n 1 | cut -d" " -f2`
    gr=`python -c "print($gR2<0.8);"`
    nn=`python -c "print($nnR2<0.8);"`
    echo $index $gR2 $nnR2 $gr $nn
    if [ $gr == "True" ]; then
	./gen_prog_run 1 $i
    fi
    if [ $nn == "True" ]; then
        ./nn2.py 50 res_pr.txt 1 0 $i 8
    fi
done
#inverse for C
for i in 1 2 3 4 5 6
do
    ./gen_prog_run 0 $i 1 2 'best/'
    ./nn2.py 50 res_pr.txt 0 0 $i 8 best/mlp_torch0_${i}_50_8.pt `expr $n_mes + 1`
    echo 0 $i >> best_on_big.txt
    ./stat.py out_gprog_0_${i}.txt >> best_on_big.txt
    ./stat.py out_nn0_${i}_50_8.txt >> best_on_big.txt
    gR2=`./stat.py best/out_gprog_0_${i}.txt | tail -n 1 | cut -d" " -f2`
    nnR2=`./stat.py best/out_nn0_${i}_50_8.txt | tail -n 1 | cut -d" " -f2`
    gr=`python -c "print($gR2<0.8);"`
    nn=`python -c "print($nnR2<0.8);"`
    echo $index $gR2 $nnR2 $gr $nn
    if [ $gr == "True" ]; then
	./gen_prog_run 0 $i
    fi
    if [ $nn == "True" ]; then
        ./nn2.py 50 res_pr.txt 0 0 $i 8
    fi
done
#inverse for P
rm res_pr.txt
ln -s res_pr_P.txt res_pr.txt
for i in 0 7 8
do
    ./gen_prog_run 0 $i 1 2 'best/'
    ./nn2.py 50 res_pr.txt 0 0 $i 8 best/mlp_torch0_${i}_50_8.pt `expr $n_mes + 1`
    echo 0 $i >> best_on_big.txt
    ./stat.py out_gprog_0_${i}.txt >> best_on_big.txt
    ./stat.py out_nn0_${i}_50_8.txt >> best_on_big.txt
    gR2=`./stat.py best/out_gprog_0_${i}.txt | tail -n 1 | cut -d" " -f2`
    nnR2=`./stat.py best/out_nn0_${i}_50_8.txt | tail -n 1 | cut -d" " -f2`
    gr=`python -c "print($gR2<0.8);"`
    nn=`python -c "print($nnR2<0.8);"`
    echo $index $gR2 $nnR2 $gr $nn
    if [ $gr == "True" ]; then
	./gen_prog_run 0 $i
    fi
    if [ $nn == "True" ]; then
        ./nn2.py 50 res_pr.txt 0 0 $i 8
    fi
done
rm res_pr.txt
ln -s res_pr_C.txt res_pr.txt

# test newly trained on small and save
rm -rf ${di}
mkdir ${di}

./convert_txt.sh res0.txt

for i in $(seq 0 $n_mes)
do
    gR2=`./stat.py best/out_gprog_1_${i}.txt | tail -n 1 | cut -d" " -f2`
    nnR2=`./stat.py best/out_nn1_${i}_50_8.txt | tail -n 1 | cut -d" " -f2`
    gr=`python -c "print($gR2<0.8);"`
    nn=`python -c "print($nnR2<0.8);"`
    echo $index $gR2 $nnR2 $gr $nn
    if [ $gr == "True" ]; then
	./gen_prog_run 1 $i 1 2
	mv best_1_${i}.txt ${di}/
	mv out_gprog_1_${i}.txt ${di}/
	mv res_pr_gprog1_minmax_1_${i}.txt ${di}/
    fi
    if [ $nn == "True" ]; then
        ./nn2.py 50 res_pr.txt 1 0 $i 8 mlp_torch1_${i}_50_8.pt `expr $n_mes + 1`
	mv mlp_torch1_${i}_50_8.pt ${di}/
	mv out_nn1_${i}_50_8.txt ${di}/
	mv out_nn_norms_x_1_${i}_50_8.txt ${di}/
	mv out_nn_norms_y_1_${i}_50_8.txt ${di}/
    fi
done
#inverse for C
for i in 1 2 3 4 5 6
do
    gR2=`./stat.py best/out_gprog_0_${i}.txt | tail -n 1 | cut -d" " -f2`
    nnR2=`./stat.py best/out_nn0_${i}_50_8.txt | tail -n 1 | cut -d" " -f2`
    gr=`python -c "print($gR2<0.8);"`
    nn=`python -c "print($nnR2<0.8);"`
    echo $index $gR2 $nnR2 $gr $nn
    if [ $gr == "True" ]; then
	./gen_prog_run 0 $i 1 2
	mv best_0_${i}.txt ${di}/
	mv out_gprog_0_${i}.txt ${di}/
	mv res_pr_gprog1_minmax_0_${i}.txt ${di}/
    fi
    if [ $nn == "True" ]; then
        ./nn2.py 50 res_pr.txt 0 0 $i 8 mlp_torch0_${i}_50_8.pt `expr $n_mes + 1`
	mv mlp_torch0_${i}_50_8.pt ${di}/
	mv out_nn0_${i}_50_8.txt ${di}/
	mv out_nn_norms_x_0_${i}_50_8.txt ${di}/
	mv out_nn_norms_y_0_${i}_50_8.txt ${di}/
    fi
done
#inverse for P
rm res_pr.txt
ln -s res_pr_P.txt res_pr.txt
for i in 0 7 8
do
    gR2=`./stat.py best/out_gprog_0_${i}.txt | tail -n 1 | cut -d" " -f2`
    nnR2=`./stat.py best/out_nn0_${i}_50_8.txt | tail -n 1 | cut -d" " -f2`
    gr=`python -c "print($gR2<0.8);"`
    nn=`python -c "print($nnR2<0.8);"`
    echo $index $gR2 $nnR2 $gr $nn
    if [ $gr == "True" ]; then
	./gen_prog_run 0 $i 1 2
	mv best_0_${i}.txt ${di}/
	mv out_gprog_0_${i}.txt ${di}/
	mv res_pr_gprog1_minmax_0_${i}.txt ${di}/
    fi
    if [ $nn == "True" ]; then
        ./nn2.py 50 res_pr.txt 0 0 $i 8 mlp_torch0_${i}_50_8.pt `expr $n_mes + 1`
	mv mlp_torch0_${i}_50_8.pt ${di}/
	mv out_nn0_${i}_50_8.txt ${di}/
	mv out_nn_norms_x_0_${i}_50_8.txt ${di}/
	mv out_nn_norms_y_0_${i}_50_8.txt ${di}/
    fi
done
rm res_pr.txt
ln -s res_pr_C.txt res_pr.txt

# cleanup
rm res_pr_gprog*.txt
