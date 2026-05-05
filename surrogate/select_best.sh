#!/bin/bash
best_dir=$1 

if [ -z "$best_dir" ]
then
    best_dir='best'
fi
# best/best2
rm -rf ${best_dir}
mkdir ${best_dir}

dirs=$2
if [ -z "$dirs" ]
then
    dirs="0/ 1/ 2/ 3/ 4/" # 5_big/" 
fi

for index in 0 1 2 3 4 5 6 7 8
do
    bestR=""
    bestN=""
    for d in $dirs
    do
    if [ -f ${d}out_gprog_0_${index}.txt ]
    then
	if [ "$bestR" == "" ]
	then
	    gR2=`./stat.py ${d}out_gprog_0_${index}.txt | tail -n 1 | cut -d" " -f2`
	    bestR=$d
	else
	    g2R2=`./stat.py ${d}out_gprog_0_${index}.txt | tail -n 1 | cut -d" " -f2`
	    gr=`python -c "print($g2R2>$gR2);"`
	    if [ $gr == "True" ]
	    then
		bestR=$d
		gR2=$g2R2
	    fi
	fi
	echo 0 GP $index $d $gR2
    fi
    if [ -f ${d}out_nn0_${index}_50_8.txt ]
    then
	if [ "$bestN" == "" ]
	then
	    nnR2=`./stat.py ${d}out_nn0_${index}_50_8.txt | tail -n 1 | cut -d" " -f2`
	    bestN=$d
	else
	    nn2R2=`./stat.py ${d}out_nn0_${index}_50_8.txt | tail -n 1 | cut -d" " -f2`
	    gr=`python -c "print($nn2R2>$nnR2);"`
	    if [ $gr == "True" ]
	    then
		bestN=$d
		nnR2=$nn2R2
	    fi
	fi
	echo 0 NN $index $d $nnR2
    fi
    done
    ln -s ../${bestR}best_0_${index}.txt ${best_dir}/best_0_${index}.txt
    ln -s ../${bestR}out_gprog_0_${index}.txt ${best_dir}/out_gprog_0_${index}.txt
    ln -s ../${bestR}res_pr_gprog1_minmax_0_${index}.txt ${best_dir}/res_pr_gprog1_minmax_0_${index}.txt
    ln -s ../${bestN}mlp_torch0_${index}_50_8.pt ${best_dir}/mlp_torch0_${index}_50_8.pt
    ln -s ../${bestN}out_nn0_${index}_50_8.txt ${best_dir}/out_nn0_${index}_50_8.txt
    ln -s ../${bestN}out_nn_norms_x_0_${index}_50_8.txt ${best_dir}/out_nn_norms_x_0_${index}_50_8.txt
    ln -s ../${bestN}out_nn_norms_y_0_${index}_50_8.txt ${best_dir}/out_nn_norms_y_0_${index}_50_8.txt
    echo $bestR $gR2 $bestN $nnR2
done

n_mes=`awk -F"," 'NR==1 { print NF }' res_pr.txt`
n_mes=`expr $n_mes - 12`
n_mes=`expr $n_mes / 3`
n_mes=`expr $n_mes + 1`
echo $n_mes
n_mes=`expr $n_mes - 1`
for index in $(seq 0 $n_mes)
do
    bestR=""
    bestN=""
    for d in $dirs
    do
    if [ -f ${d}out_gprog_1_${index}.txt ]
    then
	if [ "$bestR" == "" ]
	then
	    gR2=`./stat.py ${d}out_gprog_1_${index}.txt | tail -n 1 | cut -d" " -f2`
	    bestR=$d
	else
	    g2R2=`./stat.py ${d}out_gprog_1_${index}.txt | tail -n 1 | cut -d" " -f2`
	    gr=`python -c "print($g2R2>$gR2);"`
	    if [ $gr == "True" ]
	    then
		bestR=$d
		gR2=$g2R2
	    fi
	fi
	echo 1 GP $index $d $gR2
    fi
    if [ -f ${d}out_nn1_${index}_50_8.txt ]
    then
	if [ "$bestN" == "" ]
	then
	    nnR2=`./stat.py ${d}out_nn1_${index}_50_8.txt | tail -n 1 | cut -d" " -f2`
	    bestN=$d
	else
	    nn2R2=`./stat.py ${d}out_nn1_${index}_50_8.txt | tail -n 1 | cut -d" " -f2`
	    gr=`python -c "print($nn2R2>$nnR2);"`
	    if [ $gr == "True" ]
	    then
		bestN=$d
		nnR2=$nn2R2
	    fi
	fi
	echo 1 NN $index $d $nnR2
    fi
    done
    ln -s ../${bestR}best_1_${index}.txt ${best_dir}/best_1_${index}.txt
    ln -s ../${bestR}out_gprog_1_${index}.txt ${best_dir}/out_gprog_1_${index}.txt
    ln -s ../${bestR}res_pr_gprog1_minmax_1_${index}.txt ${best_dir}/res_pr_gprog1_minmax_1_${index}.txt
    ln -s ../${bestN}mlp_torch1_${index}_50_8.pt ${best_dir}/mlp_torch1_${index}_50_8.pt
    ln -s ../${bestN}out_nn1_${index}_50_8.txt ${best_dir}/out_nn1_${index}_50_8.txt
    ln -s ../${bestN}out_nn_norms_x_1_${index}_50_8.txt ${best_dir}/out_nn_norms_x_1_${index}_50_8.txt
    ln -s ../${bestN}out_nn_norms_y_1_${index}_50_8.txt ${best_dir}/out_nn_norms_y_1_${index}_50_8.txt
    echo $bestR $gR2 $bestN $nnR2
done

# generate through surrogate - check inverse
./gen_through_surrogate.sh 100 ${best_dir}/ inv
./stat.py res_pr_mixed.txt "," 0 > res_pr_mixed_r2.txt
cp res_pr_mixed.txt ${best_dir}/
cp res_pr_mixed_r2.txt ${best_dir}/
# generate through surrogate - check inverse - GP only
./gen_through_surrogate.sh 100 ${best_dir}/ inv ${best_dir}/ 1
./stat.py res_pr_mixed.txt "," 0 > res_pr_mixed_gp_r2.txt
mv res_pr_mixed.txt res_pr_mixed_gp.txt
cp res_pr_mixed_gp.txt ${best_dir}/
cp res_pr_mixed_gp_r2.txt ${best_dir}/
 