#!/bin/sh
out=$1
if [ -z "$out" ]
then
    out='0'
fi

./convert_txt.sh res0.txt

# process real dataset
n_mes=`awk -F"," 'NR==1 { print NF }' res_pr.txt`
n_mes=`expr $n_mes - 12`
n_mes=`expr $n_mes / 3`
n_mes=`expr $n_mes + 1`
echo $n_mes
n_mes=`expr $n_mes - 1`
#direct
for i in $(seq 0 $n_mes)
do
    ./gen_prog_run 1 $i
    ./nn2.py 50 res_pr.txt 1 0 $i 8
done
#inverse for C
for i in 1 2 3 4 5 6
do
    ./gen_prog_run 0 $i
    ./nn2.py 50 res_pr.txt 0 0 $i 8
done
#inverse for P
rm res_pr.txt
ln -s res_pr_P.txt res_pr.txt
for i in 0 7 8
do
    ./gen_prog_run 0 $i
    ./nn2.py 50 res_pr.txt 0 0 $i 8
done
rm res_pr.txt
ln -s res_pr_C.txt res_pr.txt

#move results
rm -rf ${out}
mkdir ${out}
mv best*.txt ${out}/
mv mlp_torch*.pt ${out}/
mv out_*.txt ${out}/
mv *minmax*.txt ${out}/

# cleanup
rm res_pr_gprog*.txt

# generate through surrogate - check inverse
./gen_through_surrogate.sh 100 ${out}/ inv
./stat.py res_pr_mixed.txt "," 0 > res_pr_mixed_r2.txt
cp res_pr_mixed.txt ${out}/
cp res_pr_mixed_r2.txt ${out}/
