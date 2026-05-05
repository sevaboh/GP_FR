#!/bin/bash
rm res_pr.txt
ln -s best2/res_pr_generated_join.txt res_pr.txt

n_mes=`awk -F"," 'NR==1 { print NF }' res_pr.txt`
n_mes=`expr $n_mes - 12`
n_mes=`expr $n_mes / 3`
n_mes=`expr $n_mes + 1`
echo $n_mes
n_mes=`expr $n_mes - 1`

#direct
for i in $(seq 0 $n_mes)
do
    gR2=`./stat.py best2/out_gprog_1_${i}.txt | tail -n 1 | cut -d" " -f2`
    gr=`python -c "print($gR2<0.8);"`
    echo $index $gR2 $gr
    if [ $gr == "True" ]; then
	./gen_prog_run 1 $i
    fi
done
#move results
rm -rf 1_gen
mkdir 1_gen
mv best*.txt 1_gen/
mv out_*.txt 1_gen/
mv *minmax*.txt 1_gen/
rm 1_gen/out_nn*.txt

# cleanup
rm res_pr_gprog*.txt

rm res_pr.txt
ln -s res_pr_C.txt res_pr.txt

# test on real
for i in $(seq 0 $n_mes)
do
if [ -f "1_gen/best_1_${i}.txt" ]
then
     echo $i
    ./gen_prog_run 1 $i 1 2 '1_gen/'
fi
done
rm 1_gen/out_gprog*.txt
mv out_gprog*.txt 1_gen/

# cleanup
rm res_pr_gprog*.txt

