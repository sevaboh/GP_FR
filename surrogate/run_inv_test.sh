#!/bin/bash

./convert_txt.sh res1.txt

t=$1
if [ -z "$t" ]
then
    t="0"
fi

if [ "$t" == "0" ]
then
    di='0_rest_t'
    # all up to t=1.5 0_rest_t
    cut -d"," -f1-114 res_pr_C.txt > res_pr_rest_C.txt
    cut -d"," -f1-114 res_pr_P.txt > res_pr_rest_P.txt
else
    di='0_rest'
    # x=0 0_rest 115-117,130-132,145-147,160-162,175-177,190-192
    cut -d"," -f1-9,10-12,25-27,40-42,55-57,70-72,85-87,100-102 res_pr_C.txt > res_pr_rest_C.txt
    cut -d"," -f1-9,10-12,25-27,40-42,55-57,70-72,85-87,100-102 res_pr_P.txt > res_pr_rest_P.txt
fi
rm res_pr.txt
ln -s res_pr_rest_C.txt res_pr.txt
#inverse for C
for i in 1 2 3 4 5 6
do
    ./gen_prog_run 0 $i
    ./nn2.py 50 res_pr.txt 0 0 $i 8
done
#inverse for P
rm res_pr.txt
ln -s res_pr_rest_P.txt res_pr.txt
for i in 0 7 8
do
    ./gen_prog_run 0 $i
    ./nn2.py 50 res_pr.txt 0 0 $i 8
done
rm res_pr.txt
ln -s res_pr_C.txt res_pr.txt

#move results
rm -rf ${di}
mkdir ${di}
mv best*.txt ${di}/
mv mlp_torch*.pt ${di}/
mv out_*.txt ${di}/
mv *minmax*.txt ${di}/

# cleanup
rm res_pr_gprog*.txt

# generate through surrogate - check inverse
./gen_through_surrogate.sh 100 ${di}/ inv 'best3/'
./stat.py res_pr_mixed.txt "," 0 > res_pr_mixed_r2.txt
cp res_pr_mixed.txt ${di}/
cp res_pr_mixed_r2.txt ${di}/
# generate through surrogate - check inverse - gp only
./gen_through_surrogate.sh 100 ${di}/ inv 'best3/' 1
mv res_pr_mixed.txt res_pr_mixed_gp.txt
./stat.py res_pr_mixed_gp.txt "," 0 > res_pr_mixed_gp_r2.txt
cp res_pr_mixed_gp.txt ${di}/
cp res_pr_mixed_gp_r2.txt ${di}/
