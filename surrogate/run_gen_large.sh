#!/bin/sh
out=$2
if [ -z "$out" ]
then
    out='0'
fi
# generate through surrogate
./gen_through_surrogate.sh $1 ${out}/
head -n -1 res_pr_generated_join.txt > res_gen.txt
mv res_gen.txt  ${out}/res_pr_generated_join.txt
