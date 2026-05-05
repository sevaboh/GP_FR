#!/bin/sh
nn=`wc -l $2 | cut -d" " -f1`
nn=`expr $nn - 1`
head -n 1 $1 > tmp1.txt
tail -n $nn $1 > tmp2.txt
cat tmp1.txt tmp2.txt > tmp3.txt
awk -F',' -v OFS=',' '
NR==FNR {
    for (i=1; i<=NF; i++) a[FNR,i] = $i
    na[FNR] = NF
    next
}
{
    row = FNR
    max = (na[row] < NF ? na[row] : NF)

    for (i=1; i<=max; i++)
        printf "%s%s%s%s", a[row,i], OFS, $i, (i<max?OFS:ORS)
}' tmp3.txt $2
rm tmp1.txt tmp2.txt tmp3.txt