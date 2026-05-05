#!/bin/bash
directory=$2
directory2=$2
if [ "$4" != "" ]
then
   directory2=$4
fi
gp_only=$5
if [ -z "$gp_only" ]
then
    gp_only=0
fi
echo $directory $directory2
# generate model parameters
rm res_pr_generated_gprog.txt
inv=0
if [ "$3" == "t_series" ]
then
    cat gp_t_series4/gp_constants.txt | awk '{print (NR,$2,$3,$4,$5,$6,$7,$8,$9,$10,NR)}' > res_pr_generated_gprog.txt
else
if [ "$3" == "this" ]
then
    tail -n +2 res_pr.txt | sed 's/,/ /g' | awk '{print (NR,$1,$2,$3,$4,$5,$6,$7,$8,$9,NR)}' > res_pr_generated_gprog.txt
    inv=1
else
if [ "$3" == "inv" ]
then
    echo "get from inverted"
    # combine generated values into one file
    s=""
    for index in 0 1 2 3 4 5 6 7 8
    do
	# get R2 on testing dataset
	gR2=`./stat.py ${directory}out_gprog_0_${index}.txt | tail -n 1 | cut -d" " -f2`
	nnR2=`./stat.py ${directory}out_nn0_${index}_50_8.txt | tail -n 1 | cut -d" " -f2`
	gr=`python -c "print($gR2>$nnR2);"`
	if [ "$gp_only" == "1" ]
	then
	    gr="True"
	fi
        echo $index $gR2 $nnR2 $gr
	if [ $gr == "True" ]; then
	    # gprog
	    s=`echo $s "${directory}out_gprog_0_${index}.txt"`
	else
	    # NN
	    s=`echo $s "${directory}out_nn0_${index}_50_8.txt"`
	fi
    done
    paste $s | sed 's/\t/ /g' | awk '{print (NR,$1,$3,$5,$7,$9,$11,$13,$15,$17,NR)}'  >> res_pr_generated_gprog.txt
    inv=1
else
    echo "generate random"
    m=`echo '*'`
    i=0
    for y in $(seq 1 $1)
    do
        a=`echo "scale=6; 0.5 + 0.5 ${m} ( \`expr ${RANDOM} % 1000\` ) / 1000.0" |bc | sed 's/^\./0./'`
        b=`echo "scale=6; 1.5 + 0.5 ${m} ( \`expr ${RANDOM} % 1000\`)/1000.0;"|bc | sed 's/^\./0./'`
        tau_r=`echo "scale=6; 0.0 + 0.5 ${m} ( \`expr ${RANDOM} % 1000\`)/1000.0;"|bc | sed 's/^\./0./'`
        ae=`echo "scale=6; 0.01 + 0.199 ${m} ( \`expr ${RANDOM} % 1000\`)/1000.0;"|bc | sed 's/^\./0./'`
        u_inp=`echo "scale=6; 0.05 + 0.95 ${m} ( \`expr ${RANDOM} % 1000\`)/1000.0;"|bc | sed 's/^\./0./'`
        c_inp_dur=`echo "scale=6; 0.05 + 0.95 ${m} ( \`expr ${RANDOM} % 1000\`)/1000.0;"|bc | sed 's/^\./0./'`
        c_inp=`echo "scale=6; 0.05 + 0.95 ${m} ( \`expr ${RANDOM} % 1000\`)/1000.0;"|bc | sed 's/^\./0./'`
        D=`echo "scale=6; 0.02 + 0.04 ${m} ( \`expr ${RANDOM} % 1000\`)/1000.0;"|bc | sed 's/^\./0./'`
        kmu=`echo "scale=6; 0.1 + 0.4 ${m} ( \`expr ${RANDOM} % 1000\`)/1000.0;"|bc | sed 's/^\./0./'`
        echo $i $u_inp $c_inp_dur $c_inp $D $kmu $a $b $ae $tau_r $i >> res_pr_generated_gprog.txt
	i=$(( $i + 1 ))
    done
fi
fi
fi

echo "x1,x2,x3,x4,x5,x6,x7,x8,x9">res_pr_generated_for_NN.txt
cat "res_pr_generated_gprog.txt" | sed 's/\t/ /g' | head -n -1 | awk '{print ($2,$3,$4,$5,$6,$7,$8,$9,$10)}' | sed 's/ /,/g' >> res_pr_generated_for_NN.txt

#process generated set through gprog or NN surrogate (who is better on testing dataset) 
n_mes=`awk -F"," 'NR==1 { print NF }' res_pr.txt`
n_mes=`expr $n_mes - 12`
n_mes=`expr $n_mes / 3`
n_mes=`expr $n_mes + 1`
echo $n_mes
n_mes=`expr $n_mes - 1`
for index in $(seq 0 $n_mes)
do
# get R2 on testing dataset
    gR2=`./stat.py ${directory2}out_gprog_1_${index}.txt | tail -n 1 | cut -d" " -f2`
    nnR2=`./stat.py ${directory2}out_nn1_${index}_50_8.txt | tail -n 1 | cut -d" " -f2`
    gr=`python -c "print($gR2>$nnR2);"`
    if [ "$gp_only" == "1" ]
    then
        gr="True"
    fi
    echo $index $gR2 $nnR2 $gr
if [ $gr == "True" ]; then
# gprog
    # normalize inputs
    ./normalize_from_file.py res_pr_generated_gprog.txt res_pr_generated_gprog_n.txt ${directory2}res_pr_gprog1_minmax_1_${index}.txt
    # calculate
    ./gen_prog_test config_gprog.txt res_pr_generated_gprog_n.txt 9 ${directory2}best_1_${index}.txt 1 > out_gprog.txt
    # get results
    awk "NR<30 {}; NR>=30 {v1=\$5; v2=\$3; print (v1,v2);}" out_gprog.txt > out_gprog1.txt
    ./denormalize_gprog.py out_gprog1.txt ${directory2}res_pr_gprog1_minmax_1_${index}.txt res_pr_generated_gprog${index}.txt 9
else
# NN
    ./nn2.py 100 res_pr_generated_for_NN.txt 1 0 $index 8 ${directory2}mlp_torch1_${index}_50_8.pt `expr $n_mes + 1`
    mv out_nn1_${index}_100_8.txt res_pr_generated_gprog${index}.txt
    rm out_nn_norms_x_1_${index}_100_8.txt
    rm out_nn_norms_y_1_${index}_100_8.txt
fi
done

# combine generated values into one file
rm res_pr_generated_join.txt
s="res_pr_generated_gprog.txt"
for index in $(seq 0 $n_mes)
do
    s=`echo $s "res_pr_generated_gprog${index}.txt"`
done
paste $s | sed 's/\t/ /g' | awk '{printf "%s %s %s %s %s %s %s %s %s", $2,$3,$4,$5,$6,$7,$8,$9,$10;
n= NF; for (i=12;i<=NF;i+=2) {printf  " %d %d %s",i,i,$i }; printf "%s",ORS; }' | sed 's/ /,/g'  >> res_pr_generated_join.txt

awk -F',' '
NR==2 {
    n = NF
    for (i=1; i<=n; i++) {
        printf "x%d%s", i, (i<n ? "," : "\n")
    }
}
NR>=2 { print }
' res_pr_generated_join.txt > tmp.csv && mv tmp.csv res_pr_generated_join.txt

if [ $inv -eq 0 ]
then
    grep -v "nan" res_pr_generated_join.txt > tmp.txt
    rm res_pr_generated_join.txt
    mv tmp.txt res_pr_generated_join.txt
else
    ./mixfiles.sh res_pr.txt res_pr_generated_join.txt > res_pr_mixed.txt
    rm res_pr_generated_join.txt
fi

#cleanup
rm res_pr_generated_gprog.txt
rm res_pr_generated_for_NN.txt
rm res_pr_generated_gprog_n.txt
for index in $(seq 0 $n_mes)
do
    rm res_pr_generated_gprog${index}.txt
done
rm best.txt
rm out_gprog.txt
rm out_gprog1.txt


