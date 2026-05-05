#!/bin/bash
cmpr=0 # 1 - GP-GP or 0 - GP-best(GP,NN)
ds="inv" # t_series - t dataset, "inv" - inverse through best(GP,NN) 

./convert_txt.sh res1.txt

# generate values with per point GP surrogate for inputs for f(t) GP surrogate
./gen_through_surrogate.sh 100 0_rest_t/ $ds best3/ $cmpr

# extract for x=0.3 and convert from logs to absolute values 
# columns 15 30 45 60 75 90 105 120 135 150 165 180 195 - t=0.001 + 0.250
if [ $ds == "t_series" ]
then
    tail -n+2 res_pr_generated_join.txt | sed 's/,/ /g' | awk '{print($1,$2,$3,$4,$5,$6,$7,$8,$9,exp($15),exp($30),exp($45),exp($60),exp($75),exp($90),exp($105),exp($120),exp($135),exp($150),exp($165),exp($180),exp($195));}' > res_gp_point.txt
fi
if [ $ds == "inv" ]
then
    tail -n+2 res_pr_mixed.txt | sed 's/,/ /g' | awk '{print($2,$4,$6,$8,$10,$12,$14,$16,$18,exp($29),exp($59),exp($89),exp($119),exp($149),exp($179),exp($209),exp($239),exp($269),exp($299),exp($329),exp($359),exp($389),exp($30),exp($60),exp($90),exp($120),exp($150),exp($180),exp($210),exp($240),exp($270),exp($300),exp($330),exp($360),exp($390));}' > res_gp_point.txt
fi
# calculate value through f(t) GP surrogate
directory=gp_t_series4
input=log3d.txt
acc_param=1.0
n_cand=10

n=`grep u_inp $input | wc -l`
echo $input $n

# find varying inputs
n_inps=0
inps_list=""
for i in {2..10}
do
    v=$( awk "NR==1 { v=\$$i; var=0; } { if (\$$i != v) var=1;} END {printf \"%d\",var;}" ${directory}/gp_constants.txt )
    if [ $v -eq 1 ]
    then
	n_inps=$(( $n_inps + 1 ))
	if [ "$inps_list" == "" ]
	then
	    inps_list=$i
	else
	    inps_list=$inps_list,$i
	fi
    fi
done
echo inputs $n_inps $inps_list

cols_list=`cat ${directory}/gp_t_cols_list.txt`
max_list=`cat ${directory}/gp_t_max.txt`
min_list=`cat ${directory}/gp_t_min.txt`
echo cols list: $cols_list
echo maxs list: $max_list
echo mins list: $min_list

# run
rm res_gp_t.txt
if [ $ds == "inv" ]
then
    n=`wc -l res_gp_point.txt | cut -d " " -f1`
    echo $n
fi
for i in $(seq 1 $n)
do
    # get data set
    echo $i
    if [ $ds == "t_series" ]
    then
	ex_i=$i
    fi
    if [ $ds == "inv" ]
    then
	ex_i=1
    fi
    tail -n +$(( 122*($ex_i-1)+2 )) $input | head -n 121 | cut -d" "  -f12,20 > res_t_03_c.txt 
    ./normalize_from_file.py res_t_03_c.txt res_t_03_c_n.txt ${directory}/log3a_minmax.txt
    # calculate constants
    cp ${directory}/best_p2_1.txt best_test.txt
    c=1
    for ii in $cols_list
    do
	if [ $ds == "t_series" ]
	then
	    cut -d" " -f 1,$inps_list,$ii ${directory}/gp_constants.txt | head -n $i | tail -n 1 > gp_constants1.txt
	fi
	if [ $ds == "inv" ]
	then
	    echo -n "1 " > gp_constants1.txt
	    cut -d" " -f 1,2,3,4,5,6,7,8,9,10 res_gp_point.txt | head -n $i | tail -n 1 >> gp_constants1.txt
	fi
	v=$( ./gen_prog_test_v2 config_gprog.txt gp_constants1.txt $n_inps ${directory}/best_c${ii}.txt 1 | tail -n 1 | cut -d" " -f 5 )
	# denormalize
	min=`echo $min_list | cut -d " " -f $c`
	max=`echo $max_list | cut -d " " -f $c`
	v="`python -c "print(($v) * (($max)-($min))+($min));"`"
	# insert into the tree
	awk "{ if (2*(NR-1)+11 == $ii) \$2=$v; if (2*(NR-1)+12 == $ii) \$3=$v; printf \"%d %g %g\n\",\$1,\$2,\$3}" best_test.txt > best_test2.txt
	rm best_test.txt 
	mv best_test2.txt best_test.txt
	c=$(( $c + 1 ))
    done
    ./gen_prog_test_v2 config_gprog2.txt res_t_03_c_n.txt 0 best_test.txt 1 | tail -n 121 | cut -d" " -f 1,3,5 | grep "^0 \\|^0\\.0833333 \\|^0\\.166667 \\|^0\\.25 \\|^0\\.333333 \\|^0\\.416667 \\|^0\\.5 \\|^0\\.583333 \\|^0\\.666667 \\|^0\\.75 \\|^0\\.833333 \\|^0\\.916667 \\|^1 " > out_gprog_t_${i}.txt
    ./denormalize_gprog.py out_gprog_t_${i}.txt ${directory}/log3a_minmax.txt out_gprog_t1_${i}.txt 0
    if [ $ds == "t_series" ]
    then
        s1=`cut -d" " -f2,3 out_gprog_t1_${i}.txt | tr '\n' ' '`
        s2=`cut -d" " -f 2-10 ${directory}/gp_constants.txt| head -n $i | tail -n 1`
        echo $s2 $s1>> res_gp_t.txt
    fi
    if [ $ds == "inv" ]
    then
        s1=`cut -d" " -f3 out_gprog_t1_${i}.txt | tr '\n' ' '`
        echo $s1>> res_gp_t.txt
    fi
    rm out_gprog_t_${i}.txt out_gprog_t1_${i}.txt
done

# merge two files and rearrange - params, {calculated}+, {GP f(t)}+, {GP point}+
if [ $ds == "t_series" ]
then

awk '
BEGIN { FS=OFS=" " }

FNR==NR {
    key=$1 FS $2 FS $3 FS $4 FS $5 FS $6 FS $7 FS $8 FS $9
    a[key]=$0
    next
}

{
    key=$1 FS $2 FS $3 FS $4 FS $5 FS $6 FS $7 FS $8 FS $9
    if (key in a) {
        printf "%s", a[key]
        for(i=10;i<=NF;i++)
            printf "%s%s", OFS, $i
        printf "\n"
    }
}' res_gp_point.txt res_gp_t.txt | awk '{print($1,$2,$3,$4,$5,$6,$7,$8,$9,$23,$25,$27,$29,$31,$33,$35,$37,$39,$41,$43,$45,$47,$24,$26,$28,$30,$32,$34,$36,$38,$40,$42,$44,$46,$48,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22);}' > res_gp_compare_${ds}_${cmpr}.txt

fi
if [ $ds == "inv" ]
then
    cut -d " " -f 1-22 res_gp_point.txt  > res_gp1.txt
    cut -d " " -f 23- res_gp_point.txt > res_gp2.txt
    paste res_gp1.txt res_gp_t.txt res_gp2.txt | tr '\t' ' ' > res_gp_compare_${ds}_${cmpr}.txt
    rm res_gp1.txt res_gp2.txt
fi

# clean up
rm best.txt
rm best_test.txt
rm gp_constants1.txt
rm res_t_03_c.txt
rm res_t_03_c_n.txt
rm res_pr_generated_join.txt
rm res_pr.txt
rm res_pr_C.txt
rm res_pr_P.txt
rm res_gp_point.txt res_gp_t.txt
