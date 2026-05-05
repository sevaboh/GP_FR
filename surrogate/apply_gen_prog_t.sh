#!/bin/bash
# the script makes forecast using f(t) GP surrogate for a parameters set $ds 
export LC_NUMERIC=C
# calculate value through f(t) GP surrogate
directory=gp_t_series4
input=log3d.txt
ds=4
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

    i=$ds
    # get data set
    echo $i
    ex_i=$i
    tail -n +$(( 122*($ex_i-1)+2 )) $input | head -n 121 | cut -d" "  -f12,20 > res_t_03_c.txt 
    # add time points to res_t_03_c.txt
    for x in $(seq 3.026 0.025 10);    
    do
	echo $x 0 >> res_t_03_c.txt
    done
    ./normalize_from_file.py res_t_03_c.txt res_t_03_c_n.txt ${directory}/log3a_minmax.txt
    # calculate constants
    cp ${directory}/best_p2_1.txt best_test.txt
    c=1
    for ii in $cols_list
    do
	cut -d" " -f 1,$inps_list,$ii ${directory}/gp_constants.txt | head -n $i | tail -n 1 > gp_constants1.txt
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
    ./gen_prog_test_v2 config_gprog2.txt res_t_03_c_n.txt 0 best_test.txt 1 | tail -n 401 | cut -d" " -f 3,5 > out_gprog_t_${i}.txt
    ./denormalize_gprog.py out_gprog_t_${i}.txt ${directory}/log3a_minmax.txt out_gprog_t1_${i}.txt 0
rm best.txt
rm best_test.txt
rm gp_constants1.txt
rm out_gprog_t_${i}.txt
rm res_t_03_c.txt
rm res_t_03_c_n.txt