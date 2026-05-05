#!/bin/bash
M=99
tau=0.001
n=3001
n_out=25
gr='x 0.3 '

RANDOM=$$
decay_rate=0.1
u_inp_a=0
u_inp_b=1.57079632679
for y in {1..1000}
do
a=`python -c "print(0.5+0.5*($RANDOM%1000)/1000.0);"`
b=`python -c "print(1.5+0.5*($RANDOM%1000)/1000.0);"`
tau_r=`python -c "print(0.0+0.5*($RANDOM%1000)/1000.0);"`
ae=`python -c "print(0.01+0.199*($RANDOM%1000)/1000.0);"`
u_inp=`python -c "print(0.05+0.95*($RANDOM%1000)/1000.0);"`
c_inp_dur=`python -c "print(0.05+0.95*($RANDOM%1000)/1000.0);"`
c_inp=`python -c "print(0.05+0.95*($RANDOM%1000)/1000.0);"`
D=`python -c "print(0.02+0.04*($RANDOM%1000)/1000.0);"`
kmu=`python -c "print(0.1+0.4*($RANDOM%1000)/1000.0);"`
./fractal-fract-PC $a $b $tau $n $n_out $tau_r $ae $M 0 1 0 $u_inp $c_inp_dur $c_inp $D $kmu $decay_rate $u_inp_a $u_inp_b > log.txt
head -n 1 log.txt
grep "$gr" log.txt
done

