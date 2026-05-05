#!/bin/bash
M=99
tau=0.001
n=3001
n_out=1000

a=0.75
b=1.75
ae=0.1
u_inp=0.5
c_inp_dur=1.0
c_inp=0.5
D=0.03
kmu=0.25
tau_r=0.0
decay_rate=0
u_inp_a=0
u_inp_b=1.57079632679
opencl=0

# 1
#kmu=0.1
#./fractal-fract-PC $a $b $tau $n $n_out $tau_r $ae $M 0 1 0 $u_inp $c_inp_dur $c_inp $D $kmu > log.txt
#kmu=0.25
#./fractal-fract-PC $a $b $tau $n $n_out $tau_r $ae $M 0 1 0 $u_inp $c_inp_dur $c_inp $D $kmu >> log.txt
#kmu=0.4
#./fractal-fract-PC $a $b $tau $n $n_out $tau_r $ae $M 0 1 0 $u_inp $c_inp_dur $c_inp $D $kmu >> log.txt
#exit
# 2
decay_rate=0
./fractal-fract-PC $a $b $tau $n $n_out $tau_r $ae $M 0 1 $opencl $u_inp $c_inp_dur $c_inp $D $kmu $decay_rate $u_inp_a $u_inp_b > log_${opencl}.txt
exit
decay_rate=0.1
./fractal-fract-PC $a $b $tau $n $n_out $tau_r $ae $M 0 1 $opencl $u_inp $c_inp_dur $c_inp $D $kmu $decay_rate $u_inp_a $u_inp_b  >> log_${opencl}.txt
decay_rate=0.2
./fractal-fract-PC $a $b $tau $n $n_out $tau_r $ae $M 0 1 $opencl $u_inp $c_inp_dur $c_inp $D $kmu $decay_rate $u_inp_a $u_inp_b  >> log_${opencl}.txt

