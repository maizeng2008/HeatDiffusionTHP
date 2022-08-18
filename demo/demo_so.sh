#!/bin/bash
device=0
data=./data_so/fold3/
batch=1
n_head=16
n_layers=4
d_model=64
d_rnn=32
d_inner=128
d_k=32
d_v=32
dropout=0.1
lr=1e-3
smooth=0.1
n_taylor_terms=5
epoch=1000
ExpHyper=NumOfTaylorTerms
data_name=StackOverflow
n_samples=100
ExpFolder=ExpMultipleRuns
num_workers=10
min_lr=1e-6

i=1

echo "Start"

tb_log=/tb_so
log=/log_so_t_5_f_3.txt
final_result_log=/final_result_so_t_5_f_3.txt

dir=./
root=../../../


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python ../../../Main_v2.py -dir $dir -root $root -tb_log $tb_log -num_workers $num_workers -min_lr $min_lr -final_result_log $final_result_log -ExpFolder $ExpFolder -n_taylor_terms $n_taylor_terms -n_samples $n_samples -data $data -seed $i -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log -data_name $data_name -ExpHyper $ExpHyper