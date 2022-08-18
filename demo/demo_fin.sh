#!/bin/bash
device=0
data=./data_bookorder/fold1/
batch=1
n_head=6
n_layers=3
d_model=16
d_rnn=32
d_inner=64
d_k=16
d_v=16
dropout=0.1
lr=1e-3
smooth=0.1
n_taylor_terms=3
epoch=1000
ExpHyper=NumOfTaylorTerms
data_name=Financial
n_samples=100
ExpFolder=ExpMultipleRuns
num_workers=10
min_lr=1e-7


i=3

echo "Start"

tb_log=/tb_fin
log=/log_fin_t_3_f_2.txt
final_result_log=/final_result_fin_t_3_f_2.txt

dir=./
root=../../../


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python ../../../Main_v2.py -dir $dir -root $root -tb_log $tb_log -num_workers $num_workers -min_lr $min_lr -final_result_log $final_result_log -ExpFolder $ExpFolder -n_taylor_terms $n_taylor_terms -n_samples $n_samples -data $data -seed $i -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log -data_name $data_name -ExpHyper $ExpHyper