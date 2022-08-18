#!/bin/bash

device=0
data=./data_meme/fold1/
batch=512
n_head=3
n_layers=3
d_model=16
d_rnn=32
d_inner=32
d_k=16
d_v=16
dropout=0.1
lr=1e-3
smooth=0.1
n_taylor_terms=5
epoch=50
ExpHyper=NumOfTaylorTerms
data_name=Meme
n_samples=100
ExpFolder=ExpMultipleRuns
min_lr=1e-6
num_workers=0

i=4

echo "Start"

tb_log=/tb_meme
log=/log_meme_t_5_f_1_s_${i}.txt
final_result_log=/final_result_meme_5_f_1_s_${i}.txt

dir=./
root=../../../



CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python ../../../Main_v2.py -dir $dir -root $root -tb_log $tb_log -num_workers $num_workers -min_lr $min_lr -final_result_log $final_result_log -ExpFolder $ExpFolder -n_taylor_terms $n_taylor_terms -n_samples $n_samples -data $data -seed $i -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log -data_name $data_name -ExpHyper $ExpHyper
