device=0
data=./data_retweet/fold1/
batch=8
n_head=16
n_layers=3
d_model=64
d_rnn=64
d_inner=256
d_k=16
d_v=16
dropout=0.1
lr=5e-3
smooth=0.1
epoch=80
n_taylor_terms=5
ExpHyper=NumOfTaylorTerms
data_name=Retweet
n_samples=100
ExpFolder=ExpMultipleRuns
min_lr=1e-6
num_workers=10

i=1

echo "Start"

tb_log=/tb_retweet
log=/log_retweet_t_5_f_1_s_${i}.txt
final_result_log=/final_result_retweet_t_5_f_1_s_${i}.txt

dir=./
root=../../../

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python ../../../Main_v2.py -dir $dir -root $root -tb_log $tb_log -num_workers $num_workers -min_lr $min_lr -final_result_log $final_result_log -ExpFolder $ExpFolder -n_taylor_terms $n_taylor_terms -n_samples $n_samples -data $data -seed $i -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log -data_name $data_name -ExpHyper $ExpHyper