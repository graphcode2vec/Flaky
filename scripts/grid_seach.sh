#!/bin/bash -l
#SBATCH -n 5
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J jsc
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log


conda activate graph
gnn_type=gat
pretrainpath="pretrained_models/context/gat/model_0"
gp=attention
device=0
num_class=2
cd ../../
for lr in 0.001 0.005 0.0001 
do
for dropratio in 0.1 0.2 0.3 0.4 0.5
do
for warmup in yes
do
for batch_size in 128 256 512 64 
do
output_folder=results/mutants_class_${num_class}_lr_${lr}_drop_${dropratio}_warm_${warmup}_batch_${batch_size}/context
output_prefix=${output_folder}/
output=${output_prefix}/${gnn_type}
sw=lstm
jk=sum
lstm_emb_dim=150
mkdir -p $output
python mutants_classification.py --batch_size $batch_size --num_workers 5  --epochs 50 --num_layer 5 \
--subword_embedding  $sw \
--lstm_emb_dim $lstm_emb_dim \
--graph_pooling $gp \
--JK $jk \
--saved_model_path ${output} \
--log_file ${output}/log.txt \
--gnn_type $gnn_type \
--sub_token_path ./tokens/jars \
--emb_file emb_100.txt \
--dataset DV_PDG \
--input_model_file ${pretrainpath} \
--device ${device} \
--num_class ${num_class} \
--lr $lr \
--dropratio $dropratio \
--warmup_schedule yes \
--lazy yes \
--mutant_type yes \
--grid_search no
done
done
done
done