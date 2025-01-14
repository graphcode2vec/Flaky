#!/bin/bash -l

#SBATCH -n 3
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=normal
#SBATCH -J jsc
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wei.ma@uni.lu
#SBATCH -o %x-%j.log
#SBATCH -C volta32

conda activate graph
device=0
num_class=2
i=1
for d in 0.1 0.15 0.2 0.3
do
for b in 256 512 1024
do
for w in yes no
do
output_folder=results/flaky${num_class}_${w}_${d}_${b}/context
bash patch_prediction.sh gat "pretrained_models/context/gat/model_0" ${output_folder}/ attention $device $num_class $d $b $w
done
done
done


# output_folder=results/flaky${num_class}/context_${i}
# bash patch_prediction.sh gcn "pretrained_models/context/gcn/model_0" ${output_folder}/ attention $device $num_class
# bash patch_prediction.sh gin "pretrained_models/context/gin/model_0" ${output_folder}/ attention $device $num_class
# bash patch_prediction.sh graphsage "pretrained_models/context/graphsage/model_0" ${output_folder}/ attention $device $num_class

# output_folder=results/flaky${num_class}/node_${i}
# bash patch_prediction.sh gat "pretrained_models/node/gat/model_0" ${output_folder}/ attention $device $num_class
# bash patch_prediction.sh gcn "pretrained_models/node/gcn/model_0" ${output_folder}/ attention $device $num_class
# bash patch_prediction.sh gin "pretrained_models/node/gin/model_0" ${output_folder}/ attention $device $num_class
# bash patch_prediction.sh graphsage "pretrained_models/node/graphsage/model_0" ${output_folder}/ attention $device $num_class

# output_folder=results/flaky${num_class}/vgae_${i}
# bash patch_prediction.sh gat "pretrained_models/vgae/gat/model_1" ${output_folder}/ attention $device $num_class
# bash patch_prediction.sh gcn "pretrained_models/vgae/gcn/model_1" ${output_folder}/ attention $device $num_class
# bash patch_prediction.sh gin "pretrained_models/vgae/gin/model_1" ${output_folder}/ attention $device $num_class
# bash patch_prediction.sh graphsage "pretrained_models/vgae/graphsage/model_1" ${output_folder}/ attention $device $num_class
