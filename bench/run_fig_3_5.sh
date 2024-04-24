# Script for motivational experiment

system="cpu_gpu"
iterations=30
num_gathers=1
batch_size=2048
description="fig_3_5"
numactl_use=1
locality="uniform"
numa_cmd="numactl --cpunodebind=0 --membind=0"

result_path="$PATH_LAZYDP/result"
if [ -e "$result_path/merged_result/${description}.csv" ]; then
    rm "$result_path/merged_result/${description}.csv"
fi

# MLPerf DLRM training configuration
model_config="mlperf"
arch_emb_size="39884406-39043-17289-7420-20263-3-7120-1543-63-38532951-2953546-403346-10-2208-11938-155-4-976-14-39979771-25641295-39664984-585935-12972-108-36"
arch_mlp_bot="13-512-256-128"
arch_mlp_top="1024-1024-512-256-1"
arch_sparse_feature_size=128
model_cmd="--model-config=$model_config --arch-sparse-feature-size=$arch_sparse_feature_size --arch-embedding-size=$arch_emb_size --arch-mlp-bot=$arch_mlp_bot --arch-mlp-top=$arch_mlp_top"

emb_scale_list="
            0.001
            0.01
            0.1
            1
            "

training_mode_list="
                    sgd
                    dpsgd_b
                    dpsgd_r
                    dpsgd_f
                    "

for emb_scale in $emb_scale_list
do
    for training_mode in $training_mode_list
    do
        $numa_cmd python $pdb_cmd ../dlrm/dlrm_s_pytorch.py $model_cmd --emb-scale=$emb_scale --num-batches=$iterations --mini-batch-size=$batch_size --use-gpu --num-indices-per-lookup=$num_gathers --num-indices-per-lookup-fixed=True --dpsgd-mode=$training_mode  --disable-poisson-sampling --system=$system --description=$description --path-lazydp=$PATH_LAZYDP --locality=$locality --path-model-weight=$PATH_MODEL_WEIGHT
    done
done
