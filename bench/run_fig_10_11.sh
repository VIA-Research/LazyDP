# Script for main experiment

system="cpu_gpu"
iterations=30
num_gathers=1
emb_scale=1
description="fig_10_11"
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

batch_size_list="
            1024
            2048
            4096
            "

training_mode_list="
                    sgd
                    lazydp
                    dpsgd_f
                    "

for training_mode in $training_mode_list
    do
    for batch_size in $batch_size_list
    do
        if [ $training_mode == "lazydp" ] ; then
            $numa_cmd python $pdb_cmd ../dlrm/dlrm_s_pytorch_lazydp.py $model_cmd --emb-scale=$emb_scale --num-batches=$iterations --mini-batch-size=$batch_size --use-gpu   --num-indices-per-lookup=$num_gathers --num-indices-per-lookup-fixed=True --dpsgd-mode=$training_mode --disable-poisson-sampling --system=$system --description=$description --path-lazydp=$PATH_LAZYDP --locality=$locality --path-model-weight=$PATH_MODEL_WEIGHT
        else
            $numa_cmd python $pdb_cmd ../dlrm/dlrm_s_pytorch.py $model_cmd --emb-scale=$emb_scale --num-batches=$iterations --mini-batch-size=$batch_size --use-gpu --num-indices-per-lookup=$num_gathers --num-indices-per-lookup-fixed=True --dpsgd-mode=$training_mode  --disable-poisson-sampling --system=$system --description=$description --path-lazydp=$PATH_LAZYDP --locality=$locality --path-model-weight=$PATH_MODEL_WEIGHT
        fi
    done
done