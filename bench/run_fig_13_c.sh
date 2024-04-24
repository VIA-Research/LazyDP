# Script for sensitivity test: alternative DLRM model configurations

system="cpu_gpu"
iterations=30
batch_size=2048
emb_scale=1
num_gathers=1
description="fig_13_c"
numactl_use=1
locality="uniform"
numa_cmd="numactl --cpunodebind=0 --membind=0"

result_path="$PATH_LAZYDP/result"
if [ -e "$result_path/merged_result/${description}.csv" ]; then
    rm "$result_path/merged_result/${description}.csv"
fi

model_configs_list="
                rmc1
                rmc2
                rmc3
                "

training_mode_list="
                    sgd
                    lazydp
                    dpsgd_f
                    "

for model_config in $model_configs_list
do
    for training_mode in $training_mode_list
    do
        echo $model_config
        if [ $model_config == "mlperf" ] ; then
            arch_emb_size="39884406-39043-17289-7420-20263-3-7120-1543-63-38532951-2953546-403346-10-2208-11938-155-4-976-14-39979771-25641295-39664984-585935-12972-108-36"
            arch_mlp_bot="13-512-256-128"
            arch_mlp_top="1024-1024-512-256-1"
            arch_sparse_feature_size=128

            model_cmd="--model-config=$model_config --arch-sparse-feature-size=$arch_sparse_feature_size --arch-embedding-size=$arch_emb_size --arch-mlp-bot=$arch_mlp_bot --arch-mlp-top=$arch_mlp_top"
        elif [ $model_config == "rmc1" ] ; then
            arch_mlp_bot="13-256-128-32"
            arch_mlp_top="256-64-1"
            arch_sparse_feature_size=32
            num_gathers=80
            n_table=10

            model_cmd="--model-config=$model_config --arch-sparse-feature-size=$arch_sparse_feature_size --arch-mlp-bot=$arch_mlp_bot --arch-mlp-top=$arch_mlp_top --n-table=$n_table"
        elif [ $model_config == "rmc2" ] ; then
            arch_mlp_bot="13-256-128-32"
            arch_mlp_top="512-128-1"
            arch_sparse_feature_size=32
            num_gathers=80
            n_table=40

            model_cmd="--model-config=$model_config --arch-sparse-feature-size=$arch_sparse_feature_size --arch-mlp-bot=$arch_mlp_bot --arch-mlp-top=$arch_mlp_top --n-table=$n_table"
        elif [ $model_config == "rmc3" ] ; then
            arch_mlp_bot="13-2560-512-32"
            arch_mlp_top="512-128-1"
            arch_sparse_feature_size=32
            num_gathers=20
            n_table=10

            model_cmd="--model-config=$model_config --arch-sparse-feature-size=$arch_sparse_feature_size --arch-mlp-bot=$arch_mlp_bot --arch-mlp-top=$arch_mlp_top --n-table=$n_table"
        else # basic model configuration
            arch_mlp_bot="13-512-256-64"
            arch_mlp_top="512-512-256-1"
            arch_sparse_feature_size=64
            
            model_cmd="--model-config=$model_config --arch-sparse-feature-size=$arch_sparse_feature_size --arch-mlp-bot=$arch_mlp_bot --arch-mlp-top=$arch_mlp_top --n-table=$n_table"
        fi

        if [ $training_mode == "lazydp" ] ; then
            $numa_cmd python $pdb_cmd ../dlrm/dlrm_s_pytorch_lazydp.py $model_cmd --emb-scale=$emb_scale --num-batches=$iterations --mini-batch-size=$batch_size --use-gpu   --num-indices-per-lookup=$num_gathers --num-indices-per-lookup-fixed=True --dpsgd-mode=$training_mode --disable-poisson-sampling --system=$system --description=$description --path-lazydp=$PATH_LAZYDP --locality=$locality --path-model-weight=$PATH_MODEL_WEIGHT
        else
            $numa_cmd python $pdb_cmd ../dlrm/dlrm_s_pytorch.py $model_cmd --emb-scale=$emb_scale --num-batches=$iterations --mini-batch-size=$batch_size --use-gpu --num-indices-per-lookup=$num_gathers --num-indices-per-lookup-fixed=True --dpsgd-mode=$training_mode  --disable-poisson-sampling --system=$system --description=$description --path-lazydp=$PATH_LAZYDP --locality=$locality --path-model-weight=$PATH_MODEL_WEIGHT
        fi
    done
done
