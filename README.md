# LazyDP
LazyDP is an algorithm-software co-design that addresses the compute and memory challenges of training RecSys with DP-SGD.

For more details about this work, please refer to our paper published in [ASPLOS-2024](https://www.asplos-conference.org/asplos2024/main-program/).
- "[LazyDP: Co-Designing Algorithm-Software for Scalable Training of Differentially Private Recommendation Models](https://arxiv.org/abs/2404.08847)" 

## Setup
To save the time for experiments, it is better to store the initial model weights and reuse them. Because the maximum model size is 192 GB, you should choose a path where hundreds of GBs data can be stored (e.g., `/raid/model_weight/`).
```bash
mkdir {absolute_path_to_model_weight_directory}
export PATH_MODEL_WEIGHT={absolute_path_to_model_weight_directory}
```

Then, create docker container from official docker image.
```bash
docker run -ti --gpus all --shm-size 5g --name lazydp -v $PATH_MODEL_WEIGHT:/model_weight -e "PATH_LAZYDP=/workspace/LazyDP" -e "PATH_MODEL_WEIGHT=/model_weight" --cap-add SYS_NICE nvcr.io/nvidia/pytorch:23.09-py3
```

In the docker container, clone this repository.
```bash
cd /workspace
git clone https://gitlab.com/via-research/LazyDP.git
```

Run `setup.sh` in cloned directory to extend PyTorch and install required packages.
```bash
cd $PATH_LAZYDP
./setup.sh
```

## How to reproduce the results of paper

This repository provides several shell scripts to reproduce the results in LazyDP paper. The number may be slightly different because this repository utilizes the more recent version of libraries (e.g., PyTorch and other python packages)

In the `$PATH_LAZYDP/bench` directory, there are several shell scripts to run the experiments in LazyDP paper.

Files started with `run_fig_{XXX}` are used to generate experimental data that is utilized to draw corresponding figures in the paper. (`{XXX}` means the figure number in paper)

Simply, run the codes as bellow:
```bash
cd $PATH_LAZYDP/bench
./run_fig_{XXX}.sh
```

Then, the result csv/txt files will be generated in `$PATH_LAZYDP/result`.

- (MAIN RESULT) In `$PATH_LAZYDP/result/merged_result` directory, `fig_{XXX}.csv` file is generated and this file is the main result to draw graphs.
    - Each column represents the result for a certain model/training configuration.
    - First four rows represent the latency breakdown for 3 parts of model training.
        1. Forward propagation
        2. First backpropagation
        3. Second backpropagation
        4. Model update
    - Next six rows represent the detailed latency breakdown of model update stage.
    - Last row is to measure the time for executing certain part of code you want to examine. (If not required, just ignore)
- In `$PATH_LAZYDP/result/log` directory, `{AAA}_{BBB}_s_{CCC}_B_{DDD}_L_{EEE}_{FFF}.txt` file is generated and this file is the log file which records the execution time and shows the training progress.
    - `{AAA}`: model type
        1. basic: model whose tables have same size
        2. mlperf: model follows the configuration in MLPerf (v2.1)
        3. rmc1, rmc2, rmc3: models follow the configuration in [DeepRecSys](https://arxiv.org/abs/2001.02772).
    - `{BBB}`: locality of embedding access
        1. zipf:
        2. kaggle:
        3. uniform:
    - `{CCC}`: scaling factor of table size
        - When the value is 1, then the total size of tables is 96 GB.
        - When the value is $x (\not=1)$, then the tables are scaled by $x$.
    - `{DDD}`: batch size
    - `{EEE}`: pooling factor
    - `{FFF}`: training type
        1. sgd: standard SGD training
        2. [dpsgd_b](https://arxiv.org/abs/1607.00133): basic DP-SGD training , referred to as DP-SGD(B)
        3. [dpsgd_r](https://arxiv.org/abs/2009.03106): reweighted DP-SGD training, referred to as DP-SGD(R)
        4. [dpsgd_f](https://arxiv.org/abs/2211.11896): fast-DP-SGD training (same with ghost clipping), referred to as DP-SGD(F). This training is used as our baseline method.
        5. lazydp: our proposal

- In `$PATH_LAZYDP/result/detailed_latency_breakdown` directory, there are csv files and each file records more detailed latency breakdown result for single training.
    - Comments in `$PATH_LAZYDP/custom_utils.py` describe the breakdown of this result file.

## Miscellaneous: correctness tests
To verify that the correctness of implementations for DP-SGD(B,R,F) and LazyDP training, run script files whose prefixes are `run_correctness_test_`:
```bash
cd $PATH_LAZYDP/bench
./run_correctness_test_{NNN}.sh
```
1. `run_correctness_test_1.sh`
    - Compare the results of DP-SGD(B), DP-SGD(R) and DP-SGD(F) with skipping the noise addition.
    - Only check whether the clipping procedure works well.
2. `run_correctness_test_2.sh`
    - Compare the results of DP-SGD(F) and LazyDP with using 1 instead of single noise.
    - Check whether the delayed noise works well.
3. `run_correctness_test_3.sh`
    - Compare the results of SGD and DP-SGD(B) with skipping both the noise addition and gradient clipping.
    - Because we modified the Opacus implementation, we verified that there is no mistake in our modification through this test.

The results of the above correctness tests will be displayed in the terminal.

## Miscellaneous: code structure for using LazyDP
This section explains the overview of the usage of LazyDP.

Firstly, generate DP-SGD-enabled model, optimizer and data_loader with below code. This is similar to use of [Opacus](https://opacus.ai/#quickstart).

```python
dlrm, optimizer, train_ld = privacy_engine.make_private_with_epsilon(
            module=dlrm,
            optimizer=optimizer,
            data_loader=train_ld,
            epochs=args.nepochs,
            target_epsilon=EPSILON,
            target_delta=DELTA,
            max_grad_norm=MAX_GRAD_NORM,
            disable_poisson_sampling=args.disable_poisson_sampling
        )
```
Secondly, construct the training code. You should slightly modify the original training loop like below (This code is transformed version of the main code, `LazyDP/dlrm/dlrm_s_pytorch_lazydp.py`, for clear explanation):
```python
for input_data in enumerate(train_ld): # Training loop
    # 1: load trainng data for the next iteration = "next_input"
    next_input = input_data
    
    if i == 0: # Only for first iteration 
        # 2: tranfer "next_input" to the optimizer
        optimizer.set_lS_i(next_input)

        # 3: update the history table (HT)
        optimizer.set_HT_increase_cnt_iter()

        # 4: use pre-loaded "next_input" as "curent_input"
        current_input = next_input

    else: # After first iteration
        # 2: forward propagation with "current_input"
        loss = dlrm(current_input)
            
        # 3: part of backpropagation (calculating l2 norm of per-example grad.)
        loss.backward()

        # 4: tranfer "next_input" to the optimizer
        optimizer.set_lS_i(next_input)

        # 5: part of backpropagation (calculating clipped per-example grad.) and model update
        optimizer.step(loss)

        # 6: update the history table (HT)
        optimizer.set_HT_increase_cnt_iter()

        # 7: use pre-loaded "next_input" as "curent_input"
        current_input = next_input
```

## Citation
Juntaek Lim, Youngeun Kwon, Ranggi Hwang, Kiwan Maeng, Edward Suh, and Minsoo Rhu, "[LazyDP: Co-Designing Algorithm-Software for Scalable Training of Differentially Private Recommendation Models](https://arxiv.org/abs/2404.08847)", 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), April 2024.