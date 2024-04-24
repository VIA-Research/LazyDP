# class "LatencyMeter" is for an automatic measurement of latency
# 
# Case 1: SGD
# _________________________________________________________________________________________________________________________________________________________________
# | FW_entire                                                                                                           | BW_entire              | Update_entire   |
# | FW_input_to_gpu | FW_add_mlp_bias | FW_bottom_mlp | FW_emb | FW_emb_cpu_to_gpu | FW_interact | FW_top_mlp | FW_loss | BW_zero_grad | BW_grad | Update_original |
# |_________________|_________________|_______________|________|___________________|_____________|____________|_________|______________|_________|_________________|
#
# Forward = FW_entire
# Backward = 0
# Backward_2= BW_grad
# Update = BW_zero_grad + Update
#
# Case 2: DP-SGD (Fast-DP-SGD)
# 
# Forward and Backward are same with "Case 1: SGD"
#__________________________________________________________________________________________________________________________________
#  | Update_entire                                                                                                                 |
#  | per-example & network-wise norm | loss clipping | 2nd backprop | p.summed_grad to reuse code | Update_noise | Update_original | 
# _|_________________________________|_______________|______________|_____________________________|______________|_________________|
#
# Forward = FW_entire
# Backward = BW_grad
# Backward_2 = 2nd backprop
# Update = BW_zero_grad + Update_entire - 2nd backprop
#
# Case 3: Lazy Update
# 
# Forward and Backward are same with "Case 1: SGD"
# __________________________________________________________________________________________________________________________________________________________
# | set_lS_i | Update_entire                                                                                                                                |
# |          | per-example & network-wise norm | loss clipping | 2nd backprop | p.summed_grad to reuse code | Update_noise | Update_set_emb_to_noise_update | 
# |__________|_________________________________|_______________|______________|_____________________________|______________|________________________________|
# ___________________________________________________________________
#  | Update_entire                                                   |
#  | Update_delayed_noise_update | Update_original | set_HT_increase |
#__|_____________________________|_________________|_________________|
#
# Forward = FW_entire
# Backward_1 = BW_grad
# Bacward_2 = 2nd backprop
# Update = BW_zero_grad + set_lS_i + Update_entire - 2nd backprop

import torch
import time
import pandas as pd
import os.path
import custom_api_cpp

def aggregate(mean_records: torch.Tensor, indices: list):
        result = 0
        for i in indices:
            result += mean_records[i].item()
        return result * 1000 # unit: msec
    
class LatencyMeter:
    def __init__(self, mode, result_name, iters, description, result_path):
        if(mode == "sgd" or mode == "dpsgd_b" or mode == "dpsgd_r" or mode == "dpsgd_f" or mode == "lazydp" or mode == "eana"):
            self.mode = mode
        else:
            assert False
        
        self.columns = ["FW_input_to_gpu", "FW_add_mlp_bias", "FW_bottom_mlp", "FW_emb", "FW_emb_cpu_to_gpu", "FW_interact", "FW_top_mlp", "FW_loss", "BW_zero_grad", "BW_grad"] # 0~9
        
        if(mode == "sgd"):
            self.columns += ["Update_original", "backward", "coalesce"]
            assert len(self.columns) == 13
        elif(mode == "dpsgd_b"):
            self.columns += ["Update_per_sample_clip_factor", "Update_clip_and_reduce", "Update_noise", "Update_original", "clip", "coalesce", "grad_to_summedgrad", "generate_noise_mlp", "add_noise_mlp", "generate_noise_emb", "add_noise_emb"]
            assert len(self.columns) == 21
        elif(mode in ["dpsgd_r", "dpsgd_f", "eana"]):
            self.columns += ["clipping_factor", "loss_clipping", "2nd_backprop","summedgrad_to_grad", "Update_noise", "Update_original", "backward", "coalesce", "generate_noise_mlp", "add_noise_mlp", "generate_noise_emb", "add_noise_emb"]
            assert len(self.columns) == 22
        # elif(mode == "lazy_update"):
        #     self.columns += ["set_lS_i", "clipping_factor", "loss_clipping", "2nd_backprop", "summedgrad_to_grad", "Update_noise", "Update_set_emb_to_noise_update", "Update_delayed_noise_update", "Update_original", "set_HT_increase", "backward", "coalesce", "generate_noise_mlp", "add_noise_mlp", "bypass_emb", "generate_noise_emb", "add_noise_emb"]
        #     assert len(self.columns) == 27
        elif(mode == "lazydp"):
            self.columns += ["set_lS_i", "clipping_factor", "loss_clipping", "2nd_backprop", "summedgrad_to_grad", "Update_noise", "Update_set_emb_to_noise_update", "Update_delayed_noise_update", "Update_original", "set_HT_increase", "backward", "generate_noise_mlp", "add_noise_mlp", "bypass_emb", "generate_noise_emb", "add_noise_emb", "coalesce"]
            assert len(self.columns) == 27
        else:
            assert False
            
        # debugging
        self.columns += ["test"]
        
        self.columns_num = len(self.columns)
                
        self.iters = iters
        self.cur_iter = 0
        self.records = torch.zeros(self.columns_num, self.iters)
        
        self.start_time = None
        self.current_column = None
        self.start_time_l2 = None
        self.current_column_l2 = None
        
        self.result_name = result_name
        self.description = description
        self.result_path = result_path
        self.merged_file_path = "%s/merged_result/%s.csv" %(self.result_path, self.description)
        self.detailed_file_path = "%s/detailed_latency_breakdown/%s.csv" %(self.result_path, self.result_name)
    
    def start(self, column):
        assert self.start_time == None, "invalid start-end pair (start)"
        assert self.current_column == None, "invalid start-end pair (start), column"
        self.current_column = column
        torch.cuda.synchronize()
        self.start_time = time.time()
    
    def end(self, column):
        torch.cuda.synchronize()
        t = time.time() - self.start_time
        assert self.start_time != None, "invalid start-end pair (end), time"
        assert self.current_column == column, "invalid start-end pair (end)"
        self.records[self.columns.index(column)][self.cur_iter] = t
        self.start_time = None
        self.current_column = None
        
    # this timestamp functions are for level-2 (e.g., "generate_noise" and "add_noise"
    # are in level-1 named "Update_noise")
    def start_l2(self, column):
        assert self.start_time_l2 == None, "invalid start-end pair (start)"
        assert self.current_column_l2 == None, "invalid start-end pair (start), column"
        self.current_column_l2 = column
        torch.cuda.synchronize()
        self.start_time_l2 = time.time()
    
    def end_l2(self, column):
        torch.cuda.synchronize()
        t = time.time() - self.start_time_l2
        assert self.start_time_l2 != None, "invalid start-end pair (end), time"
        assert self.current_column_l2 == column, "invalid start-end pair (end)"
        self.records[self.columns.index(column)][self.cur_iter] += t
        self.start_time_l2 = None
        self.current_column_l2 = None
    
    def increase_iter(self):
        self.cur_iter += 1
    
    def save(self):
        assert self.cur_iter == self.iters, "save only all iterations are done"
        index = self.columns
        columns = torch.arange(self.iters).tolist()
        df = pd.DataFrame(self.records, columns=columns, index=index)
        df.to_csv("%s" %self.detailed_file_path)
        
        if self.mode != "lazydp":
            mean_records = self.records[:, -10:].mean(dim=1)
        elif self.mode == "lazydp":
            mean_records = self.records[:, -11:-1].mean(dim=1)
        else:
            assert False
            
        assert mean_records.shape[0] == self.columns_num
        
        # Entire breakdown
        fwd = aggregate(mean_records, [0, 1, 2, 3, 4, 5, 6, 7])
        if self.mode == "sgd":
            bwd_example = 0
            bwd_batch = aggregate(mean_records, [11])
            update = aggregate(mean_records, [8, 12, 10])
        elif self.mode == "dpsgd_b":
            bwd_example = aggregate(mean_records, [9, 10, 14])
            bwd_batch = 0
            update = aggregate(mean_records, [8, 15, 16, 17, 18, 19, 20, 13]) # 12 -> 17, 18, 19, 20
        elif self.mode in ["dpsgd_r", "dpsgd_f", "eana"]:
            bwd_example = aggregate(mean_records, [9, 10, 11])
            bwd_batch = aggregate(mean_records, [16])
            update = aggregate(mean_records, [8, 17, 13, 18, 19, 20, 21, 15]) # 14 -> 18, 19, 20, 21
        elif self.mode == "lazydp":
            bwd_example = aggregate(mean_records, [9, 11, 12])
            bwd_batch = aggregate(mean_records, [20])
            update = aggregate(mean_records, [8, 10, 14, 21, 22, 23, 16, 24, 25, 26, 18, 19]) # 15 -> 21, 22, 23 / 17 -> 24, 25, 26
        else:
            assert False
            
        # Update breakdown
        if self.mode == "sgd":
            coalesce = aggregate(mean_records, [12])
            noise_identify = aggregate(mean_records, [])
            noise_sampling = aggregate(mean_records, [])
            merging = aggregate(mean_records, [])
            model_parameter_update = aggregate(mean_records, [10])
            metadata_update = aggregate(mean_records, [])
            overhead = noise_identify + metadata_update
            else_ = aggregate(mean_records, [8])
        elif self.mode == "dpsgd_b":
            coalesce = aggregate(mean_records, [15])
            noise_identify = aggregate(mean_records, [])
            noise_sampling = aggregate(mean_records, [17, 19])
            merging = aggregate(mean_records, [18, 20])
            model_parameter_update = aggregate(mean_records, [13])
            metadata_update = aggregate(mean_records, [])
            overhead = noise_identify + metadata_update
            else_ = aggregate(mean_records, [8, 16])
        elif self.mode in ["dpsgd_r", "dpsgd_f", "eana"]:
            coalesce = aggregate(mean_records, [17])
            noise_identify = aggregate(mean_records, [])
            noise_sampling = aggregate(mean_records, [18, 20])
            merging = aggregate(mean_records, [19, 21])
            model_parameter_update = aggregate(mean_records, [15])
            metadata_update = aggregate(mean_records, [])
            overhead = noise_identify + metadata_update
            else_ = aggregate(mean_records, [8, 13])
        elif self.mode == "lazydp":
            coalesce = aggregate(mean_records, [26])
            noise_identify = aggregate(mean_records, [16])
            noise_sampling = aggregate(mean_records, [21, 24])
            merging = aggregate(mean_records, [22, 25])
            model_parameter_update = aggregate(mean_records, [18])
            metadata_update = aggregate(mean_records, [10, 19])
            overhead = noise_identify + metadata_update
            else_ = aggregate(mean_records, [8, 23, 14])
        else:
            assert False
            
        if os.path.isfile(self.merged_file_path):
            past_result = pd.read_csv(self.merged_file_path, header=0, index_col=0)
            past_result_value = torch.tensor(past_result.values)
            past_columns = past_result.columns.tolist()
        else:
            # past_result_value = torch.empty(10 + 12 + 1,0)
            past_result_value = torch.empty(10 + 1,0)
            past_columns = []
            
        additional_result = torch.tensor([fwd, bwd_example, bwd_batch, update, coalesce, noise_sampling, merging, model_parameter_update, overhead, else_] + [mean_records[-1] * 1000]).view(-1, 1)
        new_result = torch.cat([past_result_value, additional_result], dim=1)
        df_2 = pd.DataFrame(new_result, columns=past_columns + ["%s" % self.result_name], index=["Fwd", "Bwd(per-example)", "Bwd(per-batch)", "Update", "Gradient coalesce", "Noise sampling", "Noisy gradient generation", "Model parameter update", "Overhead", "Else", "test"])
        df_2.to_csv(self.merged_file_path)
    