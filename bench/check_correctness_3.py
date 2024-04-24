import torch
import argparse
import math

def run():
    parser = argparse.ArgumentParser(
        description="Correctness test 3: SGD vs DP-SGD(B) with using one as noise"
    )
    parser.add_argument("--path-model-weight", type=str, default="/")
    args = parser.parse_args()
    
    path_model_weight = args.path_model_weight
    
    param_list_s = torch.load("%s/dlrm_sgd" %path_model_weight)
    param_list_b = torch.load("%s/dlrm_dpsgd_b" %path_model_weight)
        
    assert len(param_list_s) == len(param_list_b)
    
    length = len(param_list_b)
    
    for i in range(length):
        param_s = param_list_s[i]
        param_b = param_list_b[i]
        
        assert param_s.shape == param_b.shape
        
        cmp_s_b = torch.isclose(param_s, param_b, rtol=1e-03, atol=1e-06)
        
        if not torch.all(cmp_s_b):
            print("Correctness fail: SGD and DP-SGD(B)")
            assert False
        else:
            continue
        
    print("Correctness success.")
    
if __name__ == "__main__":
    run()