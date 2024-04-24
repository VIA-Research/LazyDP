import torch
import argparse
import math

def run():
    parser = argparse.ArgumentParser(
        description="Correctness test 1: DP-SGD(B) vs DP-SGD(R) vs DP-SGD(F) without adding noise"
    )
    parser.add_argument("--path-model-weight", type=str, default="/")
    args = parser.parse_args()
    
    path_model_weight = args.path_model_weight
    
    param_list_b = torch.load("%s/dlrm_dpsgd_b" %path_model_weight)
    param_list_r = torch.load("%s/dlrm_dpsgd_r" %path_model_weight)
    param_list_f = torch.load("%s/dlrm_dpsgd_f" %path_model_weight)
    
    assert len(param_list_b) == len(param_list_r)
    assert len(param_list_r) == len(param_list_f)
    
    length = len(param_list_b)
    
    for i in range(length):
        param_b = param_list_b[i]
        param_r = param_list_r[i]
        param_f = param_list_f[i]
        
        assert param_b.shape == param_r.shape
        assert param_r.shape == param_f.shape
        
        cmp_b_r = torch.isclose(param_b, param_r, rtol=1e-03, atol=1e-06)
        cmp_r_f = torch.isclose(param_r, param_f, rtol=1e-03, atol=1e-06)
        
        if not torch.all(cmp_b_r):
            print("Correctness fail: DP-SGD(B) and DP-SGD(R)")
            assert False
        elif not torch.all(cmp_r_f):
            print("Correctness fail: DP-SGD(R) and DP-SGD(F)")
            assert False
        else:
            continue
        
    print("Correctness success.")
    
if __name__ == "__main__":
    run()