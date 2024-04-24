import torch
import argparse
import math

def run():
    parser = argparse.ArgumentParser(
        description="Correctness test 2: DP-SGD(F) vs LazyDP with using one as noise"
    )
    parser.add_argument("--path-model-weight", type=str, default="/")
    args = parser.parse_args()
    
    path_model_weight = args.path_model_weight
    
    param_list_f = torch.load("%s/dlrm_dpsgd_f" %path_model_weight)
    param_list_l = torch.load("%s/dlrm_lazydp" %path_model_weight)
        
    assert len(param_list_f) == len(param_list_l)
    
    length = len(param_list_f)
    
    for i in range(length):
        param_f = param_list_f[i]
        param_l = param_list_l[i]
        
        assert param_f.shape == param_l.shape
        
        cmp_f_l = torch.isclose(param_f, param_l, rtol=1e-03, atol=1e-06)
        
        if not torch.all(cmp_f_l):
            print("Correctness fail: DP-SGD(F) and LazyDP")
            assert False
        else:
            continue
        
    print("Correctness success.")
    
if __name__ == "__main__":
    run()