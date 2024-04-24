import numpy as np
import time

import torch

import random
from numpy.random import default_rng
from torch import Tensor
import argparse
import pandas as pd

import os

import numpy as np

class AccessGenerator:
    def __init__(self, gpu_num, E, B, L, alpha=None):
        self.E = E
        self.B = B
        self.L = L
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda", gpu_num)
        else:
            self.device = torch.device("cpu")
            
        if alpha == 1:
            self.embedding_probs = torch.ones(E)
            self.embedding_probs /= E
        elif alpha > 1:
            self.embedding_probs = 1/(torch.pow((torch.arange(E) + 1), alpha))
            self.embedding_probs /= self.embedding_probs.sum().item()  # Normalize probabilities
        else:
            assert False

        self.embedding_probs_device = self.embedding_probs.to(self.device)
        
    def generate_accesses(self):
        indices = torch.zeros((self.B, self.L), device=self.device)
        for i in range(self.B):
            index = torch.multinomial(self.embedding_probs_device, self.L, replacement=False)
            indices[i][:] = index
        
        # indices = torch.multinomial(self.embedding_probs_device, self.B, replacement=True)
        return indices.to(torch.int64)
import pdb

def run():
    parser = argparse.ArgumentParser(
        description="Synthetic data generation for DLRM training"
    )
    parser.add_argument("--B", type=int, default=2048)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--E", type=int, default=10000000)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--niters", type=int, default=100)
    parser.add_argument("--gpu-num", type=int, default=0)
    args = parser.parse_args()

    # assert(args.L == 1)

    iter = 100

    for a in [1.2]:
        n = 0
        union = 0
        access_gen = AccessGenerator(0, args.E, args.B, args.L, a)
        past = torch.ones(1)
        for i in range(iter):
            indices = access_gen.generate_accesses()
            unique_indices = indices.unique()
            # if i != 0:
            #     union += torch.cat((past, unique_indices)).unique().shape[0]
            
            # past = unique_indices
            n += unique_indices.shape[0]
            
        union += torch.cat((past, unique_indices)).unique().shape[0]
        
        print("%f, %f" %(n/iter, union/iter))
    
     


    # avg_reaccess_interval = torch.zeros(args.niters)
    
    # accesse_gen = AccessGenerator(args.gpu_num, args.E, args.B, args.L, args.a)
    # history = torch.ones(args.E) * (-1)
    # start = time.time()
    # for i in range(args.niters):
    #     # if i % 100 == 0:
    #     #     print(i)
    #     indices = accesse_gen.generate_accesses()
    #     unique_indices = indices.unique()
    #     reaccess_intervals = i - history[unique_indices]
    #     avg_reaccess_interval[i] = reaccess_intervals.mean().item()
    #     history[unique_indices] = i
    # print(time.time() - start)
    # df = pd.DataFrame(avg_reaccess_interval, index=(torch.arange(args.niters)).tolist(), columns=["%.2f_%d" %(args.a, args.E)])
    # df.to_csv("./%.2f_%d.csv" %(args.a, args.E))
    
    # result = torch.zeros(len(list_a), int(args.niter/scale))
    # for i in range(len(list_a)):
    #     a = list_a[i]
    #     n_accessed = 0
    #     for j in range(args.niter):
    #         accesse_gen = AccessGenerator(args.gpu_num, args.E, args.B, args.L, a)
    #         indices = accesse_gen.generate_accesses()
    #         unique_indices = indices.unique()
    #         if j % scale == 0:
    #             result[i][int(j/scale)] = n_accessed + args.E
    #         n_accessed += unique_indices.shape[0]
            
    
    # df = pd.DataFrame(avg_reaccess_interval, index=(torch.arange(int(args.niters/5))*5).tolist(), columns=["%.3f_%d_%d_%d" %(args.a, args.E, args.B, args.L)])
    # df.to_csv("./%.2f_%d_%d_%d_%d_independently.csv" %(args.a, args.E, args.B, args.L, args.niters))
    
    print(time.time()-start)
if __name__ == "__main__":
    run()