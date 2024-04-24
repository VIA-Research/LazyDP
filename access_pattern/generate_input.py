import numpy as np
import torch
import pdb
import time

table_size="39884406-39043-17289-7420-20263-3-7120-1543-63-38532951-2953546-403346-10-2208-11938-155-4-976-14-39979771-25641295-39664984-585935-12972-108-36"
ln_emb = np.fromstring(table_size, dtype=int, sep="-")
ln_emb = torch.from_numpy(ln_emb)
batch_size = 4096
pooling_factor = 60

num_gathers_list = torch.zeros(len(ln_emb), dtype=torch.int64)
for i in range(len(ln_emb)):
    if ln_emb[i].item() < pooling_factor:
        num_gathers_list[i] = ln_emb[i].item()
    else:
        num_gathers_list[i] = pooling_factor
        
access_pdfs = list()    
for i in range(len(ln_emb)):
    pdf = np.ones(ln_emb[i].item())/ln_emb[i].item()
    access_pdfs.append(pdf)


pdb.set_trace()
start = time.time()

# lS_i = list()
# for i in range(len(ln_emb)):
#     lS_i_table = list()
#     for j in range(batch_size):
#         lS_i_example = None
        
#         goal = num_gathers_list[i].item()
#         num = 0
#         while num < goal:
#             tmp = torch.randint(0, ln_emb[i].item(), (goal-num, ), dtype=torch.int64)
#             if lS_i_example != None:
#                 lS_i_example = torch.cat([lS_i_example, tmp]).unique()
#             else:
#                 lS_i_example = tmp.unique()
#             num = lS_i_example.shape[0]
        
#         lS_i_table.append(lS_i_example)
#     lS_i.append(torch.cat(lS_i_table))
    
"""
lS_i = list()
for i in range(len(ln_emb)):
    print(i)
    lS_i_table = list()
    for j in range(batch_size):
        tmp = np.random.choice(ln_emb[i].item(), num_gathers_list[i].item(), replace=False, p=access_pdfs[i])
        lS_i_table.append(tmp)
    lS_i.append(torch.from_numpy(np.concatenate(lS_i_table)))
"""

print("time: %2f" %(time.time() - start))
for i in range(len(lS_i)):
    assert lS_i[i].shape[0] == num_gathers_list[i].item()  * batch_size, "%d, %d" %(lS_i[i].shape[0], num_gathers_list[i].item())
    
pdb.set_trace()
assert True