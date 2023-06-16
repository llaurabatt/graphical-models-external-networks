# #%%
# """Main script for training the model."""
# import debugpy
# debugpy.listen(5678)
# print('Waiting for debugger')
# debugpy.wait_for_client()
# print('Debugger attached')
#%%
# imports
import sys
import os
import re
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import jax
import numpyro
numpyro.set_platform('cpu')
print(jax.lib.xla_bridge.get_backend().platform)
#%%
# paths
_ROOT_DIR = "/home/user/graphical-models-external-networks/"
os.chdir(_ROOT_DIR)
sys.path.append("/home/user/graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/COVID/Pre-processed Data/'
data_save_path = '/home/user/mounted_folder/NetworkSS_results/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)


# %%
i = 0
for f_ix, f in enumerate(sorted(os.listdir(data_save_path))):
    if '1mcmc' in f:
        print(f)
        if i==0:
            with open(data_save_path + f, 'rb') as fr:
                res = pickle.load(fr)
            samples = {k:[] for k in res.keys()}
        with open(data_save_path + f, 'rb') as fr:
            res = pickle.load(fr)
        for k in res.keys():
            samples[k].append(res[k])
        i += 1

# %%
with open(data_save_path + f'NetworkSS_1mcmc_p629_s4100_aggregate0.sav' , 'wb') as f:
    pickle.dump((samples), f)
%%

# with open(data_save_path + 'NetworkSS_1mcmc_p629_s4100_aggregate0.sav', 'rb') as fr:
#     samples = pickle.load(fr)
# # # %%
# sampless = {}
# for k,v in samples.items():
#     print(k)
#     try:
#         sampless[k] = np.vstack(np.array(v, dtype=object))
#     except:
#         sampless[k] = np.vstack(np.array([s[:,None] for s in v], dtype=object))
# del samples
# # sampless = {k:np.vstack(np.array(v, dtype=object)) for k,v in samples.items()}
# # %%
# tot_samples = sampless[k].shape[0]
# print(tot_samples)
# with open(data_save_path + f'NetworkSS_1mcmc_p629_s{tot_samples}_aggregate.sav' , 'wb') as f:
#     pickle.dump((sampless), f)