#%%
import debugpy
debugpy.listen(5678)
print('Waiting for debugger')
debugpy.wait_for_client()
print('Debugger attached')
#%%
# imports
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle
import jax
import numpyro
numpyro.set_platform('cpu')
print(jax.lib.xla_bridge.get_backend().platform)
import jax.numpy as jnp
from numpyro.util import enable_x64

#%%
# paths
_ROOT_DIR = "/home/paperspace/"
os.chdir(_ROOT_DIR + 'graphical-models-external-networks/')
sys.path.append(_ROOT_DIR + "graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/Stock/Pre-processed Data/'
# data_save_path = _ROOT_DIR + 'stock_NetworkSS_results/'
# if not os.path.exists(data_save_path):
#     os.makedirs(data_save_path, mode=0o777)

# load models and functions
import models
import my_utils

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)

stock_df = pd.read_csv(data_path + 'stock_trans_SP.csv', index_col='Unnamed: 0')
E_pears = pd.read_csv(data_path + 'E_pears_SP.csv', index_col='Unnamed: 0')
P_pears = pd.read_csv(data_path + 'P_pears_SP.csv', index_col='Unnamed: 0')
p = E_pears.shape[0]

def diag_scale_network(A,p):
    diag_idx = jnp.diag_indices(n=p)
        
    A = A.at[diag_idx].set(0)
    
    scaled_net = my_utils.network_scale(A)
    
    A = scaled_net.at[diag_idx].set(0)
    return A

# E_pears
E_pears_vals = jnp.array(E_pears.values)
E_pears_clean = diag_scale_network(E_pears_vals, p=p)

# P_pears
P_pears_vals = jnp.array(P_pears.values)
P_pears_clean = diag_scale_network(P_pears_vals, p=p)

jnp.save(data_path + f'E_pears_clean_SP_{p}.npy', E_pears_clean)
jnp.save(data_path + f'P_pears_clean_SP_{p}.npy', P_pears_clean)
