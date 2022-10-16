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

# paths
import os
os.chdir("/Users/")
sys.path.append("./functions")
data_save_path = './data/stock_market_data/'

# load models and functions
import models
import my_utils

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)

stock_df = pd.read_csv(data_save_path + 'stock_trans.csv', index_col='Unnamed: 0')
E_pears = pd.read_csv(data_save_path + 'E_pears.csv', index_col='Unnamed: 0')
P_pears = pd.read_csv(data_save_path + 'P_pears.csv', index_col='Unnamed: 0')
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

jnp.save(data_save_path + 'E_pears_clean.npy', E_pears_clean)
jnp.save(data_save_path + 'P_pears_clean.npy', P_pears_clean)