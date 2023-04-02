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
_ROOT_DIR = "/home/user/graphical-models-external-networks/"
os.chdir(_ROOT_DIR)
sys.path.append("/home/user/graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/COVID/Pre-processed Data/'
data_save_path = './Network_Spike_and_Slab/numpyro/NetworkSS_results/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)

# load models and functions
import models
import my_utils

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)

covid_df = pd.read_csv(data_path + 'COVID_629_meta.csv', index_col='Unnamed: 0')
geodist = pd.read_csv(data_path + 'geodist__629_meta.csv', index_col='Unnamed: 0')
sci_idx = pd.read_csv(data_path + 'sci_index__629_meta.csv', index_col='Unnamed: 0')

n,p = covid_df.shape

def diag_scale_network(log_A,p):
    diag_idx = jnp.diag_indices(n=p)
        
    log_A = log_A.at[diag_idx].set(0)
    
    scaled_net = my_utils.network_scale(log_A)
    
    log_A = scaled_net.at[diag_idx].set(0)
    return log_A

# geo
geo_vals = geodist.values
inv_log_geo = 1/jnp.log(geo_vals)
geo_clean = diag_scale_network(log_A=inv_log_geo, p=p)

sci_vals = sci_idx.values
log_sci = jnp.log(sci_vals)
sci_clean = diag_scale_network(log_A=log_sci, p=p)

jnp.save(data_save_path + 'GEO_clean_629.npy', geo_clean)
jnp.save(data_save_path + 'SCI_clean_629.npy', sci_clean)
