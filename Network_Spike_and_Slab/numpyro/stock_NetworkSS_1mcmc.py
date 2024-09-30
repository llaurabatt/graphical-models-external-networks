# #%%
"""Main script for training the model."""
# import debugpy
# debugpy.listen(5678)
# print('Waiting for debugger')
# debugpy.wait_for_client()
# print('Debugger attached')
#%%
# imports
import sys
import os
from absl import flags
import re
#%%
import pandas as pd
import jax
import numpyro
# numpyro.set_platform('cpu')
print(jax.lib.xla_bridge.get_backend().platform)
import jax.numpy as jnp
import numpyro.infer.util
from numpyro.util import enable_x64
from NetworkSS_1mcmc_init import mcmc1_init
from NetworkSS_1mcmc_add import mcmc1_add
#%%
# paths
_ROOT_DIR = "/home/paperspace/"
os.chdir(_ROOT_DIR + 'graphical-models-external-networks/')
sys.path.append(_ROOT_DIR + "graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/Stock/Pre-processed Data/'

#%%
# load models and functions
import models
import my_utils

# define flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('thinning', None, 'Thinning between MCMC samples.')
flags.DEFINE_integer('n_samples', None, 'Number of total samples to run (excluding warmup).')
flags.DEFINE_string('data_save_path', None, 'Path for saving results.')
flags.DEFINE_integer('SEED', None, 'Random seed.')
flags.DEFINE_string('Y', 'stock_trans_SP.csv', 'Name of file where data for dependent variable is stored.')
flags.DEFINE_string('model', 'models.NetworkSS_repr_etaRepr_loglikRepr', 'Name of model to be run.')
flags.DEFINE_string('init_strategy', 'init_to_value', "Either 'init_to_value' (default) or 'init_to_feasible'.")
flags.DEFINE_multi_string('network_list', ['E_pears_clean_SP_366.npy', 'P_pears_clean_SP_366.npy'], 'Name of file where network data is stored. Flag can be called multiple times. Order of calling IS relevant.')
flags.mark_flags_as_required(['n_samples', 'thinning', 'data_save_path', 'SEED'])
FLAGS(sys.argv)


enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
#%%
n_samples = FLAGS.n_samples
my_model = eval(FLAGS.model)
thinning = FLAGS.thinning
batch_size = 500
stock_vals_name = FLAGS.Y
SEED = FLAGS.SEED
init_strategy = FLAGS.init_strategy
network_names = FLAGS.network_list
print(network_names)
print(FLAGS.model)


data_save_path = FLAGS.data_save_path
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)
print(f'Save in {data_save_path}')

# load data
stock_vals = jnp.array(pd.read_csv(data_path + stock_vals_name, index_col='Unnamed: 0').values)
E_clean = jnp.array(jnp.load(data_path + network_names[0]))
P_clean = jnp.array(jnp.load(data_path + network_names[1]))

# stock_vals = stock_vals[:,:100].copy()
# E_clean = E_clean[:100, :100].copy()
# P_clean = P_clean[:100, :100].copy()
A_list = [E_clean, P_clean]

covid_mcmc_args = {"A_list":A_list, 
                "eta0_0_m":0., "eta0_0_s":0.0015864, 
        "eta0_coefs_m":0., "eta0_coefs_s":0.0015864,
        "eta1_0_m":-2.1972246, "eta1_0_s":0.3, 
        "eta1_coefs_m":0., "eta1_coefs_s":0.3,
        "eta2_0_m":-7.7894737, "eta2_0_s":1.0263158, 
        "eta2_coefs_m":0., "eta2_coefs_s":1.0263158,
        "mu_m":0., "mu_s":1.} 

# mcmc_args = {"A_list":A_list, 
#                 "eta0_0_m":0., "eta0_0_s":0.0021276, 
#         "eta0_coefs_m":0., "eta0_coefs_s":0.0021276,
#         "eta1_0_m":-2.1972246, "eta1_0_s":0.35, 
#         "eta1_coefs_m":0., "eta1_coefs_s":0.35,
#         "eta2_0_m":-10.1578947, "eta2_0_s":1.8157895, 
#         "eta2_coefs_m":0., "eta2_coefs_s":1.8157895,
#         "mu_m":0., "mu_s":1.} 

# TO-DO: stop-and-start chain not working at the moment
# def get_init_file(dir, checkpoint):
#     '''Retrieve init file'''
#     CPs = []
#     for f in os.listdir(dir):
#         if re.search(r'(_CP)\d+', f):
#             CP = int(re.search(r'(_CP)\d+', f)[0][3:])
#             CPs.append(CP)
#     try:
#         CP_max = max(CPs)
#         for f in os.listdir(dir):
#             if re.search(fr'.*_CP{CP_max}\.sav$', f):
#                 return f, CP_max
#     except:
#         pass

# try:        
#     filename, CP_init = get_init_file(dir=data_save_path, checkpoint=n_samples)
# except:
#     CP_init = 0

mcmc1_init(my_model=my_model, thinning=thinning, my_vals=stock_vals,
        my_model_args=covid_mcmc_args, n_samples=n_samples,
        root_dir=_ROOT_DIR, data_save_path=data_save_path, seed=SEED, init_strategy=init_strategy,
        no_networks=False, scale_spike_fixed=0.0033341, b_init=None,
        store_warmup=False, init_all_path=None)

# TO-DO: stop-and-start chain not working at the moment
# if CP_init >= n_samples:
#     print(f'Checkpoint at {CP_init} number of samples already exists.')
#     sys.exit()
# elif (CP_init < 500):
#     mcmc1_init(my_model=my_model, thinning=thinning, my_vals=stock_vals,
#                 my_model_args=mcmc_args,
#                root_dir=_ROOT_DIR, data_save_path=data_save_path, seed=SEED, init_strategy=init_strategy)
#     n_rounds = (n_samples-500)/batch_size
#     batches = [batch_size]*int(n_rounds) + ([(n_samples-500)%batch_size] if (n_samples-500)%batch_size!=0 else [])
#     for s_ix, s in enumerate(batches):
#         mcmc1_add(my_model=my_model, thinning=thinning, my_vals=stock_vals,
#                   my_model_args=mcmc_args,
#               checkpoint= 500 + sum(batches[:s_ix+1]) , n_warmup=1000, n_samples=s,
#               root_dir=_ROOT_DIR, data_save_path=data_save_path)
# else:
#     n_rounds = (n_samples-CP_init)/batch_size
#     batches = [batch_size]*int(n_rounds) + ([(n_samples-CP_init)%batch_size] if (n_samples-CP_init)%batch_size!=0 else []) 
#     for s_ix, s in enumerate(batches):
#         mcmc1_add(my_model=my_model, thinning=thinning, my_vals=stock_vals,
#                   my_model_args=mcmc_args,
#               checkpoint=CP_init + sum(batches[:s_ix+1]), n_warmup=1000, n_samples=s,
#               root_dir=_ROOT_DIR, data_save_path=data_save_path)
 


