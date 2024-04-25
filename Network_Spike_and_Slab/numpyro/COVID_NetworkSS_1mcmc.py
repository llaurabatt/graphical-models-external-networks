# #%%
"""Main script for training the model."""
import debugpy
debugpy.listen(5678)
print('Waiting for debugger')
debugpy.wait_for_client()
print('Debugger attached')
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

data_path = './Data/COVID/Pre-processed Data/'

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
flags.DEFINE_string('Y', 'COVID_332_meta_pruned.csv', 'Name of file where data for dependent variable is stored.')
flags.DEFINE_string('X', None, 'Name of file where data for covariate variables is stored.')
flags.DEFINE_string('model', 'models.NetworkSS_repr_etaRepr_loglikRepr', 'Name of model to be run.')
flags.DEFINE_string('init_strategy', 'init_to_value', "Either 'init_to_value' (default) or 'init_to_feasible'.")
flags.DEFINE_multi_string('network_list', ['GEO_meta_clean_332.npy', 'SCI_meta_clean_332.npy', 'flights_meta_clean_332.npy'], 'Name of file where network data is stored. Flag can be called multiple times. Order of calling IS relevant.')
flags.mark_flags_as_required(['n_samples', 'thinning', 'data_save_path', 'SEED'])
FLAGS(sys.argv)


enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
#%%
n_samples = FLAGS.n_samples
my_model = eval(FLAGS.model)
thinning = FLAGS.thinning
batch_size = 500
covid_vals_name = FLAGS.Y
covariates_name = FLAGS.X
SEED = FLAGS.SEED
init_strategy = FLAGS.init_strategy
network_names = FLAGS.network_list
print(network_names)
print(FLAGS.model)
print(init_strategy)


data_save_path = FLAGS.data_save_path
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)
print(f'Save in {data_save_path}')

# load data
covid_vals = jnp.array(pd.read_csv(data_path + covid_vals_name, index_col='Unnamed: 0').values)
geo_clean = jnp.array(jnp.load(data_path + network_names[0]))
sci_clean = jnp.array(jnp.load(data_path + network_names[1]))
flights_clean = jnp.array(jnp.load(data_path + network_names[2]))
if covariates_name:
    covariates = jnp.array(jnp.load(data_path + covariates_name))

# covid_vals = covid_vals[:,:100].copy()
# geo_clean = geo_clean[:100, :100].copy()
# sci_clean = sci_clean[:100, :100].copy()
A_list = [geo_clean, sci_clean, flights_clean]

# mcmc_args = {"A_list":A_list, 
#                 "eta0_0_m":0., "eta0_0_s":0.0015864, 
#         "eta0_coefs_m":0., "eta0_coefs_s":0.0015864,
#         "eta1_0_m":-2.1972246, "eta1_0_s":0.3, 
#         "eta1_coefs_m":0., "eta1_coefs_s":0.3,
#         "eta2_0_m":-7.7894737, "eta2_0_s":1.0263158, 
#         "eta2_coefs_m":0., "eta2_coefs_s":1.0263158,
#         } 
mcmc_args = {"A_list":A_list, 
                "eta0_0_m":0., "eta0_0_s":0.003, 
        "eta0_coefs_m":0., "eta0_coefs_s":0.003,
        "eta1_0_m":-2.1972246, "eta1_0_s":0.65, 
        "eta1_coefs_m":0., "eta1_coefs_s":0.65,
        "eta2_0_m":-11.737, "eta2_0_s":4.184, 
        "eta2_coefs_m":0., "eta2_coefs_s":4.184,
        } 

if covariates_name is not None:
     mcmc_args["X"] = covariates
     mcmc_args.update({"b_m":0., "b_s":5.})
else:
     mcmc_args.update({"mu_m":0., "mu_s":1.})

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

mcmc1_init(my_model=my_model, thinning=thinning, my_vals=covid_vals,
        my_model_args=mcmc_args, n_samples=n_samples,
        root_dir=_ROOT_DIR, data_save_path=data_save_path, seed=SEED, init_strategy=init_strategy)

# TO-DO: stop-and-start chain not working at the moment
# if CP_init >= n_samples:
#     print(f'Checkpoint at {CP_init} number of samples already exists.')
#     sys.exit()
# elif (CP_init < 500):
#     mcmc1_init(my_model=my_model, thinning=thinning, my_vals=covid_vals,
#                 my_model_args=mcmc_args,
#                root_dir=_ROOT_DIR, data_save_path=data_save_path, seed=SEED, init_strategy=init_strategy)
#     n_rounds = (n_samples-500)/batch_size
#     batches = [batch_size]*int(n_rounds) + ([(n_samples-500)%batch_size] if (n_samples-500)%batch_size!=0 else [])
#     for s_ix, s in enumerate(batches):
#         mcmc1_add(my_model=my_model, thinning=thinning, my_vals=covid_vals,
#                   my_model_args=mcmc_args,
#               checkpoint= 500 + sum(batches[:s_ix+1]) , n_warmup=1000, n_samples=s,
#               root_dir=_ROOT_DIR, data_save_path=data_save_path)
# else:
#     n_rounds = (n_samples-CP_init)/batch_size
#     batches = [batch_size]*int(n_rounds) + ([(n_samples-CP_init)%batch_size] if (n_samples-CP_init)%batch_size!=0 else []) 
#     for s_ix, s in enumerate(batches):
#         mcmc1_add(my_model=my_model, thinning=thinning, my_vals=covid_vals,
#                   my_model_args=mcmc_args,
#               checkpoint=CP_init + sum(batches[:s_ix+1]), n_warmup=1000, n_samples=s,
#               root_dir=_ROOT_DIR, data_save_path=data_save_path)
 


