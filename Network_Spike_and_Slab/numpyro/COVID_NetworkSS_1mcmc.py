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
# from NetworkSS_1mcmc_add import mcmc1_add
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
flags.DEFINE_boolean('store_warmup', False, 'If true, it will store warmup samples.')
flags.DEFINE_boolean('no_networks', False, 'If true, network information will not be used.')
flags.DEFINE_integer('thinning', 1, 'Thinning between MCMC samples. Defaults to 1.')
flags.DEFINE_float('scale_spike_fixed', 0.0033341, 'Fixed value of the scale of the spike.')
flags.DEFINE_integer('n_samples', None, 'Number of total samples to run (excluding warmup).')
flags.DEFINE_string('data_save_path', None, 'Path for saving results.')
flags.DEFINE_integer('SEED', None, 'Random seed.')
flags.DEFINE_string('Y', 'COVID_332_meta_pruned.csv', 'Name of file where data for dependent variable is stored.')
flags.DEFINE_string('X', None, 'Name of file where data for covariate variables is stored.')
flags.DEFINE_string('model', 'models.NetworkSS_repr_etaRepr_loglikRepr', 'Name of model to be run.')
flags.DEFINE_string('init_strategy', 'init_to_value', "Either 'init_to_value' (default) or 'init_to_feasible'.")
flags.DEFINE_string('b_init', None, "Initial values for regression coefficents. Defaults to zeros.")
flags.DEFINE_string('bhat', None, "Value for centering regression coefficents. Defaults to None.")
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
scale_spike_fixed = FLAGS.scale_spike_fixed
covid_vals_name = FLAGS.Y
covariates_name = FLAGS.X
b_init = FLAGS.b_init
bhat = FLAGS.bhat
SEED = FLAGS.SEED
init_strategy = FLAGS.init_strategy
network_names = FLAGS.network_list
store_warmup = FLAGS.store_warmup
no_networks = FLAGS.no_networks
print(network_names)
print(FLAGS.model)
print(init_strategy)
print(f'Seed: {SEED}')


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
    _, _, q = covariates.shape
if b_init is not None:
    b_init = jnp.array(pd.read_csv(data_path + b_init).values).flatten()
if bhat is not None:
    bhat = jnp.array(pd.read_csv(data_path + bhat).values).flatten()

# p_red = 50
# covid_vals = covid_vals[:,:p_red].copy()
# geo_clean = geo_clean[:p_red, :p_red].copy()
# sci_clean = sci_clean[:p_red, :p_red].copy()
# flights_clean = flights_clean[:p_red, :p_red].copy()
# if covariates_name:
#     covariates = covariates[:,:p_red,:].copy()
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
if bhat is not None:
    mcmc_args['bhat'] = bhat
else:
     mcmc_args.update({"mu_m":0., "mu_s":1.})


mcmc1_init(my_model=my_model, thinning=thinning, my_vals=covid_vals,
        my_model_args=mcmc_args, scale_spike_fixed=scale_spike_fixed, n_samples=n_samples,
        root_dir=_ROOT_DIR, data_save_path=data_save_path, seed=SEED, 
        init_strategy=init_strategy, b_init=b_init,
        store_warmup=store_warmup, no_networks=no_networks)


