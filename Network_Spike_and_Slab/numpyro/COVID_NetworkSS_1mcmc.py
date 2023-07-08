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
from absl import flags
import re
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KernelDensity
import jax
import numpyro
# numpyro.set_platform('cpu')
print(jax.lib.xla_bridge.get_backend().platform)
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS 
from jax.random import PRNGKey as Key
from numpyro.infer import init_to_feasible, init_to_value
from numpyro.handlers import condition, block
import numpyro.infer.util
from jax.random import PRNGKey as Key
from numpyro.util import enable_x64
from numpyro.infer import init_to_feasible, init_to_value
import jax.nn as nn
from numpyro import plate, sample,  factor
import numpyro.distributions as dist
from numpyro.distributions import ImproperUniform, constraints
import numpyro.infer.util 
from numpyro.primitives import deterministic
from COVID_NetworkSS_1mcmc_init import mcmc1_init
from COVID_NetworkSS_1mcmc_add import mcmc1_add
#%%
# paths
_ROOT_DIR = "/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/graphical-models-external-networks/"
os.chdir(_ROOT_DIR)
sys.path.append("/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/COVID/Pre-processed Data/'
data_save_path = '/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/NetworkSS_results/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)
#%%
# load models and functions
import models
import my_utils

# define flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('thinning', 0, 'Thinning between MCMC samples.')
flags.DEFINE_integer('n_samples', 0, 'Number of total samples to run (excluding warmup).')
flags.DEFINE_string('Y', 'COVID_629_meta.csv', 'Name of file where data for dependent variable is stored.')
flags.DEFINE_multi_string('network_list', ['GEO_clean_629.npy', 'SCI_clean_629.npy'], 'Name of file where network data is stored. Flag can be called multiple times. Order of calling IS relevant.')
flags.mark_flags_as_required(['n_samples'])
FLAGS(sys.argv)


enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
#%%
n_samples = FLAGS.n_samples
thinning = FLAGS.thinning
covid_vals_name = FLAGS.Y
network_names = FLAGS.network_list
print(network_names)
# load data
covid_vals = jnp.array(pd.read_csv(data_path + covid_vals_name, index_col='Unnamed: 0').values)
geo_clean = jnp.array(jnp.load(data_path + network_names[0]))
sci_clean = jnp.array(jnp.load(data_path + network_names[1]))

covid_vals = covid_vals[:,:20].copy()
geo_clean = geo_clean[:20, :20].copy()
sci_clean = sci_clean[:20, :20].copy()
A_list = [geo_clean, sci_clean]



def get_init_file(dir, checkpoint):
    '''Retrieve init file'''
    CPs = []
    for f in os.listdir(dir):
        if re.search(r'(_CP)\d+', f):
            CP = int(re.search(r'(_CP)\d+', f)[0][3:])
            CPs.append(CP)
    try:
        CP_max = max(CPs)
        for f in os.listdir(dir):
            if re.search(fr'.*_CP{CP_max}\.sav$', f):
                return f, CP_max
    except:
        pass

try:        
    filename, CP_init = get_init_file(dir=data_save_path, checkpoint=n_samples)
except:
    CP_init = 0

if CP_init >= n_samples:
    print(f'Checkpoint at {CP_init} number of samples already exists.')
    sys.exit()
elif (CP_init < 500):
    mcmc1_init(thinning=thinning, covid_vals=covid_vals, A_list=A_list)
    n_rounds = (n_samples-500)/400
    batches = [400]*int(n_rounds) + [(n_samples-500)%400]
    for s_ix, s in enumerate(batches):
        mcmc1_add(thinning=thinning, covid_vals=covid_vals, A_list=A_list,
              checkpoint= 500 + sum(batches[:s_ix+1]) , n_warmup=50, n_samples=s)
else:
    n_rounds = (n_samples-CP_init)/400
    batches = [400]*int(n_rounds) + [(n_samples-CP_init)%400]
    for s_ix, s in enumerate(batches):
        mcmc1_add(thinning=thinning, covid_vals=covid_vals, A_list=A_list,
              checkpoint=CP_init + sum(batches[:s_ix+1]), n_warmup=50, n_samples=s)
 


