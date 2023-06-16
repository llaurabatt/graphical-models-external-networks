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
from sklearn.neighbors import KernelDensity
import jax
import numpyro
numpyro.set_platform('gpu')
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
#%%
# paths
_ROOT_DIR = "/home/user/graphical-models-external-networks/"
os.chdir(_ROOT_DIR)
sys.path.append("/home/user/graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/COVID/Pre-processed Data/'
data_save_path = '/home/user/mounted_folder/NetworkSS_results/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)
# data_init_path = './data/sim_GLASSO_data/'
#%%
# load models and functions
import models
import my_utils
import imp
imp.reload(models)

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
cpus = jax.devices("cpu")
gpus = jax.devices("gpu")
#%%
def purge(dir, checkpoint, curr_no_s):
    for f in os.listdir(dir):
        if re.search(fr'.*_{checkpoint}\.sav$', f):
            no = re.search(r'(_s)\d+', f)[0][2:]
            if int(no) < curr_no_s:
                os.remove(os.path.join(dir, f))

def to_open(dir, checkpoint):
    nos = {}
    for f in os.listdir(dir):
        if re.search(fr'.*_{checkpoint}\.sav$', f):
            no_w = int(re.search(r'(_w)\d+', f)[0][2:])
            no_s = int(re.search(r'(_s)\d+', f)[0][2:])
            nos[no_s] = no_w
    no_s = max(nos, key=int)
    no_w = nos[no_s]
    return no_s, no_w

#%%
def mcmc1_add(checkpoint, n_warmup, n_samples):
    # load data 
    covid_vals = jnp.array(pd.read_csv(data_path + 'COVID_629_meta.csv', index_col='Unnamed: 0').values)
    geo_clean = jnp.array(jnp.load(data_path + 'GEO_clean_629.npy'))
    sci_clean = jnp.array(jnp.load(data_path + 'SCI_clean_629.npy'))
    #%%
    n,p = covid_vals.shape
    print(f"NetworkSS, n {n} and p {p}")

    #%%
    ## params
    n_warmup = n_warmup
    n_samples = n_samples
    n_batches = 1
    batch = int(n_samples/n_batches)
    mu_m=0.
    mu_s=1.
    verbose = True
    my_model = models.NetworkSS_repr_etaRepr_loglikRepr #models.NetworkSS_repr_etaRepr
    is_dense=False
    #%%
    if (checkpoint-1)==0:
        with open(data_save_path + f'NetworkSS_eff_p629_w1000_s500.sav', 'rb') as fr:
            res = jax.device_put(pickle.load(fr), jax.devices("cpu")[0])
    else:
        print(f'checkpoint {checkpoint}')
        no_s, no_w = to_open(dir=data_save_path, checkpoint=checkpoint-1)
        filename = data_save_path + fr'NetworkSS_1mcmc_p629_w{no_w}_s{no_s}_{checkpoint-1}.sav'
        print(f'init from {filename}')
        with open(filename, 'rb') as fr:
            res = jax.device_put(pickle.load(fr), jax.devices("cpu")[0])

    #%%
    ## first element for p=10, second element for p=50
    fix_params=True
    mu_fixed=jnp.zeros((p,))
    scale_spike_fixed =0.003
    fixed_params_dict = {"scale_spike":scale_spike_fixed, "mu":mu_fixed}
    blocked_params_list = ["scale_spike", "mu"]

    eta0_0_m=0. 
    eta0_0_s=0.145
    eta0_coefs_m=0.
    eta0_coefs_s=0.145

    eta1_0_m=-2.197
    eta1_0_s=0.661
    eta1_coefs_m=0.
    eta1_coefs_s=0.661

    eta2_0_m=-9.368
    eta2_0_s=4.184
    eta2_coefs_m=0.
    eta2_coefs_s=4.184
    #%%

    # init strategy
    my_init_strategy = init_to_value(values={'rho_tilde':jax.device_put(res['rho_tilde'][-1], gpus[0]), 
                                                'u':jax.device_put(res['u'][-1], gpus[0]),
                                                'mu':mu_fixed, 
                                                'sqrt_diag':jax.device_put(res['sqrt_diag'][-1], gpus[0]), 
                                                'tilde_eta0_0':jax.device_put(res['tilde_eta0_0'][-1], gpus[0]),
                                                'tilde_eta1_0':jax.device_put(res['tilde_eta1_0'][-1], gpus[0]),
                                                'tilde_eta2_0':jax.device_put(res['tilde_eta2_0'][-1], gpus[0]),                                     
                                                'tilde_eta0_coefs':jax.device_put(res['tilde_eta0_coefs'][-1], gpus[0]),
                                                'tilde_eta1_coefs':jax.device_put(res['tilde_eta1_coefs'][-1], gpus[0]),
                                                'tilde_eta2_coefs':jax.device_put(res['tilde_eta2_coefs'][-1], gpus[0]),})




    A_list = [geo_clean, sci_clean]
    my_model_args = {"A_list":A_list, "eta0_0_m":eta0_0_m, "eta0_0_s":eta0_0_s, 
                "eta0_coefs_m":eta0_coefs_m, "eta0_coefs_s":eta0_coefs_s,
                "eta1_0_m":eta1_0_m, "eta1_0_s":eta1_0_s, 
                "eta1_coefs_m":eta1_coefs_m, "eta1_coefs_s":eta1_coefs_s,
                "eta2_0_m":eta2_0_m, "eta2_0_s":eta2_0_s, 
                "eta2_coefs_m":eta2_coefs_m, "eta2_coefs_s":eta2_coefs_s,
                "mu_m":mu_m, "mu_s":mu_s} 

    #%%
    # run model
    if fix_params:
        my_model_run = block(condition(my_model, fixed_params_dict), hide=blocked_params_list)
    else:
        my_model_run = my_model
    #%%    

    nuts_kernel = NUTS(my_model_run, init_strategy=my_init_strategy, dense_mass=is_dense)
    mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=batch)
    mcmc.run(rng_key = Key(8), Y=covid_vals, **my_model_args,
            extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))
    # for b in range(n_batches-1):
    #     sample_batch = mcmc.get_samples()
    #     mcmc.post_warmup_state = mcmc.last_state
    #     mcmc.run(mcmc.post_warmup_state.rng_key, Y=covid_vals, **my_model_args,
    #         extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))  # or mcmc.run(random.PRNGKey(1))


    # %%git 

    s = jax.device_put(mcmc.get_samples(), cpus[0])
    with open(data_save_path + f'NetworkSS_1mcmc_p{629}_w{n_warmup}_s{n_samples}_{checkpoint}.sav' , 'wb') as f:
        pickle.dump((s), f)
    purge(dir=data_save_path, checkpoint=checkpoint, curr_no_s=n_samples)
