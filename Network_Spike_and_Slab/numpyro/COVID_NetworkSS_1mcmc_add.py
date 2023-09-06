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
from typing import Optional

#%%
# paths
_ROOT_DIR = "/home/paperspace/"
os.chdir(_ROOT_DIR + 'graphical-models-external-networks/')
sys.path.append(_ROOT_DIR + "graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/COVID/Pre-processed Data/'
data_save_path = _ROOT_DIR + 'NetworkSS_results/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)

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

def get_init_file(dir, checkpoint):
    '''Retrieve init file'''
    CPs = []
    for f in os.listdir(dir):
        if re.search(r'(_CP)\d+', f):
            CP = int(re.search(r'(_CP)\d+', f)[0][3:])
            CPs.append(CP)
    CP_max = max(CPs)
    for f in os.listdir(dir):
        if re.search(fr'.*_CP{CP_max}\.sav$', f):
            return f, CP_max


#%%
def mcmc1_add(
        covid_vals,
        A_list,
        my_model,
        checkpoint, 
        n_warmup, 
        n_samples,
        thinning:Optional[int]=0,
        ):

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
    is_dense=False
    #%%
    print(f'Checkpoint {checkpoint}')
    filename, CP_int = get_init_file(dir=data_save_path, checkpoint=checkpoint)
    print(f'Init from {filename}')
    with open(data_save_path + filename, 'rb') as fr:
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




    # A_list = [geo_clean, sci_clean]
    my_model_args = {"A_list":A_list, "eta0_0_m":eta0_0_m, "eta0_0_s":eta0_0_s, 
                "eta0_coefs_m":eta0_coefs_m, "eta0_coefs_s":eta0_coefs_s,
                "eta1_0_m":eta1_0_m, "eta1_0_s":eta1_0_s, 
                "eta1_coefs_m":eta1_coefs_m, "eta1_coefs_s":eta1_coefs_s,
                "eta2_0_m":eta2_0_m, "eta2_0_s":eta2_0_s, 
                "eta2_coefs_m":eta2_coefs_m, "eta2_coefs_s":eta2_coefs_s,
                "mu_m":mu_m, "mu_s":mu_s} 
    if my_model == models.NetworkSS_repr_etaRepr_loglikRepr:
        y_bar = covid_vals.mean(axis=0) #p
        S_bar = covid_vals.T@covid_vals/n - jnp.outer(y_bar, y_bar) #(p,p)
        my_model_args.update({"y_bar":y_bar, "S_bar":S_bar, "n":n, "p":p,})
    elif my_model == models.NetworkSS_repr_etaRepr:
        my_model_args.update({"Y":covid_vals,})

    #%%
    # run model
    if fix_params:
        my_model_run = block(condition(my_model, fixed_params_dict), hide=blocked_params_list)
    else:
        my_model_run = my_model
    #%%    

    nuts_kernel = NUTS(my_model_run, init_strategy=my_init_strategy, dense_mass=is_dense)
    mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=batch)
    mcmc.run(rng_key = Key(8), **my_model_args,
            extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))
    # for b in range(n_batches-1):
    #     sample_batch = mcmc.get_samples()
    #     mcmc.post_warmup_state = mcmc.last_state
    #     mcmc.run(mcmc.post_warmup_state.rng_key, Y=covid_vals, **my_model_args,
    #         extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))  # or mcmc.run(random.PRNGKey(1))


    # %%

    mask = (jnp.arange(n_samples)%thinning==0)
    s = jax.device_put(mcmc.get_samples(), cpus[0])
    # why doesn't the following work with dictionary comprehension?
    ss = {}
    for k,v in s.items():
        ss[k] = v[mask]
    with open(data_save_path + f'NetworkSS_1mcmc_p{p}_w{n_warmup}_s{n_samples}_CP{checkpoint}.sav' , 'wb') as f:
        pickle.dump((ss), f)

