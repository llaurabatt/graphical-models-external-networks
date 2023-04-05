#%%
# """Main script for training the model."""
import debugpy
debugpy.listen(5678)
print('Waiting for debugger')
debugpy.wait_for_client()
print('Debugger attached')
#%%
# imports
import sys
import os
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
cpus = jax.devices("cpu")
gpus = jax.devices("gpu")

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

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
#%%
################## MAP ###############
hyperpars = ['eta0_0', 'eta0_coefs', 'eta1_0', 'eta1_coefs', 'eta2_0', 'eta2_coefs']

# Empircal Bayes marginal MAP estimates

best_params = {'eta0_0':{'bandwidth': 0.1, 'kernel': 'linear'}, 
                       'eta0_coefs':{'bandwidth': 0.1, 'kernel': 'linear'},
                       'eta1_0':{'bandwidth': 0.3088843596477481, 'kernel': 'linear'}, 
                       'eta1_coefs':{'bandwidth': 0.13257113655901093, 'kernel': 'linear'}, 
                       'eta2_0':{'bandwidth': 1.2648552168552958, 'kernel': 'linear'}, 
                       'eta2_coefs':{'bandwidth': 0.2559547922699536, 'kernel': 'gaussian'}}

#x_ranges = {'eta0_0':np.linspace(-1, 1, 1000), 'eta0_coefs':np.linspace(-1, 1, 1000),
x_ranges = {'eta0_0':np.linspace(-0.2, 0.2, 1000), 'eta0_coefs':np.linspace(-0.2, 0.2, 1000),
           'eta1_0':np.linspace(-7, -2, 1000), 'eta1_coefs':np.linspace(-2, 1, 1000),
           'eta2_0':np.linspace(-10, 7, 1000), 'eta2_coefs':np.linspace(-3, 7, 1000)}
#%%         
etas_MAPs = {'eta0_0':0, 
                       'eta0_coefs':0,
                       'eta1_0':0, 
                       'eta1_coefs':0, 
                       'eta2_0':0, 
                       'eta2_coefs':0}

with open(data_save_path + f'NetworkSS_1mcmc_p629_s1500.sav', 'rb') as fr:
    res = jax.device_put(pickle.load(fr), jax.devices("cpu")[0])
    # res = pickle.load(fr)
#%% 
for par in hyperpars:
    if 'coefs' in par:
        samples = res[par][:,0].flatten()

        kde = KernelDensity(**best_params[par])
        kde.fit(samples[:, None])

        logdensity = kde.score_samples(x_ranges[par][:, None])
        density = jnp.exp(logdensity)
        MAP_1 = x_ranges[par][jnp.argmax(density)]
        post_mean_1 = samples.mean()
        print(f'{par} A1: MAP {MAP_1}, post. mean {post_mean_1}')

        ############
        samples = res[par][:,1].flatten()

        kde = KernelDensity(**best_params[par])
        kde.fit(samples[:, None])

        logdensity = kde.score_samples(x_ranges[par][:, None])
        density = jnp.exp(logdensity)
        MAP_2 = x_ranges[par][jnp.argmax(density)]
        post_mean_2 = samples.mean()
        etas_MAPs[par] = jnp.hstack([MAP_1, MAP_2])
        print(f'{par} A2: MAP {MAP_2}, post. mean {post_mean_2}')
    
    else:

        samples = res[par].flatten()

        kde = KernelDensity(**best_params[par])
        kde.fit(samples[:, None])

        logdensity = kde.score_samples(x_ranges[par][:, None])
        density = jnp.exp(logdensity)
        MAP = x_ranges[par][jnp.argmax(density)]
        post_mean = samples.mean()
        etas_MAPs[par] = MAP
        print(f'{par} P: MAP {MAP}, post. mean {post_mean}')

#%%
################# 2mcmc ##############


def SVI_init_strategy_golazo_ss(A_list, mcmc_res, fixed_params_dict):

    all_chains = jnp.hstack([mcmc_res['eta0_0'][:,None], 
                             mcmc_res['eta0_coefs'],
                             mcmc_res['eta1_0'][:,None], 
                             mcmc_res['eta1_coefs'],
                            mcmc_res['eta2_0'][:,None], 
                             mcmc_res['eta2_coefs']])

    eta0_0_MAP = fixed_params_dict["eta0_0"]
    eta0_coefs_MAP = fixed_params_dict["eta0_coefs"]
    eta1_0_MAP = fixed_params_dict["eta1_0"]
    eta1_coefs_MAP = fixed_params_dict["eta1_coefs"]
    eta2_0_MAP = fixed_params_dict["eta2_0"]
    eta2_coefs_MAP = fixed_params_dict["eta2_coefs"]
    my_MAP = jnp.hstack([jnp.array([eta0_0_MAP]),eta0_coefs_MAP, 
                         jnp.array([eta1_0_MAP]),eta1_coefs_MAP,
                        jnp.array([eta2_0_MAP]),eta2_coefs_MAP])
    
    dists = my_utils.abs_dist(vec=my_MAP, mat=all_chains)
    rho_lt_init = mcmc_res['rho_lt'][jnp.argmin(dists)]
    
    p = A_list[0].shape[0]
    tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
    A_tril_arr = jnp.array([A[tril_idx] for A in A_list]) # (a, p, p)
    A_tril_mean0_MAP = 0.
    for coef, A in zip(eta0_coefs_MAP.T,A_tril_arr):
        A_tril_mean0_MAP += coef*A
    
    A_tril_mean1_MAP = 0.
    for coef, A in zip(eta1_coefs_MAP.T,A_tril_arr):
        A_tril_mean1_MAP += coef*A
        
    A_tril_mean2_MAP = 0.
    for coef, A in zip(eta2_coefs_MAP.T,A_tril_arr):
        A_tril_mean2_MAP += coef*A

    mean_slab = eta0_0_MAP+A_tril_mean0_MAP
    scale_slab = fixed_params_dict["scale_spike"]*(1+jnp.exp(-eta1_0_MAP-A_tril_mean1_MAP))
    w_slab = (1+jnp.exp(-eta2_0_MAP-A_tril_mean2_MAP))**(-1)
    
    u_init = mcmc_res['u'][jnp.argmin(dists)]
    is_spike = my_utils.my_sigmoid(u_init, beta=100., alpha=w_slab)

    rho_tilde_init = (rho_lt_init-mean_slab*(1-is_spike))/(is_spike*scale_spike_fixed + (1-is_spike)*scale_slab)
    sqrt_diag_init = mcmc_res['sqrt_diag'][jnp.argmin(dists)]

    rho_lt_MAP = rho_tilde_init*is_spike*scale_spike_fixed + (1-is_spike)*(rho_tilde_init*scale_slab + mean_slab)
    rho_mat_tril = jnp.zeros((p,p))
    rho_mat_tril = rho_mat_tril.at[tril_idx].set(rho_lt_MAP)
    rho = rho_mat_tril + rho_mat_tril.T + jnp.identity(p)
    rho_cpu = jax.device_put(rho, jax.devices("cpu")[0])
    enter = jax.device_put(jnp.all(jnp.linalg.eigvals(rho_cpu) > 0), jax.devices("cpu")[0])
    print(f'Is rho init p.d.:{enter}')
    
    return u_init, rho_tilde_init, sqrt_diag_init
#%%
# load data 
# covid_vals = jax.device_put(jnp.array(pd.read_csv(data_path + 'COVID_629_meta.csv', index_col='Unnamed: 0').values), jax.devices("cpu")[0])
covid_vals = jnp.array(pd.read_csv(data_path + 'COVID_629_meta.csv', index_col='Unnamed: 0').values)
geo_clean = jnp.array(jnp.load(data_path + 'GEO_clean_629.npy'))
sci_clean = jnp.array(jnp.load(data_path + 'SCI_clean_629.npy'))
#%%
n,p = covid_vals.shape
print(f"NetworkSS, n {n} and p {p}")
#%%

#%%
## params
n_warmup = 1000
n_samples = 500
n_batches = 1
batch = int(n_samples/n_batches)

my_model = models.NetworkSS_repr_etaRepr
is_dense=False
#%%
mu_fixed=jnp.zeros((p,))
mu_m=0.
mu_s=1.

scale_spike_fixed =0.003

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
fix_params = True
fixed_params_dict = {"mu":mu_fixed, "scale_spike":scale_spike_fixed,
                     "eta0_0":etas_MAPs["eta0_0"], 
                     "eta0_coefs":jnp.array(etas_MAPs["eta0_coefs"]),
                    "eta1_0":etas_MAPs["eta1_0"], 
                     "eta1_coefs":jnp.array(etas_MAPs["eta1_coefs"]),
                    "eta2_0":etas_MAPs["eta2_0"], 
                     "eta2_coefs":jnp.array(etas_MAPs["eta2_coefs"])}

blocked_params_list = ["mu", "scale_spike", "eta0_0", "eta0_coefs", "eta1_0", "eta1_coefs", 
                       "eta2_0", "eta2_coefs"]

A_list = [geo_clean, sci_clean]
my_model_args = {"A_list":A_list, "eta0_0_m":eta0_0_m, "eta0_0_s":eta0_0_s, 
         "eta0_coefs_m":eta0_coefs_m, "eta0_coefs_s":eta0_coefs_s,
         "eta1_0_m":eta1_0_m, "eta1_0_s":eta1_0_s, 
         "eta1_coefs_m":eta1_coefs_m, "eta1_coefs_s":eta1_coefs_s,
         "eta2_0_m":eta2_0_m, "eta2_0_s":eta2_0_s, 
         "eta2_coefs_m":eta2_coefs_m, "eta2_coefs_s":eta2_coefs_s,
         "mu_m":mu_m, "mu_s":mu_s} 
#%%
u_init_cpu, rho_tilde_init_cpu, sqrt_diag_init_cpu = SVI_init_strategy_golazo_ss(A_list=A_list, mcmc_res=res, 
                                                                 fixed_params_dict=fixed_params_dict)

u_init = jax.device_put(u_init_cpu, jax.devices("gpu")[0])
rho_tilde_init = jax.device_put(rho_tilde_init_cpu, jax.devices("gpu")[0])
sqrt_diag_init = jax.device_put(sqrt_diag_init_cpu, jax.devices("gpu")[0])
#%%
my_init_strategy = init_to_value(values={'u':u_init, 'rho_tilde':rho_tilde_init,'sqrt_diag':sqrt_diag_init})

#%%
# run model
if fix_params:
    my_model_run = block(condition(my_model, fixed_params_dict), hide=blocked_params_list)
else:
    my_model_run = my_model
#%%    

nuts_kernel = NUTS(my_model_run, init_strategy=my_init_strategy, dense_mass=is_dense)
mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=batch)
mcmc.run(rng_key = Key(5), Y=covid_vals, **my_model_args,
        extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))
# for b in range(n_batches-1):
#     sample_batch = mcmc.get_samples()
#     mcmc.post_warmup_state = mcmc.last_state
#     mcmc.run(mcmc.post_warmup_state.rng_key, Y=covid_vals, **my_model_args,
#         extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))  # or mcmc.run(random.PRNGKey(1))


# %%git 


s = jax.device_put(mcmc.get_samples(), cpus[0])
f_dict = jax.device_put(fixed_params_dict, cpus[0])
s.update({'fixed_params_dict':f_dict})
with open(data_save_path + f'NetworkSS_2mcmc_p{p}_s{n_warmup+n_samples}.sav' , 'wb') as f:
    pickle.dump((s), f)