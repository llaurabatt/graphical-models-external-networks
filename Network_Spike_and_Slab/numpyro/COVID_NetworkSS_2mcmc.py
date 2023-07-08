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
import re
from absl import flags
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import jax
import numpyro
numpyro.set_platform('cpu')
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
cpus = gpus = jax.devices("cpu")
# gpus = jax.devices("gpu")

# paths
# _ROOT_DIR = "/home/user/graphical-models-external-networks/"
_ROOT_DIR = "/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/graphical-models-external-networks/"
os.chdir(_ROOT_DIR)
# sys.path.append("/home/user/graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")
sys.path.append("/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/COVID/Pre-processed Data/'
# data_save_path = '/home/user/mounted_folder/NetworkSS_results/'
data_save_path = '/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/NetworkSS_results/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)
#%%
# load models and functions
import models
import my_utils

# define flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('search_MAP_best_params', False, 'If true, it will optimise kernel and bandwith for KDE on etas.')
flags.DEFINE_integer('thinning', None, 'Thinning between MCMC samples.')
flags.DEFINE_integer('n_samples', 1000, 'Number of total samples to run (excluding warmup).')
flags.DEFINE_string('model', 'models.NetworkSS_repr_etaRepr_loglikRepr', 'Name of model to be run.')
flags.DEFINE_string('Y', 'COVID_629_meta.csv', 'Name of file where data for dependent variable is stored.')
flags.DEFINE_multi_string('network_list', ['GEO_clean_629.npy', 'SCI_clean_629.npy'], 'Name of file where network data is stored. Flag can be called multiple times. Order of calling IS relevant.')
flags.mark_flags_as_required(['thinning'])
FLAGS(sys.argv)

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
#%%
# load data 
my_model = eval(FLAGS.model)
search_MAP_best_params = FLAGS.search_MAP_best_params
n_samples_2mcmc = FLAGS.n_samples
thinning = FLAGS.thinning
covid_vals_name = FLAGS.Y
network_names = FLAGS.network_list
print(network_names)
# load data
covid_vals = jnp.array(pd.read_csv(data_path + covid_vals_name, index_col='Unnamed: 0').values)
geo_clean = jnp.array(jnp.load(data_path + network_names[0]))
sci_clean = jnp.array(jnp.load(data_path + network_names[1]))

covid_vals = covid_vals[:,:100].copy()
geo_clean = geo_clean[:100, :100].copy()
sci_clean = sci_clean[:100, :100].copy()
A_list = [geo_clean, sci_clean]

# # covid_vals = jax.device_put(jnp.array(pd.read_csv(data_path + 'COVID_629_meta.csv', index_col='Unnamed: 0').values), jax.devices("cpu")[0])

#%%
n,p = covid_vals.shape
print(f"NetworkSS, n {n} and p {p}")
#%%
################## MAP ###############
for f in os.listdir(data_save_path):
    if 'aggregate.sav' in f: 
        print(f'Init 2mcmc from {f}')
        with open(data_save_path + f, 'rb') as fr:
            res = jax.device_put(pickle.load(fr), cpus[0])

#%% 
# Empircal Bayes marginal MAP estimates

hyperpars = ['eta0_0', 'eta0_coefs', 'eta1_0', 'eta1_coefs', 'eta2_0', 'eta2_coefs']

if search_MAP_best_params:
    print('Search for MAP best params...')
    bandwidths = 10 ** np.linspace(-1, 1, 50)
    kernels = ['gaussian', 'exponential', 'linear',]

    best_params = {}
    for par in hyperpars:
        print(par)
        if 'coefs' in par:
            my_kern = KernelDensity()
            grid_model = GridSearchCV(my_kern, {'bandwidth': bandwidths, 'kernel':kernels})
            grid_model.fit(res[par][:,0][:,None])
            best_params[par + '_0'] = grid_model.best_params_

            my_kern = KernelDensity()
            grid_model = GridSearchCV(my_kern, {'bandwidth': bandwidths, 'kernel':kernels})
            grid_model.fit(res[par][:,1][:,None])
            best_params[par + '_1'] = grid_model.best_params_
        else:
            my_kern = KernelDensity()
            grid_model = GridSearchCV(my_kern, {'bandwidth': bandwidths, 'kernel':kernels})
            grid_model.fit(res[par])
            best_params[par] = grid_model.best_params_
    with open(data_save_path + f'MAP_best_params_p{p}.sav' , 'wb') as f:
        pickle.dump((best_params), f)
            
else:
    try:
        with open(data_save_path + 'MAP_best_params_p{p}.sav', 'rb') as fr:
            best_params = pickle.load(fr)
    except:
        best_params = {'eta0_0':{'bandwidth': 0.1, 'kernel': 'linear'}, 
                        'eta0_coefs_0':{'bandwidth': 0.1, 'kernel': 'linear'},
                        'eta0_coefs_1':{'bandwidth': 0.1, 'kernel': 'linear'},
                        'eta1_0':{'bandwidth': 0.3088843596477481, 'kernel': 'linear'}, 
                        'eta1_coefs_0':{'bandwidth': 0.13257113655901093, 'kernel': 'linear'}, 
                        'eta1_coefs_1':{'bandwidth': 0.13257113655901093, 'kernel': 'linear'}, 
                        'eta2_0':{'bandwidth': 1.2648552168552958, 'kernel': 'linear'}, 
                        'eta2_coefs_0':{'bandwidth': 0.2559547922699536, 'kernel': 'gaussian'},
                        'eta2_coefs_1':{'bandwidth': 0.2559547922699536, 'kernel': 'gaussian'}}

x_ranges = {'eta0_0':np.linspace(-10, 1, 10000), 'eta0_coefs':np.linspace(-4, 5, 10000),
           'eta1_0':np.linspace(-10, 5, 10000), 'eta1_coefs':np.linspace(-5, 5, 10000),
           'eta2_0':np.linspace(-20, 0, 10000), 'eta2_coefs':np.linspace(-5, 15, 10000)}
#%%         
etas_MAPs = {k:0 for k in hyperpars}


for par in hyperpars:
    if 'coefs' in par:
        samples = res[par][:,0].flatten()

        kde = KernelDensity(**best_params[par + '_0'])
        kde.fit(samples[:, None])

        logdensity = kde.score_samples(x_ranges[par][:, None])
        density = jnp.exp(logdensity)
        MAP_1 = x_ranges[par][jnp.argmax(density)]
        post_mean_1 = samples.mean()
        print(f'{par} A1: MAP {MAP_1}, post. mean {post_mean_1}')

        ############
        samples = res[par][:,1].flatten()

        kde = KernelDensity(**best_params[par+ '_1'])
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

    all_chains = jnp.hstack([mcmc_res['eta0_0'], #[:,None], 
                             mcmc_res['eta0_coefs'],
                             mcmc_res['eta1_0'], #[:,None], 
                             mcmc_res['eta1_coefs'],
                             mcmc_res['eta2_0'], #[:,None], 
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
    u_init = jnp.array(u_init, dtype=jnp.float64)
    is_spike = my_utils.my_sigmoid(u_init, beta=100., alpha=w_slab)

    rho_tilde_init = (rho_lt_init-mean_slab*(1-is_spike))/(is_spike*scale_spike_fixed + (1-is_spike)*scale_slab)
    sqrt_diag_init = mcmc_res['sqrt_diag'][jnp.argmin(dists)]
    sqrt_diag_init = jnp.array(sqrt_diag_init, dtype=jnp.float64)

    rho_lt_MAP = rho_tilde_init*is_spike*scale_spike_fixed + (1-is_spike)*(rho_tilde_init*scale_slab + mean_slab)
    rho_mat_tril = jnp.zeros((p,p))
    rho_mat_tril = rho_mat_tril.at[tril_idx].set(rho_lt_MAP)
    rho = rho_mat_tril + rho_mat_tril.T + jnp.identity(p)
    rho_cpu = jax.device_put(rho, cpus[0])
    enter = jax.device_put(jnp.all(jnp.linalg.eigvals(rho_cpu) > 0), cpus[0])
    print(f'Is rho init p.d.:{enter}')
    
    return u_init, rho_tilde_init, sqrt_diag_init


#%%
## params
n_warmup = 1000
n_samples = n_samples_2mcmc
n_batches = 1
batch = int(n_samples/n_batches)

# my_model = models.NetworkSS_repr_etaRepr
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

if my_model == models.NetworkSS_repr_etaRepr_loglikRepr:
    y_bar = covid_vals.mean(axis=0) #p
    S_bar = covid_vals.T@covid_vals/n - jnp.outer(y_bar, y_bar) #(p,p)
    my_model_args.update({"y_bar":y_bar, "S_bar":S_bar, "n":n, "p":p,})
elif my_model == models.NetworkSS_repr_etaRepr:
    my_model_args.update({"Y":covid_vals,})
#%%
u_init_cpu, rho_tilde_init_cpu, sqrt_diag_init_cpu = SVI_init_strategy_golazo_ss(A_list=A_list, mcmc_res=res, 
                                                                 fixed_params_dict=fixed_params_dict)

u_init = jax.device_put(u_init_cpu, gpus[0])
rho_tilde_init = jax.device_put(rho_tilde_init_cpu, gpus[0])
sqrt_diag_init = jax.device_put(sqrt_diag_init_cpu, gpus[0])
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
mcmc.run(rng_key = Key(5), **my_model_args,
        extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))
# for b in range(n_batches-1):
#     sample_batch = mcmc.get_samples()
#     mcmc.post_warmup_state = mcmc.last_state
#     mcmc.run(mcmc.post_warmup_state.rng_key, Y=covid_vals, **my_model_args,
#         extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))  # or mcmc.run(random.PRNGKey(1))


# %%git 


mask = (jnp.arange(n_samples)%thinning==0)
s = jax.device_put(mcmc.get_samples(), cpus[0])
# why doesn't the following work with dictionary comprehension?
ss = {}
for k,v in s.items():
    ss[k] = v[mask]
f_dict = jax.device_put(fixed_params_dict, cpus[0])
ss.update({'fixed_params_dict':f_dict})
with open(data_save_path + f'NetworkSS_2mcmc_p{p}_w{n_warmup}_s{n_samples}.sav' , 'wb') as f:
    pickle.dump((ss), f)