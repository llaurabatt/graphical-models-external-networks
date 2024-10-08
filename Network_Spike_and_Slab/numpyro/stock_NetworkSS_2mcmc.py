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
#%%
cpus = jax.devices("cpu")
gpus = jax.devices("gpu")

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
flags.DEFINE_boolean('search_MAP_best_params', False, 'If true, it will optimise kernel and bandwith for KDE on etas.')
flags.DEFINE_integer('thinning', None, 'Thinning between MCMC samples.')
flags.DEFINE_integer('SEED', None, 'Random seed.')
flags.DEFINE_string('data_save_path', None, 'Path for saving results.')
flags.DEFINE_integer('n_samples', 2000, 'Number of total samples to run (excluding warmup).')
flags.DEFINE_string('model', 'models.NetworkSS_repr_etaRepr_loglikRepr', 'Name of model to be run.')
flags.DEFINE_string('Y', 'stock_trans_SP.csv', 'Name of file where data for dependent variable is stored.')
flags.DEFINE_multi_string('network_list', ['E_pears_clean_SP_366.npy', 'P_pears_clean_SP_366.npy'], 'Name of file where network data is stored. Flag can be called multiple times. Order of calling IS relevant.')
flags.mark_flags_as_required(['thinning', 'SEED', 'data_save_path'])
FLAGS(sys.argv)

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
#%%
# load data 
my_model = eval(FLAGS.model)
search_MAP_best_params = FLAGS.search_MAP_best_params
n_samples_2mcmc = FLAGS.n_samples
thinning = FLAGS.thinning
stock_vals_name = FLAGS.Y
network_names = FLAGS.network_list
SEED = FLAGS.SEED
print(network_names)
print(FLAGS.model)

data_save_path = FLAGS.data_save_path
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)
print(f'Save in {data_save_path}')

# load data
my_vals = jnp.array(pd.read_csv(data_path + stock_vals_name, index_col='Unnamed: 0').values)
E_clean = jnp.array(jnp.load(data_path + network_names[0]))
P_clean = jnp.array(jnp.load(data_path + network_names[1]))

# stock_vals = stock_vals[:,:100].copy()
# E_clean = E_clean[:100, :100].copy()
# P_clean = P_clean[:100, :100].copy()
A_list = [E_clean, P_clean]

# stock_vals = jax.device_put(jnp.array(pd.read_csv(data_path + 'stock_trans_SP.csv', index_col='Unnamed: 0').values), jax.devices("cpu")[0])

#%%
n,p = my_vals.shape
n_nets = len(A_list)
print(f"NetworkSS, n {n}, p {p}, number of networks {n_nets}")
#%%
# get init file

CPs=[]
for f in os.listdir(data_save_path):
    # if 'aggregate.sav' in f: 
    if '1mcmc' in f: 
        if re.search(r'(_CP)\d+', f):
            CP = int(re.search(r'(_CP)\d+', f)[0][3:])
            CPs.append(CP)
try:
    CP_max = max(CPs)
except:
    pass

for f in os.listdir(data_save_path):
    if 'Merge' in f:
        print(f'Init 2mcmc from {f}')
        with open(data_save_path + f, 'rb') as fr:
                res = jax.device_put(pickle.load(fr), cpus[0])
    else:
        if '1mcmc' in f: 
            if re.search(fr'.*_CP{CP_max}\.sav$', f):
                print(f'Init 2mcmc from {f}')
                with open(data_save_path + f, 'rb') as fr:
                    res = jax.device_put(pickle.load(fr), cpus[0])

################## MAP ###############

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
            for net_ix in range(n_nets):
                my_kern = KernelDensity()
                grid_model = GridSearchCV(my_kern, {'bandwidth': bandwidths, 'kernel':kernels})
                arr = np.array(res[par])
                grid_model.fit(arr[:,net_ix][:,None])
                best_params[par + f'_{net_ix}'] = grid_model.best_params_

                # my_kern = KernelDensity()
                # grid_model = GridSearchCV(my_kern, {'bandwidth': bandwidths, 'kernel':kernels})
                # grid_model.fit(res[par][:,0][:,None])
                # best_params[par + '_0'] = grid_model.best_params_

                # my_kern = KernelDensity()
                # grid_model = GridSearchCV(my_kern, {'bandwidth': bandwidths, 'kernel':kernels})
                # grid_model.fit(res[par][:,1][:,None])
                # best_params[par + '_1'] = grid_model.best_params_
        else:
            my_kern = KernelDensity()
            grid_model = GridSearchCV(my_kern, {'bandwidth': bandwidths, 'kernel':kernels})
            try:
                _, _ = res[par].shape
                arr = np.array(res[par])
                grid_model.fit(arr)
            except:
                arr = np.array(res[par])
                grid_model.fit(arr[:,None])
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
                        'eta0_coefs_2':{'bandwidth': 0.1, 'kernel': 'linear'},
                        'eta1_0':{'bandwidth': 0.3088843596477481, 'kernel': 'linear'}, 
                        'eta1_coefs_0':{'bandwidth': 0.13257113655901093, 'kernel': 'linear'}, 
                        'eta1_coefs_1':{'bandwidth': 0.13257113655901093, 'kernel': 'linear'},
                        'eta1_coefs_2':{'bandwidth': 0.13257113655901093, 'kernel': 'linear'}, 
                        'eta2_0':{'bandwidth': 1.2648552168552958, 'kernel': 'linear'}, 
                        'eta2_coefs_0':{'bandwidth': 0.2559547922699536, 'kernel': 'gaussian'},
                        'eta2_coefs_1':{'bandwidth': 0.2559547922699536, 'kernel': 'gaussian'},
                        'eta2_coefs_2':{'bandwidth': 0.2559547922699536, 'kernel': 'gaussian'}}

x_ranges = {'eta0_0':np.linspace(-10, 1, 10000), 'eta0_coefs':np.linspace(-6, 5, 10000),
           'eta1_0':np.linspace(-10, 5, 10000), 'eta1_coefs':np.linspace(-5, 5, 10000),
           'eta2_0':np.linspace(-35, 40, 10000), 'eta2_coefs':np.linspace(-7, 15, 10000)}
#%%         
etas_MAPs = {k:0 for k in hyperpars}


for par in hyperpars:
    if 'coefs' in par:
        maps = []
        for net_ix in range(n_nets):
            arr = np.array(res[par])
            samples = arr[:,net_ix].flatten()

            kde = KernelDensity(**best_params[par + f'_{net_ix}'])
            kde.fit(samples[:, None])

            logdensity = kde.score_samples(x_ranges[par][:, None])
            density = jnp.exp(logdensity)
            MAP = x_ranges[par][jnp.argmax(density)]
            post_mean = samples.mean()
            print(f'{par} A{net_ix}: MAP {MAP}, post. mean {post_mean}')
            maps.append(MAP)
        
        etas_MAPs[par] = jnp.hstack(maps)
            # samples = res[par][:,0].flatten()

            # kde = KernelDensity(**best_params[par + '_0'])
            # kde.fit(samples[:, None])

            # logdensity = kde.score_samples(x_ranges[par][:, None])
            # density = jnp.exp(logdensity)
            # MAP_1 = x_ranges[par][jnp.argmax(density)]
            # post_mean_1 = samples.mean()
            # print(f'{par} A1: MAP {MAP_1}, post. mean {post_mean_1}')

            # ############
            # samples = res[par][:,1].flatten()

            # kde = KernelDensity(**best_params[par+ '_1'])
            # kde.fit(samples[:, None])

            # logdensity = kde.score_samples(x_ranges[par][:, None])
            # density = jnp.exp(logdensity)
            # MAP_2 = x_ranges[par][jnp.argmax(density)]
            # post_mean_2 = samples.mean()
            # etas_MAPs[par] = jnp.hstack([MAP_1, MAP_2])
            # print(f'{par} A2: MAP {MAP_2}, post. mean {post_mean_2}')
        
    else:
        arr = np.array(res[par])
        samples = arr.flatten()

        kde = KernelDensity(**best_params[par])
        kde.fit(samples[:, None])

        logdensity = kde.score_samples(x_ranges[par][:, None])
        density = jnp.exp(logdensity)
        MAP = x_ranges[par][jnp.argmax(density)]
        post_mean = samples.mean()
        etas_MAPs[par] = MAP
        print(f'{par} : MAP {MAP}, post. mean {post_mean}')

#%%
################# 2mcmc ##############


def SVI_init_strategy_golazo_ss(A_list, mcmc_res, fixed_params_dict):

    try:
        all_chains = jnp.hstack([mcmc_res['eta0_0'][:,None], 
                                mcmc_res['eta0_coefs'],
                                mcmc_res['eta1_0'][:,None], 
                                mcmc_res['eta1_coefs'],
                                mcmc_res['eta2_0'][:,None], 
                                mcmc_res['eta2_coefs']])
    except:
        all_chains = jnp.hstack([mcmc_res['eta0_0'], 
                             mcmc_res['eta0_coefs'],
                             mcmc_res['eta1_0'], 
                             mcmc_res['eta1_coefs'],
                             mcmc_res['eta2_0'], 
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

scale_spike_fixed =0.0033341

#%%
covid_mcmc_args = {"A_list":A_list, 
                "eta0_0_m":0., "eta0_0_s":0.0015864, 
        "eta0_coefs_m":0., "eta0_coefs_s":0.0015864,
        "eta1_0_m":-2.1972246, "eta1_0_s":0.3, 
        "eta1_coefs_m":0., "eta1_coefs_s":0.3,
        "eta2_0_m":-7.7894737, "eta2_0_s":1.0263158, 
        "eta2_coefs_m":0., "eta2_coefs_s":1.0263158,
        "mu_m":0., "mu_s":1.} 
my_model_args = covid_mcmc_args

# my_model_args = {"A_list":A_list, 
#                 "eta0_0_m":0., "eta0_0_s":0.0021276, 
#         "eta0_coefs_m":0., "eta0_coefs_s":0.0021276,
#         "eta1_0_m":-2.1972246, "eta1_0_s":0.35, 
#         "eta1_coefs_m":0., "eta1_coefs_s":0.35,
#         "eta2_0_m":-10.1578947, "eta2_0_s":1.8157895, 
#         "eta2_coefs_m":0., "eta2_coefs_s":1.8157895,
#         "mu_m":0., "mu_s":1.} 



fix_params = True
fixed_params_dict = {"mu":mu_fixed, "scale_spike":scale_spike_fixed,
                     "eta0_0":etas_MAPs["eta0_0"], 
                     "eta0_coefs":jnp.array(etas_MAPs["eta0_coefs"]),
                    "eta1_0":etas_MAPs["eta1_0"], 
                     "eta1_coefs":jnp.array(etas_MAPs["eta1_coefs"]),
                    "eta2_0":etas_MAPs["eta2_0"], 
                     "eta2_coefs":jnp.array(etas_MAPs["eta2_coefs"]),
                      "tilde_eta0_0":(etas_MAPs["eta0_0"]-my_model_args["eta0_0_m"])*(jnp.sqrt((p*(p-1)/2.0)/n)/my_model_args["eta0_0_s"]), 
                     "tilde_eta0_coefs":(jnp.array(etas_MAPs["eta0_coefs"])-my_model_args["eta0_coefs_m"])*(jnp.sqrt((p*(p-1)/2.0)/n)/my_model_args["eta0_coefs_s"]),
                    "tilde_eta1_0":(etas_MAPs["eta1_0"]-my_model_args["eta1_0_m"])*(jnp.sqrt((p*(p-1)/2.0)/n)/my_model_args["eta1_0_s"]), 
                     "tilde_eta1_coefs":(jnp.array(etas_MAPs["eta1_coefs"])-my_model_args["eta1_coefs_m"])*(jnp.sqrt((p*(p-1)/2.0)/n)/my_model_args["eta1_coefs_s"]),
                    "tilde_eta2_0":(etas_MAPs["eta2_0"]-my_model_args["eta2_0_m"])*(jnp.sqrt((p*(p-1)/2.0)/n)/my_model_args["eta2_0_s"]), 
                     "tilde_eta2_coefs":(jnp.array(etas_MAPs["eta2_coefs"])-my_model_args["eta2_coefs_m"])*(jnp.sqrt((p*(p-1)/2.0)/n)/my_model_args["eta2_coefs_s"])}

blocked_params_list = ["mu", "scale_spike", 
                       "tilde_eta0_0", "tilde_eta0_coefs", "tilde_eta1_0", "tilde_eta1_coefs", 
                       "tilde_eta2_0", "tilde_eta2_coefs",
                       "eta0_0", "eta0_coefs", "eta1_0", "eta1_coefs", 
                       "eta2_0", "eta2_coefs"]




if ((my_model == models.NetworkSS_repr_etaRepr_loglikRepr)|(my_model == models.NetworkSS_repr_loglikRepr)):
    y_bar = my_vals.mean(axis=0) #p
    S_bar = my_vals.T@my_vals/n - jnp.outer(y_bar, y_bar) #(p,p)
    my_model_args.update({"y_bar":y_bar, "S_bar":S_bar, "n":n, "p":p,})
elif ((my_model == models.NetworkSS_repr_etaRepr)|(my_model == models.NetworkSS_repr)):
    my_model_args.update({"Y":my_vals, "n":n, "p":p,})
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
mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=batch, thinning=thinning)
mcmc.run(rng_key = Key(SEED), **my_model_args,
        extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))
# for b in range(n_batches-1):
#     sample_batch = mcmc.get_samples()
#     mcmc.post_warmup_state = mcmc.last_state
#     mcmc.run(mcmc.post_warmup_state.rng_key, Y=covid_vals, **my_model_args,
#         extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))  # or mcmc.run(random.PRNGKey(1))


# %%git 


# mask = (jnp.arange(n_samples)%thinning==0)
s = jax.device_put(mcmc.get_samples(), cpus[0])
# why doesn't the following work with dictionary comprehension?
# ss = {}
# for k,v in s.items():
#     ss[k] = v[mask]
f_dict = jax.device_put(fixed_params_dict, cpus[0])
s.update({'fixed_params_dict':f_dict})
with open(data_save_path + f'NetworkSS_2mcmc_p{p}_w{n_warmup}_s{n_samples}_seed{SEED}.sav' , 'wb') as f:
    pickle.dump((s), f)