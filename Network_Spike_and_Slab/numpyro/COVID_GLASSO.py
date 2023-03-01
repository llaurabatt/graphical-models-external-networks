#%%
# imports
import debugpy
debugpy.listen(5678)
print('Waiting for debugger')
debugpy.wait_for_client()
print('Debugger attached')
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KernelDensity
# %%
import jax
import numpyro
numpyro.set_platform('gpu')
print(jax.lib.xla_bridge.get_backend().platform)

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax.random import PRNGKey as Key
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpyro.infer.util 
from numpyro.handlers import condition, block
from jax.random import PRNGKey as Key
from numpyro.util import enable_x64
from numpyro.infer import init_to_feasible, init_to_value
from optax import adam
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer import SVI, Trace_ELBO
import os,sys,humanize,psutil,GPUtil
import jax.profiler

# Define function
def mem_report():
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
  
#%%
# paths
_ROOT_DIR = "/home/user/graphical-models-external-networks/"
os.chdir(_ROOT_DIR)
sys.path.append("/home/user/graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/COVID/Pre-processed Data/'
data_save_path = './Network_Spike_and_Slab/numpyro/NetworkSS_results/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)
#%%
# load models and functions
import models
#%%
import my_utils
#%%
enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
#%%

TP_thresh = 3
#%%
def model_run(Y, my_model, my_model_args, fix_params,  estimates_print,
                key_no_run, TP_thresh=3, n_warmup=1000, n_samples=2000,
              algo='mcmc',  SVI_samples_low=10000, SVI_samples_high=300000,
              verbose=False, my_init_strategy=init_to_feasible, 
             fixed_params_dict=None, blocked_params_list=None, adam_start=0.005,  is_dense=False):

    res = {}
    n, p = Y.shape
    tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
    print(" ")
    print('----------------------------------------')

    # run model
    if fix_params:
        res["fixed_params_dict"] = fixed_params_dict
        my_model_run = block(condition(my_model, fixed_params_dict), hide=blocked_params_list)
    else:
        my_model_run = my_model
    jax.profiler.save_device_memory_profile("memory.prof2")
    if algo=='mcmc':
        nuts_kernel = NUTS(my_model_run, init_strategy=my_init_strategy, dense_mass=is_dense)

        mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=batch)
        mcmc.run(rng_key = Key(key_no_run+3), Y=Y, **my_model_args,
                extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))

        for b in range(n_batches-1):
            sample_batch = mcmc.get_samples()
            if b==0:
                temp = {k:[] for k in sample_batch.keys()}
                temp_keys = list(temp.keys())
            
            for k in temp_keys:
                temp[k].append(sample_batch[k])

            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, Y=Y, **my_model_args,
                extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))  # or mcmc.run(random.PRNGKey(1))
            if b==(n_batches-2):
                for k in temp_keys:
                    temp[k].append(sample_batch[k])
            jax.profiler.save_device_memory_profile(f"memory.prof_b{b}")
        


        for k in temp_keys:
            try:
                assert jnp.array(temp[k]).ndim==3
                temp[k] = jnp.vstack(jnp.array(temp[k]))
            except:
                new = jnp.array(temp[k])[:,:,None]
                temp[k] = jnp.vstack(new)

        res['all_samples'] = temp
        res['accept_prob'] = mcmc.get_extra_fields()['accept_prob']
        res['num_steps'] = mcmc.get_extra_fields()['num_steps']
        res['adapt_state'] = mcmc.get_extra_fields()['adapt_state']
        res['potential_energy'] = mcmc.get_extra_fields()['potential_energy']
        rho = res['all_samples']['rho'].mean(0)
        
    elif algo == 'svi':
        n_steps = SVI_samples_low
        diff_thresh = 100
        while ((diff_thresh > 1) & (n_steps<SVI_samples_high)):
            optimizer = adam(adam_start)
            guide = AutoDelta(my_model_run, init_loc_fn=my_init_strategy)
            svi = SVI(my_model_run, guide, optimizer, loss=Trace_ELBO())
            svi_result = svi.run(Key(key_no_run-4), num_steps = n_steps, stable_update = True, Y=Y,
                                **my_model_args)

            losses = svi_result.losses
            diff_thresh = jnp.abs(losses[-1]-losses[-2000])
            n_steps = n_steps + 100000
            if jnp.isnan(diff_thresh):
                adam_start = adam_start -0.001
                diff_thresh = 100
                SVI_samples_high += 100000
        
        rho_tilde = svi_result.params['rho_tilde_auto_loc']
        res['rho_tilde'] = rho_tilde
        sqrt_diag = svi_result.params['sqrt_diag_auto_loc']
        res['sqrt_diag'] = sqrt_diag
        rho_lt = rho_tilde*jnp.exp(-fixed_params_dict['eta1_0'])
        res['rho_lt'] = rho_lt
        
        rho_mat_tril = jnp.zeros((p,p))
        rho_mat_tril = rho_mat_tril.at[tril_idx].set(rho_lt)
        rho = rho_mat_tril + rho_mat_tril.T + jnp.identity(p)
        res['rho'] = rho

        theta = jnp.outer(sqrt_diag,sqrt_diag)*rho
        res['theta'] = theta
        
    else:
        raise ValueError("Algorithm must be set to either 'mcmc' or 'svi'")
    
    # select lower-triangle only
    tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
    nonzero_preds = (jnp.abs(rho[tril_idx]) >= 5*10**(-TP_thresh -1)) # for -5 cutoff is around beta0 = 6.

    Pos = jnp.where(nonzero_preds == True)[0].shape[0]
    Neg = jnp.where(nonzero_preds == False)[0].shape[0]
  
    res['Pos'] = Pos
    res['Neg'] = Neg

    if verbose:
        print(" ")
        if fix_params:
            for k in fixed_params_dict.keys():
                print(f"{k} is fixed to {fixed_params_dict[k]}") 
        if algo=='mcmc':
            for k in estimates_print:
                est_mean = mcmc.get_samples()[f'{k}'].mean(0)
                print(f'estimated {k} is {est_mean}')
        print(" ") 
        print(f'Total:{p*(p-1)/2}, Positives (non-zeros):{Pos}, Negatives (zeros):{Neg}')  
            
    return res

#%%
# Load data
#%%
covid_vals = pd.read_csv(data_path + 'COVID_greaterthan50000.csv', index_col='Unnamed: 0').values

n,p = covid_vals.shape
#%%
# params

n_warmup = 400
n_samples = 1000
n_batches = 10
batch = int(n_samples/n_batches)
eta1_0_m= 10.556
eta1_0_s= 3.
mu_m=0.
mu_s=1.

verbose = True
algo = 'mcmc'
my_model = models.glasso_repr
my_model_args = {"eta1_0_m":eta1_0_m, "eta1_0_s":eta1_0_s, 
"mu_m":mu_m, "mu_s":mu_s}
is_dense = False
estimates_print = ["eta1_0"]
#%%
# to fix parameters:
mu_fixed = jnp.zeros((p,))
fix_params=True
fixed_params_dict = {"mu":mu_fixed}
blocked_params_list = ["mu"]

rho_init = jnp.diag(jnp.ones((p,)))
mu_init = jnp.zeros((p,))
sqrt_diag_init = jnp.ones((p,))
my_init_strategy = init_to_feasible 
jax.profiler.save_device_memory_profile("memory.prof")

#%%
print('--------------------------------------------------------------------------------')
my_res = model_run(Y=covid_vals,  my_model=my_model, my_model_args=my_model_args,
                   TP_thresh=TP_thresh,
                   n_warmup=n_warmup, n_samples=n_samples,
                   algo=algo, fix_params=fix_params,
                   fixed_params_dict=fixed_params_dict, 
                   blocked_params_list=blocked_params_list, 
                   is_dense=is_dense, estimates_print=estimates_print,
                        my_init_strategy=my_init_strategy,  
             verbose=verbose, key_no_run=p)

with open(data_save_path + f'glasso_mcmc_975.sav' , 'wb') as f:
    pickle.dump((my_res), f)

#%%
# MAP
#%%
best_params = {'bandwidth': 0.13257113655901093, 'kernel': 'linear'}
down = -2
up = 12
x_d = np.linspace(down, up, 1000)
#%%
with open(data_save_path + f'glasso_mcmc_975.sav', 'rb') as fr:
    res = pickle.load(fr)
eta1_0_samples = res['all_samples']['eta1_0'].flatten()

kde = KernelDensity(**best_params)
kde.fit(eta1_0_samples[:, None])

logdensity = kde.score_samples(x_d[:, None])
density = jnp.exp(logdensity)
MAP = x_d[jnp.argmax(density)]
post_mean = eta1_0_samples.mean()
eta1_0_MAPs = MAP
print(f'MAP {MAP}, post. mean {post_mean}')
#%%

# params
my_model = models.glasso_repr
my_model_args = {}
verbose = True
algo = 'svi'
SVI_samples_low=10000
SVI_samples_high=320000
adam_start=0.005

estimates_print=[]
#%%
with open(data_save_path + f'glasso_mcmc_975.sav', 'rb') as fr:
    res = pickle.load(fr)
fix_params = True
fixed_params_dict = {"mu":mu_fixed, "eta1_0":eta1_0_MAPs}
blocked_params_list = ["mu", "eta1_0"]

my_init_strategy = init_to_feasible

# run model
my_res = model_run(Y=covid_vals, my_model=my_model, my_model_args=my_model_args,
                   algo=algo, fix_params=fix_params, estimates_print=estimates_print, 
                   my_init_strategy=my_init_strategy,  
                   SVI_samples_low=SVI_samples_low, SVI_samples_high=SVI_samples_high,
             fixed_params_dict=fixed_params_dict, blocked_params_list=blocked_params_list, 
                   verbose=verbose, key_no_run=p, adam_start=adam_start)

with open(data_save_path + f'glasso_map_975.sav', 'wb') as f:
    pickle.dump((my_res), f)
#%%

