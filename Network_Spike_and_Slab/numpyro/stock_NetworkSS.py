#%%
# imports
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
numpyro.set_platform('cpu')
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

#%%
# paths
os.chdir("/Users/")
sys.path.append("./functions")

data_save_path = './data/stock_market_data/'

# load models and functions
import models
import my_utils

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)

#%%
def model_mcmc_run(Y, my_model, my_model_args, fix_params,  key_no_run, estimates_print=[], 
                n_warmup=1000, n_samples=2000,n_mcmc_rounds=None, 
                temp_name=None, verbose=False, my_init_strategy=init_to_feasible, 
             fixed_params_dict=None, blocked_params_list=None, is_dense=False,):
    '''
    If you want to sample n_samples in multiple rounds, provide n_mcmc_rounds and temp_name
    '''
    res = {'all_samples':{}}
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
        
    nuts_kernel = NUTS(my_model_run, init_strategy=my_init_strategy, dense_mass=is_dense)

    if n_mcmc_rounds:
        n_samples = int(n_samples/n_mcmc_rounds)
    mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=n_samples)

    mcmc.run(rng_key = Key(key_no_run+3), Y=Y, **my_model_args,
            extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))
    
    if n_mcmc_rounds:
        mcmc_last_state = mcmc.last_state
        temp = {}
        temp['all_samples'] = mcmc.get_samples()
        temp['accept_prob'] = mcmc.get_extra_fields()['accept_prob'] # (n_samples,)
        temp['num_steps'] = mcmc.get_extra_fields()['num_steps'] # (n_samples,)
        temp['potential_energy'] = mcmc.get_extra_fields()['potential_energy'] # (n_samples,)
        with open(data_save_path + temp_name + '_' + f'res_0.sav' , 'wb') as f:
                pickle.dump((temp), f)

        for round in range(1, n_mcmc_rounds):

            mcmc.post_warmup_state = mcmc_last_state
            mcmc.run(rng_key = mcmc.post_warmup_state.rng_key, Y=Y, **my_model_args,
                extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))

            mcmc_last_state = mcmc.last_state
            with open(data_save_path + temp_name + '_' + f'mcmc_last_state.sav' , 'wb') as f:
                pickle.dump((mcmc_last_state), f)

            temp = {}
            temp['all_samples'] = mcmc.get_samples()
            temp['accept_prob'] = mcmc.get_extra_fields()['accept_prob']
            temp['num_steps'] = mcmc.get_extra_fields()['num_steps']
            temp['potential_energy'] = mcmc.get_extra_fields()['potential_energy']
            with open(data_save_path + temp_name + '_' + f'res_{round}.sav' , 'wb') as f:
                    pickle.dump((temp), f)

        temp_list = []
        for round in range(n_mcmc_rounds):
            with open(data_save_path + temp_name + '_' +f'res_{round}.sav', 'rb') as fr:
                temp = pickle.load(fr)
            temp_list.append(temp)

        for k in ['accept_prob', 'num_steps', 'potential_energy']:
            res[k] = jnp.concatenate([temp[k] for temp in temp_list])

        for k in temp['all_samples'].keys():
            res['all_samples'][k] = jnp.concatenate([temp['all_samples'][k] for temp in temp_list])
    
    else:
        res['all_samples'] = mcmc.get_samples()
        res['accept_prob'] = mcmc.get_extra_fields()['accept_prob']
        res['num_steps'] = mcmc.get_extra_fields()['num_steps']
        res['potential_energy'] = mcmc.get_extra_fields()['potential_energy']

    prob_slab_all = []
    for cs in range(n_samples):
        prob_slab = my_utils.get_prob_slab(rho_lt=res['all_samples']['rho_lt'][cs], 
                                        mean_slab=res['all_samples']['mean_slab'][cs], 
                                        scale_slab=res['all_samples']['scale_slab'][cs], 
                                        scale_spike=fixed_params_dict['scale_spike'], 
                                        w_slab=res['all_samples']['w_slab'][cs], 
                                        w_spike=(1-res['all_samples']['w_slab'])[cs])
        prob_slab_all.append(prob_slab)
    prob_slab_est = (jnp.array(prob_slab_all)).mean(0)    
    nonzero_preds_5 = (prob_slab_est>0.5).astype(int)
    nonzero_preds_95 = (prob_slab_est>0.95).astype(int)

    Pos_5 = jnp.where(nonzero_preds_5 == True)[0].shape[0]
    Neg_5 = jnp.where(nonzero_preds_5 == False)[0].shape[0]
  
    res['Pos_5'] = Pos_5
    res['Neg_5'] = Neg_5

    Pos_95 = jnp.where(nonzero_preds_95 == True)[0].shape[0]
    Neg_95 = jnp.where(nonzero_preds_95 == False)[0].shape[0]
  
    res['Pos_95'] = Pos_95
    res['Neg_95'] = Neg_95
        
    if verbose:
        print(" ")
        if fix_params:
            for k in fixed_params_dict.keys():
                print(f"{k} is fixed to {fixed_params_dict[k]}") 
        for k in estimates_print:
            est_mean = res['all_samples'][f'{k}'].mean(0)
            print(f'estimated {k} is {est_mean}')
            
        print(" ")
        print(f'Total:{p*(p-1)/2}, Positives (non-zeros):{Pos_5}, Negatives (zeros):{Neg_5}')  
        print(f'Positives 95 (non-zeros):{Pos_95}, Negatives 95 (zeros):{Neg_95}')  
    
    return res
#%%
# load data
stock_vals = pd.read_csv(data_save_path + 'stock_trans.csv', index_col='Unnamed: 0').values
E_pears_clean = jnp.array(jnp.load(data_save_path + 'E_pears_clean.npy'))
P_pears_clean = jnp.array(jnp.load(data_save_path + 'P_pears_clean.npy'))

n,p = stock_vals.shape
#%%
# Run first MCMC round
## params
n_warmup = 2000
n_samples = 10000
# n_mcmc_rounds = 5
# temp_name = f'mcmc1_{n_samples}samples_{n_mcmc_rounds}rounds'

mu_m=0.
mu_s=1.

verbose = True
my_model = models.NetworkSS_repr_etaRepr
is_dense=False
#%%
fix_params=True
mu_fixed=jnp.zeros((p,))
scale_spike_fixed =0.003
fixed_params_dict = {"scale_spike":scale_spike_fixed, "mu":mu_fixed}
blocked_params_list = ["scale_spike", "mu"]

eta0_0_m=0. # 0.
eta0_0_s=0.14
eta0_coefs_m=0.
eta0_coefs_s=0.14

eta1_0_m=-2.197
eta1_0_s=0.661
eta1_coefs_m=0.
eta1_coefs_s=0.661

eta2_0_m=-9.158
eta2_0_s=3.658
eta2_coefs_m=0.
eta2_coefs_s=3.658
#%%
# init strategy
with open(data_save_path +f'glasso_map.sav', 'rb') as fr:
    svi_glasso = pickle.load(fr)
    
rho_tilde_init = svi_glasso['rho_tilde']
mu_init = jnp.zeros((p,))
sqrt_diag_init = jnp.ones((p,))
my_init_strategy = init_to_value(values={'rho_tilde':rho_tilde_init,
                                         'mu':mu_init, 
                                         'sqrt_diag':sqrt_diag_init, 
                                         'tilde_eta0_0':0.,
                                         'tilde_eta1_0':0.,
                                        'tilde_eta2_0':0.,                                     
                                        'tilde_eta0_coefs':jnp.array([0.,0.]),
                                        'tilde_eta1_coefs':jnp.array([0.,0.]),
                                        'tilde_eta2_coefs':jnp.array([0.,0.]),})

estimates_print = ["w_slab", "mean_slab", "scale_slab"]
#%%
print('--------------------------------------------------------------------------------')   
print(" ")
A_list = [E_pears_clean, P_pears_clean]
my_model_args = {"A_list":A_list, "eta0_0_m":eta0_0_m, "eta0_0_s":eta0_0_s, 
             "eta0_coefs_m":eta0_coefs_m, "eta0_coefs_s":eta0_coefs_s,
             "eta1_0_m":eta1_0_m, "eta1_0_s":eta1_0_s, 
             "eta1_coefs_m":eta1_coefs_m, "eta1_coefs_s":eta1_coefs_s,
             "eta2_0_m":eta2_0_m, "eta2_0_s":eta2_0_s, 
             "eta2_coefs_m":eta2_coefs_m, "eta2_coefs_s":eta2_coefs_s,
             "mu_m":mu_m, "mu_s":mu_s} 

my_res = model_mcmc_run(Y=stock_vals, my_model=my_model, my_model_args=my_model_args, n_warmup=n_warmup, 
                   n_samples=n_samples, is_dense=is_dense,
                   fix_params=fix_params, fixed_params_dict=fixed_params_dict, 
                   blocked_params_list=blocked_params_list, estimates_print=estimates_print, 
                   my_init_strategy=my_init_strategy,  
             verbose=verbose, key_no_run=p)

with open(data_save_path + f'NetworkSS_E_P_1mcmc.sav' , 'wb') as f:
    pickle.dump((my_res), f)
#%%
hyperpars = ['eta0_0', 'eta0_coefs', 'eta1_0', 'eta1_coefs', 'eta2_0', 'eta2_coefs']

best_params = {'eta0_0':{'bandwidth': 0.1, 'kernel': 'linear'}, 
                       'eta0_coefs':{'bandwidth': 0.1, 'kernel': 'linear'},
                       'eta1_0':{'bandwidth': 0.3088843596477481, 'kernel': 'linear'}, 
                       'eta1_coefs':{'bandwidth': 0.13257113655901093, 'kernel': 'linear'}, 
                       'eta2_0':{'bandwidth': 1.2648552168552958, 'kernel': 'linear'}, 
                       'eta2_coefs':{'bandwidth': 0.2559547922699536, 'kernel': 'gaussian'}}

x_d = np.linspace(-10, 12, 1000)
etas_MAPs = {'eta0_0':0, 
                       'eta0_coefs':0,
                       'eta1_0':0, 
                       'eta1_coefs':0, 
                       'eta2_0':0, 
                       'eta2_coefs':0}

with open(data_save_path + f'NetworkSS_E_P_1mcmc.sav', 'rb') as fr:
    res = pickle.load(fr)
for par in hyperpars:
    if 'coefs' in par:
        samples = res['all_samples'][par][:,0].flatten()

        kde = KernelDensity(**best_params[par])
        kde.fit(samples[:, None])

        logdensity = kde.score_samples(x_d[:, None])
        density = jnp.exp(logdensity)
        MAP_1 = x_d[jnp.argmax(density)]
        post_mean_1 = samples.mean()

        ############
        samples = res['all_samples'][par][:,1].flatten()

        kde = KernelDensity(**best_params[par])
        kde.fit(samples[:, None])

        logdensity = kde.score_samples(x_d[:, None])
        density = jnp.exp(logdensity)
        MAP_2 = x_d[jnp.argmax(density)]
        post_mean_2 = samples.mean()
        etas_MAPs[par] = jnp.hstack([MAP_1, MAP_2])
        print(f'{par} P: MAP {MAP_2}, post. mean {post_mean_2}')
    
    else:

        samples = res['all_samples'][par].flatten()

        kde = KernelDensity(**best_params[par])
        kde.fit(samples[:, None])

        logdensity = kde.score_samples(x_d[:, None])
        density = jnp.exp(logdensity)
        MAP = x_d[jnp.argmax(density)]
        post_mean = samples.mean()
        etas_MAPs[par] = MAP
#%%
# Run second MCMC round
## params
n_warmup = 1000
n_samples = 5000
# n_mcmc_rounds = 2
# temp_name = f'mcmc2_{n_samples}samples_{n_mcmc_rounds}rounds'

verbose = False
algo = 'mcmc'
my_model = models.NetworkSS_repr
is_dense=False
estimates_print = []

#%%
def SVI_init_strategy_golazo_ss(A_list, mcmc_res, fixed_params_dict):

    all_chains = jnp.hstack([mcmc_res['all_samples']['eta0_0'][:,None], 
                             mcmc_res['all_samples']['eta0_coefs'],
                             mcmc_res['all_samples']['eta1_0'][:,None], 
                             mcmc_res['all_samples']['eta1_coefs'],
                            mcmc_res['all_samples']['eta2_0'][:,None], 
                             mcmc_res['all_samples']['eta2_coefs']])

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
    rho_lt_init = mcmc_res['all_samples']['rho_lt'][jnp.argmin(dists)]
    
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
    
    u_init = mcmc_res['all_samples']['u'][jnp.argmin(dists)]
    is_spike = my_utils.my_sigmoid(u_init, beta=500., alpha=w_slab)

    rho_tilde_init = (rho_lt_init-mean_slab*(1-is_spike))/(is_spike*scale_spike_fixed + (1-is_spike)*scale_slab)
    sqrt_diag_init = mcmc_res['all_samples']['sqrt_diag'][jnp.argmin(dists)]

    rho_lt_MAP = rho_tilde_init*is_spike*scale_spike_fixed + (1-is_spike)*(rho_tilde_init*scale_slab + mean_slab)
    rho_mat_tril = jnp.zeros((p,p))
    rho_mat_tril = rho_mat_tril.at[tril_idx].set(rho_lt_MAP)
    rho = rho_mat_tril + rho_mat_tril.T + jnp.identity(p)

    enter = jnp.all(jnp.linalg.eigvals(rho) > 0)
    print(f'Is rho init p.d.:{enter}')
    
    return u_init, rho_tilde_init, sqrt_diag_init

#%%
tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
tril_len = tril_idx[0].shape[0]

with open(data_save_path + f'NetworkSS_E_P_1mcmc.sav', 'rb') as fr:
    res = pickle.load(fr)

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

A_list = [E_pears_clean, P_pears_clean]
my_model_args = {"A_list":A_list, "eta0_0_m":eta0_0_m, "eta0_0_s":eta0_0_s, 
         "eta0_coefs_m":eta0_coefs_m, "eta0_coefs_s":eta0_coefs_s,
         "eta1_0_m":eta1_0_m, "eta1_0_s":eta1_0_s, 
         "eta1_coefs_m":eta1_coefs_m, "eta1_coefs_s":eta1_coefs_s,
         "eta2_0_m":eta2_0_m, "eta2_0_s":eta2_0_s, 
         "eta2_coefs_m":eta2_coefs_m, "eta2_coefs_s":eta2_coefs_s,
         "mu_m":mu_m, "mu_s":mu_s} 

u_init, rho_tilde_init, sqrt_diag_init = SVI_init_strategy_golazo_ss(A_list=A_list, mcmc_res=res, 
                                                                    fixed_params_dict=fixed_params_dict)
my_init_strategy = init_to_value(values={'u':u_init, 'rho_tilde':rho_tilde_init,'sqrt_diag':sqrt_diag_init})

my_res = model_mcmc_run(Y=stock_vals, my_model=my_model, my_model_args=my_model_args, 
                   n_warmup=n_warmup, n_samples=n_samples,fix_params=fix_params, fixed_params_dict=fixed_params_dict, 
                   blocked_params_list=blocked_params_list,  estimates_print=estimates_print,
                   my_init_strategy=my_init_strategy,  
                   verbose=verbose, key_no_run=p)

with open(data_save_path + f'NetworkSS_E_P_2mcmc.sav' , 'wb') as f:
    pickle.dump((my_res), f)

# %%
