#%%
# imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KernelDensity
import jax
import numpyro
# numpyro.set_platform('gpu')
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
from optax import adam
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

# paths
os.chdir('/home/usuario/Documents/Barcelona_Yr1/GraphicalModels_NetworkData/LiLicode/paper_code_github/')
sys.path.append("/Network_Spike_and_Slab/numpyro/functions")

sim_data_path = './Data/Simulations/'
data_save_path = './data/sim_GLASSO_data/'

# load models and functions
import models
import my_utils

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
#%%

n_sims = 50
ps = [10, 50]
n = 2000
n_cuts = [100, 200, 500]
TP_thresh = 3

# Functions

def model_run(p, n, n_cut, my_model, my_model_args, fix_params,  key_no_run, TP_thresh=3, n_warmup=1000, n_samples=2000,
              estimates_print=[], algo='mcmc',  SVI_samples_low=10000, SVI_samples_high=300000,
              verbose=False, my_init_strategy=init_to_feasible, 
             fixed_params_dict=None, blocked_params_list=None, adam_start=0.005,  is_dense=False):

    res = {}
    
    print(" ")
    print(f'Dimensions: p = {p}, n = {n_cut}')
    print('----------------------------------------')

    # load data
    with open(sim_data_path + f'sim{s}_p{p}_n{n}.sav', 'rb') as fr:
        sim_res = pickle.load(fr)
    Y = sim_res['Y']
    Y = Y[:n_cut,:]
    theta_true = sim_res['theta_true']

    # run model
    if fix_params:
        res["fixed_params_dict"] = fixed_params_dict
        my_model_run = block(condition(my_model, fixed_params_dict), hide=blocked_params_list)
    else:
        my_model_run = my_model
        
    if algo=='mcmc':
        nuts_kernel = NUTS(my_model_run, init_strategy=my_init_strategy, dense_mass=is_dense)

        mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=n_samples)
        mcmc.run(rng_key = Key(key_no_run+3), Y=Y, **my_model_args,
                extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))

        res['all_samples'] = mcmc.get_samples()
        res['accept_prob'] = mcmc.get_extra_fields()['accept_prob']
        res['num_steps'] = mcmc.get_extra_fields()['num_steps']
        res['adapt_state'] = mcmc.get_extra_fields()['adapt_state']
        res['potential_energy'] = mcmc.get_extra_fields()['potential_energy']
        rho = mcmc.get_samples()['rho'].mean(0)
        theta = mcmc.get_samples()['theta'].mean(0)      
 
    elif algo == 'svi':
        n_steps = SVI_samples_low
        diff_thresh = 100
        while ((diff_thresh > 1) & (n_steps<SVI_samples_high)):
            optimizer = adam(adam_start)#adam(exponential_decay(adam_start, n_samples, adam_end))
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
        
        tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
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
    nonzero_true = (jnp.abs(theta_true[tril_idx]) != 0.)

    TP = jnp.where((nonzero_preds == True)&(nonzero_true == True))[0].shape[0]
    FP = jnp.where((nonzero_preds == True)&(nonzero_true == False))[0].shape[0]
    FN = jnp.where((nonzero_preds == False)&(nonzero_true == True))[0].shape[0]
    TN = jnp.where((nonzero_preds == False)&(nonzero_true == False))[0].shape[0]

    res['TP'] = TP
    res['FP'] = FP
    res['FN'] = FN
    res['TN'] = TN
    
    if verbose:
        print('mse on estimated theta vs true theta: ', my_utils.get_MSE(theta, theta_true))
        print(" ")
        if fix_params:
            for k in fixed_params_dict.keys():
                print(f"{k} is fixed to {fixed_params_dict[k]}") 
        if algo=='mcmc':
            for k in estimates_print:
                est_mean = mcmc.get_samples()[f'{k}'].mean(0)
                print(f'estimated {k} is {est_mean}')
        print(" ") 
        print(f'Total:{p*(p-1)/2}, TP:{TP}, FP:{FP}, FN:{FN}, TN:{TN}')
        
        TPR = my_utils.get_TPR(TP=TP, FN=FN)
        FPR = my_utils.get_FPR(FP=FP, TN=TN)
        FNR = my_utils.get_FNR(FN=FN, TP=TP)
        print(f'TPR: {TPR}, FPR: {FPR}, FNR: {FNR}')
        try:
            FDiscR = my_utils.get_FDiscR(FP=FP, TP=TP)
            print(f'FDiscR: {FDiscR}')
        except:
            print('FDiscR: N/A, no positives')
        try:
            FNonDiscR = my_utils.get_FNonDiscR(TN=TN, FN=FN)
            print(f'FNonDiscR: {FNonDiscR}')
        except:
            print('FNonDiscR: N/A, no negatives')
    
    return res

#%%
# Run full model with MCMC

## params
n_warmup = 1000
n_samples = 5000
eta1_0_m= [7.444, 7.444]
eta1_0_s= [6.61, 6.61]
mu_m=0.
mu_s=1.

verbose = True
algo = 'mcmc'
my_model = models.glasso_repr
is_dense = False

mu_fixed=[jnp.zeros((ps[0],)), jnp.zeros((ps[1],))]
estimates_print = ["eta1_0"]

my_init_strategy = init_to_feasible 

for p_ix, p in enumerate(ps):
    for n_cut in n_cuts:
        for s in range(n_sims):
            print('--------------------------------------------------------------------------------')
            print(f"Simulation number: {s}")
            
            
            # fix parameters
            fix_params=True
            fixed_params_dict = {"mu":mu_fixed[p_ix]}
            blocked_params_list = ["mu"]         
            
            # model args
            my_model_args = {"eta1_0_m":eta1_0_m[p_ix], 
                             "eta1_0_s":eta1_0_s[p_ix], 
                             "mu_m":mu_m, "mu_s":mu_s}


            my_res = model_run(p=p, n=n, n_cut=n_cut, my_model=my_model, my_model_args=my_model_args,
                               TP_thresh=TP_thresh,
                               n_warmup=n_warmup, n_samples=n_samples,
                               algo=algo, fix_params=fix_params,
                               fixed_params_dict=fixed_params_dict, 
                               blocked_params_list=blocked_params_list, 
                               is_dense=is_dense, estimates_print=estimates_print,
                               my_init_strategy=my_init_strategy,  
                         verbose=verbose, key_no_run=s+p)

            with open(data_save_path + f'glasso_{s}_p{p}_n{n_cut}_mcmc.sav' , 'wb') as f:
                pickle.dump((my_res), f)


#%%
# Compute hyperparameter MAP from MCMC round
best_params = {'bandwidth': 0.13257113655901093, 'kernel': 'linear'}
down = 0
up = 5
x_d = np.linspace(down, up, 1000)

eta1_0_MAPs = {p:{
    n_cut:{s:0 for s in range(n_sims)} for n_cut in n_cuts
} for p in ps}

for p in ps:
    for n_cut in n_cuts:
        for s in range(n_sims):
            with open(data_save_path + f'glasso_{s}_p{p}_n{n_cut}_mcmc.sav', 'rb') as fr:
                res = pickle.load(fr)
            eta1_0_samples = res['all_samples']['eta1_0']

            kde = KernelDensity(**best_params)
            kde.fit(eta1_0_samples[:, None])

            logdensity = kde.score_samples(x_d[:, None])
            density = jnp.exp(logdensity)
            MAP = x_d[jnp.argmax(density)]
            post_mean = eta1_0_samples.mean()
            eta1_0_MAPs[p][n_cut][s] = MAP
            print(f'p {p}, n {n_cut}, s {s}: MAP {MAP}, post. mean {post_mean}')

#%%
# Run SVI round, keeping hyperparameters fixed at their MAP

## params
n_warmup = 10
my_model = models.glasso_repr
my_model_args = {}
verbose = True
algo = 'svi'
SVI_samples_low=10000
SVI_samples_high=320000
adam_start=0.005
estimates_print=[]

for p_ix, p in enumerate(ps):
    for n_cut in n_cuts:
        for s in range(n_sims):
            print('--------------------------------------------------------------------------------')
            print(f"Simulation number: {s}")

            # fix eta1_0
            with open(sim_data_path + f'sim{s}_p{p}_n{n}.sav', 'rb') as fr:
                sim_res = pickle.load(fr)
            with open(data_save_path + f'glasso_{s}_p{p}_n{n_cut}_mcmc.sav', 'rb') as fr:
                res = pickle.load(fr)
            fix_params = True
            fixed_params_dict = {"mu":mu_fixed[p_ix], "eta1_0":eta1_0_MAPs[p][n_cut][s]}
            blocked_params_list = ["mu", "eta1_0"]

            my_init_strategy = init_to_feasible

            # run model
            my_res = model_run(p=p, n=n, n_cut=n_cut, my_model=my_model, my_model_args=my_model_args,
                               algo=algo, fix_params=fix_params,
                               estimates_print=estimates_print,
                               my_init_strategy=my_init_strategy,  
                               SVI_samples_low=SVI_samples_low, SVI_samples_high=SVI_samples_high,
                         fixed_params_dict=fixed_params_dict, blocked_params_list=blocked_params_list, 
                               verbose=verbose, key_no_run=s+p, adam_start=adam_start)

            with open(data_save_path + f'glasso_{s}_p{p}_n{n_cut}_map.sav' , 'wb') as f:
                pickle.dump((my_res), f)

