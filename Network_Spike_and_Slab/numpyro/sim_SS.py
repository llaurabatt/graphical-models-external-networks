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

# paths
os.chdir("/Users/")
sys.path.append("functions")

sim_data_path = './data/sim_data/'
data_save_path = './data/sim_SS_data/'
data_init_path = './data/sim_GLASSO_data/'

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


def model_mcmc_run(p, n, n_cut, my_model, my_model_args, fix_params,  key_no_run, n_warmup=1000, n_samples=2000,
                     estimates_print=[], verbose=False, my_init_strategy=init_to_feasible, 
             fixed_params_dict=None, blocked_params_list=None, is_dense=False):

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

    # select lower-triangle only
    tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
    nonzero_true = (jnp.abs(theta_true[tril_idx]) != 0.)

    # run model
    if fix_params:
        res["fixed_params_dict"] = fixed_params_dict
        my_model_run = block(condition(my_model, fixed_params_dict), hide=blocked_params_list)
    else:
        my_model_run = my_model
        
    nuts_kernel = NUTS(my_model_run, init_strategy=my_init_strategy, dense_mass=is_dense)

    mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=n_samples)
    mcmc.run(rng_key = Key(key_no_run+3), Y=Y, **my_model_args,
            extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))

    res['all_samples'] = mcmc.get_samples()
    res['accept_prob'] = mcmc.get_extra_fields()['accept_prob']
    res['num_steps'] = mcmc.get_extra_fields()['num_steps']
    res['adapt_state'] = mcmc.get_extra_fields()['adapt_state']
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
                    
    TP_5 = jnp.where((nonzero_preds_5 == True)&(nonzero_true == True))[0].shape[0]
    FP_5 = jnp.where((nonzero_preds_5 == True)&(nonzero_true == False))[0].shape[0]
    FN_5 = jnp.where((nonzero_preds_5 == False)&(nonzero_true == True))[0].shape[0]
    TN_5 = jnp.where((nonzero_preds_5 == False)&(nonzero_true == False))[0].shape[0]

    res['TP_5'] = TP_5
    res['FP_5'] = FP_5
    res['FN_5'] = FN_5
    res['TN_5'] = TN_5

    TP_95 = jnp.where((nonzero_preds_95 == True)&(nonzero_true == True))[0].shape[0]
    FP_95 = jnp.where((nonzero_preds_95 == True)&(nonzero_true == False))[0].shape[0]
    FN_95 = jnp.where((nonzero_preds_95 == False)&(nonzero_true == True))[0].shape[0]
    TN_95 = jnp.where((nonzero_preds_95 == False)&(nonzero_true == False))[0].shape[0]

    res['TP_95'] = TP_95
    res['FP_95'] = FP_95
    res['FN_95'] = FN_95
    res['TN_95'] = TN_95
    
    if verbose:
        theta = mcmc.get_samples()['theta'].mean(0)
        print('mse on estimated theta vs true theta: ', my_utils.get_MSE(theta, theta_true))
        print(" ")
        if fix_params:
            for k in fixed_params_dict.keys():
                print(f"{k} is fixed to {fixed_params_dict[k]}") 

        for k in estimates_print:
            est_mean = mcmc.get_samples()[f'{k}'].mean(0)
            print(f'estimated {k} is {est_mean}')

        print(f'Total for threshold 0.5:{p*(p-1)/2}, TP:{TP_5}, FP:{FP_5}, FN:{FN_5}, TN:{TN_5}')
        
        TPR = my_utils.get_TPR(TP=TP_5, FN=FN_5)
        FPR = my_utils.get_FPR(FP=FP_5, TN=TN_5)
        FNR = my_utils.get_FNR(FN=FN_5, TP=TP_5)
        print(f'TPR: {TPR}, FPR: {FPR}, FNR: {FNR}')
        try:
            FDiscR = my_utils.get_FDiscR(FP=FP_5, TP=TP_5)
            print(f'FDiscR: {FDiscR}')
        except:
            print('FDiscR: N/A, no positives')
        try:
            FNonDiscR = my_utils.get_FNonDiscR(TN=TN_5, FN=FN_5)
            print(f'FNonDiscR: {FNonDiscR}')
        except:
            print('FNonDiscR: N/A, no negatives')
    
    return res
#%%
# Run first MCMC round

## params
n_warmup = 1000
n_samples = 5000
mu_m=0.
mu_s=1.
verbose = True
my_model = models.glasso_ss_repr_etaRepr
is_dense = False

mu_fixed=[jnp.zeros((ps[0],)), jnp.zeros((ps[1],))]
scale_spike_fixed=[0.003, 0.003]

eta0_0_m = [0., 0.]
eta0_0_s = [0.145, 0.152]
eta1_0_m = [-2.197, -2.197]
eta1_0_s = [0.661, 0.661]
eta2_0_m = [-2.722,-6.737]
eta2_0_s = [3.278, 3.395]


estimates_print = ["w_slab", "mean_slab", "scale_slab"]
#%%
for p_ix, p in enumerate(ps):

    # fix params
    fix_params=True
    fixed_params_dict = {"scale_spike":scale_spike_fixed[p_ix], 
                         "mu":mu_fixed[p_ix]}
    blocked_params_list = ["scale_spike", "mu"]
    for n_cut in n_cuts:
        for s in range(n_sims):
            print('--------------------------------------------------------------------------------')
            print(f"Simulation number: {s}")

            # model args
            my_model_args = {"eta0_0_m":eta0_0_m[p_ix], "eta0_0_s":eta0_0_s[p_ix],
                 "eta1_0_m":eta1_0_m[p_ix], "eta1_0_s":eta1_0_s[p_ix], 
                 "eta2_0_m":eta2_0_m[p_ix], "eta2_0_s":eta2_0_s[p_ix],
                 "mu_m":mu_m, "mu_s":mu_s}

            # init strategy
            with open(data_init_path + f'glasso_{s}_p{p}_n{n_cut}_map.sav' , 'rb') as fr:
                svi_glasso = pickle.load(fr)
            rho_tilde_init = svi_glasso['rho_tilde']
            mu_init = jnp.zeros((p,))
            sqrt_diag_init = jnp.ones((p,))
            my_init_strategy = init_to_value(values={'rho_tilde':rho_tilde_init, 
                                                 'mu':mu_init, 
                                                 'sqrt_diag':sqrt_diag_init, 
                                                 'tilde_eta0_0':0.,
                                                 'tilde_eta1_0':0.,
                                                'tilde_eta2_0':0.})


            my_res = model_mcmc_run(p=p, n=n, n_cut=n_cut, my_model=my_model, my_model_args=my_model_args, 
                               n_warmup=n_warmup, n_samples=n_samples,
                               is_dense=is_dense, fix_params=fix_params, 
                               fixed_params_dict=fixed_params_dict, blocked_params_list=blocked_params_list,
                               estimates_print=estimates_print,
                               my_init_strategy=my_init_strategy,  
                               verbose=verbose, key_no_run=s+p)

            with open(data_save_path + f'SS_{s}_p{p}_n{n_cut}_1mcmc.sav', 'wb') as f:
                pickle.dump((my_res), f)

#%%
# Compute hyperparameter MAP from first MCMC round

hyperpars = ['eta0_0', 'eta1_0', 'eta2_0']

best_params = {'eta0_0':{'bandwidth': 0.1, 'kernel': 'linear'},
               'eta1_0':{'bandwidth': 0.1, 'kernel': 'exponential'},
               'eta2_0':{'bandwidth': 0.372759372031494, 'kernel': 'gaussian'}}
               
x_d = np.linspace(-10, 12, 1000)

etas_MAPs = {p:{
    n_cut:{s:{'eta0_0':0, 'eta1_0':0, 'eta2_0':0}  
           for s in range(n_sims)} 
    for n_cut in n_cuts
} for p in ps}

for p in ps:
    for n_cut in n_cuts:
        for s in range(n_sims):
            with open(data_save_path + f'SS_{s}_p{p}_n{n_cut}_1mcmc.sav', 'rb') as fr:
                res = pickle.load(fr)
            for par in hyperpars:
                samples = res['all_samples'][par].flatten()

                kde = KernelDensity(**best_params[par])
                kde.fit(samples[:, None])

                logdensity = kde.score_samples(x_d[:, None])
                density = jnp.exp(logdensity)
                MAP = x_d[jnp.argmax(density)]
                post_mean = samples.mean()
                etas_MAPs[p][n_cut][s][par] = MAP

#%%
# Run second MCMC round, keeping hyperparameters fixed at their MAP

## params

n_warmup = 1000
n_samples = 5000
my_model = models.glasso_ss_repr
verbose = True
is_dense = False
estimates_print = []

for p_ix, p in enumerate(ps):
    for n_cut in n_cuts:
        for s in range(n_sims):
            print('--------------------------------------------------------------------------------')
            print(f"Simulation number: {s}")
            tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
            tril_len = tril_idx[0].shape[0]

            with open(sim_data_path + f'sim{s}_p{p}_n{n}.sav', 'rb') as fr:
                sim_res = pickle.load(fr)
            with open(data_save_path + f'SS_{s}_p{p}_n{n_cut}_1mcmc.sav', 'rb') as fr:
                res = pickle.load(fr)

            fix_params = True
            fixed_params_dict = {"scale_spike":scale_spike_fixed[p_ix], "mu":mu_fixed[p_ix], 
                                 "eta0_0":etas_MAPs[p][n_cut][s]["eta0_0"],
                                 "eta1_0":etas_MAPs[p][n_cut][s]["eta1_0"], 
                                 "eta2_0":etas_MAPs[p][n_cut][s]["eta2_0"]}
            blocked_params_list = ["mu", "scale_spike", "eta0_0", "eta1_0", "eta2_0"]

            # model args
            my_model_args = {"eta0_0_m":eta0_0_m[p_ix], "eta0_0_s":eta0_0_s[p_ix],
                 "eta1_0_m":eta1_0_m[p_ix], "eta1_0_s":eta1_0_s[p_ix], 
                 "eta2_0_m":eta2_0_m[p_ix], "eta2_0_s":eta2_0_s[p_ix],
                 "mu_m":mu_m, "mu_s":mu_s}
            
            # init strategy
            all_chains = jnp.hstack([res['all_samples']['eta0_0'][:,None], 
                                     res['all_samples']['eta1_0'][:,None],
                                     res['all_samples']['eta2_0'][:,None]])

            eta0_0_MAP = mean_slab = etas_MAPs[p][n_cut][s]["eta0_0"]
            eta1_0_MAP = etas_MAPs[p][n_cut][s]["eta1_0"]
            eta2_0_MAP = etas_MAPs[p][n_cut][s]["eta2_0"]
            my_MAP = jnp.hstack([jnp.array([eta0_0_MAP])[:,None],jnp.array([eta1_0_MAP])[:,None], 
                                 jnp.array([eta2_0_MAP])[:,None]])

            dists = my_utils.abs_dist(vec=my_MAP, mat=all_chains)
            rho_lt_init = res['all_samples']['rho_lt'][jnp.argmin(dists)]

            scale_slab = scale_spike_fixed[p_ix]*(1+jnp.exp(-eta1_0_MAP))
            w_slab = (1+jnp.exp(-eta2_0_MAP))**(-1)
            u_init = res['all_samples']['u'][jnp.argmin(dists)]
            is_spike = my_utils.my_sigmoid(u_init, beta=500., alpha=w_slab)

            rho_tilde_init = (rho_lt_init-mean_slab*(1-is_spike))/(is_spike*scale_spike_fixed[p_ix] + (1-is_spike)*scale_slab)

            sqrt_diag_init = res['all_samples']['sqrt_diag'][jnp.argmin(dists)]

            my_init_strategy = init_to_value(values={'u':u_init, 'rho_tilde':rho_tilde_init,
                                                    'sqrt_diag':sqrt_diag_init})
            #######

            # run model
            my_res = model_mcmc_run(p=p, n=n, n_cut=n_cut, my_model=my_model, my_model_args=my_model_args,  
                               n_samples=n_samples, n_warmup=n_warmup,  
                               fix_params=fix_params, fixed_params_dict=fixed_params_dict, 
                               blocked_params_list=blocked_params_list, estimates_print=estimates_print,
                               verbose=verbose, key_no_run=s+p, 
                               my_init_strategy=my_init_strategy)

            with open(data_save_path + f'SS_{s}_p{p}_n{n_cut}_2mcmc.sav', 'wb') as f:
                pickle.dump((my_res), f)

# %%
