#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
import jax
import numpyro
numpyro.set_platform('cpu')
print(jax.lib.xla_bridge.get_backend().platform)
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.random import PRNGKey as Key
from numpyro.util import enable_x64
import sys
import os
sys.path.append("./functions")
import models

#%%
enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
#%%
def generate_theta(A_true, w_slab_1, w_slab_0, slab_0_low, slab_0_up,  slab_1_low, slab_1_up, key_no):
    
    # select lower-triangular A
    p = A_true.shape[0]
    tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
    A_tril = A_true[tril_idx]

    A_0_idx = jnp.where(A_tril==0.)[0]
    A_1_idx = jnp.where(A_tril==1.)[0]

    # generate theta vals for A=1  
    ones_no = int(A_tril.sum())
    slab_1_no = int(ones_no*w_slab_1)

    spike_1_no = ones_no - slab_1_no
    spike_1_idx = jax.random.choice(Key(key_no+5), jnp.arange(ones_no), (spike_1_no,), replace=False)

    potential_slab_1 = jnp.linspace(slab_1_low, slab_1_up, ones_no)
    theta_1 = potential_slab_1.at[spike_1_idx].set(0.)

    # generate theta vals for A = 0
    zeros_no = int((A_tril==0.).sum())
    slab_0_no = int(zeros_no*w_slab_0)

    spike_0_no = zeros_no - slab_0_no
    spike_0_idx = jax.random.choice(Key(key_no+4), jnp.arange(zeros_no), (spike_0_no,), replace=False)

    potential_slab_0 = jnp.linspace(slab_0_low, slab_0_up, zeros_no)
    theta_0 = potential_slab_0.at[spike_0_idx].set(0.)

    # combine to obtain theta
    theta_tril = A_tril.at[A_0_idx].set(theta_0)
    theta_tril = theta_tril.at[A_1_idx].set(theta_1)

    my_theta_init = jnp.diag(jnp.zeros((p,)))
    my_theta_lt = my_theta_init.at[tril_idx].set(theta_tril)

    my_theta = my_theta_lt + my_theta_lt.T + jnp.diag(jnp.ones((p,)))
    return my_theta

#%%
def simulate_data(n_obs, p, mu_true, theta_true, key_no):

    sim_res = {}

    Y = dist.MultivariateNormal(mu_true, 
                                precision_matrix=theta_true).expand((n_obs,)).sample(Key(key_no))
    sim_res['mu_true'] = mu_true
    sim_res['theta_true'] = theta_true
    sim_res['Y'] = Y
    sim_res['n'] = n_obs
    
    return sim_res

#%%
def network_simulate(p, A_true, flip_prop=0.):
    
    triu_idx = jnp.triu_indices(n=p, k=1, m=p)
    A_lt = A_true
    A_lt = A_lt.at[triu_idx].set(-999)
    
    pos = jnp.array(jnp.where(A_lt>0.))
    flip_pos = get_flip_idx(pos, prop=flip_prop)
    print(f'Flipping {flip_pos[0].shape[0]} positives out of {pos[0].shape[0]} to zero')
    
    zeros = jnp.array(jnp.where(A_lt==0.))
    flip_zeros = get_flip_idx(zeros, prop=flip_prop)
    print(f'Flipping {flip_zeros[0].shape[0]} zeros out of {zeros[0].shape[0]} to one')
    
    A_new = A_true.at[flip_pos].set(0.)
    A_new = A_new.at[flip_zeros].set(1.)
    
    A_new = jnp.tril(A_new) + jnp.tril(A_new).T - jnp.diag(jnp.diag(A_new))
    return A_new

#%%
def network_scale(A):
    p = A.shape[0]
    n_obs = p*(p-1)/2
    A_lt = jnp.tril(A, k=-1)
    A_m = A_lt.sum()/n_obs
    A_bar_lt = jnp.tril(A_lt - A_m, k=-1)
    A_var = ((A_bar_lt)**2).sum()/(n_obs)
    
    A_scaled = (A-A_m)/jnp.sqrt(A_var)
    return A_scaled


def get_flip_idx(coordinates, prop=0.6):
    n_pos = coordinates.shape[1]
    n_flip = np.int(n_pos*prop)
    
    flip_idx = jax.random.choice(Key(5), jnp.arange(n_pos), (n_flip,), replace=False)
    flip = coordinates[:,flip_idx]
    flip = (flip[0], flip[1])
    
    return flip

#%%
def get_MSE_standard(preds, true):
    return((preds-true)**2).mean()

def get_TPR(TP, FN):
    return TP/(TP + FN) # tot true

def get_FPR(FP, TN):
    return FP/(FP + TN) # tot false

def get_FNR(FN, TP):
    return FN/(FN + TP) # false negatives/tot true

def get_FDiscR(FP, TP):
    return FP/(FP + TP) # tot positives

def get_FNonDiscR(TN, FN):
    return FN/(FN + TN) # tot negatives

def get_MSE(preds, true):
    return((cov2corr(preds)-cov2corr(true))**2).sum()

def cov2corr( A ):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T/d).T)/d
    return A

def euclidean_dist(vec, mat):
    try:
        _, hv = vec.shape
    except:
        vec = vec[:,None]
        _, hv = vec.shape

    n, hm = mat.shape
    if ((n==hv)&(hm!=hv)):
        mat = mat.T
    elif ((n!=hv)&(hm!=hv)):
        raise ValueError("Vec and mat are of incompatible shapes")
    else:
        pass

    dists = jnp.sqrt(jnp.diag((vec - mat)@(vec - mat).T))
    return dists

def abs_dist(vec, mat):
    try:
        _, hv = vec.shape
    except:
        vec = vec[None]
        _, hv = vec.shape

    n, hm = mat.shape
    if ((n==hv)&(hm!=hv)):
        mat = mat.T
    elif ((n!=hv)&(hm!=hv)):
        raise ValueError("Vec and mat are of incompatible shapes")
    else:
        pass
    
    dists = jnp.abs(mat-vec).sum(1)
    return dists
#%%
def plot_exact_ss_prior(scale_spike, scale_slab, mean_slab, w_slab, rho_lt, logprior_fn): 

    density = jnp.exp(jnp.array([logprior_fn(scale_spike=scale_spike, 
                scale_slab=scale_slab_el, 
                mean_slab=mean_slab_el, 
                w_slab=w_slab_el,
                rho_lt=rho_el) for rho_el, scale_slab_el, mean_slab_el, w_slab_el in zip(rho_lt, scale_slab,
                                                                       mean_slab, w_slab)]))

    order = jnp.argsort(rho_lt)

    fig, ax = plt.subplots( figsize=(5,5))
    plt.suptitle("Spike-Slab density:")
    ax.plot(rho_lt[order],density[order])
    plt.show()

def my_sigmoid(x, pow_no=-1, beta=1., alpha=1.):
    return jnp.power((1.+jnp.exp(-beta*x + beta*alpha)),pow_no)

#%%
def get_prob_slab(rho_lt, mean_slab, scale_slab, scale_spike, w_slab, w_spike):
    log_spike = dist.Laplace(0., scale_spike).log_prob(rho_lt) + jnp.log(w_spike)
    log_slab = dist.Laplace(mean_slab, scale_slab).log_prob(rho_lt) + jnp.log(w_slab)

    prob_slab = 1/(1 + jnp.exp(log_spike - log_slab))
    return prob_slab

#%%
def is_nonzero(rho_lt, mean_slab, scale_slab, scale_spike, w_slab, w_spike, thresh=0.5):
    log_spike = dist.Laplace(0., scale_spike).log_prob(rho_lt) + jnp.log(w_spike)
    log_slab = dist.Laplace(mean_slab, scale_slab).log_prob(rho_lt) + jnp.log(w_slab)

    prob_slab = 1/(1 + jnp.exp(log_spike - log_slab))
    is_nonzero = (prob_slab>thresh).astype(int)
    print(f'is_nonzero with thresh {thresh}', is_nonzero)
    return is_nonzero

#%%
def get_density_els(A_tril, scale_spike, w_slab, scale_slab, mean_slab, nbins):
    bins = np.histogram(A_tril, bins=nbins)[1]

    A_ints = {}
    A_mid = []
    down = -jnp.inf
    for i in range(-1, len(bins)-2):
        up = bins[i+2]
        if i==-1:
            A_mid.append(up)
        else:
            A_mid.append((down+up)/2)
        A_int_ix = jnp.where((A_tril>down)&(A_tril<=up))[0]
        if len(A_int_ix)==0:
            print(down, up, 'empty interval!')
            down = bins[i+1]
            continue

        A_key = f'{jnp.round(down,2)} to {jnp.round(up,2)}'
        down = bins[i+2]
        w_slab_int = w_slab[A_int_ix]
        scale_slab_int = scale_slab[A_int_ix]
        mean_slab_int = mean_slab[A_int_ix]
        
        A_int = A_tril[A_int_ix]
        int_n = A_int.shape[0]
        quasi_median_pos = (int_n+1)//2
        int_order = jnp.argsort(A_int)
        
        w_slab_el = w_slab_int[int_order][quasi_median_pos]
        scale_slab_el = scale_slab_int[int_order][quasi_median_pos]
        mean_slab_el = mean_slab_int[int_order][quasi_median_pos]
        A_ints[A_key] = {'scale_spike': scale_spike, 'w_slab':w_slab_el, 'scale_slab':scale_slab_el, 
                        'mean_slab':mean_slab_el}
    A_mid = jnp.array(A_mid)
    return A_ints, A_mid

#%%
def get_point_estimate_MAP_glasso(chains, fixed_params_dict, plot_keys, 
                           best_params):

    
    x_d = np.linspace(-10, 12, 1000)
    hypers = ['scale_spike', 'eta0_0', 'eta1_0', 'eta2_0']

    eta_est = {}
    for hyper in hypers:
        try:
            eta_est[hyper] = fixed_params_dict[hyper]
        except:
            samples = chains[hyper]
            kde = KernelDensity(**best_params[hyper])
            kde.fit(samples[:, None])

            logdensity = kde.score_samples(x_d[:, None])
            density = jnp.exp(logdensity)
            MAP = x_d[jnp.argmax(density)]
            eta_est[hyper] = MAP

   # means
    mean_slab = jnp.array(eta_est["eta0_0"])
    scale_slab = eta_est['scale_spike']*(1+jnp.exp(-eta_est["eta1_0"]))

    w_slab = (1+jnp.exp(-eta_est["eta2_0"]))**(-1)
    plot_dict = {'scale_spike':eta_est['scale_spike'],
                'scale_slab':scale_slab, 
                 'mean_slab':mean_slab, 'w_slab':w_slab}
    return plot_dict
#%%
def get_point_estimate_MAP_golazo(p, chains, fixed_params_dict, plot_keys, A_to_plot,
                           best_params):

    
    x_d = np.linspace(-10, 12, 1000)
    hypers = ['scale_spike', 'eta0_0', 'eta0_coefs', 'eta1_0', 'eta1_coefs',
                 'eta2_0', 'eta2_coefs']

    eta_est = {}
    for hyper in hypers:
        try:
            eta_est[hyper] = fixed_params_dict[hyper]
        except:
            try:
                samples = chains[hyper]
                kde = KernelDensity(**best_params[hyper])
                kde.fit(samples[:, None])

                logdensity = kde.score_samples(x_d[:, None])
                density = jnp.exp(logdensity)
                MAP = x_d[jnp.argmax(density)]
                eta_est[hyper] = MAP
            
            except:
                samples = chains[hyper][:,0]
                kde = KernelDensity(**best_params[hyper])
                kde.fit(samples[:, None])

                logdensity = kde.score_samples(x_d[:, None])
                density = jnp.exp(logdensity)
                MAP_1 = x_d[jnp.argmax(density)]
                
                samples = chains[hyper][:,1]
                kde = KernelDensity(**best_params[hyper])
                kde.fit(samples[:, None])

                logdensity = kde.score_samples(x_d[:, None])
                density = jnp.exp(logdensity)
                MAP_2 = x_d[jnp.argmax(density)]

                eta_est[hyper] = jnp.hstack([MAP_1, MAP_2])
        print(hyper, eta_est[hyper])

   # means

    tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
    tril_len = tril_idx[0].shape[0]

    A_tril_arr = jnp.array([A[tril_idx] for A in A_to_plot]) # (a, p, p)
 
    A_tril_mean0 = 0.
    for coef, A in zip(eta_est['eta0_coefs'],A_tril_arr):
        A_tril_mean0 += coef*A

    A_tril_mean1 = 0.
    for coef, A in zip(eta_est['eta1_coefs'],A_tril_arr):
        A_tril_mean1 += coef*A
        
    A_tril_mean2 = 0.
    for coef, A in zip(eta_est['eta2_coefs'],A_tril_arr):
        A_tril_mean2 += coef*A
        
    # scale spike
    scale_spike = jnp.ones((tril_len,))*eta_est['scale_spike']
    
    # mean slab
    mean_slab = jnp.array(eta_est['eta0_0']+A_tril_mean0)

    # scale slab
    scale_slab = scale_spike*(1+jnp.exp(-eta_est['eta1_0']-A_tril_mean1))
    
    # prob of being in the slab
    w_slab = 1/(1+jnp.exp(-eta_est['eta2_0'] -A_tril_mean2)) # 45 
    plot_dict = {'scale_spike':eta_est['scale_spike'],
                'scale_slab':scale_slab, 
                 'mean_slab':mean_slab, 'w_slab':w_slab}
    return plot_dict
#%
def from_etas_to_params(coef_dict, p, model='glasso_ss', A_list=None):
    if model=='glasso_ss':
        mean_slab = jnp.array(coef_dict["eta0_0"])
        scale_slab = coef_dict['scale_spike']*(1+jnp.exp(-coef_dict["eta1_0"]))

        w_slab = (1+jnp.exp(-coef_dict["eta2_0"]))**(-1)
        par_dict = {'scale_spike':coef_dict['scale_spike'],
                    'scale_slab':scale_slab, 
                    'mean_slab':mean_slab, 'w_slab':w_slab}
        return par_dict

    elif model=='golazo_ss':
        if p>1:
            tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
            tril_len = tril_idx[0].shape[0]

            A_tril_arr = jnp.array([A[tril_idx] for A in A_list]) # (a, p, p)
            # scale spike
            scale_spike = jnp.ones((tril_len,))*coef_dict['scale_spike']
        else:
            scale_spike = coef_dict['scale_spike']
            A_tril_arr = A_list

        A_tril_mean0 = 0.
        for coef, A in zip(coef_dict['eta0_coefs'],A_tril_arr):
            A_tril_mean0 += coef*A

        A_tril_mean1 = 0.
        for coef, A in zip(coef_dict['eta1_coefs'],A_tril_arr):
            A_tril_mean1 += coef*A
            
        A_tril_mean2 = 0.
        for coef, A in zip(coef_dict['eta2_coefs'],A_tril_arr):
            A_tril_mean2 += coef*A
            

        
        # mean slab
        mean_slab = jnp.array(coef_dict['eta0_0']+A_tril_mean0)

        # scale slab
        scale_slab = scale_spike*(1+jnp.exp(-coef_dict['eta1_0']-A_tril_mean1))
        
        # prob of being in the slab
        w_slab = 1/(1+jnp.exp(-coef_dict['eta2_0'] -A_tril_mean2)) # 45 
        par_dict = {'scale_spike':coef_dict['scale_spike'],
                    'scale_slab':scale_slab, 
                    'mean_slab':mean_slab, 'w_slab':w_slab}
        return par_dict

    else:
        raise NotImplementedError('Need to write non-ss part!Choose between glasso_ss and golazo_ss')


 #%%
def get_density_els_marginal(A_tril, A_tril_pos, len_A_list, nbins, eta_dict):
    bins = np.histogram(A_tril, bins=nbins)[1]

    A_ints = {}
    A_mids = []
    down = -jnp.inf
    for i in range(-1, len(bins)-2):
        up = bins[i+2]
        if i==-1:
            A_mid = up-(bins[i+3]-bins[i+2])/2
            A_mids.append(A_mid)
        else:
            A_mid = (down+up)/2
            A_mids.append(A_mid)
        A_int_ix = jnp.where((A_tril>down)&(A_tril<=up))[0]
        if len(A_int_ix)==0:
            print(down, up, 'empty interval!')
            down = bins[i+1]
            continue

        A_key = f'{jnp.round(down,2)} to {jnp.round(up,2)}'
        down = bins[i+2]

        zeros = jnp.zeros((len_A_list))
        A_singlevals = zeros.at[A_tril_pos].set(A_mid)

        par_dict = from_etas_to_params(coef_dict=eta_dict, p=1, 
        model='golazo_ss', A_list=A_singlevals)

        A_ints[A_key] = par_dict

    A_mids = jnp.array(A_mids)
    return A_ints, A_mids