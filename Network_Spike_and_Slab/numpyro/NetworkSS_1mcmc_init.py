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
#%%
import pickle
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
from numpyro.infer.util import log_density
from typing import Optional

#%%
# # paths
# _ROOT_DIR = "/home/paperspace/"
# os.chdir(_ROOT_DIR + 'graphical-models-external-networks/')
# sys.path.append(_ROOT_DIR + "graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

# data_path = './Data/COVID/Pre-processed Data/'
# data_save_path = _ROOT_DIR + 'NetworkSS_results_loglikrepr/'
# if not os.path.exists(data_save_path):
#     os.makedirs(data_save_path, mode=0o777)
#%%
# load models and functions
# import models
# import my_utils

enable_x64(use_x64=True)
print("Is 64 precision enabled?:", jax.config.jax_enable_x64)
cpus = jax.devices("cpu")

#%%
def mcmc1_init(my_vals,
               my_model,
               my_model_args,
               root_dir,
               data_save_path,
               scale_spike_fixed,
               seed,
               n_samples,
               no_networks,
               init_strategy:Optional[str]='init_to_value',
               thinning:Optional[int]=0,
               b_init:Optional[jnp.array]=None,
               store_warmup:Optional[bool]=False,
        ):

    #%%
    # paths
    _ROOT_DIR = root_dir
    os.chdir(_ROOT_DIR + 'graphical-models-external-networks/')
    sys.path.append(_ROOT_DIR + "graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

    import models
    import my_utils
    #%%
    n,p = my_vals.shape
    
    try:
        my_covariates = my_model_args['X'] 
        _,_, q = my_covariates.shape
    except:
        q = None
    n_nets = len(my_model_args['A_list'])
    print(f"NetworkSS, n {n}, p {p}, number of networks {n_nets}")
    if no_networks:
        print("No network information will be used.")    
    if q:
        print(f"Number of covariates {q}")
    #%%

    #%%
    ## params
    n_warmup = 1000
    n_samples = n_samples
    n_batches = 1
    batch = int(n_samples/n_batches)

    verbose = True
    is_dense=False
    #%%
    ## first element for p=10, second element for p=50
    fix_params=True
    # scale_spike_fixed =0.0033341
    
    blocked_params_list = ["scale_spike", "mu"]

    rho_tilde_init = jnp.zeros((int(p*(p-1)/2),))
    u_init = jnp.ones((int(p*(p-1)/2),))*0.5
    
    if my_covariates is not None:
        if b_init is None:
            b_init = jnp.zeros((q,))
    else:
        mu_fixed = mu_init = jnp.zeros((p,))
    sqrt_diag_init = jnp.ones((p,))


    # init strategy
    if no_networks:
        fixed_params_dict = {'tilde_eta0_coefs':jnp.array([0.]*n_nets),
                             'tilde_eta1_coefs':jnp.array([0.]*n_nets),
                             'tilde_eta2_coefs':jnp.array([0.]*n_nets),}
        
        if init_strategy=='init_to_value':
            if my_covariates is not None:
                init_dict = {'rho_tilde':rho_tilde_init, 
                                                            'u':u_init,
                                                            # "b_regression_coefs":b_init,
                                                            'tilde_b_regression_coefs':b_init, 
                                                            'sqrt_diag':sqrt_diag_init, 
                                                            # 'eta0_0':my_model_args['eta0_0_m'],
                                                            # 'eta1_0':my_model_args['eta1_0_m'],
                                                            # 'eta2_0':my_model_args['eta2_0_m'],                                     
                                                            'tilde_eta0_0':0.,
                                                            'tilde_eta1_0':0.,
                                                            'tilde_eta2_0':0.,}
                my_init_strategy = init_to_value(values=init_dict)
                fixed_params_dict = {'scale_spike':scale_spike_fixed, 
                                     'tilde_eta0_coefs':jnp.array([0.]*n_nets),
                                     'tilde_eta1_coefs':jnp.array([0.]*n_nets),
                                     'tilde_eta2_coefs':jnp.array([0.]*n_nets),}
                print(f"Fixed scale_spike: {scale_spike_fixed}",)

            else:
                init_dict = {'rho_tilde':rho_tilde_init, 
                                                'u':u_init,
                                                'mu':mu_init, 
                                                'sqrt_diag':sqrt_diag_init, 
                                                # 'eta0_0':my_model_args['eta0_0_m'],
                                                # 'eta1_0':my_model_args['eta1_0_m'],
                                                # 'eta2_0':my_model_args['eta2_0_m'],                                     
                                                'tilde_eta0_0':0.,
                                                'tilde_eta1_0':0.,
                                                'tilde_eta2_0':0.,}
                my_init_strategy = init_to_value(values=init_dict)

                fixed_params_dict = {'scale_spike':scale_spike_fixed, 
                                     'mu':mu_fixed,
                                     'tilde_eta0_coefs':jnp.array([0.]*n_nets),
                                     'tilde_eta1_coefs':jnp.array([0.]*n_nets),
                                     'tilde_eta2_coefs':jnp.array([0.]*n_nets),}
                print(f"Fixed scale_spike: {scale_spike_fixed}",)

        
        elif init_strategy=='init_to_feasible':
            my_init_strategy = init_to_feasible()
        else:
            raise ValueError("Init strategy should be set to 'init_to_value' or 'init_to_feasible")


    else:
        fixed_params_dict = {}
        if init_strategy=='init_to_value':
            if my_covariates is not None:
                init_dict = {'rho_tilde':rho_tilde_init, 
                                                            'u':u_init,
                                                            # "b_regression_coefs":b_init,
                                                            'tilde_b_regression_coefs':b_init, 
                                                            'sqrt_diag':sqrt_diag_init, 
                                                            # 'eta0_0':my_model_args['eta0_0_m'],
                                                            # 'eta1_0':my_model_args['eta1_0_m'],
                                                            # 'eta2_0':my_model_args['eta2_0_m'],                                     
                                                            # 'eta0_coefs':jnp.array([my_model_args['eta0_coefs_m']]*n_nets),
                                                            # 'eta1_coefs':jnp.array([my_model_args['eta1_coefs_m']]*n_nets),
                                                            # 'eta2_coefs':jnp.array([my_model_args['eta2_coefs_m']]*n_nets),})
                                                            'tilde_eta0_0':0.,
                                                            'tilde_eta1_0':0.,
                                                            'tilde_eta2_0':0.,                                     
                                                            'tilde_eta0_coefs':jnp.array([0.]*n_nets),
                                                            'tilde_eta1_coefs':jnp.array([0.]*n_nets),
                                                            'tilde_eta2_coefs':jnp.array([0.]*n_nets),}
                my_init_strategy = init_to_value(values=init_dict)
                fixed_params_dict = {"scale_spike":scale_spike_fixed}
                print(f"Fixed scale_spike: {scale_spike_fixed}")

            else:
                init_dict = {'rho_tilde':rho_tilde_init, 
                                                'u':u_init,
                                                'mu':mu_init, 
                                                'sqrt_diag':sqrt_diag_init, 
                                                # 'eta0_0':my_model_args['eta0_0_m'],
                                                # 'eta1_0':my_model_args['eta1_0_m'],
                                                # 'eta2_0':my_model_args['eta2_0_m'],                                     
                                                # 'eta0_coefs':jnp.array([my_model_args['eta0_coefs_m']]*n_nets),
                                                # 'eta1_coefs':jnp.array([my_model_args['eta1_coefs_m']]*n_nets),
                                                # 'eta2_coefs':jnp.array([my_model_args['eta2_coefs_m']]*n_nets),})
                                                'tilde_eta0_0':0.,
                                                'tilde_eta1_0':0.,
                                                'tilde_eta2_0':0.,                                     
                                                'tilde_eta0_coefs':jnp.array([0.]*n_nets),
                                                'tilde_eta1_coefs':jnp.array([0.]*n_nets),
                                                'tilde_eta2_coefs':jnp.array([0.]*n_nets),}
                my_init_strategy = init_to_value(values=init_dict)

                fixed_params_dict = {"scale_spike":scale_spike_fixed, "mu":mu_fixed}



        elif init_strategy=='init_to_feasible':
            my_init_strategy = init_to_feasible()
        else:
            raise ValueError("Init strategy should be set to 'init_to_value' or 'init_to_feasible")

    if ((my_model == models.NetworkSS_repr_etaRepr_loglikRepr)|(my_model == models.NetworkSS_repr_loglikRepr)):
        y_bar = my_vals.mean(axis=0) #p
        S_bar = my_vals.T@my_vals/n - jnp.outer(y_bar, y_bar) #(p,p)
        my_model_args.update({"y_bar":y_bar, "S_bar":S_bar, "n":n, "p":p,})
    elif ((my_model == models.NetworkSS_repr_etaRepr)|(my_model == models.NetworkSS_repr)):
        my_model_args.update({"Y":my_vals, "n":n, "p":p,})
    elif ((my_model == models.NetworkSS_regression_repr_etaRepr)|(my_model == models.NetworkSS_regression_repr_etaRepr_centered)):
        my_model_args.update({"Y":my_vals, "X":my_covariates,"q":q,"n":n, "p":p,})
    elif (my_model == models.NetworkSS_regression_repr_etaRepr_loglikRepr):
        S_bar_y = my_vals.T@my_vals/n #(p,p)
        S_bar_x = my_covariates.T@my_covariates/n #(q,q)
        S_bar_yx = my_vals.T@my_covariates/n #(p,q)
        my_model_args.update({"S_bar_y":S_bar_y, "S_bar_x":S_bar_x, "S_bar_yx":S_bar_yx,
                              "n":n, "p":p, "q":q})
    else:
        raise ValueError("Insert valid model name")
    
    if init_strategy=='init_to_value':
        init_dict_energy = init_dict.copy() 
        init_dict_energy.update(fixed_params_dict)
        log_joint, _ = log_density(my_model,(), my_model_args, init_dict_energy)
        print(f"Initial negative log joint: {-log_joint}")


    #%%
    # run model
    if fix_params:
        my_model_run = block(condition(my_model, fixed_params_dict), hide=blocked_params_list)
    else:
        my_model_run = my_model
    #%%    

    nuts_kernel = NUTS(my_model_run, init_strategy=my_init_strategy, dense_mass=is_dense)
    mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=batch, thinning=thinning, num_chains=1, progress_bar=verbose)
    if store_warmup:
        mcmc.warmup(rng_key=Key(seed+3), **my_model_args, collect_warmup=True)
        s_w = jax.device_put(mcmc.get_samples(), cpus[0])
    mcmc.run(rng_key = Key(seed), **my_model_args,
            extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state', 'z'))
    # for b in range(n_batches-1):
    #     sample_batch = mcmc.get_samples()
    #     mcmc.post_warmup_state = mcmc.last_state
    #     mcmc.run(mcmc.post_warmup_state.rng_key, Y=covid_vals, **my_model_args,
    #         extra_fields=('potential_energy','accept_prob', 'num_steps', 'adapt_state'))  # or mcmc.run(random.PRNGKey(1))


    # %%

    # mask = (jnp.arange(n_samples)%thinning==0)
    s = jax.device_put(mcmc.get_samples(), cpus[0])
    if store_warmup:
        s.update({'warmup':s_w})
    s.update({'potential_energy':mcmc.get_extra_fields()['potential_energy']})
    # why doesn't the following work with dictionary comprehension?
    # ss = {}
    # for k,v in s.items():
    #     ss[k] = v[mask]

    with open(data_save_path + f'NetworkSS_1mcmc_p{p}_w{n_warmup}_s{n_samples}_CP{n_samples}{"_regression" if my_covariates is not None else ""}{"_nonetworks" if no_networks is not None else ""}.sav' , 'wb') as f:
        pickle.dump((s), f)