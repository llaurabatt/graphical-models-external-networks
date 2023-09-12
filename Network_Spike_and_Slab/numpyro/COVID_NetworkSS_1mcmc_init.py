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
def mcmc1_init(covid_vals,
               A_list,
               my_model,
               my_model_args,
               root_dir,
               data_save_path,
               thinning:Optional[int]=0,
        ):

    #%%
    # paths
    _ROOT_DIR = root_dir
    os.chdir(_ROOT_DIR + 'graphical-models-external-networks/')
    sys.path.append(_ROOT_DIR + "graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

    import models
    import my_utils
    #%%
    n,p = covid_vals.shape
    n_nets = len(A_list)
    print(f"NetworkSS, n {n}, p {p}, number of networks {n_nets}")
    #%%

    #%%
    ## params
    n_warmup = 1000
    n_samples = 500
    n_batches = 1
    batch = int(n_samples/n_batches)

    verbose = True
    is_dense=False
    #%%
    ## first element for p=10, second element for p=50
    fix_params=True
    mu_fixed=jnp.zeros((p,))
    scale_spike_fixed =0.003
    fixed_params_dict = {"scale_spike":scale_spike_fixed, "mu":mu_fixed}
    blocked_params_list = ["scale_spike", "mu"]

    rho_tilde_init = jnp.zeros((int(p*(p-1)/2),))
    u_init = jnp.ones((int(p*(p-1)/2),))*0.5
    mu_init = jnp.zeros((p,))
    sqrt_diag_init = jnp.ones((p,))

    # init strategy
    my_init_strategy = init_to_value(values={'rho_tilde':rho_tilde_init, 
                                                'u':u_init,
                                                'mu':mu_init, 
                                                'sqrt_diag':sqrt_diag_init, 
                                                'eta0_0':my_model_args['eta0_0_m'],
                                                'eta1_0':my_model_args['eta1_0_m'],
                                                'eta2_0':my_model_args['eta2_0_m'],                                     
                                                'eta0_coefs':jnp.array([my_model_args['eta0_coefs_m']]*n_nets),
                                                'eta1_coefs':jnp.array([my_model_args['eta1_coefs_m']]*n_nets),
                                                'eta2_coefs':jnp.array([my_model_args['eta2_coefs_m']]*n_nets),})
                                                # 'tilde_eta0_0':0.,
                                                # 'tilde_eta1_0':0.,
                                                # 'tilde_eta2_0':0.,                                     
                                                # 'tilde_eta0_coefs':jnp.array([0.]*n_nets),
                                                # 'tilde_eta1_coefs':jnp.array([0.]*n_nets),
                                                # 'tilde_eta2_coefs':jnp.array([0.]*n_nets),})




    if ((my_model == models.NetworkSS_repr_etaRepr_loglikRepr)|(my_model == models.NetworkSS_repr_loglikRepr)):
        y_bar = covid_vals.mean(axis=0) #p
        S_bar = covid_vals.T@covid_vals/n - jnp.outer(y_bar, y_bar) #(p,p)
        my_model_args.update({"y_bar":y_bar, "S_bar":S_bar, "n":n, "p":p,})
    elif ((my_model == models.NetworkSS_repr_etaRepr)|(my_model == models.NetworkSS_repr)):
        my_model_args.update({"Y":covid_vals, "n":n, "p":p,})


    #%%
    # run model
    if fix_params:
        my_model_run = block(condition(my_model, fixed_params_dict), hide=blocked_params_list)
    else:
        my_model_run = my_model
    #%%    

    nuts_kernel = NUTS(my_model_run, init_strategy=my_init_strategy, dense_mass=is_dense)
    mcmc = MCMC(nuts_kernel, num_warmup=n_warmup, num_samples=batch)
    mcmc.run(rng_key = Key(3), **my_model_args,
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

    with open(data_save_path + f'NetworkSS_1mcmc_p{p}_w{n_warmup}_s{n_samples}_CP{n_samples}.sav' , 'wb') as f:
        pickle.dump((ss), f)