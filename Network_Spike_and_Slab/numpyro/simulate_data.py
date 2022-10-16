#%%
import sys
sys.path.append('/Users/llaurabat/opt/anaconda3/lib/python3.8/site-packages')
#%%
# imports
import sys
import os
import pickle
# import json
import jax
import numpyro
numpyro.set_platform('cpu')
print(jax.lib.xla_bridge.get_backend().platform)
import jax.numpy as jnp


# paths
os.chdir("/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/graphical-regression-with-networks/numpyro/FINAL_ALL")
sys.path.append("functions")
import my_utils
data_save_path = './data/sim_data'

# parameters
p_list = [10, 50]
n = 2000
n_sims = 50
w_slab_1 = 0.95
w_slab_0_numerator = 0.5
w_slab_1, w_slab_0_numerator

slab_0_low=-0.1
slab_0_up=0.1
slab_1_low=0.2
slab_1_up=0.5

# simulate data
for p in p_list:
    print('--------------------------------------------------------------------------------')
    print(f"Dimensions: n={n}, p={p}")
    print('--------------------------------------------------------------------------------')
    
    w_slab_0 = w_slab_0_numerator/p


    mu_true = jnp.zeros(p)
    offset_1 = jnp.ones((p-1,))
    
    print(" ")
    print("Generate A true...")
    A_true = jnp.diag(offset_1,1) + jnp.diag(offset_1,-1) + jnp.diag(jnp.ones((p,)))
    A_scaled_true = my_utils.network_scale(A_true)

    print(" ")
    print("Generate A indep...")
    A_indep = my_utils.network_simulate(p=p, A_true=A_true, flip_prop=0.5)
    A_scaled_indep = my_utils.network_scale(A_indep)

    print(" ")
    print("Generate A semi-dep...")
    A_semi_dep = my_utils.network_simulate(p=p,  A_true=A_true, flip_prop=0.25)
    A_scaled_semi_dep = my_utils.network_scale(A_semi_dep)
    
    print(" ")
    print("Generate A semi-dep85...")
    A_semi_dep85 = my_utils.network_simulate(p=p,  A_true=A_true, flip_prop=0.15)
    A_scaled_semi_dep85 = my_utils.network_scale(A_semi_dep85)

    print(" ")
    print("Generate theta...")
    theta_true = my_utils.generate_theta(A_true=A_true, w_slab_1=w_slab_1, w_slab_0=w_slab_0, 
                                slab_0_low=slab_0_low, slab_0_up=slab_0_up,  
                                slab_1_low=slab_1_low, slab_1_up=slab_1_up,
                                key_no=p+2)

    print(" ")
    for s in range(n_sims):
        print(f"Simulation number: {s}")

        sim_res = my_utils.simulate_data(n_obs=n, p=p, mu_true=mu_true, theta_true=theta_true, key_no=p+s+5)



        sim_res.update({"A_indep":A_indep, "A_scaled_indep":A_scaled_indep, 
                        "A_semi_dep":A_semi_dep, "A_scaled_semi_dep":A_scaled_semi_dep,
                        "A_semi_dep85":A_semi_dep85, "A_scaled_semi_dep85":A_scaled_semi_dep85,
                        "A_true":A_true, "A_scaled_true":A_scaled_true})
        


        # uncomment to save to JSON
#         sim_res_json = {k:np.array(v).tolist() for k,v in sim_res.items()}
#         with open(data_save_path + f'/sim{s}_p{p}_n{n}.json' , 'w') as f:
#             json.dump((sim_res_json), f)
        
        with open(data_save_path + f'/sim{s}_p{p}_n{n}.sav' , 'wb') as f:
            pickle.dump((sim_res), f)
# %%
