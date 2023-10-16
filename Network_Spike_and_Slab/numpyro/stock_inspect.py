#%%
"""Main script for training the model."""
import debugpy
debugpy.listen(5678)
print('Waiting for debugger')
debugpy.wait_for_client()
print('Debugger attached')
#%%
# imports
from absl import flags
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from IPython.display import display
from IPython.display import display_html 
import sys
import os
import jax
import numpyro
numpyro.set_platform('cpu')
print(jax.lib.xla_bridge.get_backend().platform)
import jax.numpy as jnp
import pickle

#%%
# paths
_ROOT_DIR = "/home/paperspace/"
os.chdir(_ROOT_DIR + 'graphical-models-external-networks/')
sys.path.append(_ROOT_DIR + "graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/Stock/Pre-processed Data/'
data_save_path = _ROOT_DIR + 'stock_NetworkSS_etarepr_loglikrepr_newprior_SEED6/'
# data_save_path2 = _ROOT_DIR + 'NetworkSS_results_etarepr_loglikrepr_newprior_seed6/'

#%%
# with open(data_save_path + 'NetworkSS_1mcmc_p332_s1000_aggregate.sav', 'rb') as fr:
# with open(data_save_path + 'NetworkSS_1mcmc_p332_w50_s400_CP900.sav', 'rb') as fr:
with open(data_save_path + 'NetworkSS_1mcmc_p366_w1000_s10000_CP10000.sav', 'rb') as fr:
# with open(data_save_path + 'NetworkSS_2mcmc_p332_w3_s20.sav', 'rb') as fr:
    res_ss_E_P = pickle.load(fr)
    
# with open(data_save_path2 + 'NetworkSS_1mcmc_p332_w1000_s10000_CP10000.sav', 'rb') as fr:
#     res_ss_geo_sci2 = pickle.load(fr)

uni_cols = ['eta0_0', 'eta1_0', 'eta2_0', 'tilde_eta0_0', 'tilde_eta1_0', 'tilde_eta2_0', 'potential_energy']
res_ss_E_P = {k:v[:,None] if k in uni_cols else v for k,v in res_ss_E_P.items() }
# res_ss_E_P2 = {k:v[:,None] if k in uni_cols else v for k,v in res_ss_E_P2.items() }

#%%
# res_merge = {} 
# for k in res_ss_res_ss_E_P2.keys():
#     res_merge[k] = jnp.vstack([res_ss_res_ss_E_P[k], res_ss_res_ss_E_P2[k]])


#%%
all_res = {"NetworkSS_E_P":res_ss_E_P}
net_no = 2
#%%
# NetworkSS_geo_sci 
cols = ["eta0_0", "eta1_0","eta2_0"]
cols_2 = [ "eta0_coefs", "eta1_coefs", "eta2_coefs"]
names = ['E', 'P']
etas_NetworkSS = {}
#%%
for k in cols:
    etas_NetworkSS[k] = {'mean': all_res['NetworkSS_E_P'][k].mean(0),#[0],
              'ESS': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_E_P'][k],0))['Param:0']['n_eff'],
               'r_hat': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_E_P'][k],0))['Param:0']['r_hat'],
                }
#%%
for k in cols_2:
    for net_ix in range(net_no):
        etas_NetworkSS[f'{k}_{names[net_ix]}'] = {'mean': all_res['NetworkSS_E_P'][k].mean(0)[net_ix],
                  'ESS': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_E_P'][k][:,net_ix].flatten(),0))['Param:0']['n_eff'],
                   'r_hat': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_E_P'][k][:,net_ix].flatten(),0))['Param:0']['r_hat'],
                    }
        
df_NetworkSS_etas_spec = pd.DataFrame.from_dict(etas_NetworkSS, orient='index')
df_NetworkSS_etas_spec['r_hat-1']  = df_NetworkSS_etas_spec.r_hat -1 

# %%
display(df_NetworkSS_etas_spec)
# %%
plt.suptitle('Potential energy')
plt.plot(all_res['NetworkSS_E_P']['potential_energy'])
plt.show()
# %%
rho_no = all_res['NetworkSS_E_P']['rho_lt'].shape[1]
rho_ESS = []
for rho_ix in range(rho_no):
    rho_ESS.append(numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_E_P']['rho_lt'][:,rho_ix],0))['Param:0']['n_eff'])
rho_ESS = jnp.array(rho_ESS)
# %%
rho_no = all_res['NetworkSS_E_P']['rho_lt'].shape[1]
rho_rhat = []
for rho_ix in range(rho_no):
    rho_rhat.append(numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_E_P']['rho_lt'][:,rho_ix],0))['Param:0']['r_hat'])
rho_rhat = jnp.array(rho_rhat)
# %%
print('Total rho number:', len(rho_ESS))
print('ESS stats:')
stats = {'mean':float(rho_ESS.mean()), 'std':float(rho_ESS.std()), 'median':float(jnp.median(rho_ESS)), 'max':float(rho_ESS.max()), 
'min':float(rho_ESS.min()), '<10':float(sum(rho_ESS<10)), '>100':float(sum(rho_ESS>100))}
display(stats)

print('R hat stats:')
stats = {'mean':float(rho_rhat.mean()), 'std':float(rho_rhat.std()), 'median':float(jnp.median(rho_rhat)), 'max':float(rho_rhat.max()), 
'min':float(rho_rhat.min()), '<1.1':float(sum(rho_rhat<1.1))}
display(stats)

# %%
### For the paper
print(max(np.abs(df_NetworkSS_etas_spec['r_hat-1'])))

with open(data_save_path + 'NetworkSS_2mcmc_p366_w1000_s2000_seed9.sav', 'rb') as fr:
    mcmc2_ss_E_P = pickle.load(fr)

# %%
rho_no = mcmc2_ss_E_P['rho_lt'].shape[1]
rho_ESS = []
for rho_ix in range(rho_no):
    rho_ESS.append(numpyro.diagnostics.summary(jnp.expand_dims(mcmc2_ss_E_P['rho_lt'][:,rho_ix],0))['Param:0']['n_eff'])
rho_ESS = jnp.array(rho_ESS)
# %%
rho_no = mcmc2_ss_E_P['rho_lt'].shape[1]
rho_rhat = []
for rho_ix in range(rho_no):
    rho_rhat.append(numpyro.diagnostics.summary(jnp.expand_dims(mcmc2_ss_E_P['rho_lt'][:,rho_ix],0))['Param:0']['r_hat'])
rho_rhat = jnp.array(rho_rhat)
# %%
print('2mcmc: Total rho number:', len(rho_ESS))
print('ESS stats:')
stats = {'mean':float(rho_ESS.mean()), 'std':float(rho_ESS.std()), 'median':float(jnp.median(rho_ESS)), 'max':float(rho_ESS.max()), 
'min':float(rho_ESS.min()), '<10':float(sum(rho_ESS<10)), '>100':float(sum(rho_ESS>100))}
display(stats)

print('2mcmc: R hat stats:')
stats = {'mean':float(rho_rhat.mean()), 'std':float(rho_rhat.std()), 'median':float(jnp.median(rho_rhat)), 'max':float(rho_rhat.max()), 
'min':float(rho_rhat.min()), '<1.1':float(sum(rho_rhat<1.1))}
display(stats)  

# %%
# stock_vals = jnp.array(pd.read_csv(data_path + 'COVID_332_meta_pruned.csv', index_col='Unnamed: 0').values)
# n, p = stock_vals.shape
# mu = jnp.zeros(p)
# y_bar = stock_vals.mean(axis=0) #p
# S_bar = stock_vals.T@stock_vals/n - jnp.outer(y_bar, y_bar) #(p,p)
# precision_matrix = jnp.identity(p)*0.75 + jnp.ones((p,p))*0.25
# %%
# def loglik(mu, precision_matrix, y_bar, S_bar, n, p): 
#     slogdet = jnp.linalg.slogdet(precision_matrix)
#     return n*(-0.5*p*jnp.log(2*jnp.pi) + 0.5*slogdet[0]*slogdet[1] - 0.5*((y_bar - mu).T)@precision_matrix@(y_bar - mu) - 0.5*jnp.trace(S_bar@precision_matrix))
# # %%
# loglik(mu=mu, precision_matrix=precision_matrix, 
#        y_bar=y_bar, S_bar=S_bar, n=n, p=p)
# %%
filename =  _ROOT_DIR + 'MERGE3_6_NetworkSS_results_etarepr_loglikrepr_newprior/'
if not os.path.exists(filename):
    os.makedirs(filename, mode=0o777)

with open(filename + f"Merge_NetworkSS_1mcmc_p332_w1000_s{res_merge['eta0_0'].shape[0]}.sav" , 'wb') as f:
    pickle.dump((res_merge), f)
# %%
