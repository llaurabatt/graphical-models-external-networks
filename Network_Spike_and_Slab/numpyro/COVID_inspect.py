
#%%
# imports
from absl import flags
#%%
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

data_save_path_COVID_merge = _ROOT_DIR + 'MERGE_6_9_NetworkSS_results_regression_etarepr_brepr_newprior_centered_rotated/'
data_save_path_COVID_9 = _ROOT_DIR + 'NetworkSS_results_regression_etarepr_brepr_newprior_seed9_centered_rotated/'
data_save_path_COVID_9_spikedivide5 = _ROOT_DIR + 'NetworkSS_results_regression_etarepr_brepr_newprior_seed9_centered_rotated_spikedivide5/'
data_save_path_COVID_9_spiketimes2 = _ROOT_DIR + 'NetworkSS_results_regression_etarepr_brepr_newprior_seed9_centered_rotated_times2/'
data_save_path_COVID_9_nonetworks = _ROOT_DIR + 'NetworkSS_results_regression_etarepr_brepr_newprior_seed9_centered_rotated_NONETWORKS/'

data_path = './Data/COVID/Pre-processed Data/'
data_save_path = data_save_path_COVID_9_spiketimes2 
# data_save_path2 = 

#%%


# with open(data_save_path + 'NetworkSS_2mcmc_p332_w1000_s10000_regression.sav', 'rb') as fr:
#     res_ss_geo_sci = pickle.load(fr)

# with open(data_save_path + 'NetworkSS_1mcmc_p332_w10_s10_CP10_regression_nonetworks.sav', 'rb') as fr:
#     res_ss_geo_sci = pickle.load(fr) 

with open(data_save_path + 'NetworkSS_1mcmc_p332_w1000_s10000_CP10000_regression_init_all_path.sav', 'rb') as fr:
    res_ss_geo_sci = pickle.load(fr)

# with open(data_save_path2 + 'NetworkSS_1mcmc_p332_w1000_s10000_CP10000_regression.sav', 'rb') as fr:
#     res_ss_geo_sci2 = pickle.load(fr)


uni_cols = ['eta0_0', 'eta1_0', 'eta2_0', 'tilde_eta0_0', 'tilde_eta1_0', 'tilde_eta2_0', 'potential_energy']
res_ss_geo_sci = {k:v[:,None] if k in uni_cols else v for k,v in res_ss_geo_sci.items() }
# res_ss_geo_sci2 = {k:v[:,None] if k in uni_cols else v for k,v in res_ss_geo_sci2.items() }

#%%
res_merge = {} 
for k in res_ss_geo_sci2.keys():
    if k != 'warmup':
        res_merge[k] = jnp.vstack([res_ss_geo_sci[k], res_ss_geo_sci2[k]])


#%%
all_res = {"NetworkSS_geo_sci":res_ss_geo_sci}
net_no = 3
#%%
# NetworkSS_geo_sci 
cols = ["eta0_0", "eta1_0","eta2_0"]
cols_2 = [ "eta0_coefs", "eta1_coefs", "eta2_coefs"]
names = ['geo', 'sci', 'flights']
etas_NetworkSS = {}
#%%
for k in cols:
    etas_NetworkSS[k] = {'start_val': all_res['NetworkSS_geo_sci'][k][0,:],
                         'mean': all_res['NetworkSS_geo_sci'][k].mean(0),#[0],
              'ESS': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci'][k],0))['Param:0']['n_eff'],
               'r_hat': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci'][k],0))['Param:0']['r_hat'],
                }
#%%
for k in cols_2:
    for net_ix in range(net_no):
        etas_NetworkSS[f'{k}_{names[net_ix]}'] = {'start_val': all_res['NetworkSS_geo_sci'][k][0,net_ix],
                                                  'mean': all_res['NetworkSS_geo_sci'][k].mean(0)[net_ix],
                  'ESS': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci'][k][:,net_ix].flatten(),0))['Param:0']['n_eff'],
                   'r_hat': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci'][k][:,net_ix].flatten(),0))['Param:0']['r_hat'],
                    }
        
df_NetworkSS_etas_spec = pd.DataFrame.from_dict(etas_NetworkSS, orient='index')
df_NetworkSS_etas_spec['r_hat-1']  = df_NetworkSS_etas_spec.r_hat -1 
# %%
display(df_NetworkSS_etas_spec)
# %%
reg_mean = all_res['NetworkSS_geo_sci']['tilde_b_regression_coefs'].mean(axis=0)
# reg_mean2 = res_ss_geo_sci2['tilde_b_regression_coefs'].mean(axis=0)
# %%
u_ESS = jnp.array([numpyro.diagnostics.summary(jnp.expand_dims(b,0))['Param:0']['n_eff'] for b in all_res['NetworkSS_geo_sci']['u'].T])

reg_ESS = jnp.array([numpyro.diagnostics.summary(jnp.expand_dims(b,0))['Param:0']['n_eff'] for b in all_res['NetworkSS_geo_sci']['tilde_b_regression_coefs'].T])
reg_rhat = jnp.array([numpyro.diagnostics.summary(jnp.expand_dims(b,0))['Param:0']['r_hat'] for b in all_res['NetworkSS_geo_sci']['tilde_b_regression_coefs'].T])
print('Mean ESS of u:', u_ESS.mean())
print('Mean ESS of regression coefs:', reg_ESS.mean())
print('Mean r_hat of regression coefs:', reg_rhat.mean())

# %%
plt.suptitle('Trace plot first u')
plt.plot(all_res['NetworkSS_geo_sci']['u'][:,0])
plt.legend()
plt.show()
# %%
plt.suptitle('Trace plot first sqrt_diag')
plt.plot(all_res['NetworkSS_geo_sci']['sqrt_diag'][:,0])
plt.legend()
plt.show()
# %%
plt.suptitle('Trace plot first rho tilde')
plt.plot(all_res['NetworkSS_geo_sci']['rho_tilde'][:,0])
plt.legend()
plt.show()
# %%
plt.suptitle('Trace plot first b tilde')
plt.plot(all_res['NetworkSS_geo_sci']['tilde_b_regression_coefs'][:,0])
plt.legend()
plt.show()
# %%
for k in cols:
    plt.suptitle(f'Trace plot {k}')
    plt.plot(all_res['NetworkSS_geo_sci'][f'{k}'])
    plt.legend()
    plt.show()
# %%
for k in cols_2:
    for net_ix in range(net_no):
        plt.suptitle(f'Trace plot {k} Network {net_ix}')
        plt.plot(all_res['NetworkSS_geo_sci'][k][:,net_ix])
        plt.legend()
        plt.show()
# %%
plt.suptitle('Potential energy')
plt.plot(all_res['NetworkSS_geo_sci']['potential_energy'], label='seed 9')
# plt.legend()
plt.show()
# %%
rho_no = all_res['NetworkSS_geo_sci']['rho_tilde'].shape[1]
rho_ESS = []
for rho_ix in range(rho_no):
    rho_ESS.append(numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci']['rho_tilde'][:,rho_ix],0))['Param:0']['n_eff'])
rho_ESS = jnp.array(rho_ESS)
# %%
rho_no = all_res['NetworkSS_geo_sci']['rho_tilde'].shape[1]
rho_rhat = []
for rho_ix in range(rho_no):
    rho_rhat.append(numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci']['rho_tilde'][:,rho_ix],0))['Param:0']['r_hat'])
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
print('R hat stats for rho with ESS>100:')
rho_rhat_goodESS = rho_rhat[jnp.where(rho_ESS>100)[0]]
stats = {'mean':float(rho_rhat_goodESS.mean()), 'std':float(rho_rhat_goodESS.std()), 'median':float(jnp.median(rho_rhat_goodESS)), 'max':float(rho_rhat_goodESS.max()), 
'min':float(rho_rhat_goodESS.min()), '<1.1':float(sum(rho_rhat_goodESS<1.1))}
display(stats)

# %%
### For the paper
print('mean ESS of etas', np.mean(df_NetworkSS_etas_spec['ESS']))
print('min ESS of etas', min(df_NetworkSS_etas_spec['ESS']))
print('mean rhat of etas', np.mean(df_NetworkSS_etas_spec['r_hat']))
print('max abs(rhat-1) of etas:', max(np.abs(df_NetworkSS_etas_spec['r_hat-1'])))
print('mean abs(rhat-1) of etas:', np.mean(np.abs(df_NetworkSS_etas_spec['r_hat-1'])))

# %%
with open(_ROOT_DIR + 'MERGE_3_6_COVID_SS_etarepr_newprior_newlogrepr/NetworkSS_2mcmc_p332_w1000_s4000.sav', 'rb') as fr:
    mcmc2_ss_nets = pickle.load(fr)
# %%
rho_no = mcmc2_ss_nets['rho_lt'].shape[1]
rho_ESS = []
for rho_ix in range(rho_no):
    rho_ESS.append(numpyro.diagnostics.summary(jnp.expand_dims(mcmc2_ss_nets['rho_lt'][:,rho_ix],0))['Param:0']['n_eff'])
rho_ESS = jnp.array(rho_ESS)
# %%
rho_no = mcmc2_ss_nets['rho_lt'].shape[1]
rho_rhat = []
for rho_ix in range(rho_no):
    rho_rhat.append(numpyro.diagnostics.summary(jnp.expand_dims(mcmc2_ss_nets['rho_lt'][:,rho_ix],0))['Param:0']['r_hat'])
rho_rhat = jnp.array(rho_rhat)
# %%
print('2mcmc: Total rho number:', len(rho_ESS))
print('ESS stats:')
stats = {'mean':float(rho_ESS.mean()), 'std':float(rho_ESS.std()), 'median':float(jnp.median(rho_ESS)), 'max':float(rho_ESS.max()), 
'min':float(rho_ESS.min()), '<10':float(sum(rho_ESS<10)), '>100':float(sum(rho_ESS>100))}
display(stats)

print('2mcmc: R hat stats:')
stats = {'mean':float(rho_rhat.mean()), 'std':float(rho_rhat.std()), 'median':float(jnp.median(rho_rhat)), 'max':float(rho_rhat.max()), 
'min':float(rho_rhat.min()), '<1.1':float(sum(rho_rhat<1.1)),
'mean(abs(rhat-1))':np.mean(np.abs(rho_rhat-1)), 'max(abs(rhat-1))':max(np.abs(rho_rhat-1))}
display(stats) 
# %%
# covid_vals = jnp.array(pd.read_csv(data_path + 'COVID_332_meta_pruned.csv', index_col='Unnamed: 0').values)
# n, p = covid_vals.shape
# mu = jnp.zeros(p)
# y_bar = covid_vals.mean(axis=0) #p
# S_bar = covid_vals.T@covid_vals/n - jnp.outer(y_bar, y_bar) #(p,p)
# precision_matrix = jnp.identity(p)*0.75 + jnp.ones((p,p))*0.25
# %%
# def loglik(mu, precision_matrix, y_bar, S_bar, n, p): 
#     slogdet = jnp.linalg.slogdet(precision_matrix)
#     return n*(-0.5*p*jnp.log(2*jnp.pi) + 0.5*slogdet[0]*slogdet[1] - 0.5*((y_bar - mu).T)@precision_matrix@(y_bar - mu) - 0.5*jnp.trace(S_bar@precision_matrix))
# # %%
# loglik(mu=mu, precision_matrix=precision_matrix, 
#        y_bar=y_bar, S_bar=S_bar, n=n, p=p)
# %%
filename =  _ROOT_DIR + 'MERGE_6_9_NetworkSS_results_regression_etarepr_brepr_newprior_centered_rotated/'
if not os.path.exists(filename):
    os.makedirs(filename, mode=0o777)

with open(filename + f"Merge_NetworkSS_1mcmc_p332_w1000_s{res_merge['eta0_0'].shape[0]}_regression.sav" , 'wb') as f:
    pickle.dump((res_merge), f)
# %%
