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

# paths
# _ROOT_DIR = "/home/user/graphical-models-external-networks/"
_ROOT_DIR = "/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/graphical-models-external-networks/"
os.chdir(_ROOT_DIR)
# sys.path.append("/home/user/graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")
sys.path.append("/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/graphical-models-external-networks/Network_Spike_and_Slab/numpyro/functions")

data_path = './Data/COVID/Pre-processed Data/'
# data_save_path = '/home/user/mounted_folder/NetworkSS_results/'
data_save_path = '/Users/llaurabat/Dropbox/BGSE_work/LJRZH_graphs/NetworkSS_results/'
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path, mode=0o777)
if not os.path.exists(data_save_path + 'Figures/'):
    os.makedirs(data_save_path + 'Figures/', mode=0o777)

# load models and functions
import models
import my_utils

# define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('mcmc1_file', None, 'Name of 1mcmc file to generate plots from.')
flags.DEFINE_string('mcmc2_file', None, 'Name of 2mcmc file to generate plots from.')
flags.DEFINE_string('Y', 'COVID_629_meta.csv', 'Name of file where data for dependent variable is stored.')
flags.DEFINE_multi_string('network_list', ['GEO_clean_629.npy', 'SCI_clean_629.npy'], 'Name of file where network data is stored. Flag can be called multiple times. Order of calling IS relevant.')
flags.mark_flags_as_required(['mcmc1_file', 'mcmc2_file'])
FLAGS(sys.argv)

# load data
covid_vals_name = FLAGS.Y
network_names = FLAGS.network_list
print(network_names)

covid_vals = jnp.array(pd.read_csv(data_path + covid_vals_name, index_col='Unnamed: 0').values)
geo_clean = jnp.array(jnp.load(data_path + network_names[0]))
sci_clean = jnp.array(jnp.load(data_path + network_names[1]))

covid_vals = covid_vals[:,:100].copy()
geo_clean = geo_clean[:100, :100].copy()
sci_clean = sci_clean[:100, :100].copy()

net_no = 2
scale_spike_fixed =0.003

n,p = covid_vals.shape
A_list = [geo_clean, sci_clean]
tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
tril_len = tril_idx[0].shape[0]
A_tril_arr = jnp.array([A[tril_idx] for A in A_list]) # (a, p, p)

# Compile results

# 1MCMC
output_dict_ss_geo_sci = {"eta0_0":[],"eta0_coefs":[], "eta1_0":[],"eta1_coefs":[], "eta2_0":[],"eta2_coefs":[],
                    "w_slab":[], "mean_slab":[], "scale_slab":[],"Pos":[], "Neg":[], "Pos_95":[], "Neg_95":[],}


outputs = {"NetworkSS_geo_sci":output_dict_ss_geo_sci }
        
# with open(data_save_path + f'NetworkSS_geo_sci_1mcmc.sav', 'rb') as fr:
#     res_ss_geo_sci = pickle.load(fr)

# with open(data_save_path + f'NetworkSS_1mcmc_p629_s1500.sav', 'rb') as fr:
with open(data_save_path + FLAGS.mcmc1_file, 'rb') as fr:
    res_ss_geo_sci = pickle.load(fr)

all_res = {"NetworkSS_geo_sci":res_ss_geo_sci}

for i, res in all_res.items():
    print(i)
    n_samples = res["eta1_0"].shape[0]

    outputs[i]["eta0_0"].append(res["eta0_0"].mean(0))
    outputs[i]["eta1_0"].append(res["eta1_0"].mean(0))
    outputs[i]["eta2_0"].append(res["eta2_0"].mean(0))

    outputs[i]["eta0_coefs"].append(res["eta0_coefs"].mean(0))
    outputs[i]["eta1_coefs"].append(res["eta1_coefs"].mean(0))
    outputs[i]["eta2_coefs"].append(res["eta2_coefs"].mean(0))

    mean_slab_chain = []
    for eta0_0_s, eta0_coefs_s in zip(res["eta0_0"], res["eta0_coefs"]):
        A_tril_mean0_s = 0.
        for coef, A in zip(eta0_coefs_s,A_tril_arr):
            A_tril_mean0_s += coef*A
        mean_slab_s = eta0_0_s + A_tril_mean0_s
        mean_slab_chain.append(mean_slab_s)
    mean_slab_chain = jnp.array(mean_slab_chain)
    all_res['NetworkSS_geo_sci'].update({"mean_slab":mean_slab_chain})
    mean_slab = mean_slab_chain.mean(0)
    outputs[i]["mean_slab"].append(mean_slab)

    scale_slab_chain = []
    for eta1_0_s, eta1_coefs_s in zip(res["eta1_0"], res["eta1_coefs"]):
        A_tril_mean1_s = 0.
        for coef, A in zip(eta1_coefs_s,A_tril_arr):
            A_tril_mean1_s += coef*A
        scale_slab_s = scale_spike_fixed*(1+jnp.exp(-eta1_0_s-A_tril_mean1_s))
        scale_slab_chain.append(scale_slab_s)
    scale_slab_chain = jnp.array(scale_slab_chain)
    all_res['NetworkSS_geo_sci'].update({"scale_slab":scale_slab_chain})
    scale_slab = scale_slab_chain.mean(0)
    outputs[i]["scale_slab"].append(scale_slab)

    w_slab_chain = []
    for eta2_0_s, eta2_coefs_s in zip(res["eta2_0"], res["eta2_coefs"]):
        A_tril_mean2_s = 0.
        for coef, A in zip(eta2_coefs_s,A_tril_arr):
            A_tril_mean2_s += coef*A
        w_slab_s = 1/(1+jnp.exp(-eta2_0_s -A_tril_mean2_s)) 
        w_slab_chain.append(w_slab_s)
    w_slab_chain = jnp.array(w_slab_chain)
    all_res['NetworkSS_geo_sci'].update({"w_slab":w_slab_chain})
    w_slab = w_slab_chain.mean(0)
    outputs[i]["w_slab"].append(w_slab)

    prob_slab_all = []
    for cs in range(n_samples):
        prob_slab = my_utils.get_prob_slab(rho_lt=res['rho_lt'][cs], 
                                        mean_slab=mean_slab_chain[cs], 
                                        scale_slab=scale_slab_chain[cs], 
                                        scale_spike=scale_spike_fixed, 
                                        w_slab=w_slab_chain[cs], 
                                        w_spike=(1-w_slab_chain)[cs])
        prob_slab_all.append(prob_slab)
    prob_slab_est = (jnp.array(prob_slab_all)).mean(0)    
    nonzero_preds_5 = (prob_slab_est>0.5).astype(int)
    nonzero_preds_95 = (prob_slab_est>0.95).astype(int)

    Pos_5 = jnp.where(nonzero_preds_5 == True)[0].shape[0]
    Neg_5 = jnp.where(nonzero_preds_5 == False)[0].shape[0]
  
    Pos_95 = jnp.where(nonzero_preds_95 == True)[0].shape[0]
    Neg_95 = jnp.where(nonzero_preds_95 == False)[0].shape[0]

    outputs[i]['Pos'].append(Pos_5)
    outputs[i]['Neg'].append(Neg_5)
    outputs[i]['Pos_95'].append(Pos_95)
    outputs[i]['Neg_95'].append(Neg_95)

outputs_1MCMC = outputs

# 2MCMC

output_dict_ss_geo_sci = {"eta0_0":[],"eta0_coefs":[], "eta1_0":[],"eta1_coefs":[], "eta2_0":[],"eta2_coefs":[],
                    "w_slab":[], "mean_slab":[], "scale_slab":[],"Pos":[], "Neg":[], "Pos_95":[], "Neg_95":[],}


outputs = {"NetworkSS_geo_sci":output_dict_ss_geo_sci}

# with open(data_save_path + f'NetworkSS_2mcmc_p629_s1500.sav', 'rb') as fr:
with open(data_save_path + FLAGS.mcmc2_file, 'rb') as fr:
    mcmc2_ss_geo_sci = pickle.load(fr)


all_res_2MCMC = {"NetworkSS_geo_sci":mcmc2_ss_geo_sci}
for i, res in all_res_2MCMC.items():
    print(i)

    n_samples = res["rho_lt"].shape[0]

    A_tril_mean0 = 0.
    for coef, A in zip(res['fixed_params_dict']["eta0_coefs"],A_tril_arr):
        A_tril_mean0 += coef*A
    mean_slab = res['fixed_params_dict']["eta0_0"] + A_tril_mean0

    A_tril_mean1 = 0.
    for coef, A in zip(res['fixed_params_dict']["eta1_coefs"],A_tril_arr):
        A_tril_mean1 += coef*A
    scale_slab = scale_spike_fixed*(1+jnp.exp(-res['fixed_params_dict']["eta1_0"]-A_tril_mean1))

    A_tril_mean2 = 0.
    for coef, A in zip(res['fixed_params_dict']["eta2_coefs"],A_tril_arr):
        A_tril_mean2 += coef*A
    w_slab = 1/(1+jnp.exp(-res['fixed_params_dict']["eta2_0"] -A_tril_mean2)) 

    prob_slab_all = []
    for cs in range(n_samples):
        prob_slab = my_utils.get_prob_slab(rho_lt=res['rho_lt'][cs], 
                                        mean_slab=mean_slab, 
                                        scale_slab=scale_slab, 
                                        scale_spike=scale_spike_fixed, 
                                        w_slab=w_slab, 
                                        w_spike=(1-w_slab)[cs])
        prob_slab_all.append(prob_slab)
    prob_slab_est = (jnp.array(prob_slab_all)).mean(0)    
    nonzero_preds_5 = (prob_slab_est>0.5).astype(int)
    nonzero_preds_95 = (prob_slab_est>0.95).astype(int)

    Pos_5 = jnp.where(nonzero_preds_5 == True)[0].shape[0]
    Neg_5 = jnp.where(nonzero_preds_5 == False)[0].shape[0]
  
    Pos_95 = jnp.where(nonzero_preds_95 == True)[0].shape[0]
    Neg_95 = jnp.where(nonzero_preds_95 == False)[0].shape[0]

    outputs[i]['Pos'].append(Pos_5)
    outputs[i]['Neg'].append(Neg_5)
    outputs[i]['Pos_95'].append(Pos_95)
    outputs[i]['Neg_95'].append(Neg_95)


outputs_2MCMC = outputs

# Summary

cols = ['Pos', 'Neg', 'Pos_95', 'Neg_95']

res_dict_1MCMC = {"NetworkSS_geo_sci":[]}
for k, output in outputs_1MCMC.items():
    for col in cols:
        try:
            res_dict_1MCMC[k].append(round(jnp.array(output[col]).mean(0),3))
        except:
            res_dict_1MCMC[k].append(np.nan)

cols = ['Pos', 'Neg', 'Pos_95', 'Neg_95']

res_dict_2MCMC = {"NetworkSS_geo_sci":[]}
for k, output in outputs_2MCMC.items():
    for col in cols:
        try:
            res_dict_2MCMC[k].append(round(jnp.array(output[col]).mean(0),3))
        except:
            res_dict_2MCMC[k].append(np.nan)

print('Results for 1MCMC')
df = pd.DataFrame.from_dict(res_dict_1MCMC, orient='index', 
                       columns=cols).loc[['NetworkSS_geo_sci']]
display(df)

print('Results for 2MCMC')
df = pd.DataFrame.from_dict(res_dict_2MCMC, orient='index', 
                       columns=cols).loc[['NetworkSS_geo_sci']]

df.index = ['GOLAZO_SS_geo_sci_2mcmc']
display(df)

# NetworkSS_geo_sci 
cols = ["eta0_0", "eta1_0","eta2_0"]
cols_2 = [ "eta0_coefs", "eta1_coefs", "eta2_coefs"]
names = ['E', 'P']
etas_NetworkSS = {}
for k in cols:
    etas_NetworkSS[k] = all_res_2MCMC['NetworkSS_geo_sci']['fixed_params_dict'][k]
    
for k in cols:
    etas_NetworkSS[k] = {'MAP': all_res_2MCMC['NetworkSS_geo_sci']['fixed_params_dict'][k],
               'mean': all_res['NetworkSS_geo_sci'][k].mean(0)[0],
              'ESS': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci'][k],0))['Param:0']['n_eff'],
               'r_hat': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci'][k],0))['Param:0']['r_hat'],
                }

for k in cols_2:
    for net_ix in range(net_no):
        etas_NetworkSS[f'{k}_{names[net_ix]}'] = {'MAP': all_res_2MCMC['NetworkSS_geo_sci']['fixed_params_dict'][k][net_ix],
                   'mean': all_res['NetworkSS_geo_sci'][k].mean(0)[net_ix],
                  'ESS': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci'][k][:,net_ix].flatten(),0))['Param:0']['n_eff'],
                   'r_hat': numpyro.diagnostics.summary(jnp.expand_dims(all_res['NetworkSS_geo_sci'][k][:,net_ix].flatten(),0))['Param:0']['r_hat'],
                    }
        
df_NetworkSS_etas_spec = pd.DataFrame.from_dict(etas_NetworkSS, orient='index')


display(df_NetworkSS_etas_spec)

tril_idx = jnp.tril_indices(n=p, k=-1, m=p)
A_tril_geo = geo_clean[tril_idx]
A_tril_sci = sci_clean[tril_idx]

# NetworkSS

nbins=10

fixed_params_MAP_dict = all_res_2MCMC['NetworkSS_geo_sci']['fixed_params_dict']

negative_fixed_params_MAP_dict = {k:(-v if 'eta0' in k else v) 
                              for k,v in fixed_params_MAP_dict.items()
                             }

etas_dict = negative_fixed_params_MAP_dict

A_geo_ints_10, A_geo_mids_10  = my_utils.get_density_els_marginal(A_tril=A_tril_geo, 
                                                                  A_tril_pos=0, 
                                                                len_A_list=2, nbins=nbins, 
                         eta_dict=etas_dict)

A_sci_ints_10, A_sci_mids_10 = my_utils.get_density_els_marginal(A_tril=A_tril_sci, 
                                                                 A_tril_pos=1, 
                                                                len_A_list=2, nbins=nbins, 
                         eta_dict=etas_dict)

## Final plots

def get_credible_interval(post_chain):
    # Credible Intervals
    sorted_arr = np.sort(post_chain) # for each K sort chain
    len_sample = post_chain.shape[0] # len chain

    # 2.5% percentile: if integer pick value, else average of values at pos5 and pos5+1, recall python idx at 0 (lower bound)
    pos025 = 0.025*len_sample
    if pos025 == int(pos025):
        lb025 = sorted_arr[max(int(pos025)-1,0)]
    else:
        lb025 = (sorted_arr[max(int(pos025)-1,0)] + sorted_arr[int(pos025)])/2

    # 97.5% percentile: if integer pick value, else average of values at pos95 and pos95+1, recall python idx at 0 (upper bound)
    pos975 = 0.975*len_sample
    if pos975 == int(pos975):
        ub975 = sorted_arr[int(pos975)-1]
    else:
        ub975 = (sorted_arr[(int(pos975)-1)] + sorted_arr[int(pos975)])/2
        
    return (jnp.round(lb025,3), jnp.round(ub975,3))


df_MAP_dict = {'intercept':{'eta0_m':jnp.round(df_NetworkSS_etas_spec['MAP']['eta0_0'],3),
                        'eta0_CI':get_credible_interval(res_ss_geo_sci['eta0_0']),
                        'eta1_m':jnp.round(df_NetworkSS_etas_spec['MAP']['eta1_0'],3),
                        'eta1_CI':get_credible_interval(res_ss_geo_sci['eta1_0']),
                        'eta2_m':jnp.round(df_NetworkSS_etas_spec['MAP']['eta2_0'],3),
                        'eta2_CI':get_credible_interval(res_ss_geo_sci['eta2_0']),},
               
'geo':{'eta0_m':jnp.round(df_NetworkSS_etas_spec['MAP']['eta0_coefs_E'],3),
                        'eta0_CI':get_credible_interval(res_ss_geo_sci['eta0_coefs'][:,0]),
                        'eta1_m':jnp.round(df_NetworkSS_etas_spec['MAP']['eta1_coefs_E'],3),
                        'eta1_CI':get_credible_interval(res_ss_geo_sci['eta1_coefs'][:,0]),
                        'eta2_m':jnp.round(df_NetworkSS_etas_spec['MAP']['eta2_coefs_E'],3),
                        'eta2_CI':get_credible_interval(res_ss_geo_sci['eta2_coefs'][:,0]),},

'sci':{'eta0_m':jnp.round(df_NetworkSS_etas_spec['MAP']['eta0_coefs_P'],3),
                        'eta0_CI':get_credible_interval(res_ss_geo_sci['eta0_coefs'][:,1]),
                        'eta1_m':jnp.round(df_NetworkSS_etas_spec['MAP']['eta1_coefs_P'],3),
                        'eta1_CI':get_credible_interval(res_ss_geo_sci['eta1_coefs'][:,1]),
                        'eta2_m':jnp.round(df_NetworkSS_etas_spec['MAP']['eta2_coefs_P'],3),
                        'eta2_CI':get_credible_interval(res_ss_geo_sci['eta2_coefs'][:,1]),},
}

df = pd.DataFrame.from_dict(df_MAP_dict)
display(df)

df = pd.DataFrame.from_dict(df_MAP_dict).to_latex()
display(df)

def MODIFIED_get_density_els_marginal(A_tril, A_min, A_max, A_tril_pos, len_A_list, nbins, eta_dict):
    bins = np.histogram(A_tril, bins=nbins)[1]
    delta = jnp.diff(np.histogram(A_tril, bins=nbins)[1])[0]
    if min(bins) > A_min:
        low = bins[0]
        while low>A_min:
            low -= delta
            bins = jnp.concatenate([jnp.array([low]), bins])
    if max(bins) < A_max:
        high = bins[-1]
        while high<A_max:
            high += delta
            bins = jnp.concatenate([bins, jnp.array([high])])

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

        A_key = f'{jnp.round(down,2)} to {jnp.round(up,2)}'
        down = bins[i+2]

        zeros = jnp.zeros((len_A_list))
        A_singlevals = zeros.at[A_tril_pos].set(A_mid)

        par_dict = my_utils.from_etas_to_params(coef_dict=eta_dict, p=1, 
        model='golazo_ss', A_list=A_singlevals)

        A_ints[A_key] = par_dict

    A_mids = jnp.array(A_mids)
    return A_ints, A_mids

# run to have plots in LaTeX format

params = {'font.family': 'serif',
          'text.usetex': True,
          'axes.titlesize': 13,
          'axes.labelsize': 13,
          'xtick.labelsize': 13,
          'ytick.labelsize': 13,
          'legend.fontsize': 9,
          'font.weight': 'bold'}
plt.rcParams.update(params)


################################################################################################

fig, ax = plt.subplots( figsize=(5,4))

nbins=10
MODIFIED_A_geo_ints_10, MODIFIED_A_geo_mids_10  = MODIFIED_get_density_els_marginal(A_tril=A_tril_geo, 
                                                                  A_min=-2, A_max=6.5, 
                                                                  A_tril_pos=0, 
                                                                len_A_list=2, nbins=nbins,  eta_dict=etas_dict)

MODIFIED_A_sci_ints_10, MODIFIED_A_sci_mids_10 = MODIFIED_get_density_els_marginal(A_tril=A_tril_sci, 
                                                                                   A_min=-2, A_max=6.5,
                                                                 A_tril_pos=1, 
                                                                len_A_list=2, nbins=nbins, eta_dict=etas_dict)


MODIFIED_df = pd.DataFrame.from_dict(MODIFIED_A_sci_ints_10, orient='index')
MODIFIED_df = MODIFIED_df.astype('float64')
MODIFIED_df_sci = MODIFIED_df.round(decimals = 5)

MODIFIED_df = pd.DataFrame.from_dict(MODIFIED_A_geo_ints_10, orient='index')
MODIFIED_df = MODIFIED_df.astype('float64')
MODIFIED_df_geo = MODIFIED_df.round(decimals = 5)

dfs = [MODIFIED_df_geo, MODIFIED_df_sci, ]
df_names = ['geo', 'sci', ]
ticklabs = [MODIFIED_A_geo_mids_10, MODIFIED_A_sci_mids_10 ]
legendlabs = ['Geographical Distance Network', 'Facebook Connectivity Index', ]
colors = ['black', 'gray']


for df_ix, df in enumerate(dfs):
    ax.plot(ticklabs[df_ix], df['w_slab'].values, linewidth=1.2, label=legendlabs[df_ix], color=colors[df_ix])

    ax.set_ylabel('Probability of slab')
    ax.set_xlabel('Network values')

fig.legend()
plt.yticks(rotation = 90)
fig.tight_layout()
fig.savefig(data_save_path + 'Figures/' + 'COVID_SS_prob_slab.pdf')
plt.close()
################################################################################################

fig, ax = plt.subplots( figsize=(5,4))

nbins=10
MODIFIED_A_geo_ints_10, MODIFIED_A_geo_mids_10  = MODIFIED_get_density_els_marginal(A_tril=A_tril_geo, 
                                                                  A_min=-2, A_max=6.5, 
                                                                  A_tril_pos=0, 
                                                                len_A_list=2, nbins=nbins,  eta_dict=etas_dict)

MODIFIED_A_sci_ints_10, MODIFIED_A_sci_mids_10 = MODIFIED_get_density_els_marginal(A_tril=A_tril_sci, 
                                                                                   A_min=-2, A_max=6.5,
                                                                 A_tril_pos=1, 
                                                                len_A_list=2, nbins=nbins, eta_dict=etas_dict)


MODIFIED_df = pd.DataFrame.from_dict(MODIFIED_A_sci_ints_10, orient='index')
MODIFIED_df = MODIFIED_df.astype('float64')
MODIFIED_df_sci = MODIFIED_df.round(decimals = 5)

MODIFIED_df = pd.DataFrame.from_dict(MODIFIED_A_geo_ints_10, orient='index')
MODIFIED_df = MODIFIED_df.astype('float64')
MODIFIED_df_geo = MODIFIED_df.round(decimals = 5)

dfs = [MODIFIED_df_geo, MODIFIED_df_sci, ]
df_names = ['geo', 'sci', ]
ticklabs = [MODIFIED_A_geo_mids_10, MODIFIED_A_sci_mids_10 ]
legendlabs = ['Geographical Distance Network', 'Facebook Connectivity Index', ]
colors = ['black', 'gray']

for df_ix, df in enumerate(dfs):
    ax.plot(ticklabs[df_ix], df['mean_slab'].values, linewidth=1.2, label=legendlabs[df_ix], color=colors[df_ix])

    ax.set_ylabel('Slab location')
    ax.set_xlabel('Network values')

fig.legend()
plt.yticks(rotation = 90)
fig.tight_layout()
fig.savefig(data_save_path + 'Figures/' + 'COVID_SS_mean_slab.pdf')
plt.close()
################################################################################################

fig, ax = plt.subplots(figsize=(5,4))

# with open(data_save_path + f'NetworkSS_2mcmc_p629_s1500.sav', 'rb') as fr:
#     mcmc2_ss_geo_sci = pickle.load(fr)
    
x_axis = jnp.arange(-1,1,0.001)
rho_tril = mcmc2_ss_geo_sci['rho_lt'].mean(0)

densities_geo_sci_sci = []
for A_ix, (A_k, vals) in enumerate(A_sci_ints_10.items()):
    density = jnp.exp(jnp.array([models.logprior_NetworkSS(scale_spike=vals['scale_spike'],
                                                           scale_slab=vals['scale_slab'],
                                                           w_slab=vals['w_slab'],
                                                           mean_slab=vals['mean_slab'],
                                                           rho_lt=x) for x in x_axis]))
    densities_geo_sci_sci.append(density)
    ax.plot(density+A_sci_mids_10[A_ix], x_axis, alpha=0.3, c='gray')
    
ax.scatter(A_tril_sci, -rho_tril, s=18, linewidth=0.8, alpha=0.7, color='black', facecolors='none', label='partial correlations')
#ax.set_xlim(-3,14)
ax.set_xlim(-3,12)
#ax.set_ylim(-0.5, 0.7)
ax.set_ylim(-0.3, 0.7)
ax.set_xlabel('Facebook Connectivity Index')
ax.set_ylabel('Partial correlation (Network-SS)')

plt.yticks(rotation = 90) 
fig.tight_layout()
fig.savefig(data_save_path + 'Figures/' + 'COVID_SS_partial_corrs_SCI.pdf')
plt.close()
################################################################################################

fig, ax = plt.subplots(figsize=(5,4))

# with open(data_save_path +f'NetworkSS_geo_sci_2mcmc.sav', 'rb') as fr:
#     mcmc2_ss_geo_sci = pickle.load(fr)
    
x_axis = jnp.arange(-1,1,0.001)
rho_tril = mcmc2_ss_geo_sci['rho_lt'].mean(0)

densities_geo_sci_geo = []
for A_ix, (A_k, vals) in enumerate(A_geo_ints_10.items()):
    density = jnp.exp(jnp.array([models.logprior_NetworkSS(scale_spike=vals['scale_spike'],
                                                           scale_slab=vals['scale_slab'],
                                                           w_slab=vals['w_slab'],
                                                           mean_slab=vals['mean_slab'],
                                                           rho_lt=x) for x in x_axis]))
    densities_geo_sci_geo.append(density)
    ax.plot(density+A_geo_mids_10[A_ix], x_axis, alpha=0.3, c='gray')
    
ax.scatter(A_tril_geo, -rho_tril, s=18, linewidth=0.8, alpha=0.7, color='black', facecolors='none', label='partial correlations')
#ax.set_xlim(-3,14)
ax.set_xlim(-3,12)
#ax.set_ylim(-0.5, 0.7)
ax.set_ylim(-0.3, 0.7)
ax.set_xlabel('Geographical Distance Network')
ax.set_ylabel('Partial correlation (Network-SS)')

plt.yticks(rotation = 90)
fig.tight_layout()
fig.savefig(data_save_path + 'Figures/' + 'COVID_SS_partial_corrs_GEO.pdf')
plt.close()

