#!/bin/bash
set -e
set -x


# Assume we are located at numpyro repo directory



################## COVID DATA ##################
# Paths:
COVID_main_path="/home/paperspace/NetworkSS_results_regression_etarepr_brepr_newprior_seed9_centered_rotated"
COVID_nonetworks_path="/home/paperspace/NetworkSS_results_regression_etarepr_brepr_newprior_seed9_centered_rotated_NONETWORKS"
COVID_scale_spike_times2_path="/home/paperspace/NetworkSS_results_regression_etarepr_brepr_newprior_seed9_centered_rotated_times2"
COVID_scale_spike_divide5_path="/home/paperspace/NetworkSS_results_regression_etarepr_brepr_newprior_seed9_centered_rotated_spikedivide5"

# Main model
python COVID_NetworkSS_1mcmc.py --n_samples 10000 \
                                --SEED 9 \
                                --data_save_path "$COVID_main_path/" \
                                --Y "COVID_332_meta_pruned_Y_regression.csv" \
                                --X "COVID_332_meta_pruned_X_rotated_regression.npy" \
                                --model "models.NetworkSS_regression_repr_etaRepr_centered" \
                                --bhat "COVID_332_meta_pruned_b_LS_rotated.csv" \
                                --store_warmup=True \
                                --thinning 10

python COVID_NetworkSS_2mcmc.py --n_samples 10000 \
                                --SEED 9 \
                                --data_save_path "$COVID_main_path/" \
                                --Y "COVID_332_meta_pruned_Y_regression.csv" \
                                --X "COVID_332_meta_pruned_X_rotated_regression.npy" \
                                --model "models.NetworkSS_regression_repr_etaRepr_centered" \
                                --bhat "COVID_332_meta_pruned_b_LS_rotated.csv" \
                                --thinning 10 \
                                --search_MAP_best_params=True

python COVID_summary.py --get_probs=True \
                        --data_save_path $COVID_main_path \
                        --mcmc1_path "$COVID_main_path/NetworkSS_1mcmc_p332_w1000_s10000_CP10000_regression.sav" \
                        --mcmc2_path "$COVID_main_path/NetworkSS_2mcmc_p332_w1000_s10000_regression.sav" 

# Without network data
python COVID_NetworkSS_1mcmc.py --n_samples 10000 \
                                --SEED 9 \
                                --data_save_path "$COVID_nonetworks_path/" \
                                --Y "COVID_332_meta_pruned_Y_regression.csv" \
                                --X "COVID_332_meta_pruned_X_rotated_regression.npy" \
                                --model "models.NetworkSS_regression_repr_etaRepr_centered" \
                                --bhat "COVID_332_meta_pruned_b_LS_rotated.csv" \
                                --store_warmup=True \
                                --thinning 10 \
                                --no_networks=True

python COVID_NetworkSS_2mcmc.py --n_samples 10000 \
                                --SEED 9 \
                                --data_save_path "$COVID_nonetworks_path/" \
                                --Y "COVID_332_meta_pruned_Y_regression.csv" \
                                --X "COVID_332_meta_pruned_X_rotated_regression.npy" \
                                --model "models.NetworkSS_regression_repr_etaRepr_centered" \
                                --bhat "COVID_332_meta_pruned_b_LS_rotated.csv" \
                                --thinning 10 \
                                --search_MAP_best_params=True \
                                --no_networks=True

python COVID_summary.py --get_probs=True \
                        --data_save_path "$COVID_nonetworks_path/" \
                        --mcmc1_path "$COVID_nonetworks_path/NetworkSS_1mcmc_p332_w1000_s10000_CP10000_regression_nonetworks.sav" \
                        --mcmc2_path "$COVID_nonetworks_path/NetworkSS_2mcmc_p332_w1000_s10000_regression_nonetworks.sav" 

# Zero-out etas whose confidence intervals contain zero
python COVID_NetworkSS_2mcmc.py --n_samples 10000 \
                                --SEED 9 \
                                --data_save_path "$COVID_main_path/" \
                                 --Y "COVID_332_meta_pruned_Y_regression.csv" \
                                 --X "COVID_332_meta_pruned_X_rotated_regression.npy" \
                                 --model "models.NetworkSS_regression_repr_etaRepr_centered" \
                                 --bhat "COVID_332_meta_pruned_b_LS_rotated.csv" \
                                 --thinning 10 \
                                 --search_MAP_best_params=True \
                                 --zero_out_etas=True

python COVID_summary.py --get_probs=True \
                        --mcmc1_path "$COVID_main_path/NetworkSS_1mcmc_p332_w1000_s10000_CP10000_regression.sav" \
                        --mcmc2_path "$COVID_main_path/NetworkSS_2mcmc_p332_w1000_s10000_regression_zero_out_etas.sav" \
                        --zero_out_etas=True \
                        --data_save_path "$COVID_main_path/"



# Sensitivity to the scale of the spike prior

# scale spike times 2
python COVID_NetworkSS_1mcmc.py --n_samples 10000 \
                                --SEED 9 \
                                --data_save_path "$COVID_scale_spike_times2_path/" \
                                --Y "COVID_332_meta_pruned_Y_regression.csv" \
                                --X "COVID_332_meta_pruned_X_rotated_regression.npy" \
                                --model "models.NetworkSS_regression_repr_etaRepr_centered" \
                                --bhat "COVID_332_meta_pruned_b_LS_rotated.csv" \
                                --store_warmup=True \
                                --thinning 10 \
                                --scale_spike_fixed 0.0066682 \
                                --init_all_path "$COVID_main_path/NetworkSS_1mcmc_p332_w1000_s10000_CP10000_regression.sav" 



python COVID_NetworkSS_2mcmc.py --n_samples 10000 \
                                --SEED 9 \
                                --data_save_path "$COVID_scale_spike_times2_path/" \
                                --Y "COVID_332_meta_pruned_Y_regression.csv" \
                                --X "COVID_332_meta_pruned_X_rotated_regression.npy" \
                                --model "models.NetworkSS_regression_repr_etaRepr_centered" \
                                --bhat "COVID_332_meta_pruned_b_LS_rotated.csv" \
                                --thinning 10 \
                                --search_MAP_best_params=True \
                                --scale_spike_fixed 0.0066682

python COVID_summary.py --get_probs=True \
                        --mcmc1_path "$COVID_scale_spike_times2_path/NetworkSS_1mcmc_p332_w1000_s10000_CP10000_regression_init_all_path.sav" \
                        --mcmc2_path "$COVID_scale_spike_times2_path/NetworkSS_2mcmc_p332_w1000_s10000_regression.sav" \
                        --scale_spike_fixed 0.0066682 \
                        --data_save_path "$COVID_scale_spike_times2_path/"


# scale spike divided by 5
python COVID_NetworkSS_1mcmc.py --n_samples 10000 \
                                --SEED 9 \
                                --data_save_path "$COVID_scale_spike_divide5_path/" \
                                --Y "COVID_332_meta_pruned_Y_regression.csv" \
                                --X "COVID_332_meta_pruned_X_rotated_regression.npy" \
                                --model "models.NetworkSS_regression_repr_etaRepr_centered" \
                                --bhat "COVID_332_meta_pruned_b_LS_rotated.csv" \
                                --store_warmup=True \
                                --thinning 10 \
                                --scale_spike_fixed 0.00066682

python COVID_NetworkSS_2mcmc.py --n_samples 10000 \
                                --SEED 9 \
                                --data_save_path "$COVID_scale_spike_divide5_path/" \
                                --Y "COVID_332_meta_pruned_Y_regression.csv" \
                                --X "COVID_332_meta_pruned_X_rotated_regression.npy" \
                                --model "models.NetworkSS_regression_repr_etaRepr_centered" \
                                --bhat "COVID_332_meta_pruned_b_LS_rotated.csv" \
                                --thinning 10 \
                                --search_MAP_best_params=True \
                                --scale_spike_fixed 0.00066682

python COVID_summary.py --get_probs=True  \
                        --data_save_path "$COVID_scale_spike_divide5_path/" \
                        --mcmc1_path "$COVID_scale_spike_divide5_path/NetworkSS_1mcmc_p332_w1000_s10000_CP10000_regression.sav" \
                        --mcmc2_path "$COVID_scale_spike_divide5_path/NetworkSS_2mcmc_p332_w1000_s10000_regression.sav" \
                        --scale_spike_fixed 0.00066682


################# STOCK DATA ##################

stock_main_path="/home/paperspace/stock_SS_etarepr_newprior_newlogrepr_seed6"
python stock_NetworkSS_1mcmc.py --thinning 10 \
                                --n_samples 10000 \
                                --data_save_path "$stock_main_path/" \
                                --SEED 6 \
                                --model "models.NetworkSS_repr_etaRepr_loglikRepr"




python stock_NetworkSS_2mcmc.py --search_MAP_best_params=True \
                                --thinning 10 \
                                --SEED 6 \
                                --data_save_path "$stock_main_path/" \
                                --n_samples 4000 \
                                --model "models.NetworkSS_repr_etaRepr_loglikRepr"


python stock_summary.py --get_probs=True \
                        --data_save_path "$stock_main_path/" \
                        --mcmc1_path "$stock_main_path/NetworkSS_1mcmc_p366_w1000_s10000_CP10000.sav" \
                        --mcmc2_path "$stock_main_path/NetworkSS_2mcmc_p366_w1000_s4000_seed6.sav"