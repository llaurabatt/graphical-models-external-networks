# Graphical model inference with external network data

This repository contains all of the code associated with the paper ''Graphical model inference with external network data''  Jewson, Li, Battaglia, Hansen, Rossell, and Zwiernik (2022) available at ..........

This repository is organised into three sections further explained below

+ Data
+ Network GLASSO
+ Network spike-and-slab

## Data

The `Data` folder contains all the information needed to generate the simulation examples considered in Section 5 and to preprocess the data COVID-19 and stock market data considered in Section 6

### Simulations 

The `Simulations` folder contains the Jupyter notebook `simulate_data.ipynb` that simulates data in the manner outlined in Section 5 of the paper and saves them to JSON files that can be loaded into R and python. So doing allows for the comparison of different methods in different languages on the same data.

### COVID

The `COVID` folder contains `COVID19 data preprocessing allCounties.Rmd` and `COVID19 data preprocessing allClustered.Rmd` which implements the data collection and combination outlined in Sections B.1 and B.2, the linear modelling for the mean described in Section B.3, and the goodness-of-fit checks described in Section B.4. 

The folder `Raw Data` contains the raw data files required to run this analysis. 

*Note the hydromat folder was not uploaded here and can be downloaded from https://github.com/CSSEGISandData/COVID-19_Unified-Dataset/tree/master/Hydromet*

The folder `Pre-processed Data` contains the output files of our pre-processing, those files required to estimate our graphical models.

#### Clustering the conties

I order to reproduce our clustering of the 
+ COVID19 data preprocessing allCounties.Rmd reads in the raw data for all counties in the US and preprocess the data read to be clusterd into meta-counties
+ clustering_algo_flights2.ipynb clusters the data data into meta counties
+ COVID19 data preprocessing allClustered.Rmd reads in the clustered data and produces the residuals and network matricies

#### Flight Data

For full details on the data collection for the Flight Passenger Flow network see https://github.com/unstructured-data/airtravel-data

### Stock

The `Stock` folder contains `Stock_data_preprocessing_final.Rmd` which implements the data collection and combination outlined in Sections C.1 and C.2, the linear modelling for the mean described in Section C.3, and the goodness-of-fit checks described in Section C.4. 

The folder `Raw Data` contains the raw data files required to run this analysis. 

The folder `Pre-processed Data` contains the output files of our pre-processing, those files required to estimate our graphical models.

## network GLASSO 

The `Network_GLASSO` folder contains `.Rmd` files for undertaking the network GLASSO analyses as explained in Sections 3.2 and 4.1.

While we recommend downloading the `golazo` package by calling,

```r
install_github("pzwiernik/golazo", build_vignettes=TRUE)
```

should this for any reason fail, the file `GOLAZO_function.R` contains the required functions.

The file `Network_GLASSO_Simulations` implements the GLASSO and network GLASSO for the different network matrices for the simulated experiments

`COVID_Frequentist_GLASSO_GOLAZO_inSample_allClustered_BayesOpt.Rmd` implements the full sample analysis for the COVID data, while `COVID_Frequentist_GLASSO_GOLAZO_outSample_cv10_allClustered_BayesOpt.Rmd` undertakes the 10-fold cross-validation used to provide the test set log-likelihoods in Table 2.

`STOCK_Frequentist_GLASSO_GOLAZO_inSample_allSP_BayesOpt.Rmd` and `STOCK_Frequentist_GLASSO_GOLAZO_outSample_cv10_allSP_BayesOpt.Rmd` undertake the equivalent analyse for the stock market data.

## Network spike-and-slab

The `Network_Spike_and_Slab` folder contains the code for implementing the network spike-and-slab analyses as explained in Sections 3.3, 4.2, and 4.3.

The file `priorSpecification_NetworkSS.Rmd` provides code to elicit the network spike-and-slab priors as described in Section A.3.   

### stan

The folder `Stan` provides the code to implement the network spike-and-slab in Stan

The file `stan_Network_SS_timings.Rmd` implements the timing comparison presented in Section D of implementing the network spike-and-slab in Stan and also times the implementation of the network GLASSO

### numpyro 

Finally, the folder `numpyro` provides the code to implement the network spike-and-slab in NumPyro.

The folder `functions` contains two python files `my_utils.py` some useful functions for the python analysis and `models.py` which contains all of the code to implement the models in NumPyro.

The rest of the files provide code to implement the analysis using NumPyro. In particular, for file prefixed `sim`, `Covid` and `Stock` the files 

+ `_SS.py` implemented the Spike-and-Slab model with no network data
+ `_NetworkSS.py` implemented the network spike-and-slab model 
+ `_clean_data.py` did any additional data processing needed for python (after the data was pre-processed in R)
+ `_summary.ipynb` allowed for the analysis of the results of the algorithms and the production of the plots seen in the paper  

`sim_timing.ipynb` provides the timing comparison of the network GLASSO implementation in NumPyro in Section D

## Contact Information and Acknowledgments

The code in this repository was created by Laura Battaglia, Li Li, and Jack Jewson. Any questions can be directed to jack.jewson@upf.edu.

We would like to thank Du Phan for advising us on the implementation of our model in NumPyro.




