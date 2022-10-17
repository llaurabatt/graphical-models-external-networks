# Graphical model inference with external network data

This repository contains all of the code associated with the paper ''Graphical model inference with external network data''  Jewson, Li, Battaglia, Hansen, Rossell, and Zwiernik (2022) available at ..........

This repository is organised into three sections further explained below

+ Data
+ Network GLASSO
+ Network Spike-and-Slab

## Data

The `Data` folder contains all the information needed to generate the simulation examples considered in Section 5 and to preprocess the data COVID-19 and stock market data considered in Section 6

### Simulations 

The `Simulations` folder contains the Jupyter notebook `simulate_data.ipynb` that simulates data in the manner outlined in Section 5 of the paper and saves them to JSON files that can be loaded into R and python. So doing allows for the comparison of different methods in different languages on the same data.

### COVID

The `COVID` folder contains `COVID19 data preprocessing.Rmd` which implements the data collection and combination outlined in Sections B.1 and B.2, the linear modelling for the mean described in Section B.3, and the goodness-of-fit checks described in Section B.4. 

The folder `Raw Data` contains the raw data files required to run this analysis. 

*Note the hydromat folder was not uploaded here and can be downloaded from https://github.com/CSSEGISandData/COVID-19_Unified-Dataset/tree/master/Hydromet*

The folder `Pre-processed Data` contains the output files of our pre-processing, those files required to estimate our graphical models.

### Stock

The `Stock` folder contains `Stock data preprocessing.Rmd` which implements the data collection and combination outlined in Sections C.1 and C.2, the linear modelling for the mean described in Section C.3, and the goodness-of-fit checks described in Section C.4. 

The folder `Raw Data` contains the raw data files required to run this analysis. 

The folder `Pre-processed Data` contains the output files of our pre-processing, those files required to estimate our graphical models.

## Network GLASSO 

The `Network_GLASSO` folder contains `.Rmd` files for undertaking the Network GLASSO analyses as explained in Sections 3.2 and 4.1.

While we recommend downloading the `golazo` package by calling,

```r
install_github("pzwiernik/golazo", build_vignettes=TRUE)
```

should this for any reason fail, the file `GOLAZO_function.R` contains the required functions.

The file `Network_GLASSO_Simulations` implements the GLASSO and network GLASSO for the different network matrices for the simulated experiments

`COVID_Network_GLASSO_inSample.Rmd` implements the full sample analysis for the COVID data, while `COVID_Network_GLASSO_outSample_cv10` undertakes the 10-fold cross-validation used to provide the test set log-likelihoods in Table 2.

`Stock_Network_GLASSO_inSample.Rmd` and `Stock_Network_GLASSO_outSample_cv10` undertake the equivalent analyse for the stock market data.

## Network Spike-and-Slab

The `Network_Spike_and_Slab` folder contains the code for implementing the Network Spike-and-Slab analyses as explained in Sections 3.3, 4.2, and 4.3.

The file `priorSpecification_NetworkSS.Rmd` provides code to elicit the Network Spike-and-Slab priors as described in Section A.3.   

### stan

The folder `stan` provides the code to implement the Network spike-and-slab in stan

The file `GOLAZO_SS_stan_timings.Rmd` implements the timing comparison presented in Section D of implementing the Network Spike-and-Slab in stan and also times the implementation of the Network GLASSO

### numpyro 

Finally, the folder `numpyro` provides the code to implement the Network spike-and-slab in numpyro.

The folder `functions` contains two python files `my_utils.py` some useful functions for the python analysis and `models.py` which contains all of the code to implement the models in NumPyro.

The rest of the files provide code to implement the analysis using NumPyro. In particular, for file prefixed `sim`, `Covid` and `Stock` the files 

+ `_SS.py` implemented the Spike-and-Slab model with no network data
+ `_NetworkSS.py` implemented the Network Spike-and-Slab model 
+ `_clean_data.py` did any additional data processing needed for python (after the data was pre-processed in R)
+ `_summary.ipynb` allowed for the analysis of the results of the algorithms and the production of the plots seen in the paper  

`sim_timing.ipynb` provides the timing comparison of the Network GLASSO implementation in numpyro in Section D

## Contact Information and Acknowledgments

The code in this repository was created by Laura Battaglia, Li Li, and Jack Jewson. Any questions can be directed to jack.jewson@upf.edu.

We would like to thank Du Phan for advising us on the implementation of our model in NumPyro.





