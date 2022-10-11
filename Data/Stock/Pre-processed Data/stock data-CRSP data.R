
############# stock data ##############
## remember we have network matrices: 
## (1) E_cos, P_cos
## (2) E_pears, P_pears
## where E refers to Economy, P refers to Policy 



#### Load packages ####
library(readr)
library(readxl)
library(dplyr)
library(devtools)
#install_github("pzwiernik/golazo", build_vignettes=TRUE)
#library(golazo)
library(polynom)
library(ggplot2)
library("lsa")       ## cosine similarity
library(igraph)

#### Setting Routine ####
options(max.print=1000000)
source('/Users/hanna/Desktop/network GM/Code_final version/COVID/GOLAZO_function.R')



#### Functions ####
ebic_eval <- function(n, R, U, ebic.gamma, edge.tol){
  res <- golazo(R, L = -U, U = U, verbose = FALSE)
  K <- res$K
  KR <- stats::cov2cor(K)         #to make edge count independend of scalings
  nedg <- length(which(abs(KR[upper.tri(abs(KR), diag = FALSE)]) > edge.tol))
  ebic <- -(n)*(log(det(K)) - sum(R*K)) + nedg * (log(n) + 4 * ebic.gamma * log(p))
  return(ebic)   
}

ebic_eval_network <- function(n, R, A, beta0, beta1, ebic.gamma, edge.tol){
  U <- exp(beta0 + beta1*A)          
  diag(U) <- 0
  return(ebic_eval(n, R, U, ebic.gamma, edge.tol))
}

ebic_eval_two_networks <- function(n, R, A1, A2, beta0, beta1, beta2, ebic.gamma, edge.tol){
  U <- exp(beta0 + beta1*A1 + beta2*A2)           #### remember a = beta1, b= beta0
  diag(U) <- 0
  return(ebic_eval(n, R, U, ebic.gamma, edge.tol))
}


#STANDARDISE THE MATRICIES
standardise_network_matrix_tri <- function(A) {
  ## before we ignored the symmetry which matters (slightly) for the variance
  p <- nrow(A)
  
  A_tri <- A[upper.tri(A)]
  bar_A_tri <- mean(A_tri)
  #S2_A_tri <- var(A_tri)
  S2_A_tri <- 1/length(A_tri)*sum((A_tri - bar_A_tri)^2)
  
  return((A - bar_A_tri)/sqrt(S2_A_tri))
}


## Turning the correlation matrix given by the golazo function back to a covariance matrix, useful in out-of-sample-llh
cor2cov <- function(Theta_cor, sigma2_vect){
  # Theta_cor is correlation matrix, sqrt(sigma2_vect) is the standard deviations of each variable
  p <- nrow(Theta_cor)
  Theta_cov <- matrix(NA, nrow = p, ncol = p)
  for(i in 1:p){
    Theta_cov[, i] <- Theta_cor[,i]*sqrt(sigma2_vect[i])*sqrt(sigma2_vect)   
  }
  return(Theta_cov)
}

threshold <- function(Rho_mat, threshold){
  return(Rho_mat*(abs(Rho_mat) >= threshold))
}

## No Network matrix
beta0_max_GLASSO <- function(R){
  return(log(max(abs(R - diag(diag(R))))))## check we can irgnore diags
}



## Hyperparameters ##
# ebic.gamma <- 0                         # when equals to 0, it's BIC; when equals to 0.5, it's EBIC
# edge.tol   <- 0.01     

ebic.gamma <- 0                         # when equals to 0, it's BIC; when equals to 0.5, it's EBIC
round.tol  <- 3                         # values smaller than 5*10^-4 will be treated as 0
edge.tol   <-  5*10^-(round.tol + 1)     



########################################
#### Load the Stock data ####
stock_price <- read.csv('/Users/hanna/Desktop/stock data/data/tic_price_data.csv')           ## stock data

str(stock_price$date)
stock_price$date <- as.character(stock_price$date)          ## Setting Date format
stock_price$date <- format(as.Date(stock_price$date, "%Y%m%d"),"%Y-%m-%d")
str(stock_price$date)
stock_price$date <- as.Date(stock_price$date) 
str(stock_price$date)


# install.packages('plyr')
library(plyr)

names(stock_price)[names(stock_price) == 'TICKER'] <- 'tic'
stock_freq <- count(stock_price, 'tic')    ## 3019 stocks, not all stock have 252 periods

stock_price1 <- merge(x = stock_price, y = stock_freq, by = "tic")  ##count the freq of stocks


#### Load the tic data ####
cik_ticker <- read.csv('/Users/hanna/Desktop/stock data/data/tic_cik_crosswalk.csv')           ## ticker-cik data
stock_price2 <- merge(x = stock_price1, y = cik_ticker, by = "tic")                            ## merge ticker with stock




##### load fama_factor data
fama_3factor <-read.csv('/Users/hanna/Desktop/stock data/data/F-F_Research_Data_Factors_daily.csv') 
names(fama_3factor)[names(fama_3factor) == 'X'] <- 'date'
fama_3factor$rmrf <- fama_3factor$Mkt.RF - fama_3factor$RF    ## create Rm-Rf variable

str(fama_3factor$date)                                                                ## Setting Date format
fama_3factor$date <- format(as.Date(fama_3factor$date, "%Y%m%d"),"%Y-%m-%d")
str(fama_3factor$date)
fama_3factor$date <- as.Date(fama_3factor$date) 
str(fama_3factor$date)

fama_3factor <- fama_3factor %>% filter(date >= "2019-01-02" & date <= "2019-12-31")   ##keep the data of 2019 year


######## merge fama_3factor with stock_price2
stock_price3 <-  merge(x = stock_price2, y = fama_3factor, by = "date")  

count(stock_price3, 'tic')    ## 2796 stocks

stock_252 <- stock_price3[stock_price3$freq == 252, ]   ## keep all stoc that have 252 periods

## basically, we can just use the stock data to run regression and obtain the abnormal value, but the problem is that our risk dataset might 
## not have the risk measure of some of the stocks, thus we need to check and choose the sample for analysis.


###### choose stock that in risk_index and also have 252 periods in stock_price252, and then make sure E and P are not 0 vector.
risk_index <- read.csv('/Users/hanna/Desktop/stock data/data/cik_cat_count_new2.csv')           ## risk metric data
risk_index <- select(risk_index, -X)

risk_tic <- merge(x = risk_index, y = cik_ticker, by = "cik") 

risk_tic <- risk_tic %>%
  select(tic, everything())   ## move one column to the start


# risk_tic %>% filter(tic %in% 'JPM')  ## check for one row


## In order to be able to standerdise the network matrix, we need to make sure that neither the E nor the P vectors are 0 vectors.
nonzero_row <- risk_tic[rowSums(risk_tic %>% select(ends_with("_E"))) > 0 & rowSums(risk_tic %>% select(ends_with("_P"))) > 0, ]
zero_row <- risk_tic[rowSums(risk_tic %>% select(ends_with("_E"))) == 0 | rowSums(risk_tic %>% select(ends_with("_P"))) ==0, ]


## random choose 200 stocks in stock_252 and make sure 
stock_risk <- merge(x = stock_252, y = nonzero_row, by = "tic")

count(stock_risk, 'tic')  ## 2574 stock

set.seed(1234)
library(dplyr)
Y <- stock_risk %>% 
  filter(tic %in% sample(unique(tic),200))


Y$PRC <- Y$PRC*sign(Y$PRC)

##############################
# get log return
library(tidyquant)
library(timetk)

stock_return <- Y %>%
  group_by(tic) %>%                              
  tq_transmute(select = PRC,
               mutate_fun = periodReturn,
               period = 'daily',
               type = "log",
               col_rename = 'returns')                # get simple return


count(stock_return, 'tic')   ## 200 *252


## merge with Y
library(data.table)
daily_return <- as.data.frame(setDT(Y)[setDT(stock_return), on = c("date", "tic"), 
               returns := returns])

daily_return <- dplyr::arrange(daily_return, tic, date)
str(daily_return)


daily_return <- daily_return %>%
  select(returns, everything())   ## move one column to the start




################################### linear regression
model1 <- lm(returns ~ SMB + HML + rmrf , data= daily_return)

summary(model1)



# Specifying the decimal places 
# Reproduced from ref: https://stackoverflow.com/questions/32490182/r-format-regression-output
specify_decimal <- function(x, k) format(round(x, k), nsmall=k)

# Using specify_decimal to recapitulate the regression summary
restate_summary  <- function(lmsummary, digits) {
  
  coefs <- as.data.frame(lmsummary)
  coefs[] <- lapply(coefs, function(x) specify_decimal(x, digits))
  coefs
  
}


restate_summary(summary(model1)$coefficients, 4)


#### plot
##### residual plots

pdf('/Users/hanna/Desktop/residual vs covariates.pdf')
par(mfrow = c(2, 2))

plot(x = fitted.values(model1), y = residuals(model1), xlab = "fitted value", ylab = "residual")
lines(lowess(fitted.values(model1), residuals(model1)), col='red')


plot(x= daily_return$SMB, y = residuals(model1), xlab = "SMB", ylab = "residual")
lines(lowess(daily_return$SMB, residuals(model1)), col='red')


plot(x= daily_return$HML, y = residuals(model1), xlab = "HML", ylab = "residual")
lines(lowess(daily_return$HML, residuals(model1)), col='red')


plot(x= daily_return$rmrf, y = residuals(model1), xlab = "Rm-Rf", ylab = "residual")
lines(lowess(daily_return$rmrf, residuals(model1)), col='red')


par(mfrow = c(1, 1))
dev.off()


pdf('/Users/hanna/Desktop/density of residual.pdf')
plot(density(resid(model1)))
dev.off()

pdf('/Users/hanna/Desktop/qq-plot of residual.pdf')
qqnorm(rstudent(model1))
qqline(rstudent(model1))
dev.off()


############### detect the outliers
library("car")

pdf('/Users/hanna/Desktop/detect outliers.pdf')
qqPlot(model1, labels=tic(daily_return), id.method="identify",
       simulate=TRUE, main="Q-Q Plot")   

dev.off()

stock[107, 31]   ## nn[31] "CAPR"
stock[111, 101]  ## nn[101] "HSON"

outlier <- daily_return %>% filter(tic == "CAPR" | tic == 'HSON')
outlier1 <- daily_return %>% filter(tic == "CAPR")
outlier2 <- daily_return %>% filter(tic == "HSON")



pdf('/Users/hanna/Desktop/time series.pdf')
ggplot(data = daily_return, 
       aes(x = date, y = returns, group = tic)) +
  geom_line(aes(col = tic)) +
  labs(x = "Date",  y = "Daily stock returns") +
  theme(legend.position = "none") +  
  scale_x_date(date_breaks = "1 month") +
  theme(axis.text.x=element_text(angle=60, hjust=1))

dev.off()


# Residual ACF and PACF for model without AR terms
daily_return$res <- residuals(model1)

dataacf <- tapply(daily_return$res, INDEX= daily_return$tic, FUN=acf, pl=FALSE)   #Obtain ACF for each stock
datapacf <- tapply(daily_return$res, INDEX= daily_return$tic, FUN=pacf, pl=FALSE) #Obtain PACT for eachstock

r <- pr <- matrix(NA, nrow=length(dataacf), ncol=5)  #Format as matrix for easy plotting
for (i in 1:length(dataacf)) {
  r[i,] <- dataacf[[i]][[1]][2:6,,]  #start at 2, first element stores 0 order correlation
  pr[i,] <- datapacf[[i]][[1]][1:5,,]
}



 pdf('/Users/hanna/Desktop/acf_residuals-stock.pdf')
plot(density(r[,1]), xlim=c(-0.5,1), ylim = c(0,5), main='acf_residuals', xlab='Autocorrelation', cex.axis=1.3, cex.lab=1.4)
lines(density(r[,2]), col=2)
lines(density(r[,3]), col=3)
lines(density(r[,4]), col=4)
lines(density(r[,5]), col=5)
legend('topleft', c('Order 1','Order 2','Order 3','Order 4','Order 5'), lty=1, lwd=2, col=1:5, cex=1.2)
 dev.off()


 pdf('/Users/hanna/Desktop/pacf_residuals-stock.pdf')
plot(density(pr[,1]), xlim=c(-0.5,1), ylim = c(0,5), main='pacf_residuals', xlab='Partial autocorrelation', cex.axis=1.3, cex.lab=1.4)
lines(density(pr[,2]), col=2)
lines(density(pr[,3]), col=3)
lines(density(pr[,4]), col=4)
lines(density(pr[,5]), col=5)
legend('topleft', c('Order 1','Order 2','Order 3','Order 4','Order 5'), lty=1, lwd=2, col=1:5, cex=1.2)
 dev.off()



####  PUT RESIDUALS INTO A MATRIX
nn <- unique(daily_return$tic)
ntic <- length(nn)
stock <- lapply(1:ntic, function(i) daily_return[daily_return$tic==nn[i], 'res'])
stock <- do.call(cbind, stock)
 
dim(stock)     #check: 200 tic, 252 observations
 
# write.csv(stock, file = paste("stock_27Jun", ".csv", sep = ""))
 

########################################
## monotone normalizing transform
library(huge)
stockn <- huge.npn(stock)

print(stock[1:5,1])
print(stockn[1:5,1])
######################## Network matrix  #####

#### cosine similarity for Economy and Policy categories
nn1 <- unique(daily_return$tic)
mydf <- nonzero_row[nonzero_row$tic%in%nn1, ]

mydf <- arrange(mydf, tic)
  
upd <- unique(colnames(mydf[, 3:39]))

mydf[,names(mydf) %in% upd] <- log(mydf[,names(mydf) %in% upd] + 1)  ##add 1 to the elements of 37 risk measures and take the logarithm

## Cosine Network: economy and policy
E_cos <- lsa::cosine(t(as.matrix(mydf %>% select(ends_with("_E")))))      ##cosine network for Economy measure
P_cos <- lsa::cosine(t(as.matrix(mydf %>% select(ends_with("_P")))))      ##cosine network for Policy measure



#### Pearson Network: row centring

mydf_E  <- mydf %>%select(ends_with("_E")) 
mydf_centering_E <- ( mydf_E - matrix(rowMeans(mydf_E), nrow = nrow( mydf_E), ncol = ncol( mydf_E), byrow = FALSE ))   ##row centering for E risks
rowMeans(mydf_centering_E)


mydf_P  <- mydf %>%select(ends_with("_P")) 
mydf_centering_P <- ( mydf_P - matrix(rowMeans( mydf_P), nrow = nrow( mydf_P), ncol = ncol( mydf_P), byrow = FALSE ))   ##row centering for P risks
rowMeans(mydf_centering_P)


E_pears <- lsa::cosine(t(as.matrix(mydf_centering_E)))    ## pearson network for Economy measure
P_pears <- lsa::cosine(t(as.matrix(mydf_centering_P)))    ## pearson network for Policy measure



# write.csv(E_pears, file = paste("E_pears", ".csv", sep = ""))
# write.csv(P_pears, file = paste("P_pears", ".csv", sep = ""))



#######################################
## GLASSO ##
p <- ncol(stock)   #200
n <- nrow(stock)   #252
N <-1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0

beta0 <-rep(NA, N)
beta0_grid_length <- 20
beta0_grid_min <- -3

ebic_eval_optim_GLASSO <- rep(NA, N)


for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(stock))  
  
  ## grid-search ##
  beta0_grid_max <- beta0_max_GLASSO(R)
  beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length )
  
  beta_optimise <- rep(NA, beta0_grid_length)
  
  for(b in 1:beta0_grid_length){
    cat("GLASSO b = ", b, "\n")
    beta_optimise[b] <- ebic_eval(n, R, U = exp(beta0_grid[b])*U, ebic.gamma, edge.tol)
  }
  
  ## Saving beta0
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  beta0[j] <- mean(beta0_grid[min_coord])
  
  ## Saving the EBIC 
  ebic_eval_optim_GLASSO[j] <- min(beta_optimise)
  
  #### Using the optimal beta0 ##
  GraphicalModel <- golazo (R, L = exp(beta0[j]) * L, U =exp(beta0[j])* U, verbose=FALSE)
  
  Theta_hat_GLASSO <- round(GraphicalModel$K, round.tol)   
  # Rho_hat_GLASSO <- threshold(cov2cor(GraphicalModel$K), edge.tol)
  
  
}

beta0_GLASSO <- beta0 
beta0_GLASSO     #-1.474083

ebic_eval_optim_GLASSO   #47659.62

sum(cov2cor(Theta_hat_GLASSO)[lower.tri((cov2cor(Theta_hat_GLASSO)))] != 0)   # Edges for the last repeat   528
sum(cov2cor(Theta_hat_GLASSO)[lower.tri((cov2cor(Theta_hat_GLASSO)))] == 0)   # Non-edges for the last repeat  19372


pdf('/Users/hanna/Desktop/Theta_hat.pdf')
plot(Theta_hat_GLASSO, Theta_hat_GLASSO_n)
abline(a = 0, b = 1)
dev.off()

####################################
## PLOTs GLASSO estimate R_ij vs Network A1_cos, A2_cos, A1_pears, A2_pears
cov_Theta_hat_GLASSO <- -cov2cor(Theta_hat_GLASSO)
partial_corr <- cov_Theta_hat_GLASSO[upper.tri(cov_Theta_hat_GLASSO)]

      
# partial_corr <- Rho_hat_GLASSO[upper.tri(Rho_hat_GLASSO)]


par(mar = c(3.3, 3.6, 1.5, 1.1))  # bottom, left, top, right
par(mgp = c(2.15, 1, 0))
par(cex.lab = 1.25, cex.axis = 1.25, cex.main = 1.25)



pdf('/Users/hanna/Desktop/GLASSO estimates.pdf')
par(mfrow = c(2, 2))
###############
dim( E_cos) 
dim(Theta_hat_GLASSO)
# dim(Rho_hat_GLASSO)


E_cos_s <- standardise_network_matrix_tri(E_cos)
diag(E_cos_s) <- 0
E_plot_cos <- E_cos_s[upper.tri(E_cos_s)]



#DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
plot(x = E_plot_cos, y = partial_corr, xlab = "E_cos(Economy)", ylab ="GLASSO estimates of R_ij")
l <- loess(partial_corr ~ E_plot_cos)
o <- order(l$x)
lines(l$x[o], l$fitted[o], col='red')


################
dim( P_cos) 



P_cos_s <- standardise_network_matrix_tri(P_cos) 
diag(P_cos_s) <- 0
P_plot_cos <- P_cos_s[upper.tri(P_cos_s)]


#DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
plot(x = P_plot_cos, y = partial_corr, xlab = "P_cos(Policy)", ylab ="GLASSO estimates of R_ij")
l <- loess(partial_corr ~ P_plot_cos)
o <- order(l$x)
lines(l$x[o], l$fitted[o], col='red')


##################
dim( E_pears) 



E_pears_s <- standardise_network_matrix_tri(E_pears) 
diag(E_pears_s) <- 0
E_plot_pears <- E_pears_s[upper.tri(E_pears_s)]


#DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
plot(x = E_plot_pears, y = partial_corr, xlab = "E_pears(Economy)", ylab ="GLASSO estimates of R_ij")
l <- loess(partial_corr ~ E_plot_pears)
o <- order(l$x)
lines(l$x[o], l$fitted[o], col='red')


################
dim( P_pears) 



P_pears_s <- standardise_network_matrix_tri(P_pears) 
diag(P_pears_s) <- 0
P_plot_pears <- P_pears_s[upper.tri(P_pears_s)]


#DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
plot(x = P_plot_pears, y = partial_corr, xlab = "P_pears(Policy)", ylab ="GLASSO estimates of R_ij")
l <- loess(partial_corr ~ P_plot_pears)
o <- order(l$x)
lines(l$x[o], l$fitted[o], col='red')



par(mfrow = c(1, 1))
dev.off()


####################################
### GLASSO out of sample log-likelihood
## out of sample log-likelihood
set.seed(1234)
sample_size <- floor(0.9*nrow(stock))


picked <- sample(seq_len(nrow(stock)), size = sample_size)
train_data <- stock[picked, ]
test_data  <- stock[-picked, ]


n <- sample_size
N <- 1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0

beta0 <-rep(NA, N)
beta0_grid_length <- 20
beta0_grid_min <- -3

ebic_eval_optim_GLASSO_out <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(train_data))  
  
  ## grid-search ##
  beta0_grid_max <- beta0_max_GLASSO(R)
  beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length )
  
  beta_optimise <- rep(NA, beta0_grid_length)
  
  for(b in 1:beta0_grid_length){
    cat("out GLASSO b = ", b, "\n")
    beta_optimise[b] <- ebic_eval(n, R, U = exp(beta0_grid[b])*U, ebic.gamma, edge.tol)
  }
  
  ## Saving beta0
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  beta0[j] <- mean(beta0_grid[min_coord])
  
  # JACK: Saving the EBIC 
  ebic_eval_optim_GLASSO_out[j] <- min(beta_optimise, na.rm = TRUE)
  
  #### Using the optimal beta0 ##
  GraphicalModel <- golazo (R, L = exp(beta0[j]) * L, U =exp(beta0[j])* U, verbose=FALSE)
  
  Theta_hat_GLASSO_out <- round(GraphicalModel$K, round.tol)   
  # Rho_hat_GLASSO_out <- threshold(cov2cor(GraphicalModel$K), edge.tol)
}

beta0_GLASSO_out <- beta0 
beta0_GLASSO_out  #-1.564098

ebic_eval_optim_GLASSO_out  #32774.38

sum(cov2cor(Theta_hat_GLASSO_out)[lower.tri((cov2cor(Theta_hat_GLASSO_out)))] != 0) #640
sum(cov2cor(Theta_hat_GLASSO_out)[lower.tri((cov2cor(Theta_hat_GLASSO_out)))] == 0) #11921

# sum(Rho_hat_GLASSO_out[lower.tri((Rho_hat_GLASSO_out))] != 0) #640
# sum(Rho_hat_GLASSO_out[lower.tri((Rho_hat_GLASSO_out))] == 0) #11921



############# out-of-sample log-likelihood
mu_hat <- rep(0, p)
library(mvtnorm)
out_sample_llh_GLASSO <- sum(dmvnorm(test_data, mean = mu_hat, sigma = cor2cov(solve(Theta_hat_GLASSO_out), diag(cov(train_data))), log = TRUE))
out_sample_llh_GLASSO  #10638.28


# out_sample_llh_GLASSO <- sum(dmvnorm(test_data, mean = mu_hat, sigma = solve(cor2cov(-Rho_hat_GLASSO_out, diag(solve(cov(train_data))))), log = TRUE))
# out_sample_llh_GLASSO  #10638.28



####################################
###  GOLAZO with Network Matrix A1(E_pears)
p <- ncol(stock)
n <- nrow(stock)
N <-1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0


A1 <- standardise_network_matrix_tri(E_pears) 
diag(A1) <- 0


#### grid-search 
beta <- matrix(NA, nrow = N, ncol = 2)

beta0_grid_length <- 20 
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0
beta0_grid <- sort(c(beta0_grid, 0)) 


beta1_grid_length <- 20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 

ebic_eval_optim_GOLAZO_A1_full <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(stock)) 
  
  beta_optimise <- matrix(NA, nrow = beta0_grid_length, ncol = beta1_grid_length)
  
  for(b0 in 1:beta0_grid_length){
    for(b1 in 1:beta1_grid_length){
      cat("GOLAZO with A1 b0 = ", b0, ", b1 = ", b1, "\n")
      try(beta_optimise[b0, b1] <- ebic_eval_network(n, R, A1, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], ebic.gamma, edge.tol))
    }
  }
  
  ## Jack: min_coord is a matrix rather than a vector, because there are many minimums, all with the same objective, but the if-else thing is slow
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  
  dim(min_coord) <- c(length(min_coord)/2, 2)
  beta[j, 1] <- mean(beta0_grid[min_coord[,1]])
  beta[j, 2] <- mean(beta1_grid[min_coord[,2]])
  
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A1_full[j] <- min(beta_optimise, na.rm = TRUE)
  
  # U <- exp(beta0 + beta1*A1) 
  U <- exp(beta[j,1] + beta[j,2]*A1) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R, -U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A1_full <- round(res$K, round.tol)  
  # Rho_hat_GOLAZO_A1_full <- threshold(cov2cor(res$K), edge.tol)
}

beta_GOLAZO_A1_full <- beta
beta_GOLAZO_A1_full  # -1.222222 -0.5555556

ebic_eval_optim_GOLAZO_A1_full  #35704.84

sum(cov2cor(Theta_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full)))] != 0) #  605
sum(cov2cor(Theta_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full)))] == 0) #  11798

# sum(cov2cor(Rho_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A1_full)))] != 0) #  605
# sum(cov2cor(Rho_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A1_full)))] == 0) #  11798


#####################################
#####GOLAZO with A1 out of sample log-likelihood
A1 <- standardise_network_matrix_tri(E_pears) 
diag(A1) <- 0

set.seed(1234)
sample_size <- floor(0.9*nrow(stock))


picked <- sample(seq_len(nrow(stock)),size = sample_size)
train_data <- stock[picked,]
test_data  <- stock[-picked,]

n <- sample_size
N <- 1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0


#### grid-search 
beta <- matrix(NA, nrow = N, ncol = 2)

beta0_grid_length <- 20 
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0
beta0_grid <- sort(c(beta0_grid, 0)) 

beta1_grid_length <- 20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 

ebic_eval_optim_GOLAZO_A1_full_out <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(train_data)) 
  
  ## grid-search ##
  beta_optimise <- matrix(NA, nrow = beta0_grid_length, ncol = beta1_grid_length)
  for(b0 in 1:beta0_grid_length){
    for(b1 in 1:beta1_grid_length){
      cat("out GOLAZO A1 b0 = ", b0, ", b1 = ", b1, "\n")
      try(beta_optimise[b0, b1] <- ebic_eval_network(n, R, A1, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], ebic.gamma, edge.tol))
    }
  }
  
  ## Jack: min_coord is a matrix rather than a vector, because there are many minimums, all with the same objective, but the if-else thing is slow
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  
  dim(min_coord) <- c(length(min_coord)/2, 2)
  beta[j, 1] <- mean(beta0_grid[min_coord[,1]])
  beta[j, 2] <- mean(beta1_grid[min_coord[,2]])
  # min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  # beta[j, 1] <- beta0_grid[min_coord[1]]
  # beta[j, 2] <- beta1_grid[min_coord[2]]
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A1_full_out[j] <- min(beta_optimise, na.rm = TRUE)
  
  ## use the optimal beta0 and beta1
  U <- exp(beta[j,1] + beta[j,2]*A1) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R,-U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A1_full_out <- round(res$K, round.tol)
  # Rho_hat_GOLAZO_A1_full_out <- threshold(cov2cor(res$K), edge.tol)
  
  # cat("GOLAZO-trueA", "n", n , "Repeat", j, "done", "\n")    #cat: print output to the screen or to a file. "\n" Line Break separator
}


beta_GOLAZO_A1_full_out <- beta
beta_GOLAZO_A1_full_out # -1.444444 -0.3333333

ebic_eval_optim_GOLAZO_A1_full_out # 32711.63

sum(cov2cor(Theta_hat_GOLAZO_A1_full_out )[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full_out )))] != 0) # 688
sum(cov2cor(Theta_hat_GOLAZO_A1_full_out )[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full_out )))] == 0) #11873

# sum(cov2cor(Rho_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A1_full)))] != 0) # 585
# sum(cov2cor(Rho_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A1_full)))] == 0) #  11818


#############GOLAZO A1 
#JACK: Need to turn correlation into covariance first 
mu_hat <- rep(0, p)
library(mvtnorm)
out_sample_llh_GOLAZO_A1_full <- sum(dmvnorm(test_data, mean = mu_hat, sigma = cor2cov(solve(Theta_hat_GOLAZO_A1_full_out), diag(cov(train_data))), log = TRUE))
out_sample_llh_GOLAZO_A1_full  #10656.96


# out_sample_llh_GOLAZO_A1_full <- sum(dmvnorm(test_data, mean = mu_hat, sigma = solve(cor2cov(Rho_hat_GOLAZO_A1_full_out, diag(solve(cov(train_data))))), log = TRUE))
# out_sample_llh_GOLAZO_A1_full  #10656.96




####################################
###  GOLAZO with Network Matrix A2(P_pears)
p <- ncol(stock)
n <- nrow(stock)
N <-1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0



A2 <- standardise_network_matrix_tri(P_pears) 
diag(A2) <- 0


#### grid-search 
beta <- matrix(NA, nrow = N, ncol = 2)

beta0_grid_length <- 20 
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0
beta0_grid <- sort(c(beta0_grid, 0)) 


beta1_grid_length <- 20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 

ebic_eval_optim_GOLAZO_A2_full <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(stock)) 
  
  beta_optimise <- matrix(NA, nrow = beta0_grid_length, ncol = beta1_grid_length)
  
  for(b0 in 1:beta0_grid_length){
    for(b1 in 1:beta1_grid_length){
      cat("GOLAZO with A2 b0 = ", b0, ", b1 = ", b1, "\n")
      try(beta_optimise[b0, b1] <- ebic_eval_network(n, R, A2, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], ebic.gamma, edge.tol))
    }
  }
  
  ## Jack: min_coord is a matrix rather than a vector, because there are many minimums, all with the same objective, but the if-else thing is slow
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  
  dim(min_coord) <- c(length(min_coord)/2, 2)
  beta[j, 1] <- mean(beta0_grid[min_coord[,1]])
  beta[j, 2] <- mean(beta1_grid[min_coord[,2]])
  
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A2_full[j] <- min(beta_optimise, na.rm = TRUE)
  
  # U <- exp(beta0 + beta1*A1) 
  U <- exp(beta[j,1] + beta[j,2]*A2) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R, -U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A2_full <- round(res$K, round.tol)  
  # Rho_hat_GOLAZO_A2_full <- threshold(cov2cor(res$K), edge.tol)
}

beta_GOLAZO_A2_full <- beta
beta_GOLAZO_A2_full   #-1.444444 -0.3333333

ebic_eval_optim_GOLAZO_A2_full # 36001.17

sum(cov2cor(Theta_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full)))] != 0) # 685
sum(cov2cor(Theta_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full)))] == 0) # 11718

# sum(cov2cor(Rho_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A2_full)))] != 0) # 585
# sum(cov2cor(Rho_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A2_full)))] == 0) #  11818




#####################################
#####GOLAZO with A2 out of sample log-likelihood
set.seed(1234)
sample_size <- floor(0.9*nrow(stock))


picked <- sample(seq_len(nrow(stock)),size = sample_size)
train_data <- stock[picked,]
test_data  <- stock[-picked,]

n <- sample_size
N <- 1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0


#### grid-search 
beta <- matrix(NA, nrow = N, ncol = 2)

beta0_grid_length <- 20 
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0
beta0_grid <- sort(c(beta0_grid, 0)) 

beta1_grid_length <- 20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 

ebic_eval_optim_GOLAZO_A2_full_out <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(train_data)) 
  
  ## grid-search ##
  beta_optimise <- matrix(NA, nrow = beta0_grid_length, ncol = beta1_grid_length)
  for(b0 in 1:beta0_grid_length){
    for(b1 in 1:beta1_grid_length){
      cat("out GOLAZO A2 b0 = ", b0, ", b1 = ", b1, "\n")
      try(beta_optimise[b0, b1] <- ebic_eval_network(n, R, A2, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], ebic.gamma, edge.tol))
    }
  }
  
  ## Jack: min_coord is a matrix rather than a vector, because there are many minimums, all with the same objective, but the if-else thing is slow
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  
  dim(min_coord) <- c(length(min_coord)/2, 2)
  beta[j, 1] <- mean(beta0_grid[min_coord[,1]])
  beta[j, 2] <- mean(beta1_grid[min_coord[,2]])
  # min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  # beta[j, 1] <- beta0_grid[min_coord[1]]
  # beta[j, 2] <- beta1_grid[min_coord[2]]
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A2_full_out[j] <- min(beta_optimise, na.rm = TRUE)
  
  ## use the optimal beta0 and beta1
  U <- exp(beta[j,1] + beta[j,2]*A2) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R,-U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A2_full_out <- round(res$K, round.tol)
  # Rho_hat_GOLAZO_A2_full_out <- threshold(cov2cor(res$K), edge.tol)
  
  # cat("GOLAZO-trueA", "n", n , "Repeat", j, "done", "\n")    #cat: print output to the screen or to a file. "\n" Line Break separator
}


beta_GOLAZO_A2_full_out <- beta
beta_GOLAZO_A2_full_out # -1.444444 -0.3333333

ebic_eval_optim_GOLAZO_A2_full_out # 32711.63

sum(cov2cor(Theta_hat_GOLAZO_A2_full_out )[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full_out )))] != 0) # 688
sum(cov2cor(Theta_hat_GOLAZO_A2_full_out )[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full_out )))] == 0) #11873

# sum(cov2cor(Rho_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A2_full)))] != 0) # 585
# sum(cov2cor(Rho_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A2_full)))] == 0) #  11818


#############GOLAZO A2 out-of-sample log-likelihood
#JACK: Need to turn correlation into covariance first 
mu_hat <- rep(0, p)
library(mvtnorm)
out_sample_llh_GOLAZO_A2_full <- sum(dmvnorm(test_data, mean = mu_hat, sigma = cor2cov(solve(Theta_hat_GOLAZO_A2_full_out), diag(cov(train_data))), log = TRUE))
out_sample_llh_GOLAZO_A2_full  #10656.96



####################################
###  GOLAZO with Network Matrix A3(E_cos)
p <- ncol(stock)
n <- nrow(stock)
N <-1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0


A3 <- standardise_network_matrix_tri(E_cos) 
diag(A3) <- 0


#### grid-search 
beta <- matrix(NA, nrow = N, ncol = 2)

beta0_grid_length <- 20 
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0
beta0_grid <- sort(c(beta0_grid, 0)) 


beta1_grid_length <- 20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 

ebic_eval_optim_GOLAZO_A3_full <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(stock)) 
  
  beta_optimise <- matrix(NA, nrow = beta0_grid_length, ncol = beta1_grid_length)
  
  for(b0 in 1:beta0_grid_length){
    for(b1 in 1:beta1_grid_length){
      cat("GOLAZO with A3 b0 = ", b0, ", b1 = ", b1, "\n")
      try(beta_optimise[b0, b1] <- ebic_eval_network(n, R, A3, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], ebic.gamma, edge.tol))
    }
  }
  
  ## Jack: min_coord is a matrix rather than a vector, because there are many minimums, all with the same objective, but the if-else thing is slow
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  
  dim(min_coord) <- c(length(min_coord)/2, 2)
  beta[j, 1] <- mean(beta0_grid[min_coord[,1]])
  beta[j, 2] <- mean(beta1_grid[min_coord[,2]])
  
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A3_full[j] <- min(beta_optimise, na.rm = TRUE)
  
  # U <- exp(beta0 + beta1*A1) 
  U <- exp(beta[j,1] + beta[j,2]*A3) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R, -U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A3_full <- round(res$K, round.tol)  
  # Rho_hat_GOLAZO_A3_full <- threshold(cov2cor(res$K), edge.tol)
}

beta_GOLAZO_A3_full <- beta
beta_GOLAZO_A3_full  # -1.222222 -0.5555556

ebic_eval_optim_GOLAZO_A3_full  #35639.39

sum(cov2cor(Theta_hat_GOLAZO_A3_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A3_full)))] != 0) # 585
sum(cov2cor(Theta_hat_GOLAZO_A3_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A3_full)))] == 0) #  11818

# sum(cov2cor(Rho_hat_GOLAZO_A3_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A3_full)))] != 0) # 585
# sum(cov2cor(Rho_hat_GOLAZO_A3_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A3_full)))] == 0) #  11818


#####################################
#####GOLAZO with A3 out of sample log-likelihood
set.seed(1234)
sample_size <- floor(0.9*nrow(stock))


picked <- sample(seq_len(nrow(stock)),size = sample_size)
train_data <- stock[picked,]
test_data  <- stock[-picked,]

n <- sample_size
N <- 1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0


#### grid-search 
beta <- matrix(NA, nrow = N, ncol = 2)

beta0_grid_length <- 20 
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0
beta0_grid <- sort(c(beta0_grid, 0)) 

beta1_grid_length <- 20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 

ebic_eval_optim_GOLAZO_A3_full_out <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(train_data)) 
  
  ## grid-search ##
  beta_optimise <- matrix(NA, nrow = beta0_grid_length, ncol = beta1_grid_length)
  for(b0 in 1:beta0_grid_length){
    for(b1 in 1:beta1_grid_length){
      cat("out GOLAZO A3 b0 = ", b0, ", b1 = ", b1, "\n")
      try(beta_optimise[b0, b1] <- ebic_eval_network(n, R, A3, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], ebic.gamma, edge.tol))
    }
  }
  
  ## Jack: min_coord is a matrix rather than a vector, because there are many minimums, all with the same objective, but the if-else thing is slow
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  
  dim(min_coord) <- c(length(min_coord)/2, 2)
  beta[j, 1] <- mean(beta0_grid[min_coord[,1]])
  beta[j, 2] <- mean(beta1_grid[min_coord[,2]])
  # min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  # beta[j, 1] <- beta0_grid[min_coord[1]]
  # beta[j, 2] <- beta1_grid[min_coord[2]]
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A3_full_out[j] <- min(beta_optimise, na.rm = TRUE)
  
  ## use the optimal beta0 and beta1
  U <- exp(beta[j,1] + beta[j,2]*A3) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R,-U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A3_full_out <- round(res$K, round.tol)
  # Rho_hat_GOLAZO_A3_full_out <- threshold(cov2cor(res$K), edge.tol)
  
  # cat("GOLAZO-trueA", "n", n , "Repeat", j, "done", "\n")    #cat: print output to the screen or to a file. "\n" Line Break separator
}


beta_GOLAZO_A3_full_out <- beta
beta_GOLAZO_A3_full_out # -1.444444 -0.3333333

ebic_eval_optim_GOLAZO_A3_full_out # 32711.63

sum(cov2cor(Theta_hat_GOLAZO_A3_full_out )[lower.tri((cov2cor(Theta_hat_GOLAZO_A3_full_out )))] != 0) # 688
sum(cov2cor(Theta_hat_GOLAZO_A3_full_out )[lower.tri((cov2cor(Theta_hat_GOLAZO_A3_full_out )))] == 0) #11873

# sum(cov2cor(Rho_hat_GOLAZO_A3_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A3_full)))] != 0) # 585
# sum(cov2cor(Rho_hat_GOLAZO_A3_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A3_full)))] == 0) #  11818


#############GOLAZO A3 
#JACK: Need to turn correlation into covariance first 
mu_hat <- rep(0, p)
library(mvtnorm)
out_sample_llh_GOLAZO_A3_full <- sum(dmvnorm(test_data, mean = mu_hat, sigma = cor2cov(solve(Theta_hat_GOLAZO_A3_full_out), diag(cov(train_data))), log = TRUE))
out_sample_llh_GOLAZO_A3_full  #10656.96




####################################
###  GOLAZO with Network Matrix A4(P_cos)
p <- ncol(stock)
n <- nrow(stock)
N <-1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0



A4 <- standardise_network_matrix_tri(P_cos) 
diag(A4) <- 0


#### grid-search 
beta <- matrix(NA, nrow = N, ncol = 2)

beta0_grid_length <- 20 
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0
beta0_grid <- sort(c(beta0_grid, 0)) 


beta1_grid_length <- 20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 

ebic_eval_optim_GOLAZO_A4_full <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(stock)) 
  
  beta_optimise <- matrix(NA, nrow = beta0_grid_length, ncol = beta1_grid_length)
  
  for(b0 in 1:beta0_grid_length){
    for(b1 in 1:beta1_grid_length){
      cat("GOLAZO with A4 b0 = ", b0, ", b1 = ", b1, "\n")
      try(beta_optimise[b0, b1] <- ebic_eval_network(n, R, A4, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], ebic.gamma, edge.tol))
    }
  }
  
  ## Jack: min_coord is a matrix rather than a vector, because there are many minimums, all with the same objective, but the if-else thing is slow
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  
  dim(min_coord) <- c(length(min_coord)/2, 2)
  beta[j, 1] <- mean(beta0_grid[min_coord[,1]])
  beta[j, 2] <- mean(beta1_grid[min_coord[,2]])
  
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A4_full[j] <- min(beta_optimise, na.rm = TRUE)
  
  # U <- exp(beta0 + beta1*A1) 
  U <- exp(beta[j,1] + beta[j,2]*A4) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R, -U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A4_full <- round(res$K, round.tol)  
  # Rho_hat_GOLAZO_A4_full <- threshold(cov2cor(res$K), edge.tol)
}

beta_GOLAZO_A4_full <- beta
beta_GOLAZO_A4_full   #-1.444444 -0.3333333

ebic_eval_optim_GOLAZO_A4_full   #35965.48

sum(cov2cor(Theta_hat_GOLAZO_A4_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A4_full)))] != 0) # 678
sum(cov2cor(Theta_hat_GOLAZO_A4_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A4_full)))] == 0) # 11725

# sum(cov2cor(Rho_hat_GOLAZO_A4_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A4_full)))] != 0) # 678
# sum(cov2cor(Rho_hat_GOLAZO_A4_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A4_full)))] == 0) # 11725





#####################################
#####GOLAZO with A4 out of sample log-likelihood
set.seed(1234)
sample_size <- floor(0.9*nrow(stock))


picked <- sample(seq_len(nrow(stock)),size = sample_size)
train_data <- stock[picked,]
test_data  <- stock[-picked,]

n <- sample_size
N <- 1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0


#### grid-search 
beta <- matrix(NA, nrow = N, ncol = 2)

beta0_grid_length <- 20 
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0
beta0_grid <- sort(c(beta0_grid, 0)) 

beta1_grid_length <- 20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 

ebic_eval_optim_GOLAZO_A4_full_out <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(train_data)) 
  
  ## grid-search ##
  beta_optimise <- matrix(NA, nrow = beta0_grid_length, ncol = beta1_grid_length)
  for(b0 in 1:beta0_grid_length){
    for(b1 in 1:beta1_grid_length){
      cat("out GOLAZO A4 b0 = ", b0, ", b1 = ", b1, "\n")
      try(beta_optimise[b0, b1] <- ebic_eval_network(n, R, A4, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], ebic.gamma, edge.tol))
    }
  }
  
  ## Jack: min_coord is a matrix rather than a vector, because there are many minimums, all with the same objective, but the if-else thing is slow
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  
  dim(min_coord) <- c(length(min_coord)/2, 2)
  beta[j, 1] <- mean(beta0_grid[min_coord[,1]])
  beta[j, 2] <- mean(beta1_grid[min_coord[,2]])
  # min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  # beta[j, 1] <- beta0_grid[min_coord[1]]
  # beta[j, 2] <- beta1_grid[min_coord[2]]
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A4_full_out[j] <- min(beta_optimise, na.rm = TRUE)
  
  ## use the optimal beta0 and beta1
  U <- exp(beta[j,1] + beta[j,2]*A4) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R,-U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A4_full_out <- round(res$K, round.tol)
  # Rho_hat_GOLAZO_A4_full_out <- threshold(cov2cor(res$K), edge.tol)
  
  # cat("GOLAZO-trueA", "n", n , "Repeat", j, "done", "\n")    #cat: print output to the screen or to a file. "\n" Line Break separator
}


beta_GOLAZO_A4_full_out <- beta
beta_GOLAZO_A4_full_out # -1.444444 -0.3333333

ebic_eval_optim_GOLAZO_A4_full_out # 32711.63

sum(cov2cor(Theta_hat_GOLAZO_A4_full_out )[lower.tri((cov2cor(Theta_hat_GOLAZO_A4_full_out )))] != 0) # 688
sum(cov2cor(Theta_hat_GOLAZO_A4_full_out )[lower.tri((cov2cor(Theta_hat_GOLAZO_A4_full_out )))] == 0) #11873

# sum(cov2cor(Rho_hat_GOLAZO_A4_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A4_full)))] != 0) # 585
# sum(cov2cor(Rho_hat_GOLAZO_A4_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A4_full)))] == 0) #  11818


#############GOLAZO A4 
#JACK: Need to turn correlation into covariance first 
mu_hat <- rep(0, p)
library(mvtnorm)
out_sample_llh_GOLAZO_A4_full <- sum(dmvnorm(test_data, mean = mu_hat, sigma = cor2cov(solve(Theta_hat_GOLAZO_A4_full_out), diag(cov(train_data))), log = TRUE))
out_sample_llh_GOLAZO_A4_full  #10656.96




##################################### 
###  GOLAZO with Network Matrix A1 & A2
p <- 99
n <- 97
N <-1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0


A1 <- standardise_network_matrix_tri(E_pears) 
diag(A1) <- 0

A2 <- standardise_network_matrix_tri(P_pears)
diag(A2) <- 0


beta <- matrix(NA, nrow = N, ncol = 3)

beta0_grid_length <- 20  
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0 - beta0_grid_length <- 20 already has 0
beta0_grid <- sort(c(beta0_grid, 0)) 

beta1_grid_length <- 20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 


beta2_grid_length <- 20
beta2_grid_max <- 1
beta2_grid_min <- -3

beta2_grid <- seq(beta2_grid_min, beta2_grid_max, length.out = beta2_grid_length - 1)
# JACK: want to make sure we try at 0
beta2_grid <- sort(c(beta2_grid, 0)) 

ebic_eval_optim_GOLAZO_A1A2_full <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(stock)) 
  
  beta_optimise <- array(NA, dim = c(beta0_grid_length, beta1_grid_length, beta2_grid_length))
  for(b0 in 1:beta0_grid_length)
    for(b1 in 1:beta1_grid_length){
      for(b2 in 1:beta2_grid_length){
        cat("GOLAZO_A1A2 b0 = ", b0, ", b1 = ", b1, ", b2 = ", b2, "\n")
        try(beta_optimise[b0, b1, b2] <- ebic_eval_two_networks(n, R, A1, A2, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], beta2 = beta2_grid[b2], ebic.gamma, edge.tol))
      }
    }
  
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  beta[j, 1] <- beta0_grid[min_coord[1]]
  beta[j, 2] <- beta1_grid[min_coord[2]]
  beta[j, 3] <- beta2_grid[min_coord[3]]
  
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A1A2_full[j] <- min(beta_optimise, na.rm = TRUE)
  
  # U <- exp(beta0 + beta1*A1 + beta2*A2) 
  U <- exp(beta[j,1] + beta[j,2]*A1 + beta[j,3]*A2) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R, -U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A1A2_full <- round(res$K, round.tol)  
  # Rho_hat_GOLAZO_A1A2_full <- threshold(cov2cor(res$K), edge.tol)
}

beta_GOLAZO_A1A2_full <- beta
beta_GOLAZO_A1A2_full

ebic_eval_optim_GOLAZO_A1A2_full

sum(cov2cor(Theta_hat_GOLAZO_A1A2_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1A2_full)))] != 0)
sum(cov2cor(Theta_hat_GOLAZO_A1A2_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1A2_full)))] == 0)

# sum(cov2cor(Rho_hat_GOLAZO_A1A2_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A1A2_full)))] != 0)
# sum(cov2cor(Rho_hat_GOLAZO_A1A2_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A1A2_full)))] == 0)

#### plot the EBIC  ####
## becuase i repeated 0 by accident
beta_optimise_plot <- beta_optimise

## ploting the EBIC in Bivariate Marginals - need to marginalise
library(graphics)
contour(beta1_grid, beta2_grid, apply(beta_optimise_plot, MARGIN = c(2, 3), FUN = function(x){mean(x, na.rm = TRUE)}), xlab = "beta1", ylab = "beta2")
points(beta_GOLAZO_A1A2_full[2], beta_GOLAZO_A1A2_full[3], col = "red", pch = 4, cex = 1.5, lwd = 3)

contour(beta0_grid, beta1_grid, apply(beta_optimise_plot, MARGIN = c(1, 2), FUN = function(x){mean(x, na.rm = TRUE)}), xlab = "beta0", ylab = "beta1")
points(beta_GOLAZO_A1A2_full[1], beta_GOLAZO_A1A2_full[2], col = "red", pch = 4, cex = 1.5, lwd = 3)

contour(beta0_grid, beta2_grid, apply(beta_optimise_plot, MARGIN = c(1, 3), FUN = function(x){mean(x, na.rm = TRUE)}), xlab = "beta0", ylab = "beta2")
points(beta_GOLAZO_A1A2_full[1], beta_GOLAZO_A1A2_full[3], col = "red", pch = 4, cex = 1.5, lwd = 3)


## fixing the missing at it's optimum
## ploting the EBIC in Bivariate Marginals - need to marginalise
library(graphics)
contour(beta1_grid, beta2_grid, beta_optimise_plot[min_coord[1],,], xlab = "beta1", ylab = "beta2", main = "beta0 = hat_beta0")
points(beta_GOLAZO_A1A2_full[2], beta_GOLAZO_A1A2_full[3], col = "red", pch = 4, cex = 1.5, lwd = 3)

contour(beta0_grid, beta1_grid, beta_optimise_plot[,min_coord[2],], xlab = "beta0", ylab = "beta1", main = "beta2 = hat_beta2")
points(beta_GOLAZO_A1A2_full[1], beta_GOLAZO_A1A2_full[2], col = "red", pch = 4, cex = 1.5, lwd = 3)

contour(beta0_grid, beta2_grid, beta_optimise_plot[,,min_coord[3]], xlab = "beta0", ylab = "beta2", main = "beta1 = hat_beta1")
points(beta_GOLAZO_A1A2_full[1], beta_GOLAZO_A1A2_full[3], col = "red", pch = 4, cex = 1.5, lwd = 3)


## Close to the minimum 
n_close <- 10
beta_close <- matrix(NA, nrow = n_close, ncol = 3)
for(k in 1:n_close){
  min_close_coord <- which(beta_optimise == (sort(beta_optimise)[k]), arr.ind = TRUE)
  
  beta_close[k, 1] <- beta1_grid[min_close_coord[1]]
  beta_close[k, 2] <- beta1_grid[min_close_coord[2]]
  beta_close[k, 3] <- beta2_grid[min_close_coord[3]]
}

beta_close
sort(beta_optimise)[1:n_close]





##################################### 
###  GOLAZO with Network Matrix A3 & A4
p <- 99
n <- 97
N <-1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0


A3 <- standardise_network_matrix_tri(E_cos) 
diag(A3) <- 0

A4 <- standardise_network_matrix_tri(P_cos)
diag(A4) <- 0


beta <- matrix(NA, nrow = N, ncol = 3)

beta0_grid_length <-5 #20
beta0_grid_max <- 1
beta0_grid_min <- -3

beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length - 1)
# JACK: want to make sure we try at 0 - beta0_grid_length <- 20 already has 0
beta0_grid <- sort(c(beta0_grid, 0)) 

beta1_grid_length <- 5 #20
beta1_grid_max <- 1
beta1_grid_min <- -3

beta1_grid <- seq(beta1_grid_min, beta1_grid_max, length.out = beta1_grid_length - 1)
# JACK: want to make sure we try at 0
beta1_grid <- sort(c(beta1_grid, 0)) 


beta2_grid_length <- 5 #20
beta2_grid_max <- 1
beta2_grid_min <- -3

beta2_grid <- seq(beta2_grid_min, beta2_grid_max, length.out = beta2_grid_length - 1)
# JACK: want to make sure we try at 0
beta2_grid <- sort(c(beta2_grid, 0)) 

ebic_eval_optim_GOLAZO_A3A4_full <- rep(NA, N)

for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(stock)) 
  
  beta_optimise <- array(NA, dim = c(beta0_grid_length, beta1_grid_length, beta2_grid_length))
  for(b0 in 1:beta0_grid_length)
    for(b1 in 1:beta1_grid_length){
      for(b2 in 1:beta2_grid_length){
        cat("GOLAZO_A3A4 b0 = ", b0, ", b1 = ", b1, ", b2 = ", b2, "\n")
        try(beta_optimise[b0, b1, b2] <- ebic_eval_two_networks(n, R, A3, A4, beta0 = beta0_grid[b0], beta1 = beta1_grid[b1], beta2 = beta2_grid[b2], ebic.gamma, edge.tol))
      }
    }
  
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  beta[j, 1] <- beta0_grid[min_coord[1]]
  beta[j, 2] <- beta1_grid[min_coord[2]]
  beta[j, 3] <- beta2_grid[min_coord[3]]
  
  # JACK: Saving the EBIC 
  ebic_eval_optim_GOLAZO_A3A4_full[j] <- min(beta_optimise, na.rm = TRUE)
  
  # U <- exp(beta0 + beta1*A1 + beta2*A2) 
  U <- exp(beta[j,1] + beta[j,2]*A3 + beta[j,3]*A4) 
  
  ###################################
  # we are now ready to run GOLAZO and output the optimal K
  diag(U) <- 0
  
  res <- golazo(R, -U, U, verbose=FALSE)
  
  Theta_hat_GOLAZO_A3A4_full <- round(res$K, round.tol)  
  # Rho_hat_GOLAZO_A3A4_full <- threshold(cov2cor(res$K), edge.tol)
}

beta_GOLAZO_A3A4_full <- beta
beta_GOLAZO_A3A4_full

ebic_eval_optim_GOLAZO_A3A4_full

sum(cov2cor(Theta_hat_GOLAZO_A3A4_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A3A4_full)))] != 0)
sum(cov2cor(Theta_hat_GOLAZO_A3A4_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A3A4_full)))] == 0)


# sum(cov2cor(Rho_hat_GOLAZO_A3A4_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A3A4_full)))] != 0)
# sum(cov2cor(Rho_hat_GOLAZO_A3A4_full)[lower.tri((cov2cor(Rho_hat_GOLAZO_A3A4_full)))] == 0)




####################################
#### Code to produce the Latex table

library(xtable)
options(xtable.floating = FALSE)
options(xtable.timestamp = "")



table_frame_freq <- data.frame("Method" = c("GLASSO", "GOLAZO - E_pears",  "GOLAZO - P_pears"), 
                               "EBIC" = c(ebic_eval_optim_GLASSO, 
                                          ebic_eval_optim_GOLAZO_A1_full + log(nrow(stock)), 
                                          ebic_eval_optim_GOLAZO_A2_full + log(nrow(stock))),
                               "hat{beta}_0" = c(beta0_GLASSO, 
                                                 beta_GOLAZO_A1_full[,1],
                                                 beta_GOLAZO_A2_full[,1]),
                               "hat{beta}_1" = c(NA, 
                                                 beta_GOLAZO_A1_full[,2],
                                                 NA), 
                               "hat{beta}_2" = c(NA, 
                                                 NA, 
                                                 beta_GOLAZO_A2_full[,2]),
                               "edges" = c(sum(cov2cor(Theta_hat_GLASSO)[lower.tri((cov2cor(Theta_hat_GLASSO)))] != 0), 
                                           sum(cov2cor(Theta_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full)))] != 0),
                                           sum(cov2cor(Theta_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full)))] != 0)),
                               "non_edges" = c(sum(cov2cor(Theta_hat_GLASSO)[lower.tri((cov2cor(Theta_hat_GLASSO)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full)))] == 0))
)
xtable(table_frame_freq, digits=3)




table_frame_freq <- data.frame("Method" = c("GLASSO", "GOLAZO - E_pears",  "GOLAZO - P_pears"), 
                               "out of sample-llh" = c(out_sample_llh_GLASSO,
                                                       out_sample_llh_GOLAZO_A1_full,
                                                       out_sample_llh_GOLAZO_A2_full),
                               "EBIC" = c(ebic_eval_optim_GLASSO_out, 
                                          ebic_eval_optim_GOLAZO_A1_full_out + log(nrow(stock)), 
                                          ebic_eval_optim_GOLAZO_A2_full_out + log(nrow(stock))),
                               "hat{beta}_0" = c(beta0_GLASSO_out, 
                                                 beta_GOLAZO_A1_full_out[,1],
                                                 beta_GOLAZO_A2_full_out[,1]),
                               "hat{beta}_1" = c(NA, 
                                                 beta_GOLAZO_A1_full_out[,2],
                                                 NA), 
                               "hat{beta}_2" = c(NA, 
                                                 NA, 
                                                 beta_GOLAZO_A2_full_out[,2]),
                               "edges" = c(sum(cov2cor(Theta_hat_GLASSO_out)[lower.tri((cov2cor(Theta_hat_GLASSO_out)))] != 0), 
                                           sum(cov2cor(Theta_hat_GOLAZO_A1_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full_out)))] != 0),
                                           sum(cov2cor(Theta_hat_GOLAZO_A2_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full_out)))] != 0)),
                               "non_edges" = c(sum(cov2cor(Theta_hat_GLASSO_out)[lower.tri((cov2cor(Theta_hat_GLASSO_out)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A1_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full_out)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A2_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full_out)))] == 0))
)
xtable(table_frame_freq, digits=3)






table_frame_freq <- data.frame("Method" = c("GLASSO", "GOLAZO - E_pears",  "GOLAZO - P_pears", 
                                            "GOLAZO - E_cos", "GOLAZO - P_cos"), 
                               "EBIC" = c(ebic_eval_optim_GLASSO, 
                                          ebic_eval_optim_GOLAZO_A1_full + log(nrow(stock)), 
                                          ebic_eval_optim_GOLAZO_A2_full + log(nrow(stock)),
                                          ebic_eval_optim_GOLAZO_A3_full + log(nrow(stock)), 
                                          ebic_eval_optim_GOLAZO_A4_full + log(nrow(stock))),
                               "hat{beta}_0" = c(beta0_GLASSO, 
                                                 beta_GOLAZO_A1_full[,1],
                                                 beta_GOLAZO_A2_full[,1],
                                                 beta_GOLAZO_A3_full[,1],
                                                 beta_GOLAZO_A4_full[,1]),
                               "hat{beta}_1" = c(NA, 
                                                 beta_GOLAZO_A1_full[,2],
                                                 NA,
                                                 beta_GOLAZO_A3_full[,2],
                                                 NA), 
                               "hat{beta}_2" = c(NA, 
                                                 NA, 
                                                 beta_GOLAZO_A2_full[,2],
                                                 NA,
                                                 beta_GOLAZO_A4_full[,2]),
                               "edges" = c(sum(cov2cor(Theta_hat_GLASSO)[lower.tri((cov2cor(Theta_hat_GLASSO)))] != 0), 
                                           sum(cov2cor(Theta_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full)))] != 0),
                                           sum(cov2cor(Theta_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full)))] != 0),
                                           sum(cov2cor(Theta_hat_GOLAZO_A3_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A3_full)))] != 0),
                                           sum(cov2cor(Theta_hat_GOLAZO_A4_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A4_full)))] != 0)),
                               "non_edges" = c(sum(cov2cor(Theta_hat_GLASSO)[lower.tri((cov2cor(Theta_hat_GLASSO)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A1_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A2_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A3_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A3_full)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A4_full)[lower.tri((cov2cor(Theta_hat_GOLAZO_A4_full)))] == 0))
)
xtable(table_frame_freq, digits=3)






table_frame_freq <- data.frame("Method" = c("GLASSO", "GOLAZO - E_pears",  "GOLAZO - P_pears", 
                                            "GOLAZO - E_cos", "GOLAZO - P_cos"), 
                               "out of sample-llh" = c(out_sample_llh_GLASSO,
                                                       out_sample_llh_GOLAZO_A1_full,
                                                       out_sample_llh_GOLAZO_A2_full,
                                                       out_sample_llh_GOLAZO_A3_full,
                                                       out_sample_llh_GOLAZO_A4_full),
                               "EBIC" = c(ebic_eval_optim_GLASSO_out, 
                                          ebic_eval_optim_GOLAZO_A1_full_out + log(nrow(stock)), 
                                          ebic_eval_optim_GOLAZO_A2_full_out + log(nrow(stock)),
                                          ebic_eval_optim_GOLAZO_A3_full_out + log(nrow(stock)), 
                                          ebic_eval_optim_GOLAZO_A4_full_out + log(nrow(stock))),
                               "hat{beta}_0" = c(beta0_GLASSO_out, 
                                                 beta_GOLAZO_A1_full_out[,1],
                                                 beta_GOLAZO_A2_full_out[,1],
                                                 beta_GOLAZO_A3_full_out[,1],
                                                 beta_GOLAZO_A4_full_out[,1]),
                               "hat{beta}_1" = c(NA, 
                                                 beta_GOLAZO_A1_full_out[,2],
                                                 NA,
                                                 beta_GOLAZO_A3_full_out[,2],
                                                 NA), 
                               "hat{beta}_2" = c(NA, 
                                                 NA, 
                                                 beta_GOLAZO_A2_full_out[,2],
                                                 NA,
                                                 beta_GOLAZO_A4_full_out[,2]),
                               "edges" = c(sum(cov2cor(Theta_hat_GLASSO_out)[lower.tri((cov2cor(Theta_hat_GLASSO_out)))] != 0), 
                                           sum(cov2cor(Theta_hat_GOLAZO_A1_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full_out)))] != 0),
                                           sum(cov2cor(Theta_hat_GOLAZO_A2_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full_out)))] != 0),
                                           sum(cov2cor(Theta_hat_GOLAZO_A3_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A3_full_out)))] != 0),
                                           sum(cov2cor(Theta_hat_GOLAZO_A4_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A4_full_out)))] != 0)),
                               "non_edges" = c(sum(cov2cor(Theta_hat_GLASSO_out)[lower.tri((cov2cor(Theta_hat_GLASSO_out)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A1_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A1_full_out)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A2_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A2_full_out)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A3_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A3_full_out)))] == 0),
                                               sum(cov2cor(Theta_hat_GOLAZO_A4_full_out)[lower.tri((cov2cor(Theta_hat_GOLAZO_A4_full_out)))] == 0))
)
xtable(table_frame_freq, digits=3)


##################################### 
## PLOT A1 and A2


## The GLASSO estimate for Theta
cov_Theta_hat_GLASSO <- -cov2cor(Theta_hat_GLASSO)
partial_corr_Theta_hat_GLASSO <- cov_Theta_hat_GLASSO[upper.tri(cov_Theta_hat_GLASSO)]

# partial_corr_Rho_hat_GLASSO <- Rho_hat_GLASSO[upper.tri(Rho_hat_GLASSO)]

##################################### 
par(mar = c(3.3, 3.6, 1.5, 1.1))  # bottom, left, top, right
par(mgp = c(2.15, 1, 0))
par(cex.lab = 1.25, cex.axis = 1.25, cex.main = 1.25)

################# A1 ################
A1 <- standardise_network_matrix_tri(E_pears) 
diag(A1) <- 0
A1_plot <- A1[upper.tri(A1)]


pdf('/Users/hanna/Desktop/A1-equi-obs.pdf')
## Binning the network - 
library(dplyr)
num_bins <- 10

# Quantile bins - Equi-obs bins
df_A1_plot <- data.frame(A1_plot)

df_A1_plot <- df_A1_plot %>% mutate(A1_plot_bin = cut(A1_plot, 
                                                      breaks = unique(quantile(A1_plot, probs=seq.int(0,1, by=1/num_bins))), 
                                                      include.lowest=TRUE))
df_A1_plot <- df_A1_plot %>% mutate(A1_plot_bin_cat = ntile(A1_plot, num_bins))
df_A1_plot <- df_A1_plot %>% mutate(partial_corr_Theta_hat_GLASSO = partial_corr_Theta_hat_GLASSO)

par(mfrow = c(2, 2))

plot(aggregate(df_A1_plot$A1_plot, list(df_A1_plot$A1_plot_bin_cat), mean)$x, log(aggregate(df_A1_plot$partial_corr_Theta_hat_GLASSO^2, list(df_A1_plot$A1_plot_bin_cat), mean)$x), xlab = "A1 - bin mid-point", ylab = "log E[r_ij^2 | A1 binned]", main = "Quantile (Equi-obs) bins")
# JACK: Adding the GOLAZO fitted regression line
lines(seq(-3, 3, length.out = 1000), log(2) - 2*(beta_GOLAZO_A1_full[1]  + log(n)) - 2*seq(-3, 3, length.out = 1000)*beta_GOLAZO_A1_full[2], lwd = 3, col = "blue", lty = 2)
legend("bottomright", c("Fitted"), col = c("blue"), lty = c(2), cex = 0.5)


# Equi-width bins
df_A1_plot <- data.frame(A1_plot)
Equi_bins_breaks <- seq(min(A1_plot), max(A1_plot), length.out = num_bins + 1)
A1_plot_bin_mid = (Equi_bins_breaks[1:num_bins] + Equi_bins_breaks[2:(num_bins + 1)])/2

df_A1_plot <- df_A1_plot %>% mutate(A1_plot_bin = cut(A1_plot, 
                                                      breaks = Equi_bins_breaks, 
                                                      include.lowest=TRUE))
df_A1_plot <- df_A1_plot %>% mutate(partial_corr_Theta_hat_GLASSO = partial_corr_Theta_hat_GLASSO)

plot(A1_plot_bin_mid, log(aggregate(df_A1_plot$partial_corr_Theta_hat_GLASSO^2, list(df_A1_plot$A1_plot_bin), mean)$x), xlab = "A1 - bin mid-point", ylab = "log E[r_ij^2 | A1 binned]", main = "Equi-spaced bins")
lines(seq(-3, 3, length.out = 1000), log(2) - 2*(beta_GOLAZO_A1_full[1]  + log(n)) - 2*seq(-3, 3, length.out = 1000)*beta_GOLAZO_A1_full[2], lwd = 3, col = "blue", lty = 2)
legend("bottomright", c("Fitted"), col = c("blue"), lty = c(2), cex = 0.5)

# par(mfrow = c(1, 1))
# dev.off()


################# A2 ################
A2 <- standardise_network_matrix_tri(P_pears) 
diag(A2) <- 0
A2_plot <- A2[upper.tri(A2)]

## Binning the network - 
library(dplyr)
num_bins <- 10

# Quantile bins - Equi-obs bins
df_A2_plot <- data.frame(A2_plot)

df_A2_plot <- df_A2_plot %>% mutate(A2_plot_bin = cut(A2_plot, 
                                                      breaks = unique(quantile(A2_plot, probs=seq.int(0,1, by=1/num_bins))), 
                                                      include.lowest=TRUE))
df_A2_plot <- df_A2_plot %>% mutate(A2_plot_bin_cat = ntile(A2_plot, num_bins))
df_A2_plot <- df_A2_plot %>% mutate(partial_corr_Theta_hat_GLASSO = partial_corr_Theta_hat_GLASSO)

plot(aggregate(df_A2_plot$A2_plot, list(df_A2_plot$A2_plot_bin_cat), mean)$x, log(aggregate(df_A2_plot$partial_corr_Theta_hat_GLASSO^2, list(df_A2_plot$A2_plot_bin_cat), mean)$x), xlab = "A2 - bin mid-point", ylab = "log E[r_ij^2 | A2 binned]", main = "Quantile (Equi-obs) bins")
# JACK: Adding the GOLAZO fitted regression line
lines(seq(-3, 3, length.out = 1000), log(2) - 2*(beta_GOLAZO_A2_full[1]  + log(n)) - 2*seq(-3, 3, length.out = 1000)*beta_GOLAZO_A2_full[2], lwd = 3, col = "blue", lty = 2)
legend("bottomright", c("Fitted"), col = c("blue"), lty = c(2), cex = 0.5)

# Equi-width bins
df_A2_plot <- data.frame(A2_plot)
Equi_bins_breaks <- seq(min(A2_plot), max(A2_plot), length.out = num_bins + 1)
A2_plot_bin_mid = (Equi_bins_breaks[1:num_bins] + Equi_bins_breaks[2:(num_bins + 1)])/2

df_A2_plot <- df_A2_plot %>% mutate(A2_plot_bin = cut(A2_plot, 
                                                      breaks = Equi_bins_breaks, 
                                                      include.lowest=TRUE))
df_A2_plot <- df_A2_plot %>% mutate(partial_corr_Theta_hat_GLASSO = partial_corr_Theta_hat_GLASSO)


plot(A2_plot_bin_mid, log(aggregate(df_A2_plot$partial_corr_Theta_hat_GLASSO^2, list(df_A2_plot$A2_plot_bin), mean)$x), xlab = "A2 - bin mid-point", ylab = "log E[r_ij^2 | A2 binned]", main = "Equi-spaced bins")
lines(seq(-3, 3, length.out = 1000), log(2) - 2*(beta_GOLAZO_A2_full[1]  + log(n)) - 2*seq(-3, 3, length.out = 1000)*beta_GOLAZO_A2_full[2], lwd = 3, col = "blue", lty = 2)
legend("bottomright", c("Fitted"), col = c("blue"), lty = c(2), cex = 0.5)

par(mfrow = c(1, 1))
dev.off()





####################################
## PLOT A3 and A4


par(mar = c(3.3, 3.6, 1.5, 1.1))  # bottom, left, top, right
par(mgp = c(2.15, 1, 0))
par(cex.lab = 1.25, cex.axis = 1.25, cex.main = 1.25)

################# A1 ################
A3 <- standardise_network_matrix_tri(E_cos) 
diag(A3) <- 0
A3_plot <- A3[upper.tri(A3)]


pdf('/Users/hanna/Desktop/A3 A4.pdf')
## Binning the network - 
library(dplyr)
num_bins <- 10

# Quantile bins - Equi-obs bins
df_A3_plot <- data.frame(A3_plot)

df_A3_plot <- df_A3_plot %>% mutate(A3_plot_bin = cut(A3_plot, 
                                                      breaks = unique(quantile(A3_plot, probs=seq.int(0,1, by=1/num_bins))), 
                                                      include.lowest=TRUE))
df_A3_plot <- df_A3_plot %>% mutate(A3_plot_bin_cat = ntile(A3_plot, num_bins))
df_A3_plot <- df_A3_plot %>% mutate(partial_corr_Theta_hat_GLASSO = partial_corr_Theta_hat_GLASSO)

par(mfrow = c(2, 2))

plot(aggregate(df_A3_plot$A3_plot, list(df_A3_plot$A3_plot_bin_cat), mean)$x, log(aggregate(df_A3_plot$partial_corr_Theta_hat_GLASSO^2, list(df_A3_plot$A3_plot_bin_cat), mean)$x), xlab = "A3 - bin mid-point", ylab = "log E[r_ij^2 | A3 binned]", main = "Quantile (Equi-obs) bins")
# JACK: Adding the GOLAZO fitted regression line
lines(seq(-3, 3, length.out = 1000), log(2) - 2*(beta_GOLAZO_A3_full[1]  + log(n)) - 2*seq(-3, 3, length.out = 1000)*beta_GOLAZO_A3_full[2], lwd = 3, col = "blue", lty = 2)
legend("bottomright", c("Fitted"), col = c("blue"), lty = c(2), cex = 0.5)


# Equi-width bins
df_A3_plot <- data.frame(A3_plot)
Equi_bins_breaks <- seq(min(A3_plot), max(A3_plot), length.out = num_bins + 1)
A3_plot_bin_mid = (Equi_bins_breaks[1:num_bins] + Equi_bins_breaks[2:(num_bins + 1)])/2

df_A3_plot <- df_A3_plot %>% mutate(A3_plot_bin = cut(A3_plot, 
                                                      breaks = Equi_bins_breaks, 
                                                      include.lowest=TRUE))
df_A3_plot <- df_A3_plot %>% mutate(partial_corr_Theta_hat_GLASSO = partial_corr_Theta_hat_GLASSO)

plot(A3_plot_bin_mid, log(aggregate(df_A3_plot$partial_corr_Theta_hat_GLASSO^2, list(df_A3_plot$A3_plot_bin), mean)$x), xlab = "A3 - bin mid-point", ylab = "log E[r_ij^2 | A3 binned]", main = "Equi-spaced bins")
lines(seq(-3, 3, length.out = 1000), log(2) - 2*(beta_GOLAZO_A3_full[1]  + log(n)) - 2*seq(-3, 3, length.out = 1000)*beta_GOLAZO_A3_full[2], lwd = 3, col = "blue", lty = 2)
legend("bottomright", c("Fitted"), col = c("blue"), lty = c(2), cex = 0.5)

# par(mfrow = c(1, 1))
# dev.off()


################# A2 ################
A4 <- standardise_network_matrix_tri(P_cos) 
diag(A4) <- 0
A4_plot <- A4[upper.tri(A4)]

## Binning the network - 
library(dplyr)
num_bins <- 10

# Quantile bins - Equi-obs bins
df_A4_plot <- data.frame(A4_plot)

df_A4_plot <- df_A4_plot %>% mutate(A4_plot_bin = cut(A4_plot, 
                                                      breaks = unique(quantile(A4_plot, probs=seq.int(0,1, by=1/num_bins))), 
                                                      include.lowest=TRUE))
df_A4_plot <- df_A4_plot %>% mutate(A4_plot_bin_cat = ntile(A4_plot, num_bins))
df_A4_plot <- df_A4_plot %>% mutate(partial_corr_Theta_hat_GLASSO = partial_corr_Theta_hat_GLASSO)

plot(aggregate(df_A4_plot$A4_plot, list(df_A4_plot$A4_plot_bin_cat), mean)$x, log(aggregate(df_A4_plot$partial_corr_Theta_hat_GLASSO^2, list(df_A4_plot$A4_plot_bin_cat), mean)$x), xlab = "A4 - bin mid-point", ylab = "log E[r_ij^2 | A4 binned]", main = "Quantile (Equi-obs) bins")
# JACK: Adding the GOLAZO fitted regression line
lines(seq(-3, 3, length.out = 1000), log(2) - 2*(beta_GOLAZO_A4_full[1]  + log(n)) - 2*seq(-3, 3, length.out = 1000)*beta_GOLAZO_A4_full[2], lwd = 3, col = "blue", lty = 2)
legend("bottomright", c("Fitted"), col = c("blue"), lty = c(2), cex = 0.5)

# Equi-width bins
df_A4_plot <- data.frame(A4_plot)
Equi_bins_breaks <- seq(min(A4_plot), max(A4_plot), length.out = num_bins + 1)
A4_plot_bin_mid = (Equi_bins_breaks[1:num_bins] + Equi_bins_breaks[2:(num_bins + 1)])/2

df_A4_plot <- df_A4_plot %>% mutate(A4_plot_bin = cut(A4_plot, 
                                                      breaks = Equi_bins_breaks, 
                                                      include.lowest=TRUE))
df_A4_plot <- df_A4_plot %>% mutate(partial_corr_Theta_hat_GLASSO = partial_corr_Theta_hat_GLASSO)


plot(A4_plot_bin_mid, log(aggregate(df_A4_plot$partial_corr_Theta_hat_GLASSO^2, list(df_A4_plot$A4_plot_bin), mean)$x), xlab = "A4 - bin mid-point", ylab = "log E[r_ij^2 | A4 binned]", main = "Equi-spaced bins")
lines(seq(-3, 3, length.out = 1000), log(2) - 2*(beta_GOLAZO_A4_full[1]  + log(n)) - 2*seq(-3, 3, length.out = 1000)*beta_GOLAZO_A4_full[2], lwd = 3, col = "blue", lty = 2)
legend("bottomright", c("Fitted"), col = c("blue"), lty = c(2), cex = 0.5)

par(mfrow = c(1, 1))
dev.off()





####################################
## PLOTs GLASSO estimate R_ij vs Individual Network B1-B4


par(mar = c(3.3, 3.6, 1.5, 1.1))  # bottom, left, top, right
par(mgp = c(2.15, 1, 0))
par(cex.lab = 1.25, cex.axis = 1.25, cex.main = 1.25)


#######################################
# cov_Theta_hat_GLASSO <- -cov2cor(Theta_hat_GLASSO)       
# partial_corr <- cov_Theta_hat_GLASSO[upper.tri(cov_Theta_hat_GLASSO)]

 partial_corr <- Rho_hat_GLASSO[upper.tri(Rho_hat_GLASSO)]

mydf2  <- mydf1 %>%select(starts_with("list"))               ## no centering

namefile1 <- names(mydf2)
namefile1

####################################
pdf('/Users/hanna/Desktop/B1-B4.pdf')
par(mfrow = c(2, 2))

for (i in 1:4){
  
  Brisk <- standardise_network_matrix_tri(allmatrix[[i]]) 
  diag(Brisk) <- 0
  Brisk_plot <- Brisk[upper.tri(Brisk)]
  
  #DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
  # plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "_Risk", i), ylab ="GLASSO estimates of R_ij")
  plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "(", namefile1[i], ")"), ylab ="GLASSO estimates of R_ij")
  l <- loess(partial_corr ~ Brisk_plot)
  o <- order(l$x)
  lines(l$x[o], l$fitted[o], col='red')

}

par(mfrow = c(1, 1))
dev.off()



####################################
pdf('/Users/hanna/Desktop/B5-B8.pdf')
par(mfrow = c(2, 2))

for (i in 5:8){
  
  Brisk <- standardise_network_matrix_tri(allmatrix[[i]]) 
  diag(Brisk) <- 0
  Brisk_plot <- Brisk[upper.tri(Brisk)]
  
  #DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
  plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "_Risk", i), ylab ="GLASSO estimates of R_ij")
  l <- loess(partial_corr ~ Brisk_plot)
  o <- order(l$x)
  lines(l$x[o], l$fitted[o], col='red')
  
}

par(mfrow = c(1, 1))
dev.off()





####################################
pdf('/Users/hanna/Desktop/B9-B12.pdf')
par(mfrow = c(2, 2))

for (i in 9:12){
  
  Brisk <- standardise_network_matrix_tri(allmatrix[[i]]) 
  diag(Brisk) <- 0
  Brisk_plot <- Brisk[upper.tri(Brisk)]
  
  #DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
  plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "_Risk", i), ylab ="GLASSO estimates of R_ij")
  l <- loess(partial_corr ~ Brisk_plot)
  o <- order(l$x)
  lines(l$x[o], l$fitted[o], col='red')
  
}

par(mfrow = c(1, 1))
dev.off()




####################################
pdf('/Users/hanna/Desktop/B13-B16.pdf')
par(mfrow = c(2, 2))

for (i in 13:16){
  
  Brisk <- standardise_network_matrix_tri(allmatrix[[i]]) 
  diag(Brisk) <- 0
  Brisk_plot <- Brisk[upper.tri(Brisk)]
  
  #DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
  plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "_Risk", i), ylab ="GLASSO estimates of R_ij")
  l <- loess(partial_corr ~ Brisk_plot)
  o <- order(l$x)
  lines(l$x[o], l$fitted[o], col='red')
  
}

par(mfrow = c(1, 1))
dev.off()




####################################
pdf('/Users/hanna/Desktop/B17-B20.pdf')
par(mfrow = c(2, 2))

for (i in 17:20){
  
  Brisk <- standardise_network_matrix_tri(allmatrix[[i]]) 
  diag(Brisk) <- 0
  Brisk_plot <- Brisk[upper.tri(Brisk)]
  
  #DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
  plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "_Risk", i), ylab ="GLASSO estimates of R_ij")
  l <- loess(partial_corr ~ Brisk_plot)
  o <- order(l$x)
  lines(l$x[o], l$fitted[o], col='red')
  
}

par(mfrow = c(1, 1))
dev.off()





####################################
pdf('/Users/hanna/Desktop/B21-B24.pdf')
par(mfrow = c(2, 2))

for (i in 21:24){
  
  Brisk <- standardise_network_matrix_tri(allmatrix[[i]]) 
  diag(Brisk) <- 0
  Brisk_plot <- Brisk[upper.tri(Brisk)]
  
  #DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
  plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "_Risk", i), ylab ="GLASSO estimates of R_ij")
  l <- loess(partial_corr ~ Brisk_plot)
  o <- order(l$x)
  lines(l$x[o], l$fitted[o], col='red')
  
}

par(mfrow = c(1, 1))
dev.off()




####################################
pdf('/Users/hanna/Desktop/B25-B28.pdf')
par(mfrow = c(2, 2))

for (i in 25:28){
  
  Brisk <- standardise_network_matrix_tri(allmatrix[[i]]) 
  diag(Brisk) <- 0
  Brisk_plot <- Brisk[upper.tri(Brisk)]
  
  #DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
  plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "_Risk", i), ylab ="GLASSO estimates of R_ij")
  l <- loess(partial_corr ~ Brisk_plot)
  o <- order(l$x)
  lines(l$x[o], l$fitted[o], col='red')
  
}

par(mfrow = c(1, 1))
dev.off()




####################################
pdf('/Users/hanna/Desktop/B29-B32.pdf')
par(mfrow = c(2, 2))

for (i in 29:32){
  
  Brisk <- standardise_network_matrix_tri(allmatrix[[i]]) 
  diag(Brisk) <- 0
  Brisk_plot <- Brisk[upper.tri(Brisk)]
  
  #DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
  plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "_Risk", i), ylab ="GLASSO estimates of R_ij")
  l <- loess(partial_corr ~ Brisk_plot)
  o <- order(l$x)
  lines(l$x[o], l$fitted[o], col='red')
  
}

par(mfrow = c(1, 1))
dev.off()




####################################
pdf('/Users/hanna/Desktop/B33-B36.pdf')
par(mfrow = c(3, 2))

for (i in 33:37){
  
  Brisk <- standardise_network_matrix_tri(allmatrix[[i]]) 
  diag(Brisk) <- 0
  Brisk_plot <- Brisk[upper.tri(Brisk)]
  
  #DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
  plot(x = Brisk_plot, y = partial_corr, xlab = paste0("B", i, "_Risk", i), ylab ="GLASSO estimates of R_ij")
  l <- loess(partial_corr ~ Brisk_plot)
  o <- order(l$x)
  lines(l$x[o], l$fitted[o], col='red')
  
}

par(mfrow = c(1, 1))
dev.off()


pdf('/Users/hanna/Desktop/B37.pdf')
######### B37_pears
B37 <- standardise_network_matrix_tri(B37_pears) 
diag(B37) <- 0
B37_plot <- B37[upper.tri(B37)]


#DAVID. USING LOESS INSTEAD OF LOWESS. IT FITS THE DATA BETTER
plot(x = B37_plot, y = partial_corr, xlab = "B37(Risk 37)", ylab ="GLASSO estimates of R_ij")
l <- loess(partial_corr ~ B37_plot)
o <- order(l$x)
lines(l$x[o], l$fitted[o], col='red')
dev.off()




####################################
##### count the 0 corresponding to positive B1-B4
var1 <- as.vector(B1_plot)
var2 <- as.vector(B2_plot)
var3 <- as.vector(B3_plot)
var4 <- as.vector(B4_plot)
var_par <- as.vector(partial_corr)
dfB1 <- data.frame(var1, var2, var3, var3, var_par)


a1 <- count(dfB1, var1 <0 & var_par ==0)
a2 <-count(dfB1, var2 <0 & var_par ==0)
a3 <-count(dfB1, var3 <0 & var_par ==0)
a4 <-count(dfB1, var4 <0 & var_par ==0)


####################################
#### Code to produce the Latex table

library(xtable)
options(xtable.floating = FALSE)
options(xtable.timestamp = "")

table_frame_freq <- data.frame("Network" = c("B1", "B2",  "B3", "B4"), 
                               "B<0, GLASSO=0" = c(a1[2,2], a2[2,2], a3[2,2], a4[2,2]),
                               "B>0, GLASSO=0" = c(a1[1,2], a2[1,2], a3[1,2], a4[1,2]))
xtable(table_frame_freq, digits=3)






#######################################

#######################################
## GLASSO with transformed data stockn ##
p <- ncol(stockn)   #200
n <- nrow(stocn)   #252
N <-1

L <- matrix(-1,p,p)    
U <- matrix (1,p,p)
diag(U) <- diag(L) <- 0

beta0 <-rep(NA, N)
beta0_grid_length <- 20
beta0_grid_min <- -3

ebic_eval_optim_GLASSO_n <- rep(NA, N)


for(j in 1:N){
  
  #### Estimating lambda ####
  R <- stats::cov2cor(cov(stockn))  
  
  ## grid-search ##
  beta0_grid_max <- beta0_max_GLASSO(R)
  beta0_grid <- seq(beta0_grid_min, beta0_grid_max, length.out = beta0_grid_length )
  
  beta_optimise <- rep(NA, beta0_grid_length)
  
  for(b in 1:beta0_grid_length){
    cat("GLASSO transformed b = ", b, "\n")
    beta_optimise[b] <- ebic_eval(n, R, U = exp(beta0_grid[b])*U, ebic.gamma, edge.tol)
  }
  
  ## Saving beta0
  min_coord <- which(beta_optimise== min(beta_optimise, na.rm = TRUE), arr.ind = TRUE)
  beta0[j] <- mean(beta0_grid[min_coord])
  
  ## Saving the EBIC 
  ebic_eval_optim_GLASSO_n[j] <- min(beta_optimise)
  
  #### Using the optimal beta0 ##
  GraphicalModel <- golazo (R, L = exp(beta0[j]) * L, U =exp(beta0[j])* U, verbose=FALSE)
  
  Theta_hat_GLASSO_n <- round(GraphicalModel$K, round.tol)   
  # Rho_hat_GLASSO <- threshold(cov2cor(GraphicalModel$K), edge.tol)
  
  
}

beta0_GLASSO_n <- beta0 
beta0_GLASSO_n     #-1.474083

ebic_eval_optim_GLASSO_n   #47659.62

sum(cov2cor(Theta_hat_GLASSO_n)[lower.tri((cov2cor(Theta_hat_GLASSO_n)))] != 0)   # Edges for the last repeat   528
sum(cov2cor(Theta_hat_GLASSO_n)[lower.tri((cov2cor(Theta_hat_GLASSO_n)))] == 0)   # Non-edges for the last repeat  19372










