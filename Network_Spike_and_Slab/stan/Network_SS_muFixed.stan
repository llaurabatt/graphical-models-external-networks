
// Spike-and-slab prior
// reparametrisation to stop it from getting stuck in a small spike

functions {
  matrix to_triangular(vector y_basis, int K) {
    // could check rows(y) = K * (K - 1) / 2
    //matrix[K, K] y; // code works but throws warning
    matrix[K, K] y = rep_matrix(0, K, K);   
    int pos = 1;
    for(i in 2:K) {
      for(j in 1:(i-1)) {
        y[i, j] = y_basis[pos];
        pos += 1;
      }
    }
    return y;
  }
  
  /*
  int ind(real u, real w){
     return (u > (1-w));
  }
  */
  
  
   real sigmoid(real x, real k){
      // return (1/(1+exp( - k * x)));
      return exp( - log1p_exp( - k * x));
   }
   
   
   real ind(real u, real w){
      //return sigmoid(u - (1-w), 1000);
      //return sigmoid(u - (1-w), 500);
      return sigmoid(u - (1-w), 100);
   }
   
   
  
}


data {
   int n;           // Sample size
   int p;           // Observation Dimension
   matrix[n, p] Y;  // Data
   vector[p] mu; // mean
   real mu_m; // Prior hyperparameters mean
   real<lower=0> mu_s; //  "  "
   matrix[p, p] A; // Network predictor 
   real eta0_p1;    // Prior hyperparameters for eta0
   real<lower = 0> eta0_p2;
   real eta1_p1;   // Prior hyperparameters for eta1
   real<lower = 0> eta1_p2;
   real eta2_p1;   // Prior hyperparameters for eta2
   real<lower = 0> eta2_p2;
   real<lower = 0> s0;

}

parameters {
   
   
   //corr_matrix[p] Rho;
   vector<lower=0>[p] sqrt_theta_ii; 
   vector[2] tilde_eta0; // mean of "slab"
   vector[2] tilde_eta1; // GOLAZO parameter for A
   vector[2] tilde_eta2; // logistic regression coefficient
   //real tilde_Rho;
   //real u;
   vector[(p*(p-1))/2] tilde_Rho_basis;// tilde_Rho_basis -> tilde_Rho will have identity Jacobian
   vector<lower=0, upper=1>[(p*(p-1))/2] u_basis;

   
}

transformed parameters {
   vector[2] eta0; // mean of "slab"
   vector[2] eta1; // GOLAZO parameter for A
   vector[2] eta2; // logistic regression coefficient
   matrix[p, p] w;
   matrix[p, p] u;
   corr_matrix[p] Rho; 
   matrix[p, p] tri_tilde_Rho; 
   
   // Adjusting for the different priors
   eta0[1] = eta0_p1 + tilde_eta0[1] * eta0_p2 / sqrt((p*(p-1)/2.0)/n);
   eta0[2] = 0 + tilde_eta0[2] * eta0_p2 / sqrt((p*(p-1)/2.0)/n);
   eta1[1] = eta1_p1 + tilde_eta1[1] * eta1_p2 / sqrt((p*(p-1)/2.0)/n);
   eta1[2] = 0 + tilde_eta1[2] * eta1_p2 / sqrt((p*(p-1)/2.0)/n);
   eta2[1] = eta2_p1 + tilde_eta2[1] * eta2_p2 / sqrt((p*(p-1)/2.0)/n);
   eta2[2] = 0 + tilde_eta2[2] * eta2_p2 / sqrt((p*(p-1)/2.0)/n);

   for(j in 1:p){
      //for(k in 1:j){
      for(k in 1:p){
           w[j, k] = 1/(1 + exp(-(eta2[1] + eta2[2]*A[j, k])));
      }
   }
   
   u = to_triangular(u_basis, p); 

   tri_tilde_Rho = to_triangular(tilde_Rho_basis, p);
   for(j in 1:p){
      for(k in 1:p){
         if(j == k){
            Rho[j, k] = 1; 
         } 
         if(k < j){
            Rho[j, k] = (1-ind(u[j, k], w[j, k]))*(tri_tilde_Rho[j, k]) * s0 + 
            ind(u[j, k], w[j, k])*(tri_tilde_Rho[j, k]*s0*(1 + exp(-(eta1[1] + eta1[2]*A[j, k]))) + (eta0[1] + eta0[2]*A[j, k]));
         }
         if(j < k){
            Rho[j, k] = (1-ind(u[k, j], w[k, j]))*(tri_tilde_Rho[k, j]) * s0 + 
            ind(u[k, j], w[k, j])*(tri_tilde_Rho[k, j]*s0*(1 + exp(-(eta1[1] + eta1[2]*A[j, k]))) + (eta0[1] + eta0[2]*A[j, k]));
         }
      }
   }
   
}

model {

   matrix[p, p] Theta;
   for(j in 1:p){
      for(k in 1:j){
         if(j == k){
            target += inv_gamma_lpdf(sqrt_theta_ii[j] | 0.01, 0.01);
         } else{ 
             target += double_exponential_lpdf(tri_tilde_Rho[j, k] | 0, 1);
          //  target += log(ind*exp(double_exponential_lpdf(tilde_Rho[j, k] |0, 1)*s0) + ind*exp(double_exponential_lpdf(tilde_Rho[j, k] | eta0, s0*(1+exp(-eta1)))) );
         }
      }
   }
   
   target += uniform_lpdf(u_basis | 0, 1);
   
   Theta = quad_form_diag(Rho, sqrt_theta_ii);
   
   target += normal_lpdf(tilde_eta0 | 0, sqrt((p*(p-1)/2.0)/n));
   target += normal_lpdf(tilde_eta1 | 0, sqrt((p*(p-1)/2.0)/n));
   target += normal_lpdf(tilde_eta2 | 0, sqrt((p*(p-1)/2.0)/n));
  
   for (i in 1:n){
     // Y follows multi normal distribution
     target += multi_normal_prec_lpdf(Y[i,] | mu, Theta);
   }
   
}

generated quantities {
   matrix[p, p] Theta;
   Theta = quad_form_diag(Rho, sqrt_theta_ii);
   
}

