
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

transformed data {
  // Sufficient Statistics
  vector[p] x_bar;
  matrix[p, p] S_bar;  
  for(j in 1:p){
    x_bar[j] = mean(Y[,j]);
  }
  S_bar = Y'*Y/n - x_bar*x_bar';
}

parameters {
   
   vector[p] mu; // mean
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
   
   // Adjusting for the different priors - then maybe we can adjust for sample size later 
   /*
   eta0[1] = eta0_p1 + tilde_eta0[1] * eta0_p2;
   eta0[2] = 0 + tilde_eta0[2] * eta0_p2;
   eta1[1] = eta1_p1 + tilde_eta1[1] * eta1_p2;
   eta1[2] = 0 + tilde_eta1[2] * eta1_p2;
   eta2[1] = eta2_p1 + tilde_eta2[1] * eta2_p2;
   eta2[2] = 0 + tilde_eta2[2] * eta2_p2;
   */
   // So we want to add another transformation such that tilde_eta doesn't start from N(0, 1), 
   // but I guess something with a bigger variance, then we need to adjust the transformation to 
   eta0[1] = eta0_p1 + tilde_eta0[1] * eta0_p2 / sqrt((p*(p-1)/2.0)/n);
   eta0[2] = 0 + tilde_eta0[2] * eta0_p2 / sqrt((p*(p-1)/2.0)/n);
   eta1[1] = eta1_p1 + tilde_eta1[1] * eta1_p2 / sqrt((p*(p-1)/2.0)/n);
   eta1[2] = 0 + tilde_eta1[2] * eta1_p2 / sqrt((p*(p-1)/2.0)/n);
   eta2[1] = eta2_p1 + tilde_eta2[1] * eta2_p2 / sqrt((p*(p-1)/2.0)/n);
   eta2[2] = 0 + tilde_eta2[2] * eta2_p2 / sqrt((p*(p-1)/2.0)/n);
   
   
   //eta0 = tilde_eta0 / sqrt((p*(p-1)/2.0)/n);
   //eta1 = tilde_eta1 / sqrt((p*(p-1)/2.0)/n);
   //eta2 = tilde_eta2 / sqrt((p*(p-1)/2.0)/n);
   // tilde_Rho_basis has n x p observations for p(p-1)/2 parameters
   // eta has p(p-1)/2 'observations' for 6 parameters 
   //eta0 = tilde_eta0 / sqrt(((p*(p-1)/2.0)/6.0)/(p*n/(p*(p-1.0)/2.0)));
   //eta1 = tilde_eta1 / sqrt(((p*(p-1)/2.0)/6.0)/(p*n/(p*(p-1.0)/2.0)));
   //eta2 = tilde_eta2 / sqrt(((p*(p-1)/2.0)/6.0)/(p*n/(p*(p-1.0)/2.0)));

   
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
   
   /*
   target += normal_lpdf(eta0[1] | eta0_p1, eta0_p2);// intercept
   target += normal_lpdf(eta0[2] | 0, eta0_p2);// network coefficient
   target += normal_lpdf(eta1[1] | eta1_p1, eta1_p2);// intercept
   target += normal_lpdf(eta1[2] | 0, eta1_p2);// network coefficient
   target += normal_lpdf(eta2[1] | eta2_p1, eta2_p2);// intercept
   target += normal_lpdf(eta2[2] | 0, eta2_p2);// network coefficient
   */
   /*
   target += normal_lpdf(tilde_eta0 | 0, 1);
   target += normal_lpdf(tilde_eta1 | 0, 1);
   target += normal_lpdf(tilde_eta2 | 0, 1);
   */
   target += normal_lpdf(tilde_eta0 | 0, sqrt((p*(p-1)/2.0)/n));
   target += normal_lpdf(tilde_eta1 | 0, sqrt((p*(p-1)/2.0)/n));
   target += normal_lpdf(tilde_eta2 | 0, sqrt((p*(p-1)/2.0)/n));


   // target += normal_lpdf(mu | mu_m, mu_s);
   /*
   for (i in 1:n){
     // Y follows multi normal distribution
     //Y[i,] ~ multi_normal_prec(mu, Theta);
     target += multi_normal_prec_lpdf(Y[i,] | mu, Theta);
   }
   */
   //target += n*(-0.5*p*log(2*pi()) + 0.5*log_determinant(Theta) - 0.5*trace(S*Theta));
   target += n*(-0.5*p*log(2*pi()) + 0.5*log_determinant(Theta)  - 0.5*(x_bar - mu)'*Theta*(x_bar - mu) - 0.5*trace(S_bar*Theta));
   
}

generated quantities {
   //matrix[p, p] Rho;
   matrix[p, p] Theta;
   //Rho = multiply_lower_tri_self_transpose(L_Rho);
   //Theta = quad_form_diag(multiply_lower_tri_self_transpose(L_Rho), sqrt_theta_ii);
   Theta = quad_form_diag(Rho, sqrt_theta_ii);
   
}

