#' Performs GOLAZO algorithm by optimizing the dual problem.
#'
#' This function implements a simple block-coordinate descent algorithm to find the maximum of the regularized
#' Gaussiann log-likelihood  with  a an assymetric penalty of lasso type.
#' @param S Positive semidefinite matrix. This will be typically the sample covariance matrix but it can be somethink different in the dual likelihood computation or when the data follow the non-paranormal distribution.
#' @param L Matrix of lower penalties. It should have all entries non-positive (-Inf is a valid entry). The entries of L say how much negative entries of the inverse of Sigma are penalized. For GLASSO all entries should of L should be -rho and all entries of U should be rho (in both cases with zero diagonal). For positive GOLAZO L should be zero and U should be like for GLASSO.
#' @param U Matrix of upper penalties. It should have all entries non-negative (Inf is a valid entry). The entries of U say how much positive entries of the inverse of Sigma are penalized.
#' @param tol The convergence tolerance (default tol=1e-7). The algorithm termininnates when teh dual gap (guaranteed to be nonnegative) is less than tol.
#' @param verbose if TRUE (default) the output will be printed.
#' @return K the optimal value of the concentration matrix
#' @return Sig the optimal value of the covariance matrix
#' @return it the number of iterations
#' @keywords coordinate descent, concentration matrix.
#' @export
#' @examples
#' data(ability.cov)
#' S <- ability.cov$cov
#' R <- stats::cov2cor(S)
#' d <- nrow(R)
#' L <- matrix(0,d,d)
#' U <- matrix(0.2,d,d)
#' diag(U) <- 0
#' res <- golazo(R,L=L,U=U)
#' Khat <- res$K
#' print(Khat)
golazo <- function(S,L,U,tol=1e-7,verbose=TRUE){
  d <- nrow(S)
  if (verbose==TRUE){
    cat("** The function maximizes the log-likelihood function penalized with the general GOLAZO penalty.\n")
  }
  # as explained in the paper, without loss of generality big entries in L and U can be thresholded
  if (Inf %in% diag(U)){
    cat("Error: U contains Inf on the diagonal. The optimum does not exist.\n")
    return()
  } else{
    aux <- sqrt(outer(diag(S+U),diag(S+U)))
    # save the diagonals for later
    dU <- diag(U)
    dL <- diag(L)
    U <- pmin(U,-S+aux)
    L <- pmax(L,-S-aux)
    diag(U) <- dU
    diag(L) <- dL
  }
  ### compute the starting point
  SU <- S+diag(diag(U))
  # the easy case is when S is bounded away from the boundary of the PSD cone
  if (min(eigen(SU)$values)>1e-4){
    if (verbose==TRUE){
      cat("The algorithm is initiated at the input matrix.\n")
    }
    Sig <- SU
  } else{
    if (verbose==TRUE){
      cat("The input matrix is not positive definite. Computing the starting point..")
    }
    # this is the GLASSO case
    if ((min(U+diag(d))>0 && max(L-diag(d))<0)){
      Z <- diag(diag(SU))
    } else {
      if (min(U+diag(d))==0){
        cat("\n \n **Warning: This combination of L,U (with U having zero off-diagonal entries) is not supported unless S is PD..\n")
        return()
      }
      Z <- Zmatrix(SU)
    }
    t <- 1
    # perform simple backtracing to find a feasible point
    while(!(min(L<=round(t*(Z-SU),8)) && min(round(t*(Z-SU),8)<=U))){
      t <- t/2
    }
    Sig <- (1-t)*SU+t*Z # this is the starting point
    if (verbose==TRUE){
      cat(" DONE\n")
    }
  }
  K <- solve(Sig)
  it <- 0
  if (verbose==TRUE){
    cat("\n The algorithm will stop when the dual gap is below: ",tol,"\b.\n\n")
    cat("Iteration | Dual Gap\n")
  }
  dualgap <- Inf
  eqs <- neqs <- list()
  for (j in 1:d){
    eqs[[j]] <- which(L[j,-j]==U[j,-j])
    neqs[[j]] <- which(L[j,-j]<U[j,-j])
  }
  while(dualgap > tol){
    A <- rbind(diag(d-1),-diag(d-1))
    visit <- 1:d # a priori we visit all the nodes
    for (j in visit){
      if (length(eqs[[j]])==0){
        # in our canonical applications this is what we end up doing
        b <- c(S[j,-j]+L[j,-j],-(S[j,-j]+U[j,-j]))
        y <- quadprog::solve.QP(Dmat=solve(Sig[-j,-j]),dvec=rep(0,d-1),Amat=t(A),bvec=b)$solution
        Sig[j,-j] <- Sig[-j,j] <- y
      } else {
        # this looks a bit complicated but equality constraints are tricky
        # we add it for completeness but typically this is not needed in our canonical applications
        Ap <- rbind(diag(d-1)[eqs[[j]],],diag(d-1)[neqs[[j]],],-diag(d-1)[neqs[[j]],])
        ll <- S[j,-j]+L[j,-j]
        uu <- -(S[j,-j]+U[j,-j])
        b <- c(ll[eqs[[j]]],ll[neqs[[j]]],uu[neqs[[j]]])
        y <- quadprog::solve.QP(Dmat=solve(Sig[-j,-j]),dvec=rep(0,d-1),Amat=t(Ap),bvec=b,meq=length(eqs[[j]]))$solution
        Sig[j,-j] <- Sig[-j,j] <- y
      }
    }
    it <- it+1
    K <- solve(Sig)
    #roundK <- K * (abs(K)>tol)
    dualgap <- sum(S*K)-d+sum(pmax(L*(abs(K)>tol)*K,U*(abs(K)>tol)*K))
    if (verbose==TRUE){
      cat(it,"\t  | ",dualgap,"\n")
    }
  }
  return(list(K=(K+t(K))/2,Sig=(Sig+t(Sig))/2,it=it))
}