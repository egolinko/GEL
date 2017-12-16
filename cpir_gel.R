library(easypackages)
suppressWarnings(
  libraries("dplyr","Rcpp","RcppArmadillo","RcppEigen","parallel",
            "RSpectra","pryr","tidyr","data.table","Matrix")
)

# Creating weighted marginal

freqTables <- function(x){
  v <- table(x)/length(x)
  w <- data.frame(t(as.numeric(v)))
  names(w) <- names(v)
  return (w)
}

#C++

cppFunction('
            NumericMatrix IOt(NumericMatrix X){
            NumericMatrix W(X.nrow(), X.nrow());
            for(int i = 0; i < X.nrow(); i++){
              for(int j = 0; j <= i; j++){
                if(X(i,0) != X(j,0)){
                  W(i,j) = 0;
                  }
                else{
                  W(i,j) = X(i,1);
              }
                W(j,i) = W(i,j);
              }
            }
          return(W);
          }'
)

matrix_mult.cpp <- "
// [[Rcpp::depends(RcppArmadillo, RcppEigen)]]

#include <RcppArmadillo.h>
#include <RcppEigen.h>

// [[Rcpp::export]]
SEXP eigenMapMatMult(const Eigen::Map<Eigen::MatrixXd> A, 
Eigen::Map<Eigen::MatrixXd> B){
Eigen::MatrixXd C = A * B;

return Rcpp::wrap(C);
}
"

sourceCpp(code = matrix_mult.cpp)

#binary reps by class


fbyEachClass <- function(r_, f_, d_){
  
  if(is.null(dim(r_))){
    Q <- matrix(1, nrow = (ncol(d_)-1), ncol = 1)
  }
  else {
    binary_row_weights_ <- as.matrix(rbindlist(
      lapply(1:nrow(r_), function(i) freqTables(r_[i,])), fill = T))
    binary_row_weights_[is.na(binary_row_weights_)] <- 0
    binary_row_weights <- as.data.frame(binary_row_weights_)
    R <- binary_row_weights[sort(names(binary_row_weights))]
    
    binary_f_weights_ <- as.matrix(rbindlist(
      lapply(1:ncol(f_), function(i) freqTables(f_[,i])), fill = T))
    binary_f_weights_[is.na(binary_f_weights_)] <- 0
    binary_f_weights <- as.data.frame(binary_f_weights_)
    `F` <- binary_f_weights[sort(names(binary_f_weights))]
    
    Q <- eigenMapMatMult(as.matrix(`F`), as.matrix(t(R)))
  }
  return(Q/max(Q))
}  


#upper and lower 


get_upper_Fs <- function(i, d_, ccm){
  fbyEachClass(r_ = sapply(d_ %>% 
                             filter(Class == ccm$c_j[i]) %>% 
                             select(-Class), as.character), 
               f_ = sapply(d_ %>% 
                             filter(Class == ccm$c_i[i] | 
                                      Class == ccm$c_j[i]) %>% 
                             select(-Class), as.character), d_ = d_)
}

get_lower_Fs <- function(i, d_, ccm){
  fbyEachClass(r_ = sapply(d_ %>% 
                             filter(Class == ccm$c_i[i]) %>% 
                             select(-Class), as.character), 
               f_ = sapply(d_ %>% 
                             filter(Class == ccm$c_i[i] | 
                                      Class == ccm$c_j[i]) %>% 
                             select(-Class), as.character), d_ = d_)
}


#diagonal aspects

getD <- function(i, d_, dis_){
  fbyEachClass(r_ = sapply(d_[dis_[[i]],]  %>% 
                             select(-Class), as.character), 
               f_ = sapply(d_[dis_[[i]],]  %>% 
                             select(-Class), as.character),
               d_ = d_)
}

#upper and lower blocks

makeMat <- function(k_, which_diag, ccm, d_, Fs, D_){
  if(which_diag == 'upper'){
    l <- rownames(ccm[which(ccm$c_i == names(D_)[k_]),])
  }
  else{
    l <- rownames(ccm[which(ccm$c_j == names(D_)[k_]),])
  }
  
  if(length(l) == 0 & which_diag == 'upper'){
    Q_ <- D_[[n_distinct(d_$Class)]]
  }
  else if(length(l) == 0 & which_diag == 'lower'){
    Q_ <- D_[[1]]
  }
  else{
    if(which_diag == 'upper'){
      Q_ <- cbind(D_[[k_]], do.call(cbind, Fs[as.numeric(l)]))
    }
    else{
      Q_ <- cbind(do.call(cbind, Fs[as.numeric(l)]), D_[[k_]])
    }
  }
  
  if(ncol(Q_) == nrow(d_)){
    Q <- Q_
  }
  else{
    if(which_diag == 'upper'){
      Q <- cbind(matrix(0, nrow = (ncol(d_) - 1), 
                        ncol = (nrow(d_) - ncol(Q_))), Q_)
    }
    else{
      Q <- cbind(Q_, matrix(0, nrow = (ncol(d_) - 1), 
                            ncol = (nrow(d_) - ncol(Q_))))  
    }
  }
  
  return(Q)
}


cpir_gel <- function(source.data_, k = 10, class_var = NULL, 
                     learning_method = 'unsupervised'){
  
  
  if (learning_method == "supervised"){
    source.data_ <- dplyr::rename_(source.data_, Class = class_var)
    source.data_$Class <- factor(x = source.data_$Class, 
                                 names(sort(table(source.data_$Class), 
                                            decreasing = T)))
    
    source.data <- as.data.frame(apply(source.data_ %>% 
                                         arrange(Class), 2, as.character))
    W_ <- source.data
  }
  else{
    W_ <- source.data_
  }
  
  
  if (learning_method == "supervised"){
    
    class_combs <- as.data.frame.matrix(
      t(
        combn(x = unique(W_$Class), m = 2)))
    names(class_combs) <- c('c_i', 'c_j')
    
    off_diag_index_sets <- lapply(1:nrow(class_combs), function(i) 
      setdiff(rownames(W_[which(W_$Class == class_combs$c_i[i] | 
                                  W_$Class == class_combs$c_j[i]),]), 
              rownames(W_[which(W_$Class == class_combs$c_i[i]),])))
    
    
    if(n_distinct(W_$Class) == 2){
      diag_index_sets <- lapply(1:2, function(i) 
        rownames(W_[which(W_$Class == unique(W_$Class)[i]),]))
    }
    else{
      diag_index_sets <- lapply(1:n_distinct(W_$Class), function(i) 
        rownames(W_[which(W_$Class == unique(W_$Class)[i]),]))
    }
    
    upper_Fs <- lapply(1:nrow(class_combs), function(i) 
      get_upper_Fs(i, d_ = W_, ccm = class_combs))
    lower_Fs <- lapply(1:nrow(class_combs), function(i) 
      get_lower_Fs(i, d_ = W_, ccm = class_combs))
    
    D <- lapply(1:length(diag_index_sets), function(i) 
      getD(i, d_ = W_, dis_ = diag_index_sets))
    names(D) <- unique(W_$Class)
    
    upper_block <- do.call(rbind, 
                           lapply(1:n_distinct(W_$Class), 
                                  function(i) 
                                    makeMat(k_ = i, which_diag = 'upper', 
                                            ccm = class_combs, d_ = W_, 
                                            Fs = upper_Fs, D_ = D)))

    lower_block <- do.call(rbind, 
                           lapply(1:n_distinct(W_$Class), 
                                  function(i) 
                                    makeMat(k_ = i, which_diag = 'lower', 
                                            ccm = class_combs, d_ = W_, 
                                            Fs = lower_Fs, D_ = D)))
    v <- as.data.frame(
      apply(
        as.data.frame(
          source.data$Class),
        2, 
        as.character)
    )
    
    names(v) <- 'ops'
    
    ft <- freqTables(source.data$Class)
    ft <- ft[,ft!=0]
    ft_ <- data.frame(ops = names(ft), vals = as.numeric(ft))
    
    u <- v %>% 
      left_join(., ft_, by = 'ops')
    u$ops <- as.numeric(u$ops)
    
    A <- IOt(as.matrix(u))
    
    b <- bdiag(lapply(1:n_distinct(W_$Class), 
                      function(i) matrix(.5, nrow = nrow(D[[i]]), 
                                         ncol = ncol(D[[i]])))) %>% 
      as.matrix() %>% 
      apply(.,2 , function(x) ifelse(x==0, 1, x))
    
    Q_ <- (upper_block+lower_block) * b
    
    Q <- eigenMapMatMult(t(Q_), Q_)
    
    S <- eigenMapMatMult(Q/max(Q) * A, 
                         as.matrix(sapply(W_ %>% 
                                            select(-Class), 
                                          function(x) 
                                            as.numeric(as.character(x)))))
  }
  else {
    u <- fbyEachClass(r_ = sapply(W_, as.character), 
                      f_ = sapply(W_, as.character),
                      d_ = W_)
    
    Q <- eigenMapMatMult(t(u), u)
    
    S <- eigenMapMatMult(Q/max(Q),
                         as.matrix(sapply(W_ , 
                                          function(x) 
                                            as.numeric(as.character(x)))))
    
  }
  
  if(k == 'max' | k >= nrow(W_)){
    k <- nrow(W_)
  }
  else{
    k <- k
  }
  
  V <- svd(S, nv = k)$v
  
  ret <- list()
  ret$V <- V
  ret$W_ <- W_
  
  if (learning_method == "supervised"){
    ret$embed <- as.data.frame(
      as.matrix(
        sapply(
          W_ %>%
            select(-Class), function(x) as.numeric(as.character(x)))) %*% V)
    ret$embed[class_var] <- source.data$Class
  }
  else{
    ret$embed <- as.data.frame(
      as.matrix(
        sapply(
          W_, function(x) as.numeric(as.character(x)))) %*% V)
  }
  
  return(ret)}
