require(dplyr)

make_bin_data <- function(data_, class_var = NULL, comp = NULL){
  
  if (is.null(comp)){
    n5 <- function(x) {
      if (n_distinct(x) > 5) 
        x <- ntile(x = x, n = 5); 
      return(x)}
  }
  else{
    n5 <- function(x) {x}
  }
  
  v_ <- data_
  
  if (!is.null(class_var)) {
    v <- v_ %>% 
      select(setdiff(names(data_), class_var)) %>% 
      mutate_all(n5)
  }
  else{
    v <- data_ %>% mutate_all(n5)
  }
  
  X <- as.data.frame(
    sapply(v, as.factor))
  
  mm <- function(x) { 
    if(n_distinct(x) > 1){
      as.data.frame(model.matrix(~x)[, -1]) 
    }
    else{
      as.data.frame(0)
    }
  }
  
  make_bins <- function(df = df){
    Y <- list()
    for (i in 1:ncol(df)){  
      y <- mm(df[,i])
      names(y) <- paste(names(df)[i], names(y), sep = "_")
      Y[[i]] <- y
    }
    return(as.data.frame(do.call(cbind,Y)))
  }
  
  source_data <- make_bins(df = X)
  if (!is.null(class_var)){
    source_data[class_var] <- v_[class_var][,1]
    source_data[class_var] <- source_data[class_var] %>% unlist()
  }
  else{
    source_data <- source_data
  }
  
  return(source_data)}
