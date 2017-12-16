# GEL
Supports GEL algorithm

This is the code that is supporting the ongoing work of Generalized embedding for supervised, unsupervised, and online learning. Please send any comments to egolinko@fau.edu. Thank you!


car <- read.csv("https://s3-us-west-2.amazonaws.com/researchs/learn_w_cat_data/car.csv")

require(dplyr)

# for supervised
emb_sup <- cpir_gel(source.data_ = make_bin_data(data_ = car, class_var = "Class"), k = 10, class_var = "Class", learning_method = "supervised")


# for supervised
emb_unsunup <- cpir_gel(source.data_ = make_bin_data(data_ = car %>% select(-Class)), k = 10)
