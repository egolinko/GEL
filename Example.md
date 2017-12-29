Minimal example to use as supervised and unsupervised.

```{r}
car <- read.csv("https://s3-us-west-2.amazonaws.com/researchs/learn_w_cat_data/car.csv")
require(dplyr)
```

# for supervised

```{r}
emb_sup <- cpir_gel(source.data_ = make_bin_data(data_ = car, class_var = "Class"), 
                    k = 10, 
                    class_var = "Class", 
                    learning_method = "supervised"
)
```

# for supervised
```{r}
emb_unsunup <- cpir_gel(source.data_ = make_bin_data(
      data_ = car %>% 
                select(-Class)),
                k = 10
)
```
