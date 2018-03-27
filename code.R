library(keras)
use_session_with_seed(30)

# generowanie danych ---------------------------------

generate_data <- function(x = 0.5, ncol = 12) 
  list(features_train = rbind(matrix(runif(500*ncol), ncol = ncol),
                              matrix(runif(500*ncol), ncol = ncol) + x),
       target_train = matrix(c(rep(0, 500), rep(1, 500)), ncol = 1),
       features_eval = rbind(matrix(runif(50*ncol), ncol = ncol),
                             matrix(runif(50*ncol), ncol = ncol) + x),
       target_eval = matrix(c(rep(0, 50), rep(1, 50)), ncol = 1))


dat <- generate_data()

library(ggplot2)
library(dplyr)

rbind(data.frame(dat[["features_train"]], target = dat[["target_train"]], type = "train"),
      data.frame(dat[["features_eval"]], target = dat[["target_eval"]], type = "evaluation")) %>% 
  ggplot(aes(x = X1, y = X2, color = factor(target))) +
  geom_point() +
  facet_wrap(~ type) +
  theme_bw()


# tworzenie modelu  ---------------------------------
model <- keras_model_sequential()

model %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(12)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    loss = "mean_squared_error",
    optimizer = "sgd",
    metrics = c("accuracy")
  )

# historia uczenia  ---------------------------------
history <- model %>% 
  fit(dat[["features_train"]], dat[["target_train"]], 
      epochs = 20, batch_size = 128, 
      validation_split = 0.1
  )

# ocena modelu
score <- model %>% evaluate(dat[["features_eval"]], dat[["target_eval"]], batch_size = 128)

table(pred = predict_classes(model, dat[["features_eval"]]),
      target = dat[["target_eval"]])

# GitHub: https://tinyurl.com/stwur7
# architektura sieci ----------------------------------------

different_architectures <- lapply(c(4, 8, 32, 64), function(number_of_units) 
  lapply(c("binary_crossentropy", "mean_squared_error"), function(loss_function) 
    lapply(1L:10, function(replicate) {
      model <- keras_model_sequential()
      
      model %>% 
        layer_dense(units = number_of_units, activation = "relu", input_shape = c(12)) %>%
        layer_dropout(rate = 0.5) %>%
        layer_dense(units = 1, activation = "sigmoid") %>% 
        compile(
          loss = loss_function,
          optimizer = "sgd",
          metrics = c("accuracy")
        )
      
      model %>% 
        fit(dat[["features_train"]], dat[["target_train"]], 
            epochs = 20, batch_size = 128, 
            validation_split = 0.1
        )
      
      score <- model %>% evaluate(dat[["features_eval"]], dat[["target_eval"]], batch_size = 128) 
      
      data.frame(number_of_units = number_of_units,
                 loss_function = loss_function,
                 replicate = replicate,
                 acc = score[["acc"]])
    }) %>% do.call(rbind, .)
  ) %>% do.call(rbind, .)
) %>% do.call(rbind, .)

#save(different_architectures, file = "different_architectures.RData")
load("different_architectures.RData")
ggplot(different_architectures, 
       aes(x = factor(number_of_units), color = loss_function, y = acc)) +
  geom_boxplot()

# architektura sieci - 2 layers

different_architectures_2l <- lapply(c(4, 8, 32, 64), function(number_of_units) 
  lapply(c(4, 8, 32, 64), function(number_of_units2l) 
    lapply(c("binary_crossentropy", "mean_squared_error"), function(loss_function) 
      lapply(1L:3, function(replicate) {
        model <- keras_model_sequential()
        
        model %>% 
          layer_dense(units = number_of_units, activation = "relu", input_shape = c(12)) %>%
          layer_dropout(rate = 0.5) %>%
          layer_dense(units = number_of_units2l, activation = "relu") %>% 
          layer_dropout(rate = 0.5) %>%
          layer_dense(units = 1, activation = "sigmoid") %>% 
          compile(
            loss = loss_function,
            optimizer = "sgd",
            metrics = c("accuracy")
          )
        
        model %>% 
          fit(dat[["features_train"]], dat[["target_train"]], 
              epochs = 20, batch_size = 128, 
              validation_split = 0.1
          )
        
        score <- model %>% evaluate(dat[["features_eval"]], dat[["target_eval"]], batch_size = 128) 
        
        data.frame(number_of_units = number_of_units,
                   number_of_units2l = number_of_units2l,
                   loss_function = loss_function,
                   replicate = replicate,
                   acc = score[["acc"]])
      }) %>% do.call(rbind, .)
    ) %>% do.call(rbind, .)
  ) %>% do.call(rbind, .)
) %>% do.call(rbind, .)

