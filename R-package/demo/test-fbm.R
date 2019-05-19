
library(lightgbm)
library(bigstatsr)

data(agaricus.train, package = "lightgbm")

params <- list(objective="regression", metric="l2")
train <- agaricus.train

dat_mat <- lgb.Dataset(train$data, label = train$label)
set.seed(11)
model_mat <- lgb.cv(params, dat_mat, 10, nfold = 5, min_data = 1,
                learning_rate = 1, early_stopping_rounds = 10)


dat_fbm <- as_FBM(as.matrix(agaricus.train$data))
dat_fbm <- lgb.Dataset(dat_fbm, label = train$label)
set.seed(11)
model_fbm <- lgb.cv(params, dat_fbm, 10, nfold = 5, min_data = 1,
                    learning_rate = 1, early_stopping_rounds = 10)
