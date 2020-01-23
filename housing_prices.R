library(tidyverse)
library(tidymodels)
library(corrr)
library(caret)


test_dat <- read_csv("test.csv") %>% as_tibble() %>% janitor::clean_names()
test_ids <- test_dat %>% pull(id)
test_dat <- test_dat %>% select(-id)
train_dat <- read_csv("train.csv") %>% as_tibble() %>% select(-Id) %>% janitor::clean_names()

dim(train_dat)

glimpse(train_dat)

train_dat <- 
  train_dat %>% 
  select(sale_price, everything())

all_dat <- bind_rows(train_dat, test_dat)

# Visualize the outcome/target variable
all_dat %>% 
  ggplot(aes(x = sale_price)) +
  geom_histogram(bins = 100) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = scales::comma) +
  theme_bw()

top_cor <- 
  all_dat %>% 
  select_if(is.numeric) %>% 
  correlate() %>% 
  rearrange() %>%
  shave()

btm_cor <- 
  all_dat %>% 
  select_if(is.numeric) %>% 
  correlate() %>% 
  rearrange() %>%
  shave() %>% 
  slice(-1) %>% 
  select(rowname, sale_price) %>% 
  arrange(sale_price) %>% 
  slice(1:10)

all_dat %>% 
  select_if(is.numeric) %>% 
  select(sale_price, everything()) %>% 
  correlate() %>% 
  rplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot of highest correlations--categorical variables
map(
  top_cor %>% 
    slice(1, 3, 7:10) %>% 
    pull(rowname),
  ~ all_dat %>% 
    ggplot(aes(x = .data[[.x]], y = sale_price, group = .data[[.x]])) +
    geom_boxplot() +
    theme_bw()
)

# Plot of highest correlations--categorical variables
map(
  top_cor %>% 
    slice(2, 4:6) %>% 
    pull(rowname),
  ~ all_dat %>% 
    ggplot(aes(x = .data[[.x]], y = sale_price)) +
    geom_point() +
    geom_smooth(method = "lm", color="blue") +
    theme_bw()
)

# Curious about neighborhoods
all_dat %>% 
  arrange(desc(sale_price)) %>% 
  mutate(Neighborhood = forcats::as_factor(Neighborhood)) %>% 
  ggplot(aes(x = Neighborhood, y = sale_price)) +
  geom_boxplot() +
  theme_bw()

# Remove sale price prior to manipulation
all_dat <-
  all_dat %>% 
  select(-sale_price)

# Missing Values
num_vars_with_nas <- 
  map_dbl(
    all_dat %>% 
      select_if(is.numeric), 
    ~ sum(is.na(.x))
    ) %>% 
  sort(decreasing = TRUE) %>% 
  .[. > 0] %>% 
  names()

all_dat %>% 
  select(one_of(num_vars_with_nas)) %>% 
  glimpse()

map(
  all_dat %>% 
    select(one_of(num_vars_with_nas)),
  ~ unique(.x)
  )

# Predict lot frontage
all_dat %>% 
  ggplot(aes(x = lot_area, y = lot_frontage)) + 
  geom_point()

all_dat %>% 
  ggplot(aes(x = lot_shape, y = lot_frontage)) +
  geom_boxplot() +
  theme_bw()

lot_p <- lm(lot_frontage ~ lot_area + lot_shape, data = all_dat)
lot_p <- predict(lot_p, all_dat)

all_dat$lot_frontage[is.na(all_dat$lot_frontage)] <- round(lot_p[is.na(all_dat$lot_frontage)])

# Garage year built
all_dat$garage_yr_blt[is.na(all_dat$garage_yr_blt)]  <- all_dat$year_built[is.na(all_dat$garage_yr_blt)]

# For the rest of the numeric variables, impute the median
num_vars_with_nas <- num_vars_with_nas[!str_detect(num_vars_with_nas, "^lot|yr")]

all_dat <-
  all_dat %>% 
  mutate_at(
    vars(one_of(num_vars_with_nas[])),
    ~ ifelse(is.na(.), median(., na.rm = TRUE), .)
    )

# All character vars with NAs
all_dat <-
  all_dat %>% 
  mutate_if(is.character, ~ replace_na(., "None"))

# Make factors
ordinal_vars <-
  all_dat %>% 
  select_if(is.character) %>% 
  select(matches("qc|qual|cond$")) %>% 
  names()

cat_vars <-
  all_dat %>% 
  select_if(is.character) %>% 
  select(-one_of(ordinal_vars)) %>% 
  names()

all_dat <-
  all_dat %>% 
  mutate_at(vars(cat_vars), forcats::as_factor)

all_dat <-
  all_dat %>% 
  mutate_at(vars(ordinal_vars), ~ factor(., levels = c("None", "Po", "Fa", "TA", "Gd", "Ex")))

# Feature engineering ----
all_dat <-
  all_dat %>% 
  mutate(
    total_sf = total_bsmt_sf + gr_liv_area,
    outside_sf = wood_deck_sf + open_porch_sf + screen_porch + x3ssn_porch,
    age = yr_sold - year_built,
    pool = ifelse(pool_area > 0, 1, 0),
    remodeled = ifelse(year_built == year_remod_add, 0, 1), 
    total_bathrooms = full_bath + (half_bath * 0.5) + bsmt_full_bath + (bsmt_half_bath * 0.5),
    new = ifelse(yr_sold == year_built, 1, 0)
  )

# Turn numeric categorical variables into characters
all_dat <-
  all_dat %>% 
  mutate_at(
    vars(ms_sub_class, yr_sold, mo_sold, year_built, overall_qual, overall_cond, year_remod_add),
    factor
  )

# Fixed skewed variables
num_vars <- 
  all_dat %>% 
  select_if(is.numeric)

skewed <- which(map_dbl(num_vars, moments::skewness) > 0.75) %>% names()

num_vars[skewed] <-
  num_vars[skewed] %>% 
  mutate_if(~ any(. == 0), ~ log(. + 1)) %>% 
  mutate_if(~ all(. != 0), ~ log(.))

# Standardize the variables
num_vars <-
  num_vars %>% 
  preProcess(method=c("center", "scale")) %>% 
  predict(., num_vars)

# One-hot encoding of categorical variables
cat_vars <-
  all_dat %>% 
  select_if(~!is.numeric(.))

cat_vars <- as.data.frame(model.matrix(~ . -1, cat_vars))

# Putting it all back together
all_dat <- bind_cols(num_vars, cat_vars)

# Lasso
idx <- createDataPartition(1:1460, p = 0.7, list = FALSE)
train_X <- all_dat[idx, ]
test_X <- all_dat[(1:1460)[-idx], ]
train_y <- log(train_dat$sale_price[idx])
test_y <- log(train_dat$sale_price[-idx])
y <- log(train_dat$sale_price)

cntl <- trainControl(method="cv", number=5)
l_grid <- expand.grid(alpha = 1, lambda = c(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1))

set.seed(849)
lasso_caret <- 
  train(
    all_dat[1:1460, ], 
    y, 
    method = "glmnet",
    trControl = cntl,
    tuneGrid = l_grid)

lasso_caret$bestTune
min(lasso_caret$results$RMSE)

pred <- predict(lasso_caret, test_X)
sqrt(mean((pred - test_y)^2))

pred = predict(lasso_caret, newdata = all_dat[1461:2919,])
tibble("Id" = test_ids, "SalePrice" = exp(pred)) %>% 
  write_csv("submission.csv")

library(gbm)
gbm_dat <- cbind(train_X, train_y)

# Remove outlier
# gbm_dat <- gbm_dat[-907, ]

gbm_fit <-
  gbm(
    formula = train_y ~ ., 
    distribution = "gaussian", 
    data = gbm_dat, 
    n.trees = 10000,
    interaction.depth = 4, 
    shrinkage = 0.01
  )

preds <- predict(gbm_fit, newdata = test_X, n.trees = seq(100, 10000, by = 100))
preds <- as_tibble(preds)
rmses <- map_dbl(preds, ~ sqrt(mean((.x - test_y)^2)))
which(rmses == min(rmses))

preds <- predict(gbm_fit, newdata = test_X, n.trees = 4600)
sqrt(mean((preds - test_y)^2))

pred = predict(gbm_fit, newdata = all_dat[1461:2919,], n.trees = 4600)
tibble("Id" = test_ids, "SalePrice" = exp(pred)) %>% 
  write_csv("submission.csv")

# xgboost
library(xgboost)

grid_s <- 
  expand.grid(
    nrounds = 500,
    eta = c(0.01,0.005,0.001, 0.0005),
    max_depth = c(1:8),
    colsample_bytree=c(0,1,10),
    min_child_weight = 1:5,
    subsample = 1,
    gamma = 0
  )

xgb_X <- as.matrix(all_dat[1:1460, ])

xgb_c <-
  train(
    x = xgb_X,
    y = y,
    objective = "reg:linear",
    method = "xgbTree",
    trControl = trainControl(method = "cv", number = 5),
    tuneGrid = grid_s
  )

xgb_c$bestTune

xgb_params <-
  list(
    method = "xgbTree",
    objective = "reg:linear",
    eta = 0.01,
    max_depth = 7,
    subsample = 1,
    colsample_bytree = 1,
    gamma = 0,
    min_child_weight = 2
  )

xgb_train <- 
  xgb.DMatrix(
    data = as.matrix(all_dat[1:1460, ]), 
    label = as.matrix(log(train_dat$sale_price)))

xgb_cv <-
  xgb.cv(
    data = xgb_train,
    params = xgb_params,
    nrounds = 5000,
    nfold = 5,
    early_stopping_rounds = 10
  )

xgb_fit <- xgb.train(data = xgb_train, params = xgb_params, nrounds = 1207)

preds <- predict(xgb_fit, newdata = test_X %>% as.matrix())
sqrt(mean((preds - test_y)^2))


preds <- predict(xgb_fit, newdata = all_dat[1461:2919, ] %>% as.matrix())
tibble("Id" = test_ids, "SalePrice" = exp(preds)) %>% 
  write_csv("submission.csv")


# Neural networks
library(keras)

# Build model
model <-  
  keras_model_sequential() %>%
  layer_dense(units = 12, activation = "relu",
              input_shape = dim(train_X)[2]) %>%
  layer_dense(units = 8, activation = "relu") %>% 
  layer_dense(units = 1)

model %>% 
  compile(
    loss = "mse",
    optimizer = "adam",
    metrics = list("mean_squared_error")
  )

model %>% summary()

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

epochs <- 200

history <- 
  model %>%
  fit(
    train_X %>% as.matrix(),
    train_y %>% as.matrix(),
    epochs = 500,
    validation_split = 0.25,
    verbose = 0,
    callbacks = list(print_dot_callback)
  )

test_predictions <- model %>% predict(test_X %>% as.matrix())
sqrt(mean((test_predictions - test_y)^2))
