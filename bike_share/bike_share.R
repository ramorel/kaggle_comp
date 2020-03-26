library(tidyverse)
library(corrr)
library(tidymodels)

theme_set(
  theme_minimal() +
    theme(
      axis.title.x = element_text(size = 12, hjust = 1),
      axis.title.y = element_text(size = 12),
      axis.text.x = element_text(size = 10),
      axis.text.y = element_text(size = 10),
      plot.title = element_text(size = 12, colour = "grey25", face = "bold"),
      plot.subtitle = element_text(size = 11, colour = "grey45"))
)

train <- read_csv("train.csv")
test <- read_csv("test.csv")

train %>% glimpse()

# Examining the outcome variable
summary(train$count)

est_dist <- fitdistrplus::fitdist(train$count, "nbinom")

theo_dist <- rnbinom(train$count, size = est_dist$estimate["size"], mu = est_dist$estimate["mu"])

train %>% 
  select(count) %>% 
  mutate(est = theo_dist) %>% 
  pivot_longer(cols = everything()) %>% 
  ggplot(aes(x = value, group = name)) +
  geom_histogram(aes(fill = name), color = "midnightblue", alpha = 0.5, bin = 50)

train %>% 
  select(count) %>% 
  mutate(est = theo_dist) %>% 
  pivot_longer(cols = everything()) %>% 
  ggplot(aes(x = value, group = name)) + 
  geom_step(aes(color = name), 
            stat = "ecdf", 
            pad = FALSE) +
  scale_color_manual(name = "",
                     values = c("cornflowerblue", "magenta")) +
  labs(x = "", y = "", 
       title = "Empirical & Theoretical CDFs - Negative Binomial") +
  xlim(0, 1000)

train %>% 
  ggplot(aes(x = count)) +
  geom_histogram(aes(fill = factor(season)),
                 color = "midnightblue", 
                 bins = 50,
                 alpha = 0.5) +
  facet_wrap(~ season)

train %>% 
  ggplot(aes(x = count)) +
  geom_histogram(aes(fill = factor(holiday)),
                 color = "midnightblue", 
                 bins = 50,
                 alpha = 0.5) +
  facet_wrap(~ holiday)

train %>% 
  ggplot(aes(x = count)) +
  geom_histogram(aes(fill = factor(workingday)),
                 color = "midnightblue", 
                 bins = 50,
                 alpha = 0.5) +
  facet_wrap(~ workingday)

train %>% 
  ggplot(aes(x = count)) +
  geom_histogram(aes(fill = factor(weather)),
                 color = "midnightblue", 
                 bins = 50,
                 alpha = 0.5) +
  facet_wrap(~ weather)

train <- train %>% 
  mutate(time_of_day = case_when(
    between(lubridate::hour(datetime), 0, 4) ~ "late night",
    between(lubridate::hour(datetime), 5, 10) ~ "early morning",
    between(lubridate::hour(datetime), 11, 16) ~ "midday",
    between(lubridate::hour(datetime), 17, 20) ~ "evening",
    between(lubridate::hour(datetime), 21, 23) ~ "night"
  )) 

train %>% 
  ggplot(aes(x = count)) +
  geom_histogram(aes(fill = factor(time_of_day)),
                 color = "midnightblue", 
                 bins = 50,
                 alpha = 0.5) +
  facet_wrap(~ time_of_day)


train %>% 
  select(season, casual, registered) %>% 
  pivot_longer(cols = -season) %>% 
  ggplot(aes(x = factor(season), y = value, group = name)) + 
  geom_col(aes(fill = name), position = "dodge") + 
  scale_fill_manual(values = c("cornflowerblue", "magenta"))

train %>% 
  select(workingday, casual, registered) %>% 
  pivot_longer(cols = -workingday) %>% 
  ggplot(aes(x = factor(workingday), y = value, group = name)) + 
  geom_col(aes(fill = name), position = "dodge") + 
  scale_fill_manual(values = c("cornflowerblue", "magenta"))

train %>% 
  select(weather, casual, registered) %>% 
  pivot_longer(cols = -weather) %>% 
  ggplot(aes(x = factor(weather), y = value, group = name)) + 
  geom_col(aes(fill = name), position = "dodge") + 
  scale_fill_manual(values = c("cornflowerblue", "magenta"))

train %>% 
  select(time_of_day, casual, registered) %>% 
  pivot_longer(cols = -time_of_day) %>% 
  ggplot(aes(x = factor(time_of_day), y = value, group = name)) + 
  geom_col(aes(fill = name), position = "dodge") + 
  scale_fill_manual(values = c("cornflowerblue", "magenta"))

# Other vars
train %>% 
  select(-datetime, -season, -holiday, -workingday, -weather, -time_of_day) %>% 
  pivot_longer(cols = everything()) %>% 
  ggplot(aes(x = value, group = name)) +
  geom_histogram(alpha = 0.5, 
                 bins = 50,
                 fill = "cornflowerblue", 
                 color = "midnightblue") +
  facet_wrap(~ name, scales = "free")

# Correlations
train %>% 
  select(-datetime, -season, -holiday, -workingday, -weather) %>% 
  correlate() %>% 
  shave() %>% 
  rplot()

train %>% 
  ggplot(aes(x = temp, y = count)) +
  geom_point(color = "cornflowerblue", alpha = 0.7) +
  facet_wrap(~ time_of_day)

train %>% 
  ggplot() +
  geom_point(aes(x = atemp, y = registered), color = "magenta", alpha = 0.7) +
  geom_point(aes(x = atemp, y = casual), color = "cornflowerblue", alpha = 0.7) +
  facet_wrap(~ time_of_day)

atrain %>% 
  ggplot() +
  geom_point(aes(x = atemp, y = registered), color = "magenta", alpha = 0.7) +
  geom_point(aes(x = atemp, y = casual), color = "cornflowerblue", alpha = 0.7) +
  facet_wrap(~ weather)

train %>% 
  ggplot() +
  geom_point(aes(x = atemp, y = registered, color = factor(weather)), alpha = 0.7) +
  facet_wrap(~ time_of_day)

train %>% 
  ggplot() +
  geom_point(aes(x = atemp, y = registered, color = factor(workingday)), alpha = 0.7) +
  facet_wrap(~ time_of_day)

train %>% 
  ggplot() +
  geom_point(aes(x = atemp, y = registered, color = time_of_day), alpha = 0.7) +
  facet_wrap(~ workingday)

train %>% 
  group_by(season) %>% 
  summarize(mean = mean(count, na.rm = TRUE),
            sd = sd(count, na.rm = TRUE),
            sum = sum(count, na.rm = TRUE)) %>% 
  mutate(prop = sum/sum(sum))

train %>% 
  group_by(holiday) %>% 
  summarize(mean = mean(count, na.rm = TRUE),
            sd = sd(count, na.rm = TRUE))

train %>% 
  group_by(workingday) %>% 
  summarize(mean = mean(count, na.rm = TRUE),
            sd = sd(count, na.rm = TRUE))

train %>% 
  select(weather, registered, count) %>% 
  group_by(weather) %>% 
  summarize(mean = mean(count, na.rm = TRUE),
            sd = sd(count, na.rm = TRUE),
            sum = sum(count, na.rm = TRUE)) %>% 
  mutate(prop = sum/sum(sum))

# Feature engineering
all_dat <- bind_rows(train, test)

all_dat <-
  all_dat %>% 
  mutate(time_of_day = case_when(
    between(lubridate::hour(datetime), 0, 4) ~ "late night",
    between(lubridate::hour(datetime), 5, 10) ~ "early morning",
    between(lubridate::hour(datetime), 11, 16) ~ "midday",
    between(lubridate::hour(datetime), 17, 20) ~ "evening",
    between(lubridate::hour(datetime), 21, 23) ~ "night"
  ))

# Model 1: Specified negative binomial -----

rmsle <- function(df) {
  actual <- df %>% pull(actual)
  pred <- df %>% pull(pred)
  len <- nrow(df)
  ms <- sum((log(pred + 1) - log(actual + 1))^2)
  mse <- ms/len
  rmse <- sqrt(mse)
  return(rmse)
}

reg_rec <- recipe(registered ~ ., data = train) %>% 
  step_rm(count) %>% 
  step_num2factor(weather, ordered = TRUE, levels = c("fair", "mist", "precipitation", "heavy precipitation")) %>% 
  step_num2factor(season, ordered = TRUE, levels = c("spring", "summer", "fall", "winter")) %>% 
  step_other(weather, threshold = 0.1) %>% 
  step_mutate(hour = lubridate::hour(datetime)) %>% 
  step_date(datetime) %>% 
  step_log(casual, all_outcomes(), offset = 1) %>% 
  prep(retain = TRUE)

cas_rec <- recipe(casual ~ ., data = train) %>% 
  step_rm(count) %>% 
  step_num2factor(weather, ordered = TRUE, levels = c("fair", "mist", "precipitation", "heavy precipitation")) %>% 
  step_num2factor(season, ordered = TRUE, levels = c("spring", "summer", "fall", "winter")) %>% 
  step_other(weather, threshold = 0.1) %>% 
  step_mutate(hour = lubridate::hour(datetime)) %>% 
  step_date(datetime) %>% 
  step_log(registered, all_outcomes(), offset = 1) %>% 
  prep(retain = TRUE)
  
reg_train <- juice(reg_rec)
cas_train <- juice(cas_rec)

reg_split <- initial_split(reg_train, 0.25, strata = registered)
cas_split <- initial_split(cas_train, 0.25, strata = casual)

lm_reg_fit <- lm(registered ~
                   weather +
                   season + 
                   holiday +
                   factor(workingday)*weather + 
                   factor(workingday)*time_of_day + 
                   weather*time_of_day +
                   factor(datetime_year) +
                   temp + 
                   I(temp^2) +
                   windspeed, 
                 data = training(reg_split))

lm_reg_preds <- predict(lm_reg_fit, testing(reg_split))

reg_pred <- tibble(actual = testing(reg_split) %>% pull(registered) %>% exp(),
                   pred = lm_reg_preds %>% exp())

rmsle(reg_pred)

lm_cas_fit <- lm(casual ~ 
                   season + 
                   weather +
                   factor(workingday)*weather + 
                   holiday + 
                   factor(workingday)*time_of_day + 
                   weather*time_of_day +
                   factor(datetime_year) +
                   atemp + 
                   I(atemp^2) +
                   temp +
                   windspeed, 
                 data = training(cas_split))

lm_cas_preds <- predict(lm_cas_fit, testing(cas_split))

cas_pred <- tibble(actual = testing(cas_split) %>% pull(casual) %>% exp(),
                   pred = lm_cas_preds %>% exp())

rmsle(cas_pred)

# Recipe
reg_rec <- recipe(registered ~ ., data = train) %>% 
  step_rm(count, casual) %>% 
  step_mutate(workingday = factor(workingday, levels = c(0, 1), labels = c("weekend", "workingday"))) %>% 
  step_mutate(holiday = factor(holiday, levels = c(0, 1), labels = c("no", "yes"))) %>% 
  step_num2factor(weather, ordered = TRUE, levels = c("fair", "mist", "precipitation", "heavy precipitation")) %>% 
  step_num2factor(season, ordered = TRUE, levels = c("spring", "summer", "fall", "winter")) %>% 
  step_other(weather, threshold = 0.1) %>% 
  step_mutate(hour = lubridate::hour(datetime)) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_date(datetime) %>% 
  step_rm(datetime) %>% 
  step_interact(terms = ~ contains("season"):contains("weather")) %>%
  step_interact(terms = ~ contains("workingday"):matches("^weather_[a-z]+$")) %>%
  step_interact(terms = ~ matches("^workingday_[a-z]+$"):contains("time_of_day")) %>%
  step_log(all_outcomes(), offset = 1) %>% 
  step_normalize(all_numeric(), -registered) %>% 
  step_nzv(all_predictors()) %>%
  prep(retain = TRUE)

cas_rec <- recipe(casual ~ ., data = train) %>% 
  step_rm(count, registered, -time_of_day) %>% 
  step_mutate(workingday = factor(workingday, levels = c(0, 1), labels = c("weekend", "workingday"))) %>% 
  step_mutate(holiday = factor(holiday, levels = c(0, 1), labels = c("no", "yes"))) %>% 
  step_num2factor(weather, ordered = FALSE, levels = c("fair", "mist", "precipitation", "heavy precipitation")) %>% 
  step_num2factor(season, ordered = FALSE, levels = c("spring", "summer", "fall", "winter")) %>% 
  step_other(weather, threshold = 0.1) %>% 
  step_mutate(hour = lubridate::hour(datetime)) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_date(datetime) %>% 
  step_rm(datetime) %>% 
  #step_interact(terms = ~ contains("season"):contains("weather")) %>%
  #step_interact(terms = ~ contains("workingday"):matches("^weather_[a-z]+$")) %>%
  #step_interact(terms = ~ matches("^workingday_[a-z]+$"):contains("time_of_day")) %>%
  step_log(all_outcomes(), offset = 1) %>% 
  step_normalize(all_numeric(), -casual) %>% 
  step_nzv(all_predictors()) %>%
  prep(retain = TRUE)

  
reg_train <- juice(reg_rec)
cas_train <- juice(cas_rec)

set.seed(1491)
reg_split <- initial_split(reg_train, 0.25)
cas_split <- initial_split(cas_train, 0.25)
reg_cv <- vfold_cv(training(reg_split), v = 5)
cas_cv <- vfold_cv(training(cas_split), v = 5)

# Regularized models
reg_model <-
  linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet")

reg_grid <- grid_regular(penalty(), mixture(), levels = 5)

tuned <- tune_grid(formula = registered ~ ., 
                   model = reg_model, 
                   resamples = reg_cv, 
                   metrics = metric_set(rmse))

best_tune <- select_best(tuned, metric = "rmse", maximize = FALSE)

reg_fit <- finalize_model(reg_model, best_tune) %>% 
  fit(registered ~ ., training(reg_split))

reg_pred <- tibble(actual = testing(reg_split) %>% pull(registered) %>% exp(),
       pred = predict(reg_fit, testing(reg_split))[[".pred"]] %>% exp())

rmsle(reg_pred)

tuned <- tune_grid(formula = casual ~ ., 
                   model = reg_model, 
                   resamples = cas_cv, 
                   metrics = metric_set(rmse))

best_tune <- select_best(tuned, metric = "rmse", maximize = FALSE)

cas_fit <- finalize_model(reg_model, best_tune) %>% 
  fit(casual ~ ., training(cas_split))

cas_pred <- tibble(actual = testing(cas_split) %>% pull(casual),
              pred = predict(cas_fit, testing(cas_split))[[".pred"]])

rmsle(cas_pred)

overall_pred <- tibble(actual = testing(reg_split) %>% pull(registered) %>% exp() + 
                         testing(cas_split) %>% pull(casual) %>% exp(),
                       pred = predict(cas_fit, testing(cas_split))[[".pred"]] %>% exp() +
                         predict(reg_fit, testing(reg_split))[[".pred"]] %>% exp())
rmsle(overall_pred)

# Random forest
rf_model <-
  rand_forest(mode = "regression", trees = tune(), min_n = tune()) %>% 
  set_engine("ranger")

reg_grid <- grid_regular(trees(), min_n(), levels = 5)

tuned <- tune_grid(formula = registered ~ ., model = rf_model, resamples = reg_cv, metrics = metric_set(rmse))

best_tune <- select_best(tuned, metric = "rmse", maximize = FALSE)

rf_reg_fit <- finalize_model(rf_model, best_tune) %>% 
  fit(registered ~ ., training(reg_split))

rf_reg_preds <- predict(rf_reg_fit, testing(reg_split))[[".pred"]]
reg_pred <- tibble(actual = testing(reg_split) %>% pull(registered) %>% exp(),
              pred = rf_reg_preds %>% exp())

rmsle(reg_pred)

tuned <- tune_grid(formula = casual ~ ., model = rf_model, resamples = cas_cv, metrics = metric_set(rmse))

best_tune <- select_best(tuned, metric = "rmse", maximize = FALSE)

rf_cas_fit <- finalize_model(rf_model, best_tune) %>% 
  fit(casual ~ ., training(cas_split))

rf_cas_preds <- predict(rf_cas_fit, testing(cas_split))[[".pred"]]
cas_pred <- tibble(actual = testing(cas_split) %>% pull(casual) %>% exp(),
                   pred = rf_cas_preds %>% exp())

rmsle(cas_pred)

# Boosted
xgb_model <-
  boost_tree(mode = "regression", 
             trees = tune(), 
             tree_depth = tune(), 
             learn_rate = tune()) %>% 
  set_engine("xgboost")

reg_grid <- grid_regular(trees(), 
                         tree_depth(), 
                         learn_rate(), 
                         levels = 5)

tuned <- tune_grid(formula = registered ~ ., 
                   model = xgb_model, 
                   resamples = reg_cv, 
                   metrics = metric_set(rmse),
                   control = control_grid(verbose = FALSE)
                   )

best_tune <- select_best(tuned, metric = "rmse", maximize = FALSE)

xgb_reg_fit <- finalize_model(xgb_model, best_tune) %>% 
  fit(registered ~ ., training(reg_split))

xgb_reg_preds <- predict(xgb_reg_fit, testing(reg_split))[[".pred"]]
reg_pred <- tibble(actual = testing(reg_split) %>% pull(registered) %>% exp(),
              pred =  xgb_reg_preds %>% exp())

rmsle(reg_pred)
#0.3059208

tuned <- tune_grid(formula = casual ~ ., 
                   model = xgb_model, 
                   resamples = cas_cv, 
                   metrics = metric_set(rmse),
                   control = control_grid(verbose = FALSE)
)

best_tune <- select_best(tuned, metric = "rmse", maximize = FALSE)

xgb_cas_fit <- finalize_model(xgb_model, best_tune) %>% 
  fit(casual ~ ., training(cas_split))

xgb_cas_preds <- predict(xgb_cas_fit, testing(cas_split))[[".pred"]]

cas_pred <- tibble(actual = testing(cas_split) %>% pull(casual) %>% exp(),
                   pred = xgb_cas_preds %>% exp())

rmsle(cas_pred)
#0.4220675

# Together
overall_pred <- tibble(actual = testing(reg_split) %>% pull(registered) %>% exp() + 
                     testing(cas_split) %>% pull(casual) %>% exp(),
                   pred = predict(cas_fit, testing(cas_split))[[".pred"]] %>% exp() +
                     predict(reg_fit, testing(reg_split))[[".pred"]] %>% exp())
rmsle(overall_pred)
#0.2639761


# Averaging prediction ----

reg_preds <- (lm_reg_preds*0.1 + rf_reg_preds*0.1 + xgb_reg_preds*0.8) %>% exp()
cas_preds <- (lm_cas_preds*0.1 + rf_cas_preds*0.1 + xgb_cas_pred*0.8)

reg_pred <- tibble(actual = testing(reg_split) %>% pull(registered) %>% exp(),
                   pred = reg_preds)
cas_pred <- tibble(actual = testing(cas_split) %>% pull(casual) %>% exp(),
                   pred = cas_preds %>% exp())
rmsle(reg_pred)
rmsle(cas_pred)


overall_pred <- tibble(actual = testing(reg_split) %>% pull(registered) %>% exp() + 
                         testing(cas_split) %>% pull(casual) %>% exp(),
                       pred = (lm_reg_preds*0.1 + rf_reg_preds*0.1 + xgb_reg_preds*0.8) %>% exp() +
                         (lm_cas_preds*0.1 + rf_cas_preds*0.1 + xgb_cas_pred*0.8))
rmsle(overall_pred)

# Neural network
library(keras)

reg_rec <- recipe(registered ~ ., data = train) %>% 
  step_rm(count, casual) %>% 
  step_num2factor(weather, ordered = TRUE, levels = c("fair", "mist", "precipitation", "heavy precipitation")) %>% 
  step_num2factor(season, ordered = TRUE, levels = c("spring", "summer", "fall", "winter")) %>% 
  step_mutate(hour = lubridate::hour(datetime)) %>% 
  step_date(datetime) %>% 
  step_rm(datetime) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_log(all_outcomes(), offset = 1) %>% 
  step_normalize(all_predictors()) %>% 
  prep(retain = TRUE)

cas_rec <- recipe(casual ~ ., data = train) %>% 
  step_rm(count, registered) %>% 
  step_num2factor(weather, ordered = TRUE, levels = c("fair", "mist", "precipitation", "heavy precipitation")) %>% 
  step_num2factor(season, ordered = TRUE, levels = c("spring", "summer", "fall", "winter")) %>% 
  step_mutate(hour = lubridate::hour(datetime)) %>% 
  step_date(datetime) %>% 
  step_rm(datetime) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_log(all_outcomes(), offset = 1) %>% 
  step_normalize(all_predictors()) %>% 
  prep(retain = TRUE)

reg_train <- juice(reg_rec)
cas_train <- juice(cas_rec)

reg_split <- initial_split(reg_train, 0.75, strata = registered)
cas_split <- initial_split(cas_train, 0.75, strata = casual)

y_reg_train <- training(reg_split) %>% pull(registered)
X_reg_train <- training(reg_split) %>% select(-registered)
y_reg_test <- testing(reg_split) %>% pull(registered)
X_reg_test <- testing(reg_split) %>% select(-registered)

y_cas_train <- training(cas_split) %>% pull(casual)
X_cas_train <- training(cas_split) %>% select(-casual)
y_cas_test <-   testing(cas_split) %>% pull(casual)
X_cas_test <-   testing(cas_split) %>% select(-casual)


model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu",
              input_shape = dim(X_reg_train)[2]) %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.2) %>% 
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mean_squared_error", "mean_squared_logarithmic_error")
)

model %>% summary()


history <- model %>% fit(
  X_reg_train %>% as.matrix(),
  y_reg_train,
  epochs = 50,
  validation_split = 0.25,
  verbose = 1
)

pred <- predict(model, X_reg_test %>% as.matrix())

reg_pred <- tibble(actual = y_reg_test %>% exp(),
                   pred = pred[,1] %>% exp())

rmsle(reg_pred)
