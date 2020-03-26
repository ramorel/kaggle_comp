library(tidyverse)
library(tidymodels)
library(tune)

theme_set(theme_minimal() +
            theme(axis.title.x = element_text(size = 11, hjust = 1),
                  axis.title.y = element_text(size = 11),
                  axis.text.x = element_text(size = 10),
                  axis.text.y = element_text(size = 10),
                  panel.grid.major = element_line(linetype = 2),
                  panel.grid.minor = element_line(linetype = 2),
                  plot.title = element_text(size = 14, colour = "grey25", face = "bold"), 
                  plot.subtitle = element_text(size = 12, colour = "grey44")))

train <-  read_csv("train.csv") %>% as_tibble()
train_labs <-  read_csv("train_labels.csv")
specs <-  read_csv("specs.csv")
test <-  read_csv("test.csv")

train %>% glimpse()

train_labs %>% glimpse()

specs %>% glimpse()

test %>% glimpse()

length(unique(train$event_id))

event_dat <- jsonlite::parse_json(train$event_data[1]) %>% bind_rows()

# How do kids score overall on each of the assessments?
train_labs %>% 
  group_by(title) %>% 
  summarize(mean_accuracy = mean(accuracy, na.rm = T)) %>% 
  ggplot(aes(x = title, y = mean_accuracy)) +
  geom_col(fill = "cornflowerblue", alpha = 0.7, width = 0.5)

train_labs %>% 
  group_by(title) %>% 
  summarize(median_group = median(accuracy_group, na.rm = T)) %>% 
  ggplot(aes(x = title, y = median_group)) +
  geom_col(fill = "cornflowerblue", alpha = 0.7, width = 0.5)

train_labs %>% 
  ggplot(aes(x = title, y = accuracy)) +
  geom_boxplot(aes(color = title)) +
  scale_color_brewer(palette = "Set1")

train_labs %>% 
  count(title, accuracy_group) %>% 
  group_by(title) %>% 
  mutate(prop = n/sum(n)) %>% 
  ggplot(aes(x = title, y = prop)) +
  geom_col(aes(fill = factor(accuracy_group)), width = 0.5, alpha = 0.7) +
  scale_fill_brewer(palette = "Set1")

train_labs %>% 
  ggplot(aes(x = accuracy)) +
  geom_histogram(aes(x = accuracy, stat(density)), fill = "cornflowerblue", alpha = 0.7, bins = 100) +
  geom_density()


# Focusing on assessment data
train_assess <- 
  train %>% filter(type == "Assessment") %>% 
  group_by(installation_id, game_session) %>% 
  summarize(total_time = max(game_time), total_event = max(event_count)) %>% 
  right_join(train_labs)

train_assess %>% 
  mutate(time_q = ntile(total_time, 5)) %>% 
  count(time_q, accuracy_group) %>% 
  group_by(time_q) %>% 
  mutate(prop = n/sum(n)) %>% 
  ggplot(aes(x = time_q, y = prop)) +
  geom_col(aes(fill = factor(accuracy_group)), width = 0.5) +
  scale_fill_brewer(palette = "Set2")

train_assess %>% 
  mutate(event_q = ntile(total_event, 5)) %>% 
  count(event_q, accuracy_group) %>% 
  group_by(event_q) %>% 
  mutate(prop = n/sum(n)) %>% 
  ggplot(aes(x = event_q, y = prop)) +
  geom_col(aes(fill = factor(accuracy_group)), width = 0.5) +
  scale_fill_brewer(palette = "Set2")


# How far can we get with just time, events, and game title?
train_assess <-
  train_assess %>% 
  filter(total_event < 1000, total_time < 20000000) %>% 
  mutate(attempts = num_correct + num_incorrect)

prep_rec <-
  recipe(accuracy_group ~ total_time + total_event + title + attempts, data = train_assess) %>% 
  step_mutate(accuracy_group = as.character(accuracy_group)) %>% 
  step_string2factor(title) %>% 
  step_string2factor(accuracy_group, ordered = TRUE) %>% 
  step_BoxCox(total_time, total_event, attempts) %>% 
  step_dummy(title, one_hot = TRUE) %>% 
  step_interact(~ starts_with("title"):total_time) %>% 
  step_interact(~ starts_with("title"):total_event) %>% 
  step_interact(~ total_time:total_event) %>% 
  step_normalize(total_time, total_event, attempts) %>% 
  prep(retain = TRUE)

train_dat <- juice(prep_rec)
train_split <- initial_split(train_dat, prop = 0.7)
train_cv <- vfold_cv(training(train_split), v = 5)
rf_grid <-
  grid_regular(
    trees(),
    min_n(),
    levels = 3
  )

rf_model <- 
  rand_forest(mode = "classification",
              trees = tune(),
              min_n = tune()) %>% 
  set_engine("ranger")

rf_cv <-
  tune_grid(
    formula   = accuracy_group ~ .,
    model     = rf_model,
    resamples = train_cv,
    grid      = rf_grid,
    metrics   = metric_set(kap, accuracy),
    control   = control_grid(verbose = TRUE)
  )

best_rf <-
  rf_cv %>% 
  select_best("kap")

rf_fit <- 
  rf_model %>% 
  finalize_model(parameters = best_rf) %>% 
  fit(accuracy_group ~ ., training(train_split))

rf_predictions <- 
  testing(train_split) %>%
  select(accuracy_group) %>%
  bind_cols(
    predict(rf_fit, testing(train_split))
  )

print(kap(rf_predictions, accuracy_group, .pred_class))


# Multi-nomial regression
mn_grid <-
  grid_regular(
    penalty(),
    mixture(),
    levels = 2
  )

mn_model <- 
  multinom_reg(mode = "classification",
               penalty = tune(),
               mixture = tune()) %>% 
  set_engine("keras", 
             layers = 3)

mn_cv <-
  tune_grid(
    formula   = accuracy_group ~ .,
    model     = mn_model,
    resamples = train_cv,
    grid      = mn_grid,
    metrics   = metric_set(kap, accuracy),
    control   = control_grid(verbose = TRUE)
  )

best_mn <-
  mn_cv %>% 
  select_best("kap")

mn_fit <- 
  mn_model %>% 
  finalize_model(parameters = best_mn) %>% 
  fit(accuracy_group ~ ., training(train_split))

mn_predictions <- 
  testing(train_split) %>%
  select(accuracy_group) %>%
  bind_cols(
    predict(mn_fit, testing(train_split))
  )

print(kap(mn_predictions, accuracy_group, .pred_class))


# SVM
svm_grid <-
  grid_regular(
    cost(),
    rbf_sigma(),
    levels = 3
  )

svm_model <- 
  svm_rbf(mode = "classification",
          cost = tune(), 
          rbf_sigma = tune()
          ) %>% 
  set_engine("kernlab")

svm_cv <-
  tune_grid(
    formula   = accuracy_group ~ .,
    model     = svm_model,
    resamples = train_cv,
    grid      = svm_grid,
    metrics   = metric_set(kap, accuracy),
    control   = control_grid(verbose = TRUE)
  )

best_svm <-
  svm_cv %>% 
  select_best("kap")

svm_fit <- 
  svm_model %>% 
  finalize_model(parameters = best_svm) %>% 
  fit(accuracy_group ~ ., training(train_split))

svm_predictions <- 
  testing(train_split) %>%
  select(accuracy_group) %>%
  bind_cols(
    predict(svm_fit, testing(train_split))
  )

print(kap(svm_predictions, accuracy_group, .pred_class))


# Boosting
xgb_grid <-
  grid_regular(
    trees(),
    min_n(),
    tree_depth(),
    learn_rate(),
    levels = 2
  )

xgb_model <- 
  boost_tree(mode = "classification",
             trees = tune(), 
             min_n = tune(),
             tree_depth = tune(),
             learn_rate = tune()
  ) %>% 
  set_engine("xgboost")

xgb_cv <-
  tune_grid(
    formula   = accuracy_group ~ .,
    model     = xgb_model,
    resamples = train_cv,
    grid      = xgb_grid,
    metrics   = metric_set(kap, accuracy),
    control   = control_grid(verbose = TRUE)
  )

best_xgb <-
  xgb_cv %>% 
  select_best("kap")

xgb_fit <- 
  xgb_model %>% 
  finalize_model(parameters = best_xgb) %>% 
  fit(accuracy_group ~ ., training(train_split))

xgb_predictions <- 
  testing(train_split) %>%
  select(accuracy_group) %>%
  bind_cols(
    predict(xgb_fit, testing(train_split))
  )

print(kap(xgb_predictions, accuracy_group, .pred_class))




rf_model %>% fit(accuracy_group ~ total_time + total_event + title, data = train_cv)  


tmp <- train[which(train$installation_id %in% unique(train$installation_id)[1:50]), ]
world_vals <- 
  tibble(
    world = unique(tmp$world) %>% sort(),
    world_val = c("weight", "capacity/displacement", "none", "legnth/height")
  )

tmp <- tmp %>% left_join(world_vals)

tmp <-
  tmp %>% 
  mutate(date = lubridate::as_date(timestamp))

tmp_per_day <-
  tmp %>% 
  count(installation_id, date, name = "n_per_day")

tmp_n_activity <-
  tmp %>% 
  count(installation_id, type, name = "n_per_type")

tmp_n_world <-
  tmp %>% 
  count(installation_id, world, name = "n_per_world")

tmpA <- tmp %>% filter(event_code %in% c(4100, 4110))
tmpB <- tmpA$event_data %>% map(~ .x %>% jsonlite::fromJSON() %>% keep(~length(.x)==1) %>% bind_rows()) %>% bind_rows()
tmpA <- bind_cols(tmpA, tmpB)

tmpA %>% 
  mutate(incorrect = !correct) %>% 
  group_by(game_session, installation_id, title) %>% 
  summarize(num_correct = sum(correct), 
            num_incorrect = sum(incorrect))

tmp3 <- specs[specs$event_id %in% tmp$event_id, ]

tmp4 <- left_join(tmp, tmp2)