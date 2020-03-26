library(tidyverse)
library(tidymodels)
library(tune)
library(janitor)
library(patchwork)

# Define theme for ggplot
theme_set(
  theme_minimal() +
    theme(
      axis.title.x = element_text(size = 12, hjust = 1),
      axis.title.y = element_text(size = 12),
      axis.text.x = element_text(size = 10, angle = 45, hjust = 1),
      axis.text.y = element_text(size = 10),
      plot.title = element_text(size = 12, colour = "grey25", face = "bold"),
      plot.subtitle = element_text(size = 11, colour = "grey45"))
)

train <- read_csv("../input/train.csv") %>% clean_names() %>% select(-id)
test <- read_csv("../input/test.csv") %>% clean_names()

# Keep IDs for submission
ids <- test %>% pull(id)
test <- test %>% select(-id)

# Keep outcome variable
y <- train$sale_price

# Missing values ----
train <-
  train %>%
  group_by(neighborhood) %>%
  mutate(median_sp = median(sale_price)) %>%
  ungroup() %>%
  mutate(wealth =
           case_when(
             median_sp > quantile(median_sp, 0.8) ~ 2,
             between(median_sp, quantile(median_sp, 0.2),  quantile(median_sp, 0.8)) ~ 1,
             median_sp < quantile(median_sp, 0.2) ~ 0)) %>%
  select(-median_sp )

neighborhoods <- train %>% select(neighborhood, wealth) %>% distinct()

test <-
  test %>%
  left_join(neighborhoods)

none_vars <-
  c(
    "bsmt_cond",
    "bsmt_qual",
    "bsmt_exposure",
    "bsmt_fin_type1",
    "bsmt_fin_type2",
    "mas_vnr_type",
    "garage_type",
    "garage_finish",
    "garage_qual",
    "garage_cond",
    "alley",
    "fence",
    "fireplace_qu",
    "misc_feature",
    "pool_qc"
  )

zero_vars <-
  c(
    "mas_vnr_area",
    "bsmt_fin_sf1",
    "bsmt_fin_sf2",
    "bsmt_full_bath",
    "bsmt_half_bath",
    "bsmt_unf_sf",
    "total_bsmt_sf",
    "garage_area",
    "garage_cars",
    "garage_yr_blt"
  )

ordinal_vars <-
  train %>%
  select_if(is.character) %>%
  select(matches("qc|qual|cond$")) %>%
  names

skewed_vars <-
  map_dbl(
    bind_rows(train, test) %>% select_if(is.numeric),
    moments::skewness, na.rm = TRUE) %>%
  .[. > 0.75] %>%
  names()

skewed_vars <- skewed_vars[-1]


# Recipe ----
prep_rec <-
  
  # Specify the model and the training data
  recipe(sale_price ~ ., data = train) %>%
  
  # Impute the mode for categorical variables  
  step_modeimpute(
    ms_zoning,
    electrical,
    exterior1st,
    exterior2nd,
    functional,
    sale_type,
    utilities,
    kitchen_qual
  ) %>%
  
  # recipes seems to convert character strings to factors automatically, so need to convert back to deal with missing observations
  step_factor2string(none_vars) %>%  
  
  # Now, replace the missing observations with "none" for the none_vars
  step_mutate_at(none_vars, fn = ~ replace_na(., "None")) %>%
  
  # Convert back
  step_string2factor(all_nominal()) %>%
  
  step_knnimpute(lot_frontage, impute_with = imp_vars(neighborhood, ms_zoning, ms_sub_class, lot_area)) %>% 
  
  # Replace the missing observations with 0 for the zero vars
  step_mutate_at(zero_vars, fn = ~ replace_na(., 0)) %>%
  
  # Recode ordinal variables
  step_mutate_at(
    ordinal_vars, 
    fn = ~ factor(., 
                  levels = c("None", "Po", "Fa", "TA", "Gd", "Ex"), 
                  ordered = TRUE)
  ) %>%
  
  # Create new variables
  step_mutate(
    year_mo = paste(yr_sold, ifelse(nchar(mo_sold) == 1, paste0("0", mo_sold), mo_sold), sep = "-"),
    total_sf = total_bsmt_sf + x1st_flr_sf + x2nd_flr_sf,
    avg_rm_sf = gr_liv_area / tot_rms_abv_grd,
    total_baths = bsmt_full_bath + (bsmt_half_bath * 0.5) + full_bath + (half_bath * 0.5),
    age = yr_sold - year_built,
    #new = if_else(yr_sold == year_built, 1, 0),
    #pool = if_else(pool_area > 0, 1, 0),
    #basement = if_else(total_bsmt_sf > 0, 1, 0),
    #garage = if_else(garage_area > 0, 1, 0),
    #remodeled = if_else(year_remod_add > year_built, 1, 0),
    misc = ifelse(misc_feature == "None", 0, 1),
    porch_area = open_porch_sf + x3ssn_porch + wood_deck_sf + screen_porch + enclosed_porch,
    overall_qual2 = overall_qual^2
  ) %>%
  step_mutate(total_sf2 = total_sf^2) %>% 
  
  # Turn numeric categorical variables to strings
  step_mutate(
    ms_sub_class = factor(ms_sub_class, 
                          levels = c("20", "30",
                                     "40", "45", 
                                     "50", "60", 
                                     "70", "75", 
                                     "80","85", 
                                     "90","120",
                                     "150", "160",
                                     "180", "190")
    ), 
    mo_sold = factor(mo_sold), 
    yr_sold = factor(yr_sold),
    year_mo = factor(year_mo)
  ) %>% 

  # One-hot encoding of categorical variables
  step_dummy(all_nominal(), -ordinal_vars, one_hot = TRUE) %>%
  
  # Add interaction,
  step_interact(~ overall_qual:overall_cond) %>%
  step_interact(~ total_sf:lot_area) %>%
  step_interact(~ year_built:overall_qual) %>%
  step_interact(~ year_built:total_sf) %>%
  step_interact(~ year_built:contains("central_air")) %>%
  
  # Normalize the skewed variables
  step_BoxCox(skewed_vars, -sale_price) %>%
  
  # Center and scale the variales
  step_normalize(all_numeric(), -sale_price) %>%
  
  # Encode the factors so that they are numeric
  step_ordinalscore(ordinal_vars) %>%
  
  # Normalize the outcome variable
  step_log(sale_price) %>%
  
  # Remove sparse variables
  step_zv(all_predictors()) %>% 
  
  # It is necessary to prep the recipe before using it on a new dataset
  prep(retain = TRUE)


set.seed(1491)

train_dat <- juice(prep_rec)
test_dat <- bake(prep_rec, test) %>% select(-sale_price)

print(train_dat)
print(test_dat)

train_split <- initial_split(train_dat, prob = 0.7)
train_cv <- training(train_split) %>% vfold_cv(v = 5)

lasso_model <-
  linear_reg(
    penalty = tune(),
    mixture = 1) %>%
  set_engine("glmnet",
             standardize = FALSE)

alphas <- grid_regular(penalty(), levels = 100)

lasso_cv <-
  tune_grid(
    formula = sale_price ~ .,
    model = lasso_model,
    resamples = train_cv,
    grid = alphas,
    metrics = metric_set(rmse),
    control = control_grid(verbose = TRUE)
  )

best_lasso <-
  lasso_cv %>%
  select_best("rmse", maximize = FALSE)

print(best_lasso)

best_lasso_model <-
  lasso_model %>%
  finalize_model(parameters = best_lasso)

lasso_fit <-
  best_lasso_model %>%
  fit(sale_price ~ ., training(train_split))

lasso_predictions <- predict(lasso_fit, testing(train_split))


testing(train_split) %>%
  select(sale_price) %>%
  bind_cols(lasso_predictions) %>%
  rmse(sale_price, .pred)
