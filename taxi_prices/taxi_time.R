library(tidyverse)
library(lubridate)
library(caret)
# spatial data pacakges
library(tigris)
library(sf)
library(sp)
library(tidymodels)
library(tune)
# need to set a few options for the tigris package
options(tigris_class = "sf")
options(tigris_use_cache = TRUE)

# Define theme for ggplot
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

train_dat <- data.table::fread("train.csv", nrows = 1000000)
glimpse(train_dat)

# What's up with the fares? Summary stats, distribution
summary(train_dat)

# How many fares are less than 0? Not too many
sum(train_dat$fare_amount < 0)



train_dat %>% 
  ggplot() +
  geom_histogram(aes(x = fare_amount, stat(density)),
                 bins = 100,
                 fill = "cornflowerblue",
                 alpha = 0.7) +
  geom_density(aes(x = fare_amount), color = "tomato")

train_dat %>% 
  ggplot() +
  geom_histogram(aes(x = log(fare_amount + 1), stat(density)),
                 bins = 100,
                 fill = "cornflowerblue",
                 alpha = 0.7) +
  geom_density(aes(x = log(fare_amount + 1)), color = "tomato")


train_dat <- 
  train_dat %>% 
  filter(
    between(
      fare_amount, 1, quantile(fare_amount, 0.99) # These extreme values aren't doing anyone any favors.
    ),
    between(
      passenger_count, 1, 6 # 6 passengers is pushing it, but they do have those taxi vans...
    )
  ) %>% 
  filter(pickup_longitude < -73.70002, pickup_longitude > -74.3) %>% 
  filter(pickup_latitude < 40.7868, pickup_latitude > 40.62) %>% 
  filter(dropoff_longitude < -73, dropoff_longitude > -75) %>% 
  filter(dropoff_latitude < 42, dropoff_latitude > 40)

train_dat %>% 
  ggplot() +
  geom_histogram(aes(x = fare_amount, stat(density)),
                 bins = 100,
                 fill = "cornflowerblue",
                 alpha = 0.7) +
  geom_density(aes(x = fare_amount), color = "tomato")

train_dat %>% 
  ggplot() +
  geom_histogram(aes(x = log(fare_amount), stat(density)),
                 bins = 100,
                 fill = "cornflowerblue",
                 alpha = 0.7) +
  geom_density(aes(x = log(fare_amount)), color = "tomato")


# Now to find the distance
train_dat <- 
  train_dat %>% 
  mutate(dist = geosphere::distHaversine(
    train_dat[, c("pickup_longitude", "pickup_latitude")], 
    train_dat[, c("dropoff_longitude", "dropoff_latitude")]))
summary(train_dat$dist)

train_dat %>% 
  ggplot() +
  geom_histogram(aes(x = dist, stat(density)),
                 bins = 100,
                 fill = "cornflowerblue",
                 alpha = 0.7) +
  geom_density(aes(x = dist), color = "tomato")

train_dat <-
  train_dat %>% 
  filter(dist < quantile(dist, 0.99),
         dist > 0)

train_dat %>% 
  ggplot() +
  geom_histogram(aes(x = log(dist+1), stat(density)),
                 bins = 100,
                 fill = "cornflowerblue",
                 alpha = 0.7) +
  geom_density(aes(x = log(dist+1)), color = "tomato")


# categorize trip times
train_dat <- 
  train_dat %>% 
  mutate(day = weekdays(as_date(pickup_datetime)), 
         hour = hour(as_datetime(pickup_datetime)),
         year = year(pickup_datetime))

train_dat <- 
  train_dat %>% 
  mutate(time_of_day = 
           case_when(
             hour %in% 6:8 ~ "am_rush", 
             hour %in% 9:11 ~ "mid_am", 
             hour %in% 12:14 ~ "lunchtime", 
             hour %in% 15:17 ~ "afternoon", 
             hour %in% 18:20 ~ "pm_rush", 
             hour %in% 21:23 ~ "night", 
             hour %in% 0:2 ~ "mid_night", 
             hour %in% 3:5 ~ "late_night")
  )

# airports ----
jfk <- str_sub(call_geolocator_latlon(40.6413, -73.7781), 1, 11)
lga <- str_sub(call_geolocator_latlon(40.7769, -73.8740), 1, 11)
ewr <- str_sub(call_geolocator_latlon(40.6895, -74.1745), 1, 11)

airports <- 
  tracts(36, county = "Queens", cb = T) %>% 
  filter(GEOID %in% c(lga, jfk))

airports <-
  rbind(airports, 
        tracts(34, county = "Essex", cb = T) %>% 
          filter(GEOID == ewr)
  )

# extract the dropoff lon/lat to convert to spatial point object
pickup_coords <- 
  train_dat  %>% 
  select(pickup_longitude, pickup_latitude) %>% 
  as.matrix()

pickup_coords <- 
  pickup_coords %>% 
  st_multipoint() %>% 
  st_sfc(crs = 4326) %>% 
  st_cast("POINT")

dropoff_coords <- 
  train_dat  %>% 
  select(dropoff_longitude, dropoff_latitude) %>% 
  as.matrix()

dropoff_coords <- 
  dropoff_coords %>% 
  st_multipoint() %>% 
  st_sfc(crs = 4326) %>% 
  st_cast("POINT")

airport_pickup <- st_contains(st_transform(airports, 4326), pickup_coords)
airport_dropoff <- st_contains(st_transform(airports, 4326), dropoff_coords)

train_dat <-
  train_dat %>% 
  mutate(
    from_airport = 
      ifelse(
        row_number() %in% unlist(airport_pickup), 
        1, 0
      ),
    to_airport = 
      ifelse(
        row_number() %in% unlist(airport_dropoff), 
        1, 0
      )
  )

rm(list = ls()[str_detect(ls(), "air|coords")])


# Preprocessing ----
prep_rec <- 
  recipe(fare_amount ~ ., data = train_dat) %>% 
  step_mutate(dist2 = dist^2) %>% 
  step_rm(key, pickup_datetime, 
          contains("latitude"), contains("longitude")) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_interact(~ contains("day"):contains("time_of_day")) %>% 
  step_log(all_numeric(), offset = 1) %>%
  prep(retain = TRUE)

train <- juice(prep_rec)

train_split <- initial_split(train)

X_train <- training(train_split)
X_test <- testing(train_split)

X_train_cv <- vfold_cv(X_train, v = 5)


# LASSO ----
lasso_model <-
  linear_reg(
    penalty = tune(),
    mixture = 1) %>%
  set_engine("glmnet",
             standardize = FALSE)

alphas <- grid_regular(penalty(), levels = 25)

lasso_cv <-
  tune_grid(
    formula = fare_amount ~ .,
    model = lasso_model,
    resamples = X_train_cv,
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
  fit(fare_amount ~ ., X_test)

lasso_predictions <- predict(lasso_fit, X_test)

testing(train_split) %>%
  select(fare_amount) %>%
  bind_cols(lasso_predictions) %>%
  mutate_all(exp) %>% 
  rmse(fare_amount, .pred)

# Xgboost
xgb_model <-
  boost_tree(trees = tune(),
             tree_depth = tune(), min_n = tune()) %>% 
  set_engine("xgboost")

xgb_grid <- grid_regular(trees(),
                         tree_depth(),
                         min_n(),
                         levels = 3)

xgb_cv <-
  tune_grid(
    formula = fare_amount ~ .,
    model = xgb_model,
    resamples = X_train_cv,
    grid = xgb_grid,
    metrics = metric_set(rmse),
    control = control_grid(verbose = TRUE)
  )
