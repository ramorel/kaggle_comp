library(tidyverse)
library(tidymodels)
library(corrr)


test_dat <- read_csv("test.csv") %>% as_tibble()
train_dat <- read_csv("train.csv") %>% as_tibble()

dim(train_dat)

glimpse(train_dat)

train_dat <- 
  train_dat %>% 
  select(SalePrice, everything(), -Id)

# Visualize the outcome/target variable
train_dat %>% 
  ggplot(aes(x = SalePrice)) +
  geom_histogram(bins = 100) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = scales::comma) +
  theme_bw()

top_cor <- 
  train_dat %>% 
  select_if(is.numeric) %>% 
  select(SalePrice, everything()) %>% 
  correlate() %>% 
  rearrange() %>%
  shave() %>% 
  slice(-1) %>% 
  select(rowname, SalePrice) %>% 
  arrange(desc(SalePrice)) %>% 
  slice(1:10)

btm_cor <- 
  train_dat %>% 
  select_if(is.numeric) %>% 
  select(SalePrice, everything()) %>% 
  correlate() %>% 
  rearrange() %>%
  shave() %>% 
  slice(-1) %>% 
  select(rowname, SalePrice) %>% 
  arrange(SalePrice) %>% 
  slice(1:10)

train_dat %>% 
  select_if(is.numeric) %>% 
  select(SalePrice, everything()) %>% 
  correlate() %>% 
  rplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Plot of highest correlations--categorical variables
map(
  top_cor %>% 
    slice(1, 3, 7:10) %>% 
    pull(rowname),
  ~ train_dat %>% 
    ggplot(aes(x = .data[[.x]], y = SalePrice, group = .data[[.x]])) +
    geom_boxplot() +
    theme_bw()
)

# Plot of highest correlations--categorical variables
map(
  top_cor %>% 
    slice(2, 4:6) %>% 
    pull(rowname),
  ~ train_dat %>% 
    ggplot(aes(x = .data[[.x]], y = SalePrice)) +
    geom_point() +
    geom_smooth(method = "lm", color="blue") +
    theme_bw()
)

# Curious about neighborhoods
train_dat %>% 
  arrange(desc(SalePrice)) %>% 
  mutate(Neighborhood = forcats::as_factor(Neighborhood)) %>% 
  ggplot(aes(x = Neighborhood, y = SalePrice)) +
  geom_boxplot() +
  theme_bw()

# Missing Values
vars_with_nas <- 
  map_dbl(train_dat, ~ sum(is.na(.x))) %>% sort(decreasing = TRUE) %>% .[. > 0] %>% names()

train_dat %>% 
  select(one_of(vars_with_nas)) %>% 
  glimpse()

map(
  train_dat %>% 
    select(one_of(vars_with_nas)),
  ~ unique(.x)
  )

# Predict lot frontage
train_dat %>% 
  ggplot(aes(x = LotArea, y = LotFrontage)) + 
  geom_point()

train_dat %>% 
  ggplot(aes(x = LotShape, y = LotFrontage)) +
  geom_boxplot() +
  theme_bw()

lot_p <- lm(LotFrontage ~ LotArea + LotShape, data = train_dat)
lot_p <- predict(lot_p, train_dat)

train_dat$LotFrontage[is.na(train_dat$LotFrontage)] <- round(lot_p[is.na(train_dat$LotFrontage)])

# Garage year built
train_dat$GarageYrBlt[is.na(train_dat$GarageYrBlt)]  <- train_dat$YearBuilt[is.na(train_dat$GarageYrBlt)]

# Masonry veneer
train_dat$MasVnrArea[is.na(train_dat$MasVnrType)] <- 0

# All character vars with NAs
train_dat <-
  train_dat %>% 
  mutate_if(is.character, ~ replace_na(., "None"))

# Make factors
ordinal_vars <-
  train_dat %>% 
  select_if(is.character) %>% 
  select(matches("QC|Qual|Cond$")) %>% 
  names()

cat_vars <-
  train_dat %>% 
  select_if(is.character) %>% 
  select(-one_of(ordinal_vars)) %>% 
  names()

train_dat <-
  train_dat %>% 
  mutate_at(vars(cat_vars), forcats::as_factor)

train_dat <-
  train_dat %>% 
  mutate_at(vars(ordinal_vars), ~ factor(., levels = c("None", "Po", "Fa", "TA", "Gd", "Ex")))


set.seed(1491)
rf_fit <-
  rand_forest(trees = 100, mode = "regression") %>%
  set_engine("randomForest", importance = TRUE) %>%
  fit(SalePrice ~ ., data = train_dat)

importance(rf_fit$fit) %>% 
  as.data.frame() %>%
  rownames_to_column("variable") %>% 
  arrange(desc(`%IncMSE`)) %>% 
  top_n(20) %>% 
  ggplot(aes(x = reorder(variable, `%IncMSE`), y = `%IncMSE`)) +
  geom_bar(stat = "identity") +
  coord_flip() + 
  theme_bw()
