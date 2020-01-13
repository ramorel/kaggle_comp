library(tidyverse)
library(corrr)

test_dat <- read_csv("test.csv") %>% as_tibble()
train_dat <- read_csv("train.csv") %>% as_tibble()

dim(train_dat)

glimpse(train_dat)

train_dat <- 
  train_dat %>% 
  select(-Id)

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

train_dat <-
  train_dat %>% 
  mutate_if(is.character, ~ replace_na(., "None"))
