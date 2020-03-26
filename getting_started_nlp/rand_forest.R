library(tidyverse)
library(tidytext)
library(tidymodels)
library(caret)
library(glmnet)
library(tune)
library(text2vec)
library(keras)

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

train <- train %>% distinct(keyword,location,text,.keep_all = TRUE)

# Missing data in the training and testing data
train %>% 
  mutate(set = "training") %>% 
  bind_rows(test %>% mutate(set = "testing")) %>% 
  group_by(set) %>% 
  summarize_at(
    vars(keyword:text), 
    list(missing = ~ sum(is.na(.)))
  ) %>% 
  pivot_longer(cols = -set) %>% 
  ggplot(aes(x = name, y = value, group = set)) +
  geom_col(aes(fill = set), width = 0.7, alpha = 0.7, position = "dodge") +
  scale_fill_manual(values = c("darkorchid2", "salmon1")) +
  labs(title = "Number missing in training and testing sets", x = "", y = "")

map(
  train,
  ~ paste("% Missing:", round(sum(is.na(.x)) / length(.x), 2))
)

map(
  test,
  ~ paste("% Missing:", round(sum(is.na(.x)) / length(.x), 2))
)


# Target descriptives
train %>% 
  ggplot(aes(x = factor(target))) +
  geom_bar(width = 0.5, fill = "darkorchid2", alpha = 0.75) +
  labs(x = "", y = "Count")

# Charateristics of disaster and non-disaster tweets
train %>% 
  mutate(n = nchar(text)) %>% 
  ggplot() +
  geom_density(aes(x = n, fill = factor(target)), alpha = 0.25) +
  geom_histogram(aes(x = n, fill = factor(target), stat(density)), position = position_dodge(), alpha = 0.7, binwidth = 1) + 
  scale_fill_manual(values = c("darkorchid2", "salmon1")) +
  labs(title = "Characters per tweet", x = "", y = "")

train %>% 
  mutate(n_words = str_count(text, "\\w+")) %>% 
  ggplot() +
  geom_density(aes(x = n_words, fill = factor(target)), alpha = 0.5) +
  scale_fill_manual(values = c("darkorchid2", "salmon1")) +
  labs(title = "Words per tweet", x = "", y = "")

train %>% 
  group_split(target) %>% 
  map(~ .x %>% summarize(text = str_c(text, collapse = " "))) %>% 
  map(~ .x %>% mutate(mean_word_len = str_split(text, "\\s+", simplify = TRUE) %>% nchar() %>% mean())) %>% 
  map2(c(0,1), ~ .x %>% mutate(target = .y)) %>% 
  bind_rows() %>% 
  select(-text) %>% 
  ggplot(aes(y = mean_word_len, x = factor(target))) +
  geom_bar(aes(fill = factor(target)), stat = "identity", alpha = 0.7, width = 0.5) +
  scale_fill_manual(values = c("darkorchid2", "salmon1")) +
  labs(title = "Mean word length", x = "", y = "")

train %>% 
  mutate(has_link = as.numeric(str_detect(text, "http://"))) %>% 
  count(target, has_link) %>%
  group_by(target) %>% 
  mutate(prop = n/sum(n)) %>% 
  filter(has_link == 1) %>% 
  ggplot(aes(x = factor(target), y = prop)) + 
  geom_col(aes(fill = factor(target)), width = 0.5, alpha = 0.7) +
  scale_fill_manual(values = c("darkorchid2", "salmon1")) +
  labs(title = "Has link", x = "", y = "")

# Keywords
train %>% count(keyword)

train %>% 
  filter(!is.na(keyword)) %>% 
  count(keyword, target) %>% 
  complete(target, nesting(keyword), fill = list(n = 0)) %>%
  group_by(keyword) %>% 
  mutate(prop = n/sum(n)) %>% 
  arrange(desc(target), desc(prop)) %>% 
  ungroup() %>% 
  mutate(keyword = forcats::as_factor(keyword)) %>% 
  ggplot(aes(x = keyword, y = prop, group = target)) +
  geom_col(aes(fill = factor(target)), width = 0.5, alpha = 0.7, position = "dodge") +
  scale_fill_manual(values = c("darkorchid2", "salmon1")) +
  coord_flip()

train %>% 
  filter(!is.na(keyword)) %>% 
  count(keyword, target) %>% 
  complete(target, nesting(keyword), fill = list(n = 0)) %>%
  group_by(keyword) %>% 
  mutate(prop = n/sum(n)) %>% 
  filter(target == 1) %>% 
  arrange(desc(prop))

train %>% 
  filter(!is.na(keyword)) %>% 
  count(keyword, target) %>% 
  complete(target, nesting(keyword), fill = list(n = 0)) %>%
  group_by(keyword) %>% 
  mutate(prop = n/sum(n)) %>% 
  filter(target == 1) %>% 
  arrange(prop)


train %>% 
  filter(!is.na(keyword)) %>% 
  count(keyword, target) %>% 
  complete(target, nesting(keyword), fill = list(n = 0)) %>%
  group_by(keyword) %>% 
  mutate(prop = n/sum(n)) %>% 
  filter(target == 0) %>% 
  arrange(prop)


train %>% 
  filter(!is.na(keyword)) %>% 
  count(keyword, target) %>% 
  complete(target, nesting(keyword), fill = list(n = 0)) %>%
  group_by(keyword) %>% 
  mutate(prop = n/sum(n)) %>% 
  filter(target == 0) %>% 
  arrange(desc(prop))


# Most common words by target
train %>% 
  mutate(text = str_remove_all(text, "https?://\\S+")) %>%
  mutate(text = str_remove(text, "\\&amp")) %>% mutate(text = str_remove(text, "\n")) %>% mutate(text = str_remove(text, "\r")) %>%  
  mutate(text = str_remove(text, "\u0089Û_")) %>% 
  mutate(text = str_replace_all(text, "[0-9]", " ")) %>% mutate(text = str_replace_all(text, "@_.+ ", " ")) %>%
  mutate(target = recode(target, `0` = "Non-disaster", `1` = "Disaster")) %>% 
  unnest_tokens(word, text) %>% 
  filter(nchar(word) > 2) %>% 
  anti_join(get_stopwords(source = "smart")) %>% 
  anti_join(get_stopwords()) %>% 
  count(target, word, sort = TRUE) %>% 
  group_by(target) %>%
  top_n(20) %>%
  ungroup() %>%
  ggplot(aes(reorder_within(word, n, target), n,
             fill = target)) +
  geom_col(alpha = 0.75, show.legend = FALSE) +
  scale_fill_manual(values = c("darkorchid2", "salmon1")) +
  scale_x_reordered() +
  labs(title = "Most common keywords", x = "", y = "") +
  coord_flip() +
  facet_wrap(~target, scales = "free")

train %>% 
  mutate(text = str_remove(text, "https?://\\S+")) %>% 
  mutate(text = str_remove(text, "\\&amp")) %>% mutate(text = str_remove(text, "\n")) %>% mutate(text = str_remove(text, "\r")) %>%  
  mutate(text = str_remove(text, "\u0089Û_")) %>% 
  mutate(text = str_replace_all(text, "[0-9]", " ")) %>% 
  mutate(text = str_replace_all(text, "@_.+ ", " ")) %>%
  mutate(text = tm::removeWords(text, get_stopwords(source = "smart")$word)) %>% 
  mutate(text = tm::removeWords(text, get_stopwords()$word)) %>%
  mutate(target = recode(target, `0` = "Non-disaster", `1` = "Disaster")) %>% 
  unnest_tokens(word, text, token = "ngrams", n = 2) %>% 
  filter(!is.na(word)) %>% 
  anti_join(get_stopwords(source = "smart")) %>% 
  anti_join(get_stopwords()) %>% 
  count(target, word, sort = TRUE) %>% 
  group_by(target) %>%
  top_n(20) %>%
  ungroup() %>%
  ggplot(aes(reorder_within(word, n, target), n,
             fill = target)) +
  geom_col(alpha = 0.75, show.legend = FALSE) +
  scale_fill_manual(values = c("darkorchid2", "salmon1")) +
  scale_x_reordered() +
  labs(title = "Most common bigrams", x = "", y = "") +
  coord_flip() +
  facet_wrap(~target, scales = "free")

## Feature engineering ----
train_test <-
  list(train, test) %>% 
  map(~ .x %>% 
        mutate(
          keyword = ifelse(is.na(keyword), "None", keyword),
          location = ifelse(is.na(location), "None", location),
          character_count = nchar(text),
          hashtag_count = str_count(text, "#"),
          has_hashtag = as.numeric(str_detect(text, "#")),
          mention_count = str_count(text, "@"),
          has_mention = as.numeric(str_detect(text, "@")),
          link_count = str_count(text, "https?://"),
          has_link = as.numeric(str_detect(text, "https?://")),
          word_count = str_count(text, "\\w+"),
          unique_words = map_dbl(str_split(text, " "), ~length(unique(.x)))
        )
  )

train_dat <- train_test[[1]]
test_dat <- train_test[[2]]

train_test <- train_test %>% map2(c("train", "test"), ~ mutate(.x, set = .y)) %>% bind_rows()

## Clean text ----
clean_dat <-
  train_test %>% 
  mutate(text = str_to_lower(text)) %>% 
  mutate(text = str_replace_all(text, "%20", "")) %>% 
  mutate(text = str_replace_all(text, "https?://\\S+", " ")) %>% 
  mutate(text = str_replace_all(text, "\\&amp", " ")) %>% 
  mutate(text = iconv(text, "utf8", "ASCII", sub = " ")) %>% 
  mutate(text = str_replace_all(text, "\n", " ")) %>% 
  mutate(text = str_replace_all(text, "#\\S+", " ")) %>% 
  mutate(text = str_replace_all(text, "\r", " ")) %>%  
  mutate(text = str_replace_all(text, "[0-9]", " ")) %>% 
  mutate(text = str_replace_all(text, "@_\\S+ ", " ")) %>%
  mutate(text = str_replace_all(text, "[^[:alnum:]]", " ")) %>%
  mutate(text = tm::removeWords(text, get_stopwords()$word)) %>% 
  mutate(text = str_replace_all(text, "\\b[a-z]{1,2}\\b", " ")) %>%
  mutate(text = str_squish(text))

## Lasso ----
tokens <- word_tokenizer(clean_dat[["text"]])
it <- itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, doc_count_min = 5)
tweet_dtm <- create_dtm(it, vocab_vectorizer(vocab))

tfidf = TfIdf$new(norm = "l2")
tweet_tfidf = fit_transform(tweet_dtm, tfidf)

train_test_dat <- 
  clean_dat %>% 
  rename(y_target = target) %>% 
  select(y_target, character_count, hashtag_count, mention_count, link_count) %>% 
  cbind(tweet_tfidf %>% as.matrix())

y_train <- train_test_dat %>% 
  filter(!is.na(y_target)) %>% 
  pull(y_target)

train_dat <- train_test_dat %>% 
  filter(!is.na(y_target))

test_dat <- train_test_dat %>% 
  filter(is.na(y_target)) %>% 
  select(-y_target)

train_split <- initial_split(train_dat, strata = y_target)
X_train <- training(train_split)
X_test <- testing(train_split)

train_cv <- vfold_cv(X_train, v = 5)

rf_model <-
  rand_forest(mode = "classification", trees = tune(), min_n = tune()) %>% 
  set_engine("randomForest")

rf_tune <- grid_regular(trees(), min_n(), levels = 5)

rf_cv <-
  tune_grid(
    formula = factor(y_target) ~ .,
    model = rf_model,
    resamples = train_cv,
    grid = rf_tune,
    metrics = metric_set(accuracy),
    control = control_grid(verbose = TRUE)
  )

rf_fit <-
  rf_model %>% 
  fit(factor(y_target) ~ ., X_train)

