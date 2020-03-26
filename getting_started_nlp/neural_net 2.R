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

# EDA ----

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
  mutate(text = str_replace_all(text, "[:punct:]", " ")) %>%
  mutate(text = tm::removeWords(text, get_stopwords()$word)) %>% 
  mutate(text = str_squish(text))

## GloVE ----
lines <- readLines("glove.6B.100d.txt")

# Create an index for the word and its associated coefficients
embeddings_index <- new.env(parent = emptyenv())
for (line in lines) {
  coefs <- as.numeric(strsplit(line, ' ', fixed = TRUE)[[1]][-1])
  word <- strsplit(line, ' ', fixed = TRUE)[[1]][1]
  embeddings_index[[word]] <- coefs
  }

tokenizer <- text_tokenizer()
tokenizer %>% fit_text_tokenizer(clean_dat[["text"]])
sequences <- texts_to_sequences(tokenizer, clean_dat[["text"]])

word_index <- tokenizer$word_index

pad_texts <- pad_sequences(sequences, maxlen=50)

num_words <- min(20000, length(word_index) + 1)

prepare_embedding_matrix <- function() {
  embedding_matrix <- matrix(0L, nrow = num_words, ncol = 100)
  for (word in names(word_index)) {
    index <- word_index[[word]]
    if (index >= 20000)
      next
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector)) {
      # words not found in embedding index will be all-zeros.
      embedding_matrix[index,] <- embedding_vector
    }
  }
  embedding_matrix
}

embedding_matrix <- prepare_embedding_matrix()




y_train <- clean_dat[["target"]][!is.na(clean_dat[["target"]])]
X_train <- pad_texts[!is.na(train_test$target), ]

model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim = num_words, 
                  output_dim = 100, 
                  input_length = 50, 
                  weight = list(embedding_matrix),
                  trainable = FALSE) %>%
  layer_spatial_dropout_1d(0.3) %>%
  layer_lstm(64, dropout=0.2, recurrent_dropout=0.2) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% summary()

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

history <- model %>% fit(
  X_train,
  y_train,
  epochs = 40,
  batch_size = 512,
  validation_split = 0.3,
  verbose=1
)



# Neural net ----
tokens <-  word_tokenizer(clean_dat$text)
it <- itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)
vectorizer <- vocab_vectorizer(vocab)

vocab_size <- nrow(vocab)
tokenizer <- text_tokenizer()
tokenizer$fit_on_texts(clean_dat$text)
sequences <- tokenizer$texts_to_sequences(clean_dat$text)
pad_texts <- pad_sequences(sequences, maxlen = 50, truncating='post',padding='post')

num_words <- length(tokenizer$word_index) + 1

y_train <- clean_dat$target
X_train <- pad_texts[!is.na(train_test$target), ]

model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
  layer_dropout(0.3) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(units = 1) %>% 
  layer_activation("sigmoid")

model %>% summary()


model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

history <- model %>% fit(
  X_train,
  y_train,
  epochs = 40,
  batch_size = 512,
  validation_split = 0.3,
  verbose=1
)

preds <- model %>% predict(X_test)

tibble(
  id = test$id,
  target = ifelse(preds > 0.65, 1, 0)
) %>% 
  write_csv("submission.csv")



## Logistic regression using tweet features ----
library(rsample)
train <-
  train %>% 
  mutate(keyword = ifelse(is.na(keyword), "None", keyword),
         has_link = as.numeric(str_detect(text, "http://")),
         contains_hashtag = as.numeric(str_detect(text, "#")),
         nchar = nchar(text),
         n_words = str_count(text, "\\w+"),
         disaster_key = ifelse(str_detect(text, disaster_keys %>% pull(word) %>% paste(collapse = "|")), 1, 0),
         disaster_ngram = ifelse(str_detect(text, disaster_ngrams %>% pull(word) %>% paste(collapse = "|")), 1, 0),
         ) %>% 
  mutate(d_key_or_n = ifelse(disaster_key == 1 | disaster_ngram == 1, 1, 0))

set.seed(1491)
train_split <- initial_split(train, strata = target)

simple_fit <- glm(target ~ keyword + contains_hashtag + nchar + n_words + d_key_or_n + has_link, data = training(train_split))
preds <- predict(simple_fit, newdata = testing(train_split))

tab <-
  tibble(
    truth = factor(testing(train_split)$target),
    estimate = factor(ifelse(predict(simple_fit, type = "response", newdata = testing(train_split)) > 0.6, 1, 0))
  )

rc <- yardstick::recall(tab, truth, estimate)$.estimate
prec <- yardstick::precision(tab, truth, estimate)$.estimate

2 * ((prec * rc) / (prec + rc))

test <-
  test %>% 
  mutate(keyword = ifelse(is.na(keyword), "None", keyword),
         has_link = as.numeric(str_detect(text, "http://")),
         contains_hashtag = as.numeric(str_detect(text, "#")),
         nchar = nchar(text),
         n_words = str_count(text, "\\w+"),
         disaster_key = ifelse(str_detect(text, disaster_keys %>% pull(word) %>% paste(collapse = "|")), 1, 0),
         disaster_ngram = ifelse(str_detect(text, disaster_ngrams %>% pull(word) %>% paste(collapse = "|")), 1, 0),
  ) %>% 
  mutate(d_key_or_n = ifelse(disaster_key == 1 | disaster_ngram == 1, 1, 0))

preds <- predict(simple_fit, newdata = test)

tibble(
  id = test$id,
  target = ifelse(preds > 0.65, 1, 0)
) %>% 
  write_csv("submission.csv")


## Penalized regression with dtm ----
train_dat <- training(train_split)

train_clean <-
  train %>% 
  mutate(text = str_to_lower(text)) %>% 
  mutate(text = str_remove(text, "https?://.+")) %>% 
  mutate(text = str_remove(text, "\\&amp")) %>% 
  mutate(text = str_remove(text, "\n")) %>% 
  mutate(text = str_remove(text, "\r")) %>%  
  mutate(text = str_remove(text, "\u0089û_")) %>% 
  mutate(text = str_replace_all(text, "[0-9]", " ")) %>% 
  mutate(text = str_replace_all(text, "@_.+ ", " ")) %>%
  mutate(text = tm::removeWords(text, get_stopwords(source = "smart")$word)) %>% 
  mutate(text = tm::removeWords(text, get_stopwords()$word)) %>% 
  mutate(text = tm::removePunctuation(text)) %>% 
  mutate(text = str_squish(text)) %>% 
  mutate(keyword = ifelse(is.na(keyword), "None", keyword),
         has_link = as.numeric(str_detect(text, "http://")),
         contains_hashtag = as.numeric(str_detect(text, "#")),
         nchar = nchar(text),
         n_words = str_count(text, "\\w+"),
         disaster_key = ifelse(str_detect(text, disaster_keys %>% pull(word) %>% paste(collapse = "|")), 1, 0),
         disaster_ngram = ifelse(str_detect(text, disaster_ngrams %>% pull(word) %>% paste(collapse = "|")), 1, 0),
  ) %>% 
  mutate(d_key_or_n = ifelse(disaster_key == 1 | disaster_ngram == 1, 1, 0))

tweet_dtm <-
  train_clean %>%
  unnest_tokens(word, text) %>%
  group_by(word) %>%
  filter(n() > 10) %>%
  ungroup() 

ids_to_keep <- unique(tweet_dtm$id)

tweet_dtm <- 
  tweet_dtm %>%
  count(id, word) %>%
  cast_sparse(id, word, n)

train_dat <- cbind(tweet_dtm, train_clean %>% filter(id %in% ids_to_keep) %>% select(has_link:contains_hashtag) %>% as.matrix())
y_train <- train_clean %>% filter(id %in% ids_to_keep) %>% pull(target)

model <- cv.glmnet(train_dat, y_train,
                   alpha = 1,
                   nfolds = 5,
                   family = "binomial",
                   parallel = TRUE, keep = TRUE
)

plot(model)

pred <- predict(model, train_dat, type = "class")
MLmetrics::F1_Score(y_train, pred, positive = 1)

reg_grid <-
  grid_regular(mixture(), penalty(), levels = 5) %>% 
  rename(alpha = mixture, lambda = penalty)

train_cv <- vfold_cv()

lasso_model <-
  linear_reg(
    penalty = tune(),
    mixture = tune()) %>%
  set_engine("glmnet",
             standardize = FALSE)



## GloVe
train_clean <-
  train %>% 
  mutate(text = str_to_lower(text)) %>% 
  mutate(text = str_remove(text, "https?://.+")) %>% 
  mutate(text = str_remove(text, "\\&amp")) %>% 
  mutate(text = str_remove(text, "\n")) %>% 
  mutate(text = str_remove(text, "\r")) %>%  
  mutate(text = str_remove(text, "\u0089û_")) %>% 
  mutate(text = str_replace_all(text, "[0-9]", " ")) %>% 
  mutate(text = str_replace_all(text, "@_.+ ", " ")) %>%
  mutate(text = tm::removeWords(text, get_stopwords(source = "smart")$word)) %>% 
  mutate(text = tm::removeWords(text, get_stopwords()$word)) %>% 
  mutate(text = tm::removePunctuation(text)) %>% 
  mutate(text = str_squish(text)) %>% 
  mutate(keyword = ifelse(is.na(keyword), "None", keyword),
         has_link = as.numeric(str_detect(text, "http://")),
         contains_hashtag = as.numeric(str_detect(text, "#")),
         nchar = nchar(text),
         n_words = str_count(text, "\\w+"))

test_clean <-
  test %>% 
  mutate(text = str_to_lower(text)) %>% 
  mutate(text = str_remove(text, "https?://.+")) %>% 
  mutate(text = str_remove(text, "\\&amp")) %>% 
  mutate(text = str_remove(text, "\n")) %>% 
  mutate(text = str_remove(text, "\r")) %>%  
  mutate(text = str_remove(text, "\u0089û_")) %>% 
  mutate(text = str_replace_all(text, "[0-9]", " ")) %>% 
  mutate(text = str_replace_all(text, "@_.+ ", " ")) %>%
  mutate(text = tm::removeWords(text, get_stopwords(source = "smart")$word)) %>% 
  mutate(text = tm::removeWords(text, get_stopwords()$word)) %>% 
  mutate(text = tm::removePunctuation(text)) %>% 
  mutate(text = str_squish(text)) %>% 
  mutate(keyword = ifelse(is.na(keyword), "None", keyword),
         has_link = as.numeric(str_detect(text, "http://")),
         contains_hashtag = as.numeric(str_detect(text, "#")),
         nchar = nchar(text),
         n_words = str_count(text, "\\w+"))

tokens <- word_tokenizer(train_clean[["text"]])
it <- itoken(tokens, preprocess_function = tolower, progressbar = FALSE)
vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 5L)
train_dtm <- create_dtm(it, vocab_vectorizer(vocab))

tfidf = TfIdf$new()
dtm_train_tfidf = fit_transform(train_dtm, tfidf)

dtm_train_tfidf <- cbind(dtm_train_tfidf, train_clean %>% select(has_link:n_words) %>% as.matrix())

glmnet_fit = cv.glmnet(x = dtm_train_tfidf, y = train_clean[["target"]], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 5,
                              thresh = 1e-3,
                              maxit = 1e3)

tokens <- word_tokenizer(test_clean[["text"]])
it_test <- itoken(tokens, preprocess_function = tolower, progressbar = FALSE)
test_dtm <- create_dtm(it_test, vocab_vectorizer(vocab))
dtm_test_tfidf = fit_transform(test_dtm, tfidf)

dtm_test_tfidf <- cbind(dtm_test_tfidf, test_clean %>% select(has_link:n_words) %>% as.matrix())

preds <- predict(glmnet_fit, dtm_test_tfidf, type = "response")[,1]

tibble(
  id = test$id,
  target = ifelse(preds > 0.5, 1, 0)
) %>% 
  write_csv("submission.csv")






vocab_size <- dim(dtm_train_tfidf)[1]

model <- keras_model_sequential()
model %>% 
  layer_embedding(input_dim = vocab_size, output_dim = 16) %>%
  layer_dropout(0.3) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(units = 1) %>% 
  layer_activation("sigmoid")

model %>% summary()


model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

history <- model %>% fit(
  dtm_train_tfidf,
  train_clean[["target"]],
  epochs = 40,
  batch_size = 512,
  validation_split = 0.3,
  verbose=1
)
