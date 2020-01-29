library(tidyverse)

theme_set(theme_minimal() +
            theme(axis.title.x = element_text(size = 15, hjust = 1),
                  axis.title.y = element_text(size = 15),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12),
                  panel.grid.major = element_line(linetype = 2),
                  panel.grid.minor = element_line(linetype = 2),
                  plot.title = element_text(size = 18, colour = "grey25", face = "bold"), plot.subtitle = element_text(size = 16, colour = "grey44")))

train <-  read_csv("train.csv")
train_labs <-  read_csv("train_labels.csv")
specs <-  read_csv("specs.csv")

train %>% 
  glimpse()

train_labs %>% 
  glimpse()

event_dat <- jsonlite::parse_json(train$event_data[1]) %>% bind_rows()