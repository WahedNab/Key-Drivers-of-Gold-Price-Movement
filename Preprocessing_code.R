library(data.table)
library(lubridate)
library(dplyr)

# Create the normalize function

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Read the file

df <- read.csv("/Users/sadafmoghisi/Desktop/PhD/Terms/Term3/Data Mining:Discovery (523)/HW/Final/gld_price_data.csv")

#### Preprocess data

df$Date <- as.Date(df$Date, format = "%m/%d/%Y")
df <- filter(df, !is.na(df$Date))

## Feature Engineering

# Extract day, month, year for temporal effects (Mainly for visualizations)

df <- df |>
  mutate(day = day(Date),
         month = month(Date),
         year = year(Date))

# Create the lag features for predictions

df <- df |>
  mutate(lag_SPX = lag(SPX, n = 1),
         lag_GLD = lag(GLD, n = 1),
         lag_USO = lag(USO, n = 1),
         lag_SLV = lag(SLV, n = 1),
         lag_EUR.USD = lag(EUR.USD, n = 1))

# Normalize the dependent features for SVM

df <- df |>
  mutate(norm_SPX = normalize(SPX),
         norm_USO = normalize(USO),
         norm_SLV = normalize(SLV),
         norm_EUR.USD = normalize(EUR.USD))

# Create the gold label for classification


df <- df |>
  mutate(
    gold_label = case_when(
      lead(GLD) - GLD > 0 ~ "Up",
      lead(GLD) - GLD < 0 ~ "Down",
      lead(GLD) - GLD == 0 ~ "Same"
    ),
    gold_label = lag(gold_label),
    gold_label = ifelse(is.na(gold_label), "Same", gold_label)
  )

write.csv(df, "/Users/sadafmoghisi/Desktop/PhD/Terms/Term3/Data Mining:Discovery (523)/HW/Final/gold_price_dataset_preprocessed.csv")

