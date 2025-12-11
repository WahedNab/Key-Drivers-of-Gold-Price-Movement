
############ Stylized Time-Series Plot (Gold, Oil, Silver)

library(tidyverse)
library(zoo)        # for rolling means
library(scales)     # for date formatting

# Load data
df <- read_csv("/Users/sadafmoghisi/Desktop/PhD/Terms/Term3/Data Mining:Discovery (523)/HW/Final/gold_price_dataset_preprocessed.csv")

# Convert Date column
df$Date <- as.Date(df$Date)

# Smooth the lines (7-day centered rolling mean)
df <- df %>%
  mutate(
    GLD_smooth = rollmean(GLD, 7, fill = NA, align = "center"),
    USO_smooth = rollmean(USO, 7, fill = NA, align = "center"),
    SLV_smooth = rollmean(SLV, 7, fill = NA, align = "center")
  )

# Color palette
colors <- c(
  GLD = "#41DB56",
  USO = "#682EC4",
  SLV = "#EF521E"
)

# Base Plot
p <- ggplot(df, aes(x = Date)) +
  
  # Smooth lines
  geom_line(aes(y = GLD_smooth, color = "GLD"), linewidth = 1.2) +
  geom_line(aes(y = USO_smooth, color = "USO"), linewidth = 1.2) +
  geom_line(aes(y = SLV_smooth, color = "SLV"), linewidth = 1.2) +
  
  # Subtle dotted scatter markers every 90th point
  geom_point(data = df[seq(1, nrow(df), 90), ], 
             aes(y = GLD, color = "GLD"), size = 1.5, alpha = 0.5) +
  geom_point(data = df[seq(1, nrow(df), 90), ], 
             aes(y = USO, color = "USO"), size = 1.5, alpha = 0.5) +
  geom_point(data = df[seq(1, nrow(df), 90), ], 
             aes(y = SLV, color = "SLV"), size = 1.5, alpha = 0.5) +
  
  # Manual colors
  scale_color_manual(
    name = "",
    values = colors,
    labels = c(
      GLD = "Gold Price (GLD)",
      USO = "Oil Price (USO)",
      SLV = "Silver Price (SLV)"
    )
  ) +
  
  labs(
    title = "Stylized Trends of Gold, Oil, and Silver Prices",
    x = "Date",
    y = "Price"
  ) +
  
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    legend.background = element_rect(fill = "white", color = "black", size = 0.2),
    legend.position = "top"
  )

# Display
print(p)


############## Normalized Time Series

library(tidyverse)

# Load processed file
df <- read_csv("/Users/sadafmoghisi/Desktop/PhD/Terms/Term3/Data Mining:Discovery (523)/HW/Final/gold_price_dataset_preprocessed.csv")

# Convert Date
df$Date <- as.Date(df$Date)

# ---- ADD THE MISSING NORMALIZED GOLD COLUMN ----
df <- df %>%
  mutate(
    norm_GLD = (GLD - min(GLD, na.rm = TRUE)) / (max(GLD, na.rm = TRUE) - min(GLD, na.rm = TRUE))
  )

# Select normalized columns
norm_cols <- c("norm_GLD", "norm_SLV", "norm_USO", "norm_EUR.USD", "norm_SPX")

# Convert to long format for ggplot
df_long <- df %>%
  select(Date, all_of(norm_cols)) %>%
  pivot_longer(cols = all_of(norm_cols), names_to = "Variable", values_to = "Value")

# Plot
p <- ggplot(df_long, aes(x = Date, y = Value, color = Variable)) +
  geom_line(size = 1) +
  labs(
    title = "Normalized Time Series Trends of Gold, Silver, Oil, EUR/USD, and S&P 500",
    x = "Date",
    y = "Normalized Prices"
  ) +
  theme_minimal(base_size = 14) +
  scale_color_brewer(palette = "Set1") +
  theme(
    legend.title = element_blank(),
    plot.title = element_text(size = 18, face = "bold")
  )

print(p)

############## SCATTER-PLOT MATRIX


library(tidyverse)
library(GGally)
library(ks)
library(rlang)

# -----------------------------
# Load data
# -----------------------------
df <- read_csv("/Users/sadafmoghisi/Desktop/PhD/Terms/Term3/Data Mining:Discovery (523)/HW/Final/gold_price_dataset_preprocessed.csv")

df <- df %>% select(SPX, GLD, USO, SLV, `EUR.USD`) %>% drop_na()

cols <- c("SPX", "GLD", "USO", "SLV", "EUR.USD")

# Color palette (similar to Python tab10)
palette <- c(
  "SPX" = "#1f77b4",
  "GLD" = "#ff7f0e",
  "USO" = "#2ca02c",
  "SLV" = "#d62728",
  "EUR.USD" = "#9467bd"
)

# -----------------------------
# FIXED CUSTOM PANEL FUNCTIONS
# -----------------------------

# 1. Diagonal KDE plot
panel_kde <- function(data, mapping, ...) {
  
  var <- rlang::as_name(mapping$x)
  x <- data[[var]]
  
  # Use base density() for stable 1D KDE (avoids ks::kde vertical-line issue)
  d <- density(x, na.rm = TRUE)
  
  df_kde <- data.frame(
    x = d$x,
    y = d$y
  )
  
  ggplot(df_kde, aes(x, y)) +
    geom_line(color = palette[var], linewidth = 1.2) +
    theme_void() +
    labs(x = var)
}

# 2. Lower triangle: scatterplot
panel_scatter <- function(data, mapping, ...) {
  
  var <- rlang::as_name(mapping$y)
  
  ggplot(data, mapping) +
    geom_point(color = palette[var], alpha = 0.4, size = 1.2) +
    theme_minimal() +
    theme(
      axis.title = element_blank(),
      axis.text = element_blank(),
      panel.grid = element_blank()
    )
}

# 3. Upper triangle: correlation coefficient
panel_corr <- function(data, mapping, ...) {
  
  xvar <- rlang::as_name(mapping$x)
  yvar <- rlang::as_name(mapping$y)
  
  corr <- round(cor(data[[xvar]], data[[yvar]]), 3)
  
  ggplot() +
    annotate("text", x = 0.5, y = 0.5,
             label = paste0("Corr:\n", corr),
             fontface = "bold", size = 5) +
    theme_void()
}

# -----------------------------
# FINAL SCATTER-PLOT MATRIX
# -----------------------------
p <- ggpairs(
  df,
  columns = cols,
  upper = list(continuous = panel_corr),
  lower = list(continuous = panel_scatter),
  diag  = list(continuous = panel_kde)
)

print(p)


############## Correlation Heatmap

library(tidyverse)
library(reshape2)
library(ggplot2)

# Load data
df <- read_csv("/Users/sadafmoghisi/Desktop/PhD/Terms/Term3/Data Mining:Discovery (523)/HW/Final/gold_price_dataset_preprocessed.csv")

# Select columns for correlation
cols <- c("SPX", "GLD", "USO", "SLV", "EUR.USD")

# Compute correlation matrix
corr <- cor(df[, cols], use = "complete.obs")

# Convert to long format for ggplot2
corr_melt <- melt(corr)

# Heatmap plot
p <- ggplot(corr_melt, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", value)), size = 4) +
  scale_fill_distiller(palette = "RdBu", direction = -1, limits = c(-1, 1)) +
  labs(
    title = "Correlation Heatmap of Key Features",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
  )

print(p)













