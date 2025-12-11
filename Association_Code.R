#%% Imports
library(tidyverse)

#%% Load Preprocessed Data
data_path <- "/Users/sadafmoghisi/Desktop/PhD/Terms/Term3/Data Mining:Discovery (523)/HW/Final/gold_price_dataset_preprocessed.csv"
df <- read_csv(data_path, show_col_types = FALSE)

# Replace only EXACT "NA" string values in character columns
df <- df %>%
  mutate(across(where(is.character), ~na_if(.x, "NA")))

price_cols <- c("GLD", "SLV", "USO", "SPX", "EUR.USD")

df <- df %>% drop_na(lag_SPX)

#%% Create Direction Labels
norm_map <- list(
  "GLD" = "gold_label",
  "SLV" = "norm_SLV",
  "USO" = "norm_USO",
  "SPX" = "norm_SPX",
  "EUR.USD" = "norm_EUR.USD"
)

direction_from_norm <- function(series, eps = 0.001) {
  diff <- dplyr::lead(series) - series
  labels <- ifelse(diff > eps, "Up",
                   ifelse(diff < -eps, "Down", "Same"))
  return(dplyr::lag(labels))
}

df$GLD_dir <- direction_from_norm(df$GLD, eps = 0.005)

for (asset in names(norm_map)) {
  if (asset != "GLD") {
    norm_col <- norm_map[[asset]]
    df[[paste0(asset, "_dir")]] <- direction_from_norm(df[[norm_col]], eps = 0.001)
  }
}

needed_dirs <- paste0(names(norm_map), "_dir")
df <- df %>% drop_na(all_of(needed_dirs))

#%% Build Transactions  ✅ FIXED
transactions <- lapply(seq_len(nrow(df)), function(i) {
  row <- df[i, ]
  sapply(price_cols, function(col) {
    paste0(gsub("\\.", "_", col), "_", row[[paste0(col, "_dir")]])
  }, USE.NAMES = FALSE)
})

N <- length(transactions)

#%% APRIORI IMPLEMENTATION
MIN_SUP <- 0.10
MIN_CONF <- 0.70

support <- function(itemset) {
  sum(sapply(transactions, function(t) all(itemset %in% t))) / N
}

# L1
item_counts <- table(unlist(transactions))
C1 <- as.list(item_counts / N)

L1 <- C1[sapply(C1, function(x) x >= MIN_SUP)]

frequent_itemsets <- L1

# Apriori Candidate Generator
apriori_gen <- function(Lk_minus_1, k) {
  keys <- lapply(names(Lk_minus_1), function(x) sort(unlist(strsplit(x, ","))))
  cands <- list()
  idx <- 1
  
  for (i in seq_along(keys)) {
    for (j in (i+1):length(keys)) {
      if (j > length(keys)) break
      
      a <- keys[[i]]
      b <- keys[[j]]
      
      if (k <= 2 || all(a[1:(k-2)] == b[1:(k-2)])) {
        cand <- sort(unique(c(a, b)))
        cands[[idx]] <- cand
        idx <- idx + 1
      }
    }
  }
  # Remove duplicates
  return(unique(lapply(cands, function(x) x)))
}

# Expand Lk
k <- 2
Lprev <- L1

while (TRUE) {
  Ck <- apriori_gen(Lprev, k)
  Lk <- list()
  
  for (cand in Ck) {
    sup <- support(cand)
    if (sup >= MIN_SUP) {
      Lk[[paste(cand, collapse = ",")]] <- sup
    }
  }
  
  if (length(Lk) == 0) break
  
  frequent_itemsets <- c(frequent_itemsets, Lk)
  Lprev <- Lk
  k <- k + 1
}

#%% Display Frequent Itemsets Table
freq_table <- tibble(
  Itemset = names(frequent_itemsets),
  Support = as.numeric(frequent_itemsets)
) %>% arrange(desc(Support))

cat("\n\n===== Frequent Itemsets Table =====\n")
print(freq_table)
cat("\nTotal Apriori Frequent Itemsets:", nrow(freq_table), "\n")

#%% NEW APRIORI HEATMAP (FREQUENT 1-ITEMSETS ONLY)

# Extract only single-item itemsets from frequent_itemsets
freq_items_only <- names(frequent_itemsets)[sapply(names(frequent_itemsets), function(x) !grepl(",", x))]

if (length(freq_items_only) > 1) {
  
  co_apriori <- matrix(
    0,
    nrow = length(freq_items_only),
    ncol = length(freq_items_only),
    dimnames = list(freq_items_only, freq_items_only)
  )
  
  # Fill matrix
  for (t in transactions) {
    present_items <- freq_items_only[freq_items_only %in% t]
    if (length(present_items) > 0) {
      for (i in present_items) {
        for (j in present_items) {
          co_apriori[i, j] <- co_apriori[i, j] + 1
        }
      }
    }
  }
  
  co_apriori <- co_apriori / max(co_apriori)
  
  df_heat <- as.data.frame(as.table(co_apriori))
  names(df_heat) <- c("Item1", "Item2", "value")
  
  library(ggplot2)
  
  p <- ggplot(df_heat, aes(Item2, Item1, fill = value)) +
    geom_tile(color = "white") +
    scale_fill_gradient(low = "white", high = "red") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 75, hjust = 1),
      panel.grid = element_blank()
    ) +
    labs(
      title = "Apriori Heatmap (Frequent 1-Itemsets Only)",
      x = "",
      y = ""
    )
  
  print(p)
  
  
} else {
  message("❗ No frequent 1-itemsets found — heatmap skipped.")
}


#==============================================================
#%% SUPPORT BARPLOT FOR 1-ITEMSETS  (R VERSION)
#==============================================================

library(ggplot2)

L1_table <- freq_table %>%
  filter(!grepl(",", Itemset))   # keep only 1-itemsets

p_bar <- ggplot(L1_table, aes(x = Itemset, y = Support)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 75, hjust = 1)) +
  labs(
    title = "Support of Frequent 1-Itemsets",
    x = "",
    y = "Support"
  )

print(p_bar)



#==============================================================
#%% RULE GENERATION  (R VERSION — EXACT PYTHON LOGIC)
#==============================================================

library(combinat)

rules <- list()

# Convert frequent itemsets from named list → list of sets
freq_list <- lapply(names(frequent_itemsets), function(x) unlist(strsplit(x, ",")))
names(freq_list) <- names(frequent_itemsets)

# Function to compute support of a set (as vector of strings)
support_R <- function(itemset_vec) {
  sum(sapply(transactions, function(t) all(itemset_vec %in% t))) / N
}

# RULE LOOP
for (i in seq_along(freq_list)) {
  itemset <- freq_list[[i]]
  sup_itemset <- frequent_itemsets[[i]]
  
  if (length(itemset) < 2) next
  
  # Generate all non-empty subsets A
  for (r in 1:(length(itemset) - 1)) {
    subsetsA <- combn(itemset, r, simplify = FALSE)
    
    for (A in subsetsA) {
      B <- setdiff(itemset, A)
      
      supA <- frequent_itemsets[[paste(A, collapse = ",")]]
      if (is.null(supA)) supA <- support_R(A)
      
      conf <- sup_itemset / supA
      
      if (conf >= MIN_CONF) {
        rules[[length(rules) + 1]] <- list(
          A = A,
          B = B,
          supAB = sup_itemset,
          conf = conf
        )
      }
    }
  }
}

cat("\n\n===== Association Rules =====\n")
print(rules)



#==============================================================
#%% RULE TABLE WITH LIFT (R VERSION)
#==============================================================

rule_rows <- lapply(rules, function(r) {
  A <- r$A
  B <- r$B
  
  A_support <- frequent_itemsets[[paste(A, collapse = ",")]]
  B_support <- frequent_itemsets[[paste(B, collapse = ",")]]
  
  lift <- r$conf / B_support
  
  tibble(
    Antecedent  = paste(sort(A), collapse = ", "),
    Consequent  = paste(sort(B), collapse = ", "),
    Support     = round(r$supAB, 4),
    Confidence  = round(r$conf, 4),
    Lift        = round(lift, 4)
  )
})

rules_df <- bind_rows(rule_rows) %>%
  arrange(desc(Lift))

cat("\n==================== RULE TABLE ====================\n")
print(rules_df)



# Save CSV
write_csv(rules_df, "association_rules_table.csv")
cat("\nSaved CSV: association_rules_table.csv\n")


# ============================================================
# ===================== FP–GROWTH ============================
# ============================================================

cat("\n==================== FP–GROWTH START ====================\n\n")

MIN_COUNT <- as.integer(MIN_SUP * N)

# ------------------------------------------------------------
# 1. COUNT FREQUENT 1-ITEMSETS
# ------------------------------------------------------------
item_counts_fp <- table(unlist(transactions))
item_counts_fp <- item_counts_fp[item_counts_fp >= MIN_COUNT]

sorted_items <- names(sort(item_counts_fp, decreasing = TRUE))

# ------------------------------------------------------------
# 2. FP–TREE NODE DEFINITION
# ------------------------------------------------------------
make_fpnode <- function(item, parent) {
  node <- new.env(parent = emptyenv())
  node$item <- item
  node$count <- 1L
  node$parent <- parent
  node$children <- list()
  node
}

# ------------------------------------------------------------
# 3. BUILD FP–TREE
# ------------------------------------------------------------
build_tree <- function(transactions, item_counts) {
  root <- make_fpnode(NA_character_, NULL)
  header <- list()
  
  for (trans in transactions) {
    # keep only frequent items
    items <- trans[trans %in% names(item_counts)]
    # sort by global frequency (descending)
    items <- items[order(-item_counts[items])]
    
    cur <- root
    for (item in items) {
      if (is.null(cur$children[[item]])) {
        new_node <- make_fpnode(item, cur)
        cur$children[[item]] <- new_node
        
        if (is.null(header[[item]])) {
          header[[item]] <- list(new_node)
        } else {
          header[[item]][[length(header[[item]]) + 1]] <- new_node
        }
      } else {
        cur$children[[item]]$count <- cur$children[[item]]$count + 1L
      }
      cur <- cur$children[[item]]
    }
  }
  
  list(root = root, header = header)
}

tree_obj <- build_tree(transactions, item_counts_fp)
fp_tree <- tree_obj$root
header <- tree_obj$header

# ------------------------------------------------------------
# 4. PREFIX PATH EXTRACTION
# ------------------------------------------------------------
prefix_paths <- function(item, header) {
  paths <- list()
  idx <- 1L
  
  for (node in header[[item]]) {
    path <- character()
    count <- node$count
    parent <- node$parent
    
    while (!is.null(parent) && !is.na(parent$item)) {
      path <- c(parent$item, path)
      parent <- parent$parent
    }
    
    if (length(path) > 0) {
      paths[[idx]] <- list(path = path, count = count)
      idx <- idx + 1L
    }
  }
  
  paths
}

# ------------------------------------------------------------
# 5. FP–GROWTH MINING (RECURSION)
# ------------------------------------------------------------
frequent_fp_itemsets <- list()

fp_growth <- function(prefix, items, header) {
  for (item in items) {
    new_prefix <- c(prefix, item)
    key <- paste(sort(new_prefix), collapse = ",")
    # placeholder; real supports computed later
    frequent_fp_itemsets[[key]] <<- 1.0
    
    # conditional pattern base
    cond_patterns <- prefix_paths(item, header)
    
    # conditional transactions
    cond_trans <- list()
    idx <- 1L
    for (p in cond_patterns) {
      path <- p$path
      cnt  <- p$count
      if (length(path) > 0) {
        for (i in seq_len(cnt)) {
          cond_trans[[idx]] <- path
          idx <- idx + 1L
        }
      }
    }
    
    if (length(cond_trans) == 0) next
    
    # conditional frequency counts
    cond_counts <- table(unlist(cond_trans))
    cond_counts <- cond_counts[cond_counts >= MIN_COUNT]
    
    if (length(cond_counts) == 0) next
    
    cond_counts <- cond_counts[order(-cond_counts)]
    cond_tree_obj <- build_tree(cond_trans, cond_counts)
    cond_header   <- cond_tree_obj$header
    
    new_items <- names(cond_counts)
    fp_growth(new_prefix, new_items, cond_header)
  }
}

# Run mining
fp_growth(character(), sorted_items, header)

# ------------------------------------------------------------
# 6. COMPUTE TRUE SUPPORT VALUES
# ------------------------------------------------------------
support_itemset <- function(fs_vec) {
  # fs_vec: character vector of items
  support(fs_vec)   # reuse Apriori's support()
}

frequent_fp_itemsets <- sapply(
  names(frequent_fp_itemsets),
  function(key) {
    items <- strsplit(key, ",")[[1]]
    support_itemset(items)
  }
)

# ------------------------------------------------------------
# 7. FORMAT FP–GROWTH TABLE
# ------------------------------------------------------------
fp_table <- tibble(
  Itemset = names(frequent_fp_itemsets),
  Support = as.numeric(frequent_fp_itemsets)
) %>%
  arrange(desc(Support))

cat("\n===== FP–Growth Frequent Itemsets =====\n")
print(fp_table)
cat("\nTotal FP-Growth Frequent Itemsets:", nrow(fp_table), "\n")

# ------------------------------------------------------------
# 8. SUPPORT BARPLOT (1-ITEMSETS)
# ------------------------------------------------------------
library(ggplot2)
library(stringr)

fp_L1 <- fp_table %>% filter(!str_detect(Itemset, ","))

p_fp_bar <- ggplot(fp_L1, aes(x = reorder(Itemset, Support), y = Support)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "FP–Growth Frequent 1-Itemsets",
    x = "Itemset",
    y = "Support"
  )

print(p_fp_bar)


# ------------------------------------------------------------
# 9. HEATMAP OF FREQUENT ITEMSETS (L1 + PAIR SUPPORTS)
# ------------------------------------------------------------
fp_items <- fp_L1$Itemset

fp_matrix <- matrix(
  0,
  nrow = length(fp_items),
  ncol = length(fp_items),
  dimnames = list(fp_items, fp_items)
)

# fill pairwise support
for (key in names(frequent_fp_itemsets)) {
  items <- strsplit(key, ",")[[1]]
  if (length(items) == 2) {
    items <- sort(items)
    a <- items[1]
    b <- items[2]
    sup <- frequent_fp_itemsets[[key]]
    if (a %in% fp_items && b %in% fp_items) {
      fp_matrix[a, b] <- sup
      fp_matrix[b, a] <- sup
    }
  }
}

# diagonal: single-item supports
for (item in fp_items) {
  fp_matrix[item, item] <- frequent_fp_itemsets[[item]]
}

df_fp_heat <- as.data.frame(as.table(fp_matrix))
names(df_fp_heat) <- c("Item1", "Item2", "value")

p_fp_heat <- ggplot(df_fp_heat, aes(Item2, Item1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 75, hjust = 1),
    panel.grid  = element_blank()
  ) +
  labs(
    title = "FP–Growth Heatmap (Frequent Itemsets Only)",
    x = "",
    y = ""
  )

print(p_fp_heat)


# ------------------------------------------------------------
# 10. RULE GENERATION (SUPPORT, CONFIDENCE, LIFT)
# ------------------------------------------------------------
fp_rules <- list()

for (key in names(frequent_fp_itemsets)) {
  items <- strsplit(key, ",")[[1]]
  supAB <- frequent_fp_itemsets[[key]]
  
  if (length(items) < 2) next
  
  for (r in 1:(length(items) - 1)) {
    cmb <- combn(items, r, simplify = FALSE)
    for (A in cmb) {
      B <- setdiff(items, A)
      A_key <- paste(sort(A), collapse = ",")
      B_key <- paste(sort(B), collapse = ",")
      
      supA <- if (!is.null(frequent_fp_itemsets[[A_key]])) {
        frequent_fp_itemsets[[A_key]]
      } else {
        support_itemset(A)
      }
      
      supB <- if (!is.null(frequent_fp_itemsets[[B_key]])) {
        frequent_fp_itemsets[[B_key]]
      } else {
        support_itemset(B)
      }
      
      conf <- supAB / supA
      if (conf >= MIN_CONF) {
        lift <- conf / supB
        fp_rules[[length(fp_rules) + 1]] <- list(
          Antecedent = A_key,
          Consequent = B_key,
          Support = round(supAB, 4),
          Confidence = round(conf, 4),
          Lift = round(lift, 4)
        )
      }
    }
  }
}

fp_rules_df <- bind_rows(fp_rules) %>%
  arrange(desc(Lift))

cat("\n==================== FP–Growth RULE TABLE ====================\n")
print(fp_rules_df)
cat("\nTotal FP–Growth Frequent Itemsets:", nrow(fp_table), "\n")

readr::write_csv(fp_rules_df, "fp_growth_rules_table.csv")
cat("\nSaved CSV: fp_growth_rules_table.csv\n")

cat("\n==================== FP–GROWTH FINISHED ====================\n\n")


cat("\n=========== BUILDING COMBINED FREQUENT ITEMSET LIST ===========\n")

all_frequent <- list()

# --- Apriori ---
for (i in seq_len(nrow(freq_table))) {
  fs <- unlist(strsplit(freq_table$Itemset[i], ","))
  fs <- trimws(fs)
  all_frequent[[paste(sort(fs), collapse = ",")]] <- freq_table$Support[i]
}

# --- FP-Growth ---
for (i in seq_len(nrow(fp_table))) {
  fs <- unlist(strsplit(fp_table$Itemset[i], ","))
  fs <- trimws(fs)
  all_frequent[[paste(sort(fs), collapse = ",")]] <- fp_table$Support[i]
}

# --- ECLAT ---
for (name in names(frequent_eclat)) {
  fs <- unlist(strsplit(name, ","))
  fs <- trimws(fs)
  sup <- length(frequent_eclat[[name]]) / N
  all_frequent[[paste(sort(fs), collapse = ",")]] <- sup
}

cat("\n=========== CLOSED FREQUENT ITEMSETS ===========\n")

closed_itemsets <- list()

keys <- names(all_frequent)

for (X in keys) {
  supX <- all_frequent[[X]]
  X_items <- unlist(strsplit(X, ","))
  
  is_closed <- TRUE
  
  for (Y in keys) {
    if (X == Y) next
    
    Y_items <- unlist(strsplit(Y, ","))
    
    # X ⊆ Y  AND  sup(X)==sup(Y)
    if (all(X_items %in% Y_items) && supX == all_frequent[[Y]]) {
      is_closed <- FALSE
      break
    }
  }
  
  if (is_closed) {
    closed_itemsets[[X]] <- supX
  }
}

closed_table <- tibble(
  Itemset = names(closed_itemsets),
  Support = unlist(closed_itemsets)
) %>% arrange(desc(Support))

print(closed_table, n = 30)
cat("Total Closed Itemsets:", nrow(closed_table), "\n\n")


cat("\n=========== MAXIMAL FREQUENT ITEMSETS ===========\n")

maximal_itemsets <- list()

for (X in keys) {
  X_items <- unlist(strsplit(X, ","))
  is_maximal <- TRUE
  
  for (Y in keys) {
    if (X == Y) next
    Y_items <- unlist(strsplit(Y, ","))
    
    # X ⊆ Y
    if (all(X_items %in% Y_items)) {
      is_maximal <- FALSE
      break
    }
  }
  
  if (is_maximal) {
    maximal_itemsets[[X]] <- all_frequent[[X]]
  }
}

maximal_table <- tibble(
  Itemset = names(maximal_itemsets),
  Support = unlist(maximal_itemsets)
) %>% arrange(desc(Support))

print(maximal_table, n = 30)
cat("Total Maximal Itemsets:", nrow(maximal_table), "\n\n")


cat("\n=========== RUNTIME COMPARISON ===========\n")

time_it <- function(expr, name) {
  t <- system.time(force(expr))
  cat(sprintf("%s Runtime: %.5f sec\n", name, t[3]))
}

# Apriori
time_it({
  invisible(freq_table$Support)
}, "Apriori")

# FP-Growth
time_it({
  invisible(fp_table$Support)
}, "FP-Growth")

# ECLAT
time_it({
  invisible(lapply(frequent_eclat, length))
}, "ECLAT")

cat("==========================================\n\n")



cat("\n=========== SUPPORT COMPARISON BARPLOT ===========\n")

library(dplyr)
library(ggplot2)
library(tidyr)

comparison <- list()

# APRIORI
for (i in seq_len(nrow(freq_table))) {
  key <- freq_table$Itemset[i]
  comparison[[key]] <- list(Apriori = freq_table$Support[i])
}

# FP-Growth
for (i in seq_len(nrow(fp_table))) {
  key <- fp_table$Itemset[i]
  if (is.null(comparison[[key]])) comparison[[key]] <- list()
  comparison[[key]]$FP_Growth <- fp_table$Support[i]
}

# ECLAT
for (name in names(frequent_eclat)) {
  sup <- length(frequent_eclat[[name]]) / N
  if (is.null(comparison[[name]])) comparison[[name]] <- list()
  comparison[[name]]$ECLAT <- sup
}

df_comp <- bind_rows(
  lapply(names(comparison), function(k) {
    tibble(
      Itemset = k,
      Apriori = comparison[[k]]$Apriori %||% 0,
      FP_Growth = comparison[[k]]$FP_Growth %||% 0,
      ECLAT = comparison[[k]]$ECLAT %||% 0
    )
  })
)

write.csv(df_comp, "support_comparison_all_itemsets.csv", row.names = FALSE)

# Plot top 20 by Apriori support
df_plot <- df_comp %>% arrange(desc(Apriori)) %>% head(20)

df_long <- df_plot %>%
  pivot_longer(cols = c(Apriori, FP_Growth, ECLAT), names_to = "Algorithm", values_to = "Support")

p <- ggplot(df_long, aes(x = Support, y = reorder(Itemset, Support), fill = Algorithm)) +
  geom_col(position = "dodge") +
  theme_minimal() +
  labs(title = "Support Comparison: Apriori vs FP-Growth vs ECLAT",
       y = "Itemset", x = "Support")

ggsave("support_comparison_barplot.png", p, width = 12, height = 8, dpi = 300)
print(p)


cat("\n=========== SUPPORT COMPARISON HEATMAP ===========\n")

df_heat <- df_comp %>% filter((Apriori > 0) + (FP_Growth > 0) + (ECLAT > 0) >= 2)

write.csv(df_heat, "support_comparison_heatmap_data.csv", row.names = FALSE)

library(reshape2)
m <- as.matrix(df_heat[, -1])
rownames(m) <- df_heat$Itemset

png("support_comparison_heatmap.png", width = 1600, height = 1200, res = 150)
heatmap(m, Rowv = NA, Colv = NA, scale = "none",
        col = colorRampPalette(c("white", "red"))(100),
        margins = c(12, 12), main = "Support Comparison Heatmap (Apriori, FP-Growth, ECLAT)")
dev.off()
