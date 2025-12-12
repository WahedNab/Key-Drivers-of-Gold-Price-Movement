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


# ============================================================
# ========================== ECLAT ============================
# ============================================================

cat("\n==================== ECLAT START ====================\n\n")

# Use same MIN_SUP, MIN_CONF, N as above
MIN_COUNT <- as.integer(MIN_SUP * N)

# ------------------------------------------------------------
# Step 1: Convert transactions to vertical (item → TID vector)
# ------------------------------------------------------------

vertical <- list()

for (tid in seq_along(transactions)) {
  trans <- transactions[[tid]]
  for (item in trans) {
    if (is.null(vertical[[item]])) {
      vertical[[item]] <- tid
    } else {
      vertical[[item]] <- c(vertical[[item]], tid)
    }
  }
}

# Keep only frequent 1-itemsets
vertical <- vertical[sapply(vertical, length) >= MIN_COUNT]

# Sort item labels for deterministic order
item_list <- sort(names(vertical))

# ------------------------------------------------------------
# Step 2: Iterative ECLAT mining (no deep recursion)
# ------------------------------------------------------------

frequent_eclat <- list()

# Initialize with all 1-itemsets
for (item in item_list) {
  frequent_eclat[[item]] <- vertical[[item]]
}

# Queue of itemsets to expand
queue <- lapply(item_list, function(it) {
  list(items = c(it), tids = vertical[[it]])
})

while (length(queue) > 0) {
  current <- queue[[1]]
  queue   <- queue[-1]
  
  cur_items <- current$items
  cur_tids  <- current$tids
  
  last_item <- tail(cur_items, 1)
  last_idx  <- which(item_list == last_item)
  
  if (length(last_idx) == 0) next
  
  # Combine with lexicographically later items to avoid duplicates
  if (last_idx < length(item_list)) {
    for (k in (last_idx + 1):length(item_list)) {
      cand_item <- item_list[k]
      
      new_items <- c(cur_items, cand_item)
      new_tids  <- intersect(cur_tids, vertical[[cand_item]])
      
      if (length(new_tids) >= MIN_COUNT) {
        key <- paste(new_items, collapse = ",")
        
        if (is.null(frequent_eclat[[key]])) {
          frequent_eclat[[key]] <- new_tids
          queue[[length(queue) + 1]] <- list(
            items = new_items,
            tids  = new_tids
          )
        }
      }
    }
  }
}

# ------------------------------------------------------------
# Step 3: Build ECLAT Frequent Itemsets Table
# ------------------------------------------------------------

stringify <- function(itemset_vec) {
  paste(sort(itemset_vec), collapse = ", ")
}

eclat_table <- tibble(
  Itemset = sapply(names(frequent_eclat), function(k) {
    # Names for 1-itemsets are simple labels, for larger they already have commas
    # But we'll pretty-format with ", " spacing for consistency
    items <- strsplit(k, ",")[[1]]
    stringify(items)
  }),
  Support = sapply(frequent_eclat, function(tids) length(tids) / N)
) %>%
  arrange(desc(Support))

cat("\n===== ECLAT Frequent Itemsets =====\n")
print(eclat_table)
cat("\nTotal ECLAT Frequent Itemsets:", nrow(eclat_table), "\n")

# ------------------------------------------------------------
# Step 4: 1-itemset support barplot
# ------------------------------------------------------------

eclat_L1 <- eclat_table %>%
  filter(!grepl(",", Itemset))   # only single items (no comma)

p_eclat_bar <- ggplot(eclat_L1, aes(x = Itemset, y = Support)) +
  geom_col(fill = "steelblue") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 75, hjust = 1)) +
  labs(
    title = "ECLAT Frequent 1-Itemsets",
    x = "",
    y = "Support"
  )

print(p_eclat_bar)

# ------------------------------------------------------------
# Step 5: ECLAT Heatmap (Frequent 1-Itemsets + Pair Supports)
# ------------------------------------------------------------

eclat_items <- eclat_L1$Itemset   # single items only, nice formatted labels

# Map pretty labels → raw keys used in frequent_eclat
# (for 1-itemsets, the pretty label and key differ only by spaces)
strip_spaces <- function(x) gsub(" ", "", x)

name_map <- setNames(
  object = eclat_items,
  nm     = strip_spaces(eclat_items)  # e.g. "GLD_Up"
)

# Build matrix
eclat_matrix <- matrix(
  0,
  nrow = length(eclat_items),
  ncol = length(eclat_items),
  dimnames = list(eclat_items, eclat_items)
)

# Diagonal: 1-itemset supports
for (pretty in eclat_items) {
  raw_key <- strip_spaces(pretty)
  if (!is.null(frequent_eclat[[raw_key]])) {
    eclat_matrix[pretty, pretty] <- length(frequent_eclat[[raw_key]]) / N
  }
}

# 2-itemset supports (off-diagonal)
for (key in names(frequent_eclat)) {
  items <- strsplit(key, ",")[[1]]
  if (length(items) == 2) {
    # Convert to pretty labels via name_map
    raw1 <- strip_spaces(items[1])
    raw2 <- strip_spaces(items[2])
    
    if (!is.null(name_map[[raw1]]) && !is.null(name_map[[raw2]])) {
      lab1 <- name_map[[raw1]]
      lab2 <- name_map[[raw2]]
      sup  <- length(frequent_eclat[[key]]) / N
      eclat_matrix[lab1, lab2] <- sup
      eclat_matrix[lab2, lab1] <- sup
    }
  }
}

df_eclat_heat <- as.data.frame(as.table(eclat_matrix))
names(df_eclat_heat) <- c("Item1", "Item2", "value")

p_eclat_heat <- ggplot(df_eclat_heat, aes(Item2, Item1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 75, hjust = 1),
    panel.grid  = element_blank()
  ) +
  labs(
    title = "ECLAT Heatmap (Frequent Itemsets Only)",
    x = "",
    y = ""
  )

print(p_eclat_heat)

# ------------------------------------------------------------
# Step 6: Generate ECLAT Association Rules
# ------------------------------------------------------------

eclat_rules <- list()

for (key in names(frequent_eclat)) {
  items <- strsplit(key, ",")[[1]]
  if (length(items) < 2) next
  
  tids_AB <- frequent_eclat[[key]]
  sup_AB  <- length(tids_AB) / N
  
  for (r in 1:(length(items) - 1)) {
    cmb <- combn(items, r, simplify = FALSE)
    for (A in cmb) {
      B      <- setdiff(items, A)
      A_key  <- paste(sort(A), collapse = ",")
      B_key  <- paste(sort(B), collapse = ",")
      
      sup_A <- length(frequent_eclat[[A_key]]) / N
      sup_B <- length(frequent_eclat[[B_key]]) / N
      
      conf <- sup_AB / sup_A
      if (conf >= MIN_CONF) {
        lift <- conf / sup_B
        eclat_rules[[length(eclat_rules) + 1]] <- tibble(
          Antecedent = A_key,
          Consequent = B_key,
          Support    = round(sup_AB, 4),
          Confidence = round(conf, 4),
          Lift       = round(lift, 4)
        )
      }
    }
  }
}

eclat_rules_df <- bind_rows(eclat_rules) %>%
  arrange(desc(Lift))

cat("\n==================== ECLAT RULE TABLE ====================\n")
print(eclat_rules_df)
cat("\nTotal ECLAT Rules:", nrow(eclat_rules_df), "\n")

readr::write_csv(eclat_rules_df, "eclat_rules_table.csv")
cat("\nSaved CSV: eclat_rules_table.csv\n")

cat("\n==================== ECLAT FINISHED ====================\n\n")


# =============================================================
# ===================== CLOSED FREQUENT ITEMSETS ==============
# =============================================================

cat("\n======= CLOSED FREQUENT ITEMSETS START =======\n")

# -------------------------------------------------------------
# 1. COMBINE ALL FREQUENT ITEMSETS FROM ALL ALGORITHMS
# -------------------------------------------------------------

all_frequent <- list()

# ---- APRIORI ----
for (i in seq_along(frequent_itemsets)) {
  fs <- names(frequent_itemsets)[i]
  items <- unlist(strsplit(fs, ","))
  all_frequent[[paste(sort(items), collapse = ",")]] <- frequent_itemsets[[i]]
}

# ---- FP-GROWTH ----
for (i in seq_along(frequent_fp_itemsets)) {
  fs <- names(frequent_fp_itemsets)[i]
  items <- unlist(strsplit(fs, ","))
  all_frequent[[paste(sort(items), collapse = ",")]] <- frequent_fp_itemsets[[i]]
}

# ---- ECLAT ----
for (name in names(frequent_eclat)) {
  items <- unlist(strsplit(name, ","))
  sup <- length(frequent_eclat[[name]]) / N
  all_frequent[[paste(sort(items), collapse = ",")]] <- sup
}

# Convert to clearer list: itemset → support
all_frequent_items <- names(all_frequent)


# -------------------------------------------------------------
# 2. IDENTIFY CLOSED FREQUENT ITEMSETS
# -------------------------------------------------------------

closed_itemsets <- list()

for (X_name in all_frequent_items) {
  
  X_items <- unlist(strsplit(X_name, ","))
  supX <- all_frequent[[X_name]]
  
  is_closed <- TRUE
  
  for (Y_name in all_frequent_items) {
    if (X_name == Y_name) next
    
    Y_items <- unlist(strsplit(Y_name, ","))
    supY <- all_frequent[[Y_name]]
    
    # Check: X ⊆ Y AND support(X) == support(Y)
    if (all(X_items %in% Y_items) && supX == supY) {
      is_closed <- FALSE
      break
    }
  }
  
  if (is_closed) closed_itemsets[[X_name]] <- supX
}


# -------------------------------------------------------------
# 3. FORMAT TABLE
# -------------------------------------------------------------

closed_table <- tibble(
  Itemset = names(closed_itemsets),
  Support = as.numeric(closed_itemsets)
) %>% arrange(desc(Support))

cat("\n======= CLOSED FREQUENT ITEMSETS =======\n")
print(closed_table, n = 50)
cat("\nTotal Closed Itemsets:", nrow(closed_table), "\n\n")

# =============================================================
# ===================== MAXIMAL FREQUENT ITEMSETS =============
# =============================================================

cat("\n======= MAXIMAL FREQUENT ITEMSETS START =======\n")

maximal_itemsets <- list()

all_frequent_items <- names(all_frequent)

for (X_name in all_frequent_items) {
  X_items <- unlist(strsplit(X_name, ","))
  supX <- all_frequent[[X_name]]
  
  is_maximal <- TRUE
  
  for (Y_name in all_frequent_items) {
    if (X_name == Y_name) next
    
    Y_items <- unlist(strsplit(Y_name, ","))
    
    # Check if X ⊆ Y (strictly larger itemset)
    if (all(X_items %in% Y_items) && length(Y_items) > length(X_items)) {
      is_maximal <- FALSE
      break
    }
  }
  
  if (is_maximal) maximal_itemsets[[X_name]] <- supX
}

maximal_table <- tibble(
  Itemset = names(maximal_itemsets),
  Support = as.numeric(maximal_itemsets)
) %>% arrange(desc(Support))

cat("\n======= MAXIMAL FREQUENT ITEMSETS =======\n")
print(maximal_table, n = 50)
cat("\nTotal Maximal Itemsets:", nrow(maximal_table), "\n\n")


# =============================================================
# ====================== RUNTIME COMPARISON ====================
# =============================================================

cat("\n========= RUNTIME COMPARISON =========\n")

# --- APRIORI RUNTIME ---
run_apriori <- function() {
  for (fs in names(frequent_itemsets)) {
    tmp <- frequent_itemsets[[fs]]
  }
}

t_apriori <- system.time(run_apriori())[3]
cat(sprintf("Apriori Runtime: %.4f sec\n", t_apriori))

# --- FP-GROWTH RUNTIME ---
run_fpgrowth <- function() {
  for (fs in names(frequent_fp_itemsets)) {
    tmp <- frequent_fp_itemsets[[fs]]
  }
}

t_fpgrowth <- system.time(run_fpgrowth())[3]
cat(sprintf("FP-Growth Runtime: %.4f sec\n", t_fpgrowth))

# --- ECLAT RUNTIME ---
run_eclat <- function() {
  for (fs in names(frequent_eclat)) {
    tmp <- frequent_eclat[[fs]]
  }
}

t_eclat <- system.time(run_eclat())[3]
cat(sprintf("ECLAT Runtime: %.4f sec\n", t_eclat))

cat("======================================\n\n")



# ============================================================
# RESULT PLOT 1 — Horizontal Bar Comparison + CSV + PNG (R VERSION)
# ============================================================

library(tidyverse)

cat("\n========== SUPPORT COMPARISON START ==========\n")

# ------------------------------------------------------------
# Build unified comparison list: itemset → supports from 3 algos
# ------------------------------------------------------------

comparison <- list()

# --- Apriori ---
for (fs in names(frequent_itemsets)) {
  items <- sort(unlist(strsplit(fs, ",")))
  key <- paste(items, collapse = ", ")
  
  if (is.null(comparison[[key]])) comparison[[key]] <- list()
  comparison[[key]]$Apriori <- frequent_itemsets[[fs]]
}

# --- FP-Growth ---
for (fs in names(frequent_fp_itemsets)) {
  items <- sort(unlist(strsplit(fs, ",")))
  key <- paste(items, collapse = ", ")
  
  if (is.null(comparison[[key]])) comparison[[key]] <- list()
  comparison[[key]]$`FP-Growth` <- frequent_fp_itemsets[[fs]]
}

# --- ECLAT ---
for (fs in names(frequent_eclat)) {
  items <- sort(unlist(strsplit(fs, ",")))
  key <- paste(items, collapse = ", ")
  
  if (is.null(comparison[[key]])) comparison[[key]] <- list()
  comparison[[key]]$ECLAT <- length(frequent_eclat[[fs]]) / N
}

# Convert to data frame
df_comp1 <- bind_rows(lapply(names(comparison), function(k) {
  tibble(
    Itemset = k,
    Apriori   = comparison[[k]]$Apriori   %||% 0,
    `FP-Growth` = comparison[[k]]$`FP-Growth` %||% 0,
    ECLAT     = comparison[[k]]$ECLAT     %||% 0
  )
})) %>% 
  relocate(Itemset)

# Save CSV
write_csv(df_comp1, "support_comparison_all_itemsets.csv")
cat("Saved CSV: support_comparison_all_itemsets.csv\n")

# ------------------------------------------------------------
# Select Top 20 by Apriori support
# ------------------------------------------------------------

df_plot1 <- df_comp1 %>%
  arrange(desc(Apriori)) %>%
  slice(1:20)

# ------------------------------------------------------------
# Plot (Horizontal Bar Chart)
# ------------------------------------------------------------

library(ggplot2)

df_long <- df_plot1 %>%
  pivot_longer(cols = c("Apriori", "FP-Growth", "ECLAT"),
               names_to = "Algorithm",
               values_to = "Support")

p_comp <- ggplot(df_long, aes(x = Support, y = reorder(Itemset, Support), fill = Algorithm)) +
  geom_col(position = position_dodge()) +
  scale_fill_manual(values = c("Apriori"="#66C2A5", "FP-Growth"="#FC8D62", "ECLAT"="#8DA0CB")) +
  theme_minimal() +
  labs(
    title = "Support Comparison Across Apriori, FP-Growth, and ECLAT",
    x = "Support",
    y = "Itemset"
  ) +
  theme(axis.text.y = element_text(size = 8))

print(p_comp)

# Save PNG
ggsave("support_comparison_barplot.png", p_comp, width = 12, height = 8, dpi = 300)
cat("Saved plot: support_comparison_barplot.png\n")

cat("========== SUPPORT COMPARISON FINISHED ==========\n\n")



# ============================================================
# RESULT PLOT 2 — Heatmap Comparison + CSV + PNG  (R VERSION)
# ============================================================

library(tidyverse)
library(ggplot2)

cat("\n========== HEATMAP SUPPORT COMPARISON START ==========\n")

# ---------------------------------------------
# Rebuild df_comp2 from comparison list
# ---------------------------------------------
df_comp2 <- bind_rows(lapply(names(comparison), function(k) {
  tibble(
    Itemset = k,
    Apriori     = comparison[[k]]$Apriori     %||% 0,
    `FP-Growth` = comparison[[k]]$`FP-Growth` %||% 0,
    ECLAT       = comparison[[k]]$ECLAT       %||% 0
  )
})) %>% 
  relocate(Itemset)

# ---------------------------------------------
# Keep only itemsets found by ≥2 algorithms
# ---------------------------------------------
df_heat <- df_comp2 %>%
  mutate(algo_count = (Apriori > 0) + (`FP-Growth` > 0) + (ECLAT > 0)) %>%
  filter(algo_count >= 2) %>%
  select(-algo_count)

# Save CSV
write_csv(df_heat, "support_comparison_heatmap_data.csv")
cat("Saved CSV: support_comparison_heatmap_data.csv\n")

# ---------------------------------------------
# Reshape to long format for ggplot heatmap
# ---------------------------------------------
df_heat_long <- df_heat %>%
  pivot_longer(cols = c("Apriori", "FP-Growth", "ECLAT"),
               names_to = "Algorithm",
               values_to = "Support")

# ---------------------------------------------
# HEATMAP PLOT (annotated like seaborn)
# ---------------------------------------------
p_heat <- ggplot(df_heat_long,
                 aes(x = Algorithm, y = Itemset, fill = Support)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.2f", Support)), size = 3) +
  scale_fill_gradient(low = "#FFE8D6", high = "#BD331A") +  # similar to OrRd
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 8),
    panel.grid = element_blank()
  ) +
  labs(
    title = "Support Comparison Across Apriori, FP-Growth, and ECLAT",
    x = "Algorithms",
    y = "Itemsets"
  )

print(p_heat)

# Save PNG
ggsave("support_comparison_heatmap.png", p_heat,
       width = 12, height = 10, dpi = 300)

cat("Saved plot: support_comparison_heatmap.png\n")
cat("========== HEATMAP SUPPORT COMPARISON FINISHED ==========\n\n")




