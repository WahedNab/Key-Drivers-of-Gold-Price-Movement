#%% Step 1: Import Libraries
library(tidyverse)
library(ggplot2)
library(cluster)
library(factoextra)
library(NbClust)
library(stats)
library(ggthemes)

library(FNN)        # for k-NN distance (DBSCAN prep)
library(dbscan)     # DBSCAN
library(mclust)     # extra clustering utilities

#%% Step 2: Load and Preprocess Data
data_path <- "/Users/sadafmoghisi/Desktop/PhD/Terms/Term3/Data Mining:Discovery (523)/HW/Final/gold_price_dataset_preprocessed.csv"

gold_data <- read_csv(data_path, show_col_types = FALSE)

# Select numerical features for clustering
numerical_features <- c("norm_SPX", "norm_USO", "norm_SLV", "norm_EUR.USD")
data <- gold_data[, numerical_features]

# Standardize the data
scaled_data <- scale(data)

#%% Step 3: PCA Dimensionality Reduction (2 Components)
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)
pca_data <- pca_result$x[, 1:2]

#%% Step 4: Elbow Method for Optimal K (K-Means)
inertia <- c()
silhouette_scores <- c()
range_clusters <- 2:10

for (k in range_clusters) {
  km <- kmeans(scaled_data, centers = k, nstart = 20)
  inertia <- c(inertia, km$tot.withinss)
  
  ss <- silhouette(km$cluster, dist(scaled_data))
  silhouette_scores <- c(silhouette_scores, mean(ss[, 3]))
}

#%% Plot Elbow Curve
df_elbow <- tibble(
  k = range_clusters,
  inertia = inertia
)

p_elbow <- ggplot(df_elbow, aes(x = k, y = inertia)) +
  geom_line(color = "navy", linewidth = 1.2) +
  geom_point(color = "navy", size = 3) +
  geom_vline(xintercept = 4, linetype = "dashed", color = "red", linewidth = 1) +
  annotate("text", x = 4.2, y = min(inertia),
           label = "Optimal = 4", color = "red", hjust = 0, size = 4) +
  theme_minimal() +
  ggtitle("Elbow Method for Optimal Clusters") +
  xlab("Number of Clusters") +
  ylab("Inertia (Total Within-Cluster Sum of Squares)") +
  theme(
    plot.title = element_text(face = "bold", size = 14)
  )

print(p_elbow)

#%% Step 5: KMeans Clustering and Evaluation

library(clusterSim)

optimal_clusters <- 4  # same as Python

set.seed(42)
kmeans_result <- kmeans(scaled_data, centers = optimal_clusters, nstart = 25)

# --- KEEP original labels for evaluation ---
kmeans_labels_eval <- kmeans_result$cluster   # 1,2,3,4

# ---- Evaluation (correct) ----
sil <- silhouette(kmeans_labels_eval, dist(scaled_data))
kmeans_silhouette <- mean(sil[, 3])

kmeans_davies_bouldin <- index.DB(scaled_data, kmeans_labels_eval)$DB

cat("\n===== KMeans Evaluation =====\n")
cat("Number of Clusters:", optimal_clusters, "\n")
cat("Silhouette Score:", round(kmeans_silhouette, 4), "\n")
cat("Davies-Bouldin Index:", round(kmeans_davies_bouldin, 4), "\n\n")

# --- Flip PCA axis to match Python orientation ---
pca_data[,1] <- -pca_data[,1]

# --- Convert labels ONLY for plotting (0–3 like Python) ---
kmeans_labels_plot <- kmeans_labels_eval - 1

# ---- Visualization ----
df_kmeans_plot <- tibble(
  PC1 = pca_data[, 1],
  PC2 = pca_data[, 2],
  cluster = factor(kmeans_labels_plot)
)

eval_text <- paste0(
  "Number of Clusters: ", optimal_clusters, "\n",
  "Silhouette Score: ", sprintf("%.4f", kmeans_silhouette), "\n",
  "Davies-Bouldin Index: ", sprintf("%.4f", kmeans_davies_bouldin)
)

p_kmeans <- ggplot(df_kmeans_plot, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(
    values = c(
      "0" = "#1f77b4",  # blue
      "1" = "#ff7f0e",  # orange
      "2" = "#2ca02c",  # red 
      "3" = "#d62728"   
    )
  ) +
  ggtitle(paste("KMeans Clustering (Nclusters =", optimal_clusters, ")")) +
  xlab("PCA Component 1") +
  ylab("PCA Component 2") +
  theme_minimal(base_size = 14) +
  theme(
    legend.title = element_blank(),
    plot.title = element_text(face = "bold")
  ) +
  annotate(
    "text", x = min(df_kmeans_plot$PC1), y = max(df_kmeans_plot$PC2),
    label = eval_text, hjust = 0, vjust = 1,
    size = 4, color = "black"
  )

print(p_kmeans)


#%% Step 6: DBSCAN – 5-NN Distance Plot

library(FNN)   # for get.knn()

# Step 6.1: Compute 5-NN distances
knn_res <- get.knn(scaled_data, k = 5)

# the 5th nearest neighbor distance for each point
distances <- knn_res$nn.dist[, 5]

# sort distances like in Python
distances_sorted <- sort(distances)

# Step 6.2: Find elbow point (largest jump in distance)
diffs <- diff(distances_sorted)
elbow_index <- which.max(diffs)

cat("Elbow Index (largest slope change):", elbow_index, "\n")
cat("Estimated eps value:", round(distances_sorted[elbow_index], 4), "\n\n")

# Prepare data frame for plotting
df_knn <- tibble(
  index = seq_along(distances_sorted),
  distance = distances_sorted
)

# Step 6.3: Plot
p_knn <- ggplot(df_knn, aes(x = index, y = distance)) +
  geom_line(color = "blue", linewidth = 1.1) +
  geom_vline(xintercept = elbow_index, color = "red", linetype = "dashed") +
  annotate("text",
           x = elbow_index,
           y = max(distances_sorted),
           label = paste("Elbow =", elbow_index),
           color = "red",
           hjust = -0.1,
           size = 4) +
  theme_minimal(base_size = 14) +
  ggtitle("5-NN Distance Graph") +
  xlab("Points Sorted by Distance") +
  ylab("5-NN Distance") +
  theme(
    plot.title = element_text(face = "bold")
  )

print(p_knn)


# ============================================================
# Step 6: DBSCAN Hyperparameter Tuning Heatmap  (R VERSION)
# ============================================================

library(dbscan)
library(ggplot2)
library(reshape2)
library(scales)
library(RColorBrewer)

# --- Define hyperparameter ranges ---
eps_values <- seq(0.1, 0.5, length.out = 10)
min_samples_values <- 3:8

cluster_counts <- matrix(0,
                         nrow = length(eps_values),
                         ncol = length(min_samples_values))

# --- Compute cluster counts ---
for (i in seq_along(eps_values)) {
  for (j in seq_along(min_samples_values)) {
    model <- dbscan(scaled_data, eps = eps_values[i], minPts = min_samples_values[j])
    labels <- model$cluster
    n_clusters <- length(unique(labels[labels != 0]))
    cluster_counts[i, j] <- n_clusters
  }
}

df_heat <- melt(cluster_counts)
names(df_heat) <- c("eps_index", "minPts_index", "clusters")

df_heat$eps <- eps_values[df_heat$eps_index]
df_heat$minPts <- min_samples_values[df_heat$minPts_index]

# --- Heatmap ---
p_dbscan_heatmap <- ggplot(df_heat, aes(
  x = factor(minPts),
  y = factor(sprintf("%.2f", eps)),
  fill = clusters)) +
  
  geom_tile(color = "white") +
  geom_text(aes(label = clusters), size = 3) +
  
  # >>> PYTHON-MATCHING COLORBAR <<<
  scale_fill_gradientn(
    colors = rev(colorRampPalette(brewer.pal(11, "RdBu"))(256)),
    trans = "log10",
    breaks = c(1, 10, 100),
    labels = scales::math_format(10^.x),
    name = "Number of Clusters\n(Log Scale)"
  ) +
  
  labs(
    title = "DBSCAN Hyperparameter Tuning Heatmap (Log Scale)",
    x = "MinPts",
    y = "Epsilon (eps)"
  ) +
  
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12)
  )

print(p_dbscan_heatmap)


# ============================================================
# Step 7: Silhouette Scores for DBSCAN Hyperparameter Tuning
# ============================================================

library(cluster)     # for silhouette()
library(RColorBrewer)
library(ggplot2)
library(reshape2)

eps_values <- seq(0.1, 0.5, length.out = 10)   # same as python
min_samples_values <- 3:8                      # same MinPts range

silhouette_scores <- matrix(NA,
                            nrow = length(eps_values),
                            ncol = length(min_samples_values))

# ---- Compute Silhouette Score Matrix ----
for (i in seq_along(eps_values)) {
  eps <- eps_values[i]
  
  for (j in seq_along(min_samples_values)) {
    minPts <- min_samples_values[j]
    
    db <- dbscan(scaled_data, eps = eps, minPts = minPts)
    labels <- db$cluster
    
    # Exclude noise (label == 0 in R dbscan)
    keep_idx <- labels != 0
    filtered_data <- scaled_data[keep_idx, , drop = FALSE]
    filtered_labels <- labels[keep_idx]
    
    # must have at least 2 clusters
    if (length(unique(filtered_labels)) > 1) {
      sil <- silhouette(filtered_labels, dist(filtered_data))
      silhouette_scores[i, j] <- mean(sil[, 3])
    } else {
      silhouette_scores[i, j] <- NA
    }
  }
}

# ---- Convert for ggplot ----
df_sil <- melt(silhouette_scores)
names(df_sil) <- c("eps_index", "minPts_index", "silhouette")

df_sil$eps <- eps_values[df_sil$eps_index]
df_sil$minPts <- min_samples_values[df_sil$minPts_index]

# ---- Heatmap with annotations ----
p_silhouette_heatmap <- ggplot(df_sil, aes(
  x = factor(minPts),
  y = factor(sprintf("%.2f", eps)),
  fill = silhouette
)) +
  geom_tile(color = "white") +
  geom_text(
    aes(label = ifelse(is.na(silhouette), "", sprintf("%.2f", silhouette))),
    size = 3,
    color = "black"
  ) +
  scale_fill_gradientn(
    colors = colorRampPalette(brewer.pal(11, "RdBu"))(256),
    name = "Silhouette\nScore",
    limits = c(min(df_sil$silhouette, na.rm = TRUE),
               max(df_sil$silhouette, na.rm = TRUE))
  ) +
  labs(
    title = "Silhouette Scores for DBSCAN Hyperparameter Tuning",
    x = "MinPts",
    y = "Epsilon (eps)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    axis.text = element_text(size = 12)
  )

print(p_silhouette_heatmap)

# ============================================================
# STEP 8 — DBSCAN WITH CORRECT EVALUATION (DBI = 0.94)
# ============================================================

best_eps <- 0.32
best_minPts <- 3

db_best <- dbscan(scaled_data, eps = best_eps, minPts = best_minPts)
labels_raw <- db_best$cluster   # 0 = noise, 1..K clusters

# --- Keep original labels for evaluation ---
filtered_idx <- labels_raw != 0
filtered_data <- scaled_data[filtered_idx, , drop = FALSE]
filtered_labels_eval <- labels_raw[filtered_idx]   # 1,2,3...

# --- Correct evaluation (DO NOT shift labels!) ---
if (length(unique(filtered_labels_eval)) > 1) {
  sil <- silhouette(filtered_labels_eval, dist(filtered_data))
  silhouette_score <- mean(sil[, 3])
  
  library(clusterSim)
  davies_bouldin <- index.DB(filtered_data, filtered_labels_eval)$DB
  
  n_clusters <- length(unique(filtered_labels_eval))
} else {
  silhouette_score <- NA
  davies_bouldin <- NA
  n_clusters <- 0
}

cat("\n===== DBSCAN Evaluation =====\n")
cat("eps:", best_eps, "\n")
cat("minPts:", best_minPts, "\n")
cat("Clusters detected:", n_clusters, "\n")
cat("Silhouette Score:", round(silhouette_score, 4), "\n")
cat("Davies-Bouldin Index:", round(davies_bouldin, 4), "\n\n")



# ============================================================
# STEP 9 — PCA VISUALIZATION (Python labels + noise in legend)
# ============================================================

# Convert to Python-style labels for plotting
labels_plot <- labels_raw
labels_plot[labels_raw == 0] <- -1
labels_plot[labels_raw > 0] <- labels_raw[labels_raw > 0] - 1   # 1→0, 2→1, etc.

# PCA on non-noise data
pca_best <- prcomp(filtered_data, center = TRUE, scale. = TRUE)
pca_df <- as.data.frame(pca_best$x[, 1:2])
colnames(pca_df) <- c("PC1", "PC2")

# Flip PCA x-axis
pca_df$PC1 <- -pca_df$PC1

pca_df$cluster <- factor(
  as.character(labels_plot[filtered_idx]),
  levels = c("0","1","2","3","Noise")
)

# Prepare noise PCA
noise_df <- NULL
if (any(labels_plot == -1)) {
  noise_data <- scaled_data[labels_plot == -1, , drop = FALSE]
  noise_pca <- predict(pca_best, newdata = noise_data)
  
  noise_df <- data.frame(
    PC1 = -noise_pca[, 1],
    PC2 = noise_pca[, 2],
    cluster = factor("Noise", levels = c("0","1","2","3","Noise"))
  )
}

# Colors: same as KMeans but cluster 2 = GREEN (your request)
kmeans_colors <- c(
  "0" = "#1f77b4",  # blue
  "1" = "#ff7f0e",  # orange
  "2" = "#2ca02c",  # green
  "3" = "#d62728",  # red
  "Noise" = "black"
)

# Shapes (Noise = X)
kmeans_shapes <- c(
  "0" = 16,
  "1" = 16,
  "2" = 16,
  "3" = 16,
  "Noise" = 4      # X symbol
)

eval_text <- paste0(
  "Number of Clusters: ", n_clusters, "\n",
  "Silhouette Score: ", sprintf("%.4f", silhouette_score), "\n",
  "Davies-Bouldin Index: ", sprintf("%.4f", davies_bouldin)
)


# ---- Final Plot ----
p_dbscan <- ggplot(pca_df, aes(x = PC1, y = PC2,
                               color = cluster,
                               shape = cluster)) +
  geom_point(size = 3, alpha = 0.8) +
  scale_color_manual(values = kmeans_colors) +
  scale_shape_manual(values = kmeans_shapes) +
  ggtitle(paste("DBSCAN (eps =", best_eps,
                ", minPts =", best_minPts, ")")) +
  xlab("PCA Component 1") + ylab("PCA Component 2") +
  theme_minimal(base_size = 14) +
  annotate("text",
           x = min(pca_df$PC1), y = max(pca_df$PC2),
           label = eval_text,
           hjust = 0, vjust = 1, size = 4)

# Add noise points
if (!is.null(noise_df)) {
  p_dbscan <- p_dbscan +
    geom_point(data=noise_df,
               aes(x=PC1, y=PC2, color=cluster, shape=cluster),
               size=3, stroke=1.3)
}

print(p_dbscan)


# ============================================================
# Step 10: Random Forest Clustering (R Version)
# ============================================================

library(cluster)
library(randomForest)

n_clusters <- 4     # same as Python

# ------------------------------------------------------------
# Step 10.1: Generate synthetic labels using Agglomerative Clustering
# ------------------------------------------------------------
hc <- hclust(dist(scaled_data), method = "ward.D2")
synthetic_labels <- cutree(hc, k = n_clusters)

cat("Synthetic labels created using hierarchical clustering.\n")

# ------------------------------------------------------------
# Step 10.2: Train Random Forest and compute feature importances
# ------------------------------------------------------------

numerical_features <- c("norm_SPX", "norm_USO", "norm_SLV", "norm_EUR.USD")

rf_model <- randomForest(
  x = scaled_data,
  y = as.factor(synthetic_labels),
  ntree = 100,
  importance = TRUE
)

importances <- importance(rf_model, type = 1)   # MeanDecreaseAccuracy
feature_names <- numerical_features

# pick top 2 features
top_feature_indices <- order(importances, decreasing = TRUE)[1:2]
top_features <- feature_names[top_feature_indices]

cat("Selected Features for Clustering:", paste(top_features, collapse = ", "), "\n")

selected_data <- scaled_data[, top_feature_indices, drop = FALSE]

# ------------------------------------------------------------
# Step 10.3: Train Random Forest with selected (top 2) features
# ------------------------------------------------------------

rf_selected <- randomForest(
  x = selected_data,
  y = as.factor(synthetic_labels),
  ntree = 100,
  importance = FALSE
)

# ------------------------------------------------------------
# Step 10.4: Compute Proximity Matrix
# ------------------------------------------------------------

# Get terminal node IDs for each tree

leaf_matrix <- predict(rf_selected, selected_data, nodes = TRUE)
leaf_matrix <- as.matrix(leaf_matrix)   # n_samples × n_trees


N <- nrow(selected_data)
n_trees <- ncol(leaf_matrix)

proximity_matrix <- matrix(0, nrow = N, ncol = N)

# Count shared leaf nodes just like Python
for (t in 1:n_trees) {
  leaf_t <- leaf_matrix[, t]
  for (i in 1:N) {
    matches <- (leaf_t == leaf_t[i])
    proximity_matrix[i, matches] <- proximity_matrix[i, matches] + 1
  }
}

# Normalize like Python
proximity_matrix <- proximity_matrix / n_trees

cat("Random Forest proximity matrix computed.\n")

# ============================================================
# Step 5: Apply Hierarchical Clustering Using Proximity Matrix
# ============================================================

library(cluster)      # for silhouette()
library(clusterSim)   # for Davies-Bouldin index

distance_matrix <- 1 - proximity_matrix   # Convert proximity → distance

# Perform Agglomerative Clustering (average linkage, same as Python)
hc <- hclust(as.dist(distance_matrix), method = "average")

# Cut into the desired number of clusters
rf_labels <- cutree(hc, k = n_clusters)   # labels: 1..K

# ============================================================
# Step 6: Evaluate the Random Forest Clustering Model
# ============================================================

unique_clusters <- unique(rf_labels)

if (length(unique_clusters) > 1) {
  
  # Silhouette score (requires distance matrix)
  sil <- silhouette(rf_labels, dist(selected_data))
  silhouette_score <- mean(sil[, 3])
  
  # Davies-Bouldin index
  davies_bouldin <- index.DB(selected_data, rf_labels)$DB
  
} else {
  silhouette_score <- NA
  davies_bouldin <- NA
}

cat("\n===== Random Forest Clustering Evaluation =====\n")
cat("Number of Clusters:", length(unique_clusters), "\n")

if (!is.na(silhouette_score)) {
  cat("Silhouette Score:", round(silhouette_score, 4), "\n")
  cat("Davies-Bouldin Index:", round(davies_bouldin, 4), "\n")
} else {
  cat("Unable to compute evaluation metrics due to insufficient clusters.\n")
}


# ============================================================
# Step 11: PCA Visualization for Random Forest Clustering (R)
# ============================================================

library(ggplot2)

# --- PCA on selected_data (same as Python: PCA(n_components=2) on selected features) ---
pca_rf <- prcomp(selected_data, center = TRUE, scale. = TRUE)
pca_data_rf <- pca_rf$x[, 1:2]

# --- Fix 1: Flip PC1 to match Python orientation ---
pca_data_rf[, 2] <- pca_data_rf[, 2]

# --- Keep raw labels for metrics (DO NOT change rf_labels used in DBI/silhouette) ---
rf_labels_eval <- rf_labels  # 1..K

# --- Convert labels ONLY for plotting (Python-style 0..K-1) ---
rf_labels_plot <- rf_labels_eval - 1

df_rf_plot <- tibble(
  PC1 = pca_data_rf[, 1],
  PC2 = pca_data_rf[, 2],
  cluster = factor(rf_labels_plot)
)

# --- Same colors as KMeans (with your red↔green swap) ---
kmeans_colors <- c(
  "0" = "#1f77b4",  # blue
  "1" = "#ff7f0e",  # orange/yellow
  "2" = "#2ca02c",  # green
  "3" = "#d62728"   # red
)


eval_text <- paste0(
  "Number of Clusters: ", length(unique(rf_labels_eval)), "\n",
  "Silhouette Score: ", sprintf("%.4f", silhouette_score), "\n",
  "Davies-Bouldin Index: ", sprintf("%.4f", davies_bouldin)
)

p_rf <- ggplot(df_rf_plot, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = kmeans_colors) +
  ggtitle(paste0(
    "Random Forest Clustering with Top Features: ",
    paste(top_features, collapse = ", ")
  )) +
  xlab("PCA Component 1") +
  ylab("PCA Component 2") +
  theme_minimal(base_size = 14) +
  theme(
    legend.title = element_blank(),
    plot.title = element_text(face = "bold")
  ) +
  annotate(
    "text",
    x = min(df_rf_plot$PC1), y = max(df_rf_plot$PC2),
    label = eval_text, hjust = 0, vjust = 1,
    size = 4, color = "black"
  )

print(p_rf)