## --------------------------------------------------------------
## Gold Price Movement Classification (Up vs Down) - R version
## Uses caret + SMOTE (via sampling="smote") for balancing
## Models: Random Forest, SVM (RBF), Naive Bayes
## Metrics: Accuracy, Error rate, Sensitivity, Specificity, Macro F1, AUC
## All plots are saved AND shown in RStudio Plots tab.
## --------------------------------------------------------------

rm(list = ls())

library(caret)
library(randomForest)
library(e1071)
library(klaR)
library(pROC)
library(ggplot2)
library(reshape2)
library(smotefamily)

DATA_PATH  <- "gold_price_dataset_preprocessed.csv"
OUTPUT_DIR <- "classification_outputs_R"
set.seed(42)

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR)

## ---------------- 1. Load & prepare data ----------------

df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

# Drop index column if present and remove NAs
if ("Unnamed..0" %in% names(df)) df$Unnamed..0 <- NULL
df <- na.omit(df)

# Keep only Up / Down classes
df <- df[df$gold_label %in% c("Up", "Down"), ]

# Make target a factor; Up as positive class (first level)
df$gold_label <- factor(df$gold_label, levels = c("Up", "Down"))

# Drop non-predictive columns
if ("Date" %in% names(df)) df$Date <- NULL

cat("\nDataset loaded:\n")
print(table(df$gold_label))

## ---------------- 2. Quick plots ----------------

# Class distribution
p_class <- ggplot(df, aes(x = gold_label)) +
  geom_bar() +
  ggtitle("Gold Movement Class Distribution (Up vs Down)") +
  xlab("Class") + ylab("Count")

print(p_class)  # show in Plots tab
ggsave(file.path(OUTPUT_DIR, "class_distribution_R.png"),
       p_class, width = 6, height = 4, dpi = 300)

# Correlation heatmap for numeric predictors
num_df <- df[, sapply(df, is.numeric), drop = FALSE]
corr_mat <- cor(num_df)

p_corr <- ggplot(melt(corr_mat), aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                       limits = c(-1, 1)) +
  ggtitle("Feature Correlation Heatmap") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        axis.title.x = element_blank(),
        axis.title.y = element_blank())

print(p_corr)  # show in Plots tab
ggsave(file.path(OUTPUT_DIR, "correlation_heatmap_R.png"),
       p_corr, width = 10, height = 8, dpi = 300)

## ---------------- 3. Train / test split ----------------

set.seed(42)
train_idx   <- createDataPartition(df$gold_label, p = 0.7, list = FALSE)
train_data  <- df[train_idx, ]
test_data   <- df[-train_idx, ]

## ---------------- 4. caret control with SMOTE + repeated CV ----------------

ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,  # uses ROC
  sampling        = "smote",
  savePredictions = "final"
)

## ---------------- 5. Model definitions (with scaling options) ----------------

p <- ncol(train_data) - 1  # number of predictors

models_list <- list(
  RandomForest = list(
    method     = "rf",
    tuneGrid   = expand.grid(
      mtry = c(2, floor(sqrt(p)), floor(p / 3))
    ),
    preProcess = NULL,     # RF doesn't need scaling
    extraArgs  = list(ntree = 300)
  ),
  SVM = list(
    method     = "svmRadial",
    tuneGrid   = expand.grid(
      sigma = c(0.01, 0.03, 0.05, 0.1),
      C     = c(1, 5, 10, 20)
    ),
    preProcess = c("center", "scale"),  # like Python StandardScaler
    extraArgs  = list()
  ),
  NaiveBayes = list(
    method     = "nb",
    tuneGrid   = expand.grid(
      fL        = c(0, 0.5, 1),
      usekernel = c(TRUE),
      adjust    = c(0.5, 1, 2)
    ),
    preProcess = c("center", "scale"),
    extraArgs  = list()
  )
)

## ---------------- 6. Train + evaluate ----------------

metrics_summary <- data.frame()
results <- list()

for (mname in names(models_list)) {
  cat("\n=====================================\n")
  cat("Training:", mname, "\n")
  cat("=====================================\n")
  
  mconf <- models_list[[mname]]
  
  set.seed(42)
  fit <- do.call(
    train,
    c(
      list(
        form       = gold_label ~ .,
        data       = train_data,
        method     = mconf$method,
        metric     = "ROC",
        trControl  = ctrl,
        tuneGrid   = mconf$tuneGrid,
        preProcess = mconf$preProcess
      ),
      mconf$extraArgs
    )
  )
  
  best_params <- fit$bestTune
  print(best_params)
  
  # Predictions on hold-out test set
  pred_class <- predict(fit, newdata = test_data)
  pred_prob  <- predict(fit, newdata = test_data, type = "prob")[, "Up"]
  
  cm <- table(Actual = test_data$gold_label, Predicted = pred_class)
  print(cm)
  
  # Ensure 2x2 matrix ordered (Up, Down)
  cm_full <- matrix(0, nrow = 2, ncol = 2,
                    dimnames = list(
                      Actual    = c("Up", "Down"),
                      Predicted = c("Up", "Down")
                    ))
  cm_full[rownames(cm), colnames(cm)] <- cm
  
  TP <- cm_full["Up",   "Up"]
  FN <- cm_full["Up",   "Down"]
  FP <- cm_full["Down", "Up"]
  TN <- cm_full["Down", "Down"]
  
  accuracy    <- (TP + TN) / (TP + TN + FP + FN)
  error_rate  <- 1 - accuracy
  sensitivity <- ifelse(TP + FN > 0, TP / (TP + FN), NA)
  specificity <- ifelse(TN + FP > 0, TN / (TN + FP), NA)
  
  f1_up   <- ifelse(2 * TP + FP + FN > 0, 2 * TP / (2 * TP + FP + FN), NA)
  f1_down <- ifelse(2 * TN + FN + FP > 0, 2 * TN / (2 * TN + FN + FP), NA)
  macro_f1 <- mean(c(f1_up, f1_down), na.rm = TRUE)
  
  cat(sprintf("Accuracy   : %.3f\n", accuracy))
  cat(sprintf("Error Rate : %.3f\n", error_rate))
  cat(sprintf("Sensitivity: %.3f\n", sensitivity))
  cat(sprintf("Specificity: %.3f\n", specificity))
  cat(sprintf("Macro F1   : %.3f\n", macro_f1))
  
  # ROC & AUC (specificity on x-axis, sensitivity on y-axis)
  roc_obj <- roc(
    response  = test_data$gold_label,
    predictor = pred_prob,
    levels    = c("Down", "Up"),  # Down = negative, Up = positive
    direction = "<"
  )
  auc_val <- as.numeric(auc(roc_obj))
  cat(sprintf("AUC        : %.3f\n", auc_val))
  
  # Confusion matrix plot
  cm_df <- as.data.frame(as.table(cm_full))
  p_cm <- ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), colour = "white") +
    scale_fill_gradient(low = "lightblue", high = "darkblue") +
    ggtitle(paste("Confusion Matrix -", mname)) +
    theme_minimal()
  
  print(p_cm)  # show in Plots tab
  ggsave(file.path(OUTPUT_DIR, paste0("cm_", mname, "_R.png")),
         p_cm, width = 5, height = 4, dpi = 300)
  
  # ROC curve: first show in Plots tab, then save to file
  plot(roc_obj, main = paste(mname, "ROC Curve"), col = "black", lwd = 2)
  abline(a = 0, b = 1, lty = 2, col = "grey")
  dev.copy(
    png,
    filename = file.path(OUTPUT_DIR, paste0("roc_", mname, "_R.png")),
    width = 6, height = 5, units = "in", res = 300
  )
  dev.off()
  
  metrics_summary <- rbind(
    metrics_summary,
    data.frame(
      model         = mname,
      test_accuracy = accuracy,
      error_rate    = error_rate,
      sensitivity   = sensitivity,
      specificity   = specificity,
      macro_f1      = macro_f1,
      AUC           = auc_val,
      best_params   = paste(names(best_params),
                            as.character(best_params),
                            collapse = "; "),
      stringsAsFactors = FALSE
    )
  )
  
  results[[mname]] <- list(fit = fit, cm = cm_full, roc = roc_obj)
}

## ---------------- 7. Save summary & comparison plots ----------------

write.csv(metrics_summary,
          file.path(OUTPUT_DIR, "metrics_summary_R.csv"),
          row.names = FALSE)

# Accuracy / Sensitivity / Specificity bar chart
metrics_long <- melt(
  metrics_summary[, c("model", "test_accuracy", "sensitivity", "specificity")],
  id.vars = "model",
  variable.name = "Metric",
  value.name = "Score"
)

p_metrics <- ggplot(metrics_long, aes(x = model, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  ylim(0, 1.05) +
  ggtitle("Model Comparison: Accuracy, Sensitivity, Specificity") +
  xlab("Model") + ylab("Score") +
  theme_minimal()

print(p_metrics)  # show in Plots tab
ggsave(file.path(OUTPUT_DIR, "model_metric_scores_R.png"),
       p_metrics, width = 8, height = 5, dpi = 300)

# Error rate bar chart
p_error <- ggplot(metrics_summary, aes(x = model, y = error_rate)) +
  geom_bar(stat = "identity", fill = "tomato") +
  ylim(0, 1) +
  ggtitle("Error Rate Comparison") +
  xlab("Model") + ylab("Error Rate") +
  theme_minimal()

print(p_error)  # show in Plots tab
ggsave(file.path(OUTPUT_DIR, "model_error_rates_R.png"),
       p_error, width = 6, height = 4, dpi = 300)

cat("\nAll R results saved in:", normalizePath(OUTPUT_DIR), "\n")

