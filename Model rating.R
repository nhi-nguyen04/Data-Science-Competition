# Misclassification and Residual Analysis for Vaccine Predictions
# This code analyzes prediction errors to identify improvement opportunities

library(tidyverse)
library(tidymodels)
library(corrplot)
library(ggplot2)
library(patchwork)

# -----------------------------------------------
# 1. GENERATE DETAILED PREDICTIONS ON EVALUATION SET
# -----------------------------------------------

# Get both probabilities and class predictions
eval_results_h1n1 <- eval_data %>%
  bind_cols(
    predict(fit_h1n1, eval_data, type = "prob"),
    predict(fit_h1n1, eval_data, type = "class")
  ) %>%
  rename(pred_class_h1n1 = .pred_class) %>%
  mutate(
    actual_h1n1 = as.numeric(as.character(h1n1_vaccine)),
    pred_prob_h1n1 = .pred_1,
    # Calculate residuals (actual - predicted probability)
    residual_h1n1 = actual_h1n1 - pred_prob_h1n1,
    # Identify misclassifications
    misclassified_h1n1 = (pred_class_h1n1 != h1n1_vaccine),
    # Calculate absolute residuals
    abs_residual_h1n1 = abs(residual_h1n1)
  )

eval_results_seas <- eval_data %>%
  bind_cols(
    predict(fit_seas, eval_data, type = "prob"),
    predict(fit_seas, eval_data, type = "class")
  ) %>%
  rename(pred_class_seas = .pred_class) %>%
  mutate(
    actual_seas = as.numeric(as.character(seasonal_vaccine)),
    pred_prob_seas = .pred_1,
    residual_seas = actual_seas - pred_prob_seas,
    misclassified_seas = (pred_class_seas != seasonal_vaccine),
    abs_residual_seas = abs(residual_seas)
  )

# -----------------------------------------------
# 2. BASIC PERFORMANCE METRICS
# -----------------------------------------------

cat("=== H1N1 VACCINE MODEL PERFORMANCE ===\n")
h1n1_accuracy <- mean(!eval_results_h1n1$misclassified_h1n1)
h1n1_misclass_rate <- mean(eval_results_h1n1$misclassified_h1n1)
cat("Accuracy:", round(h1n1_accuracy * 100, 2), "%\n")
cat("Misclassification Rate:", round(h1n1_misclass_rate * 100, 2), "%\n")
cat("Mean Absolute Residual:", round(mean(eval_results_h1n1$abs_residual_h1n1), 4), "\n")

cat("\n=== SEASONAL VACCINE MODEL PERFORMANCE ===\n")
seas_accuracy <- mean(!eval_results_seas$misclassified_seas)
seas_misclass_rate <- mean(eval_results_seas$misclassified_seas)
cat("Accuracy:", round(seas_accuracy * 100, 2), "%\n")
cat("Misclassification Rate:", round(seas_misclass_rate * 100, 2), "%\n")
cat("Mean Absolute Residual:", round(mean(eval_results_seas$abs_residual_seas), 4), "\n")

# -----------------------------------------------
# 3. CONFUSION MATRICES
# -----------------------------------------------

cat("\n=== CONFUSION MATRICES ===\n")
cat("H1N1 Vaccine Confusion Matrix:\n")
print(table(Actual = eval_results_h1n1$h1n1_vaccine, 
            Predicted = eval_results_h1n1$pred_class_h1n1))

cat("\nSeasonal Vaccine Confusion Matrix:\n")
print(table(Actual = eval_results_seas$seasonal_vaccine, 
            Predicted = eval_results_seas$pred_class_seas))

# -----------------------------------------------
# 4. ANALYZE MISCLASSIFICATIONS BY FEATURE VALUES
# -----------------------------------------------

analyze_misclassifications <- function(data, misclass_col, target_name) {
  cat("\n=== MISCLASSIFICATION ANALYSIS FOR", target_name, "===\n")
  
  # Separate correctly classified vs misclassified
  correct <- data[!data[[misclass_col]], ]
  misclass <- data[data[[misclass_col]], ]
  
  cat("Total misclassified cases:", nrow(misclass), "out of", nrow(data), "\n")
  
  # Analyze numeric features
  numeric_features <- intersect(numeric_cols, names(data))
  
  cat("\n--- Numeric Feature Differences (Misclassified vs Correct) ---\n")
  for (col in numeric_features) {
    if (col %in% names(data)) {
      correct_mean <- mean(correct[[col]], na.rm = TRUE)
      misclass_mean <- mean(misclass[[col]], na.rm = TRUE)
      difference <- misclass_mean - correct_mean
      
      if (abs(difference) > 0.1) {  # Only show meaningful differences
        cat(sprintf("%-25s: Correct=%.2f, Misclass=%.2f, Diff=%.2f\n", 
                    col, correct_mean, misclass_mean, difference))
      }
    }
  }
  
  # Analyze categorical features
  cat("\n--- Categorical Feature Patterns ---\n")
  for (col in categorical_cols) {
    if (col %in% names(data) && !all(is.na(data[[col]]))) {
      # Calculate misclassification rate by category
      misclass_by_cat <- data %>%
        group_by(!!sym(col)) %>%
        summarise(
          count = n(),
          misclass_rate = mean(!!sym(misclass_col), na.rm = TRUE),
          .groups = 'drop'
        ) %>%
        filter(count >= 10) %>%  # Only categories with enough samples
        arrange(desc(misclass_rate))
      
      if (nrow(misclass_by_cat) > 1) {
        cat("\nMisclassification rates by", col, ":\n")
        print(misclass_by_cat, n = 5)
      }
    }
  }
}

# Run misclassification analysis
analyze_misclassifications(eval_results_h1n1, "misclassified_h1n1", "H1N1")
analyze_misclassifications(eval_results_seas, "misclassified_seas", "SEASONAL")

# -----------------------------------------------
# 5. RESIDUAL ANALYSIS PLOTS
# -----------------------------------------------

# Create residual plots
p1 <- ggplot(eval_results_h1n1, aes(x = pred_prob_h1n1, y = residual_h1n1)) +
  geom_point(alpha = 0.6, aes(color = misclassified_h1n1)) +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "H1N1 Residuals vs Predicted Probability",
       x = "Predicted Probability", y = "Residual (Actual - Predicted)",
       color = "Misclassified") +
  theme_minimal()

p2 <- ggplot(eval_results_seas, aes(x = pred_prob_seas, y = residual_seas)) +
  geom_point(alpha = 0.6, aes(color = misclassified_seas)) +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Seasonal Residuals vs Predicted Probability",
       x = "Predicted Probability", y = "Residual (Actual - Predicted)",
       color = "Misclassified") +
  theme_minimal()

# Distribution of residuals
p3 <- ggplot(eval_results_h1n1, aes(x = residual_h1n1)) +
  geom_histogram(bins = 30, alpha = 0.7, fill = "blue") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "H1N1 Residual Distribution", x = "Residual", y = "Count") +
  theme_minimal()

p4 <- ggplot(eval_results_seas, aes(x = residual_seas)) +
  geom_histogram(bins = 30, alpha = 0.7, fill = "red") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Seasonal Residual Distribution", x = "Residual", y = "Count") +
  theme_minimal()

# Combine plots
combined_plot <- (p1 | p2) / (p3 | p4)
print(combined_plot)

# -----------------------------------------------
# 6. IDENTIFY HIGH-ERROR CASES
# -----------------------------------------------

cat("\n=== HIGH-ERROR CASES ANALYSIS ===\n")

# Find cases with largest absolute residuals
high_error_h1n1 <- eval_results_h1n1 %>%
  arrange(desc(abs_residual_h1n1)) %>%
  slice_head(n = 10)

high_error_seas <- eval_results_seas %>%
  arrange(desc(abs_residual_seas)) %>%
  slice_head(n = 10)

cat("Top 10 H1N1 prediction errors (by absolute residual):\n")
print(high_error_h1n1 %>% 
        select(respondent_id, actual_h1n1, pred_prob_h1n1, residual_h1n1, misclassified_h1n1))

cat("\nTop 10 Seasonal prediction errors (by absolute residual):\n")
print(high_error_seas %>% 
        select(respondent_id, actual_seas, pred_prob_seas, residual_seas, misclassified_seas))

# -----------------------------------------------
# 7. FEATURE IMPORTANCE FOR MISCLASSIFIED CASES
# -----------------------------------------------

# Analyze which features are most different in misclassified cases
analyze_feature_importance_for_errors <- function(data, misclass_col, target_name) {
  cat("\n=== FEATURE PATTERNS IN MISCLASSIFIED", target_name, "CASES ===\n")
  
  numeric_features <- intersect(numeric_cols, names(data))
  feature_diffs <- tibble()
  
  for (col in numeric_features) {
    if (col %in% names(data)) {
      correct_vals <- data[!data[[misclass_col]], col]
      misclass_vals <- data[data[[misclass_col]], col]
      
      # Calculate effect size (Cohen's d)
      pooled_sd <- sqrt(((length(correct_vals) - 1) * var(correct_vals, na.rm = TRUE) + 
                           (length(misclass_vals) - 1) * var(misclass_vals, na.rm = TRUE)) / 
                          (length(correct_vals) + length(misclass_vals) - 2))
      
      cohens_d <- (mean(misclass_vals, na.rm = TRUE) - mean(correct_vals, na.rm = TRUE)) / pooled_sd
      
      feature_diffs <- bind_rows(feature_diffs, 
                                 tibble(feature = col, cohens_d = cohens_d))
    }
  }
  
  feature_diffs <- feature_diffs %>%
    arrange(desc(abs(cohens_d))) %>%
    filter(abs(cohens_d) > 0.2)  # Medium effect size threshold
  
  cat("Features with largest differences (Cohen's d > 0.2):\n")
  print(feature_diffs)
  
  return(feature_diffs)
}

h1n1_error_features <- analyze_feature_importance_for_errors(eval_results_h1n1, "misclassified_h1n1", "H1N1")
seas_error_features <- analyze_feature_importance_for_errors(eval_results_seas, "misclassified_seas", "SEASONAL")

# -----------------------------------------------
# 8. RECOMMENDATIONS SUMMARY
# -----------------------------------------------

cat("\n" + rep("=", 60) + "\n")
cat("IMPROVEMENT RECOMMENDATIONS:\n")
cat(rep("=", 60) + "\n")

cat("\n1. MODEL CALIBRATION:\n")
cat("   - Check if predicted probabilities are well-calibrated\n")
cat("   - Consider probability calibration techniques if needed\n")

cat("\n2. FEATURE ENGINEERING:\n")
cat("   - Focus on features that show large differences in misclassified cases\n")
cat("   - Consider interaction terms between important features\n")
cat("   - Look for non-linear relationships in high-error regions\n")

cat("\n3. DATA QUALITY:\n")
cat("   - Investigate high-error cases for potential data quality issues\n")
cat("   - Consider additional data collection for underrepresented groups\n")

cat("\n4. MODEL COMPLEXITY:\n")
cat("   - Try different regularization parameters\n")
cat("   - Consider ensemble methods to reduce variance\n")
cat("   - Experiment with different model types (Random Forest, XGBoost)\n")

cat("\n5. CLASS IMBALANCE:\n")
cat("   - Check if misclassifications are concentrated in minority class\n")
cat("   - Consider resampling techniques or cost-sensitive learning\n")