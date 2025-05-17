#Packages
library(readr)
library(dplyr)
library(ggplot2)




#STEP 1: UNDERSTAND THE PROBLEM STATEMENT

# -----------  ----------- Clarify Objectives & Business Impact: 
# Task Type: Classification
# Participants are asked to predict whether individuals received the H1N1 and seasonal flu vaccines, making it a binary classification task for each vaccine:
#   
# Target values: 0 (did not receive the vaccine) or 1 (did receive the vaccine).

#The aim is to understand which factors are associated with vaccine uptake, which can support public health campaigns and policies
# Success means building a model that can accurately predict vaccination status for both H1N1 and seasonal flu vaccines. More specifically:
#   
#   Your predictions should score well based on the evaluation metric used (e.g., log loss).
# 
# A successful submission would identify key predictors of vaccination behavior and perform well on the private leaderboard.

#-----------  -----------Target Variable: 

# In the Flu Shot Learning competition, the targets are defined as follows:
#   
#   h1n1_vaccine: Indicates whether a respondent received the H1N1 flu vaccine.
# 
# seasonal_vaccine: Indicates whether a respondent received the seasonal flu vaccine.
# 
# Both are binary variables: 1 signifies that the individual received the respective vaccine,
#while 0 means they did not. This setup constitutes a multilabel classification problem, as each respondent can have received none, one, or both vaccines.

# -----------  ----------- Constraints & Permissions: 

# -----------  ----------- Action Points:

# Requirements:
#   
#   Build a model to predict two binary outcomes:
#   
#   h1n1_vaccine: Did the respondent get the H1N1 flu shot?
#   
#   seasonal_vaccine: Did the respondent get the seasonal flu shot?
#   
#   Submit predictions as probabilities, not binary labels.
# 
# Use provided survey data, which includes demographics, behaviors, opinions, and access to healthcare.
# 
# Optimize for an evaluation metric like log loss, which rewards well-calibrated probability estimates.
# 
# Assumptions:
#   
#   The training data is representative of the test data (i.e., drawn from the same distribution).
# 
# Survey responses are assumed to be truthful and accurate.
# 
# Missing data is either ignorable or missing at random (though this should be tested).
# 
# The relationship between features and vaccination behavior in 2009 is generalizable to similar contexts.




#-----------  ----------------------  ----------------------  ----------------------  ----------------------  ----------------------  --------------|
#Problem Restated in Plain Terms

#You're given data from a 2009 U.S. survey where people answered questions about their background, health, and opinions. 
# Your job is to build a machine learning model that predicts the likelihood of each person having received the H1N1 flu vaccine 
# and the seasonal flu vaccine. The predictions should be probabilities (from 0 to 1) and as accurate and well-calibrated as possible, 
# even in the face of class imbalances.           
  
#  -----------  ----------------------  ----------------------  ----------------------  ----------------------  ----------------------  -----------|
  
  
  
  
#  STEP 2: STUDY THE EVALUATION METRIC


# -----------  ----------- Metric Matters
#   Metric: Log loss (average over two binary targets)
#   Success = Well-calibrated probability predictions
#   Model tuning should focus on probability accuracy, not just classification accuracy



#  STEP 3: PERFORM AN INITIAL DATA ANALYSIS (EDA)

# -----------  -----------  Data profiling:Summarize shape, basic statistics, feature and target distribution.
library(readr)
# you can do this or import the data in r studio
training_set_features <- read_csv("Data/training_set_features.csv")
training_set_labels

#view the data
View(training_set_features)
View(training_set_labels)



# View first few rows
head(training_set_features)
head(training_set_labels)

# View structure
str(training_set_features)
str(training_set_labels)

# View column names
names(training_set_features)
names(training_set_labels)

# Check dimensions
dim(training_set_features)
dim(training_set_labels)

# Summary statistics for all columns
summary(training_set_features)
summary(training_set_labels)

# Check for missing values per column
colSums(is.na(training_set_features))
colSums(is.na(training_set_labels))

#-----Feature and Target Distribution

features <- read_csv("Data/training_set_features.csv")
labels <- read_csv("Data/training_set_labels.csv")

# Merge on respondent_id
# They have the same  respondent_id
df <- left_join(features, labels, by = "respondent_id")

# Proportional distribution of H1N1 vaccine
cat("H1N1 Vaccine Distribution:\n")
print(prop.table(table(df$h1n1_vaccine)))

# Proportional distribution of Seasonal vaccine
cat("Seasonal Vaccine Distribution:\n")
print(prop.table(table(df$seasonal_vaccine)))


# Plot: H1N1 vaccine
ggplot(df, aes(x = factor(h1n1_vaccine))) +
  geom_bar(fill = "skyblue") +
  labs(title = "H1N1 Vaccine Target Distribution", x = "H1N1 Vaccine", y = "Count") +
  theme_minimal()

# Plot: Seasonal vaccine
ggplot(df, aes(x = factor(seasonal_vaccine))) +
  geom_bar(fill = "skyblue") +
  labs(title = "Seasonal Vaccine Target Distribution", x = "Seasonal Vaccine", y = "Count") +
  theme_minimal()

#Top frequencies of each categorical feature
cat_cols <- names(df)[sapply(df, is.character)]

cat("\nTop 3 categories per categorical feature:\n")
for (col in cat_cols) {
  cat(paste0("\n", col, ":\n"))
  print(sort(table(df[[col]]), decreasing = TRUE)[1:3])
}


# ----- ------- Quality Checks: Identify missing values, outliers, or suspicious patterns; confirm no hidden data leakage.

#data quality checks on your merged dataset (features + labels) df
   #1. Missing Values Summary
missing_summary <- colSums(is.na(df))
missing_summary[missing_summary > 0]

  #2. Outlier Detection (Numeric Features)
 #Using IQR (Interquartile Range) method to flag outliers:
#You used the IQR method, which is designed to detect unusually high or low values compared to the bulk of the data

num_cols <- names(df)[sapply(df, is.numeric)]
num_cols <- setdiff(num_cols, c("respondent_id", "h1n1_vaccine", "seasonal_vaccine"))

outlier_check <- function(column) {
  q1 <- quantile(column, 0.25, na.rm = TRUE)
  q3 <- quantile(column, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  outliers <- column < (q1 - 1.5 * iqr) | column > (q3 + 1.5 * iqr)
  sum(outliers, na.rm = TRUE)
}

outlier_counts <- sapply(df[num_cols], outlier_check)
outlier_counts[outlier_counts > 0]

#Outcome:No need to remove or impute any values based on IQR outliers alone.
# Instead, focus on:
#   
#   Class imbalance if applicable (e.g., for targets or binary features)
# 
# Handling missing values
# 
# Feature scaling if you're using models sensitive to magnitude (e.g., logistic regression, SVM)

  #3. Suspicious Patterns
constant_columns <- sapply(df, function(x) length(unique(na.omit(x))) == 1)
names(df)[constant_columns]


cat_cols <- names(df)[sapply(df, is.character)]
for (col in cat_cols) {
  freq <- sort(table(df[[col]]), decreasing = TRUE)
  if (freq[1] / sum(freq) > 0.95) {
    cat(paste0("Highly skewed: ", col, "\n"))
    print(freq)
  }
}
#Don’t drop them — instead, treat them as categorical (factor) in modeling.
#No features have only a single unique (non-missing) value → ✅ Good.
#Your script didn’t flag any columns, which means no single value dominates >95% of any categorical column.
#That’s a good sign — no suspicious or imbalanced category columns.

  #4. Data Leakage Detection
# Correlation with H1N1 vaccine
cor_df <- df %>% 
  select(all_of(num_cols), h1n1_vaccine) %>%
  cor(use = "complete.obs")

# Sort by correlation with the target
sort(abs(cor_df[,"h1n1_vaccine"]), decreasing = TRUE)


# What Would Count as Data Leakage?
#   Data leakage means a feature contains or implies the label due to data being collected after the target event, or being too directly tied to it.
# 
# Checkpoints:
#   
#   ✅ No feature is >0.8 correlated with the target.
# 
# ✅ All high-correlation features are reasonable (e.g., doctor's recommendation, vaccine opinions).
# 
# 🚫 No derived field like received_h1n1_vaccine, which would leak the label.
# 
# ✅ No perfect or near-perfect predictors (cor = 1 or close).

# Load necessary libraries
library(tibble)
library(knitr)

# Create quality check summary table
quality_summary <- tribble(
  ~Check,                    ~Status,        ~Notes,
  "Missing values",          "⚠️ Present",   "Several features have missing values (e.g., health_insurance, employment_industry). Consider imputation.",
  "Outliers",                "⚠️ Present",   "Outliers exist in some numeric columns. Use IQR or Z-score for handling.",
  "Constant features",       "✅ None found","All features show some variation.",
  "Skewed categorical",      "⚠️ Some skew", "Columns like race and sex are skewed. Acceptable but consider during modeling.",
  "Data leakage",            "✅ None found","No feature is suspiciously predictive or correlated (> 0.8) with the target."
)

# Print the table
kable(quality_summary, caption = "Summary of Data Quality Checks")



#STEP 4: DEVELOP A BASELINE MODEL
library(tidyverse)
library(caret)
install.packages("ROCR")
library(ROCR)

# -----------  ----------- Model development
#Load and Prepare Training Data
features <- read_csv("Data/training_set_features.csv")
labels <- read_csv("Data/training_set_labels.csv")
df <- left_join(features, labels, by = "respondent_id") %>%
  select(-respondent_id)

# Drop columns with >30% NAs
df <- df %>% select(where(~ mean(!is.na(.)) > 0.7))

# Impute missing values
for (col in names(df)) {
  if (is.numeric(df[[col]])) {
    df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
  } else {
    df[[col]][is.na(df[[col]])] <- names(sort(table(df[[col]]), decreasing = TRUE))[1]
  }
}

# Convert character columns to factors
df <- df %>% mutate(across(where(is.character), as.factor))

# Separate models for each vaccine
model_h1n1 <- glm(h1n1_vaccine ~ ., data = df, family = binomial)
model_seasonal <- glm(seasonal_vaccine ~ ., data = df, family = binomial)



######### Evaluate Both Models

# Predictions
pred_h1n1 <- predict(model_h1n1, type = "response")
pred_seasonal <- predict(model_seasonal, type = "response")

# Convert to labels
pred_h1n1_label <- ifelse(pred_h1n1 > 0.5, 1, 0)
pred_seasonal_label <- ifelse(pred_seasonal > 0.5, 1, 0)

# Confusion Matrices
conf_h1n1 <- confusionMatrix(factor(pred_h1n1_label), factor(df$h1n1_vaccine))
conf_seasonal <- confusionMatrix(factor(pred_seasonal_label), factor(df$seasonal_vaccine))

# AUC
auc_h1n1 <- performance(prediction(pred_h1n1, df$h1n1_vaccine), "auc")@y.values[[1]]
auc_seasonal <- performance(prediction(pred_seasonal, df$seasonal_vaccine), "auc")@y.values[[1]]

# Output
cat("H1N1 Vaccine Model:\n")
print(conf_h1n1)
cat(sprintf("ROC-AUC: %.3f\n\n", auc_h1n1))

cat("Seasonal Flu Vaccine Model:\n")
print(conf_seasonal)
cat(sprintf("ROC-AUC: %.3f\n", auc_seasonal))



# -----------  -----------Define pipeline checklist
pipeline_checklist <- tribble(
  ~Step,                        ~Description,                                                                 ~Status,
  "1. Data Loading",            "Loaded training_set_features.csv and training_set_labels.csv using read_csv()", "✅",
  "2. Merge on ID",             "Merged on respondent_id using left_join()",                                     "✅",
  "3. ID Removal",              "Dropped respondent_id to avoid leakage",                                        "✅",
  "4. Missing Data Handling",   "Dropped columns >30% NA, imputed rest (median/mode)",                           "✅",
  "5. Data Type Conversion",    "Converted character columns to factors",                                        "✅",
  "6. Validation Split",        "Used provided training data only — no additional splitting",                    "✅",
  "7. Target Columns",          "Both h1n1_vaccine and seasonal_vaccine are binary (0/1)",                       "✅",
  "8. Model Type",              "Used logistic regression (glm, family = binomial)",                             "✅",
  "9. Evaluation Metrics",      "Used Confusion Matrix and ROC-AUC",                                             "✅",
  "10. No Data Leakage",        "Verified via correlation — no suspiciously predictive features",                "✅"
)

# Display the checklist
kable(pipeline_checklist, caption = "Pipeline Structure Checklist")




# ✅ 1. Start with a Naive Model (Already Done)
# You're using:
# 
# Logistic regression (GLM) with default parameters
# 
# No feature engineering or model tuning
# 
# Baseline set using training_set_features.csv only
# 
# Status: ✅ Complete

####Use cross validation for reability
#for h1n1_vaccine
set.seed(123)

cv_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Convert target to factor with labels (for caret)
df$h1n1_vaccine <- factor(df$h1n1_vaccine, labels = c("No", "Yes"))

cv_model_h1n1 <- train(
  h1n1_vaccine ~ ., data = df,
  method = "glm",
  family = binomial,
  trControl = cv_control,
  metric = "ROC"
)

print(cv_model_h1n1)



#for seasonal_vaccine
df$seasonal_vaccine <- factor(df$seasonal_vaccine, labels = c("No", "Yes"))

cv_model_seasonal <- train(
  seasonal_vaccine ~ ., data = df,
  method = "glm",
  family = binomial,
  trControl = cv_control,
  metric = "ROC"
)

print(cv_model_seasonal)


#Analyze Misclassifications / Residuals

# Predict probabilities for H1N1
prob_h1n1 <- predict(cv_model_h1n1, df, type = "prob")[, "Yes"]
pred_h1n1 <- ifelse(prob_h1n1 > 0.5, "Yes", "No")

# Actual vs predicted
misclassified_h1n1 <- df %>%
  mutate(predicted = pred_h1n1) %>%
  filter(predicted != h1n1_vaccine)

# Show top few misclassifications
head(misclassified_h1n1)

# Visual: Which age groups or income levels are often misclassified?
library(ggplot2)
ggplot(misclassified_h1n1, aes(x = age_group)) +
  geom_bar(fill = "red") +
  ggtitle("H1N1 Misclassifications by Age Group") +
  theme_minimal()





baseline_metrics <- tibble::tribble(
  ~Target, ~Model, ~CrossVal_ROC,
  "H1N1 Vaccine", "Logistic Regression", max(cv_model_h1n1$results$ROC),
  "Seasonal Vaccine", "Logistic Regression", max(cv_model_seasonal$results$ROC)
)

knitr::kable(baseline_metrics, caption = "Baseline ROC-AUC Scores (5-fold CV)")

------------------------------------------------------------------------------------------------------------------------------------------------
  
  
#  Second attempt at model
  
  
  ## Baseline Logistic Regression for H1N1 and Seasonal Vaccine Prediction in R
  
  # Ensure required package versions
  if (!requireNamespace("rlang", quietly = TRUE) || packageVersion("rlang") < "1.1.5") {
    install.packages("rlang")
  }

# Load necessary libraries
library(tidymodels)
library(readr)
library(dplyr)

# 1. Read datasets
train_features <- read_csv("/Data/training_set_features.csv")
train_labels   <- read_csv("/Data/training_set_labels.csv")
test_features  <- read_csv("/Data/test_set_features.csv")

# 2. Merge training features and labels
data <- train_features %>%
  inner_join(train_labels, by = "respondent_id")

# 3. Define targets
targets <- c("h1n1_vaccine", "seasonal_vaccine")

# 4. Preprocessing recipe template: impute, scale, encode
base_recipe <- recipe(~ ., data = data) %>%
  update_role(respondent_id, new_role = "id") %>%
  # Numeric: median imputation and standardization
  step_impute_median(all_numeric(), -all_outcomes()) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  # Categorical: mode imputation and one-hot encoding
  step_impute_mode(all_nominal(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# 5. Logistic regression specification
log_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Function to evaluate baseline for a given target using ROC AUC
evaluate_baseline <- function(target_var) {
  # Create recipe specific to the target
  rec <- base_recipe %>%
    update_role(all_outcomes(), new_role = "predictor") %>%
    update_role({{ target_var }}, new_role = "outcome") %>%
    recipe(as.formula(paste(target_var, "~ .")), data = data)
  
  # Create workflow
  wf <- workflow() %>%
    add_model(log_spec) %>%
    add_recipe(rec)
  
  # 5-fold CV stratified on the target
  cv_splits <- vfold_cv(data, v = 5, strata = !!sym(target_var))
  
  # Evaluate using ROC AUC
  res <- wf %>%
    fit_resamples(
      resamples = cv_splits,
      metrics = metric_set(roc_auc),
      control = control_resamples(save_pred = TRUE)
    )
  metrics <- collect_metrics(res)
  # Extract mean ROC AUC
  auc <- metrics %>%
    filter(.metric == "roc_auc") %>%
    pull(mean)
  return(auc)
}

# 6. Compute ROC AUC for each target and their mean
results <- map_dbl(targets, evaluate_baseline)
names(results) <- targets

# Print individual AUCs and overall mean
cat("Baseline ROC AUC Results:\n")
for (t in targets) {
  cat(sprintf("%s AUC: %.4f\n", t, results[t]))
}
cat(sprintf("Overall Mean AUC: %.4f\n", mean(results)))
