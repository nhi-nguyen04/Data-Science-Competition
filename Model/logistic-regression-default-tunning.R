# -----------------------------------------------
#0. Author: Nhi Nguyen
# -----------------------------------------------

# -----------------------------------------------
# 1. SET UP ENVIRONMENT
# -----------------------------------------------
library(tidyverse)
library(tidymodels)
library(baguette)
library(tune)
library(future)
library(vip) 
library(skimr)
set.seed(6)


# -----------------------------------------------
# 2. LOAD DATA
# -----------------------------------------------
train_features <- read_csv("Data/training_set_features.csv")
train_labels <- read_csv("Data/training_set_labels.csv")
train_df <- left_join(train_features, train_labels, by = "respondent_id")
test_df <- read_csv("Data/test_set_features.csv")


# -----------------------------------------------
# 3. DATA PREPARATION 
# -----------------------------------------------
train_df <- train_df %>%
  mutate(
    h1n1_vaccine = factor(h1n1_vaccine, levels = c(1, 0)),
    seasonal_vaccine = factor(seasonal_vaccine, levels = c(1, 0))
  )

# IDENTIFY NUMERIC VS. CATEGORICAL BY TYPE
numeric_vars <- train_df %>% select(where(is.numeric)) %>% names()
categorical_vars <- train_df %>% select(where(is.character), where(is.factor)) %>% names()

# Remove the target + ID 
numeric_vars <- setdiff(numeric_vars, c("respondent_id"))
categorical_vars <- setdiff(categorical_vars, c("respondent_id", "h1n1_vaccine", "seasonal_vaccine"))
# -----------------------------------------------
# 4. CREATE TWO SEPARATE SPLITS (ONE PER TARGET)---> Avoids class imbalance
# -----------------------------------------------
#Ensures random split with similar distribution of the outcome variable 
set.seed(9678)

#Ensures random split with similar distribution of the outcome variable 
data_split_h1n1 <- initial_split(train_df, prop = 0.8, strata = h1n1_vaccine)
train_data_h1n1 <- training(data_split_h1n1)
eval_data_h1n1  <- testing(data_split_h1n1)

set.seed(92398)

data_split_seas <- initial_split(train_df, prop = 0.8, strata = seasonal_vaccine)
train_data_seas <- training(data_split_seas)
eval_data_seas  <- testing(data_split_seas)

# -----------------------------------------------
# 5. SPECIFY BASE MODEL (LOGISTIC REGRESSION)
# -----------------------------------------------
log_spec <- logistic_reg(
  penalty = 1,
  mixture = 0
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# -----------------------------------------------
# 6. R E C I P E  
# -----------------------------------------------
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  # Remove the other target (seasonal) if it’s present
  step_rm(seasonal_vaccine) %>%
  # Impute all numeric predictors by median:
  step_impute_median(all_numeric_predictors()) %>%
  # Create an "unknown" level for any missing factor
  step_unknown(all_nominal_predictors()) %>%
  # One‐hot encode all factors
  step_dummy(all_nominal_predictors()) %>%
  # Drops any predictors that have zero variance
  step_zv(all_predictors()) %>% 
  # Normalize numeric columns
  step_normalize(all_numeric_predictors()) 



seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# -----------------------------------------------
# 7. CREATE WORKFLOWS
# -----------------------------------------------
lr_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(log_spec)

lr_wf_h1n1

lr_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(log_spec)

# -----------------------------------------------
# 8. TRAIN & EVALUATE ON SPLIT
# -----------------------------------------------
lr_h1n1_dt_wkfl_fit <- lr_wf_h1n1 %>% 
  last_fit(split = data_split_h1n1)

lr_seas_dt_wkfl_fit <- lr_wf_seas %>% 
  last_fit(split = data_split_seas)


# -----------------------------------------------
# 9.Model Evaluation -->Calculate performance metrics on test data(20% of the trainning data)
# -----------------------------------------------
#predictions and metrics are generated with test dataset
lr_metrics_h1n1 <- lr_h1n1_dt_wkfl_fit %>% 
  collect_metrics()


lr_metrics_seas <- lr_seas_dt_wkfl_fit %>% 
  collect_metrics()

# Pull out predictions (with class‐probabilities)
lr_h1n1_preds <- collect_predictions(lr_h1n1_dt_wkfl_fit)
lr_h1n1_preds
lr_seas_preds <- collect_predictions(lr_seas_dt_wkfl_fit)

# Compute ROC curve data
lr_roc_h1n1 <- roc_curve(lr_h1n1_preds, truth = h1n1_vaccine, .pred_1)
lr_roc_h1n1
lr_roc_seas <- roc_curve(lr_seas_preds, truth = seasonal_vaccine, .pred_1)

# Plot separately  ROC curves
autoplot(lr_roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve")

autoplot(lr_roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve")

# Calculate the ROC AUC VALUES
roc_auc(lr_h1n1_preds, truth = h1n1_vaccine, .pred_1)

roc_auc(lr_seas_preds, truth = seasonal_vaccine, .pred_1)


#confusion matrix

# Compute confusion matrix with Heatmap for counts
lr_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

lr_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "heatmap")

# Compute confusion matrix with mosaic plots for visualization of sensitivity and specitivity
lr_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "mosaic")

lr_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "mosaic")

#custom metric predictions 
custom_metrics <- metric_set(accuracy,sens,spec,roc_auc)

custom_metrics(lr_h1n1_preds,truth = h1n1_vaccine, estimate = .pred_class,.pred_1)
custom_metrics(lr_seas_preds,truth = seasonal_vaccine, estimate = .pred_class,.pred_1)

# -----------------------------------------------
# 10.Cross Validation--> Estimating performance with CV
# -----------------------------------------------
#Cross-validation gives you a more robust estimate of your out-of-sample performance without 
#the statistical pitfalls - it assesses your model more profoundly.

#For speed
plan(multisession, workers = 4) 

set.seed(290)
h1n1_folds <- vfold_cv(train_data_h1n1, 
                       v = 10,
                       strata = h1n1_vaccine)

h1n1_folds

set.seed(291)

seasonal_folds <- vfold_cv(train_data_seas, 
                           v = 10,
                           strata = seasonal_vaccine)

seasonal_folds

# Create custom metrics function
data_metrics <- metric_set(accuracy,roc_auc, sens, spec)


# Fit resamples
lr_h1n1_dt_rs <- lr_wf_h1n1 %>% 
  fit_resamples(resamples = h1n1_folds,
                metrics = data_metrics)

lr_seasonal_dt_rs <- lr_wf_seas %>% 
  fit_resamples(resamples = seasonal_folds,
                metrics = data_metrics)

lr_rs_metrics_h1n1 <- lr_h1n1_dt_rs %>% 
  collect_metrics()

lr_rs_metrics_seas <- lr_seasonal_dt_rs %>% 
  collect_metrics()


# Detailed cross validation results
lr_h1n1_dt_rs_results <- lr_h1n1_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for decision tree
lr_h1n1_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate),
            sd = sd(.estimate))


lr_seasonal_dt_rs_results <- lr_seasonal_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for decision tree
lr_seasonal_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate),
            sd = sd(.estimate))
# -----------------------------------------------
# 11.Hyperparameter tuning
# -----------------------------------------------
lr_dt_tune_model <- logistic_reg(
  penalty = tune(),   
  mixture = tune()    
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")


lr_dt_tune_model


# Create a tuning workflow
lr_h1n1_tune_wkfl <- lr_wf_h1n1 %>% 
  # Replace model
  update_model(lr_dt_tune_model)

lr_h1n1_tune_wkfl


lr_seas_tune_wkfl <- lr_wf_seas %>% 
  # Replace model
  update_model(lr_dt_tune_model)

lr_seas_tune_wkfl

cv_folds_lr_h1n1 <- vfold_cv(train_data_h1n1, v = 5)
cv_folds_lr_seas <- vfold_cv(train_data_seas, v = 5)

set.seed(123)
plan(multisession, workers = 4)

lr_h1n1_dt_tuning <- lr_h1n1_tune_wkfl %>%
  tune_grid(
    resamples = cv_folds_lr_h1n1,
    grid = 50,                 
    metrics = data_metrics
  )

lr_seas_dt_tuning <- lr_seas_tune_wkfl %>%
  tune_grid(
    resamples = cv_folds_lr_seas,
    grid = 50,
    metrics = data_metrics
  )



# View results
lr_h1n1_dt_tuning %>% 
  collect_metrics()


# View results
lr_seas_dt_tuning %>% 
  collect_metrics()


# Collect detailed tuning results
lr_h1n1_dt_tuning_results <- lr_h1n1_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
lr_h1n1_dt_tuning_results %>% 
  filter(.metric == "roc_auc") %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))

# Collect detailed tuning results
lr_seas_dt_tuning_results <- lr_seas_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
lr_seas_dt_tuning_results %>% 
  filter(.metric == "roc_auc") %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))



# -----------------------------------------------
# 12.Selecting the best model
# -----------------------------------------------
# Display 5 best performing models
lr_h1n1_dt_tuning %>% 
  show_best(metric = "roc_auc", n = 5)

lr_seas_dt_tuning %>% 
  show_best(metric = "roc_auc", n = 5)


# Select based on best performance
lr_best_h1n1_dt_model <- lr_h1n1_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

lr_best_h1n1_dt_model


lr_best_seas_dt_model <- lr_seas_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

lr_best_seas_dt_model

# -----------------------------------------------
# 13.Finalize your workflow
# -----------------------------------------------
lr_final_h1n1_tune_wkfl <- lr_h1n1_tune_wkfl %>% 
  finalize_workflow(lr_best_h1n1_dt_model)

lr_final_h1n1_tune_wkfl


lr_final_seas_tune_wkfl <- lr_seas_tune_wkfl %>% 
  finalize_workflow(lr_best_seas_dt_model)

lr_final_seas_tune_wkfl

# -----------------------------------------------
# 14. LAST_FIT ON THE HELD-OUT SPLITS
# -----------------------------------------------
#Here Training and test dataset are created
#recipe trained and applied
#Tune logistic regression trained with entire training dataset
lr_h1n1_final_fit <- 
  lr_final_h1n1_tune_wkfl %>% 
  last_fit(split = data_split_h1n1)

lr_seas_final_fit <- 
  lr_final_seas_tune_wkfl %>% 
  last_fit(split = data_split_seas)



#-----------------------------------------------
# 15. COLLECT METRICS
# -----------------------------------------------
#predictions and metrics are generated with test dataset
lr_h1n1_final_fit %>% collect_metrics()
lr_seas_final_fit  %>% collect_metrics()

# -----------------------------------------------
# 16. ROC CURVE VISUALIZATION (via last_fit results)
# -----------------------------------------------

# Pull out predictions (with probabilities)
lr_aftr_tunning_h1n1_preds <- lr_h1n1_final_fit %>% 
  collect_predictions()
lr_aftr_tunning_seas_preds <- lr_seas_final_fit  %>%
  collect_predictions()

# Compute ROC curve data
lr_aftr_tunning_roc_h1n1 <- roc_curve(lr_aftr_tunning_h1n1_preds, truth = h1n1_vaccine, .pred_1)
lr_aftr_tunning_roc_seas <- roc_curve(lr_aftr_tunning_seas_preds, truth = seasonal_vaccine, .pred_1)

# Plot separately
autoplot(lr_aftr_tunning_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (Random Forest)")
autoplot(lr_aftr_tunning_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (Random Forest)")

# -----------------------------------------------
# 17. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
lr_final_h1n1 <- fit(lr_final_h1n1_tune_wkfl, train_df)
lr_final_seas <- fit(lr_final_seas_tune_wkfl, train_df)
# -----------------------------------------------
# 18. MAKE PREDICTIONS ON TEST DATA
# -----------------------------------------------
# Add missing columns to test data to match training structure
test_df_prepared <- test_df %>%
  mutate(
    h1n1_vaccine = factor(NA, levels = c(1, 0)),
    seasonal_vaccine = factor(NA, levels = c(1, 0)),
    strata = NA_character_
  )

test_pred_h1n1_log_reg <- predict(lr_final_h1n1, test_df_prepared, type = "prob") %>% pull(.pred_1)
test_pred_seas_log_reg <- predict(lr_final_seas, test_df_prepared, type = "prob") %>% pull(.pred_1)

head(test_pred_h1n1_log_reg)
head(test_pred_seas_log_reg)

# -----------------------------------------------
# 19. CREATE SUBMISSION FILE
# -----------------------------------------------
submission_log_reg <- tibble(
  respondent_id = test_df$respondent_id,
  h1n1_vaccine = test_pred_h1n1_log_reg,
  seasonal_vaccine = test_pred_seas_log_reg
)

# -----------------------------------------------
# 20. SAVE SUBMISSION
# -----------------------------------------------
#Already available
#write_csv(submission_log_reg, "logistic_reg__default-tunning.csv")