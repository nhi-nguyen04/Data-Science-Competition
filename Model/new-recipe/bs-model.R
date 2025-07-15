# Predicting H1N1 and Seasonal Flu Vaccine Uptake
# A tidymodels approach with ridge logistic regression


# -----------------------------------------------
# 1. SET UP ENVIRONMENT
# -----------------------------------------------
library(tidyverse)
library(tidymodels)
library(baguette)
library(tune)
library(future)
library(vip) # for variable importance
library(skimr)
set.seed(6)


# -----------------------------------------------
# 2. LOAD DATA
# -----------------------------------------------
train_features <- read_csv("Data/training_set_features.csv")
train_labels   <- read_csv("Data/training_set_labels.csv")
train_df       <- left_join(train_features, train_labels, by = "respondent_id")
test_df        <- read_csv("Data/test_set_features.csv")

glimpse(train_df)
#skim(train_df)
#View(train_df)

# -----------------------------------------------
# 3. DATA PREPARATION 
# -----------------------------------------------
train_df <- train_df %>%
  mutate(
    h1n1_vaccine     = factor(h1n1_vaccine, levels = c(1, 0)),
    seasonal_vaccine = factor(seasonal_vaccine, levels = c(1, 0))
  )


ordinal_vars <- c(
  "h1n1_concern",                    # 0-3 scale
  "h1n1_knowledge",                  # 0-2 scale  
  "opinion_h1n1_vacc_effective",     # 1-5 scale
  "opinion_h1n1_risk",               # 1-5 scale
  "opinion_h1n1_sick_from_vacc",     # 1-5 scale
  "opinion_seas_vacc_effective",     # 1-5 scale
  "opinion_seas_risk",               # 1-5 scale
  "opinion_seas_sick_from_vacc"      # 1-5 scale
)

# Variables that should be treated as nominal (unordered factors)
nominal_vars <- c(
  "age_group", "education", "race", "sex", "income_poverty",
  "marital_status", "rent_or_own", "employment_status",
  "hhs_geo_region", "census_msa", "employment_industry",
  "employment_occupation"
)

# Binary variables (0/1) that should be factors
binary_vars <- c(
  "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask",
  "behavioral_wash_hands", "behavioral_large_gatherings", "behavioral_outside_home",
  "behavioral_touch_face", "doctor_recc_h1n1", "doctor_recc_seasonal",
  "chronic_med_condition", "child_under_6_months", "health_worker", "health_insurance"
)

# Count variables (should remain numeric)
count_vars <- c("household_adults", "household_children")

# Apply transformations
train_df <- train_df %>%
  mutate(
    # Convert ordinal variables to ordered factors
    across(all_of(ordinal_vars), ~ factor(.x, ordered = TRUE)),
    
    # Convert nominal variables to factors
    across(all_of(nominal_vars), ~ factor(.x)),
    
    # Convert binary variables to factors with meaningful labels
    across(all_of(binary_vars), ~ factor(.x, levels = c(0, 1)))
  )

# Apply same transformations to test data
test_df <- test_df %>%
  mutate(
    # Convert ordinal variables to ordered factors
    across(all_of(ordinal_vars), ~ factor(.x, ordered = TRUE)),
    
    # Convert nominal variables to factors
    across(all_of(nominal_vars), ~ factor(.x)),
    
    # Convert binary variables to factors with meaningful labels
    across(all_of(binary_vars), ~ factor(.x, levels = c(0, 1)))
  )
glimpse(train_df)

# -----------------------------------------------
# 4. CREATE TWO SEPARATE SPLITS (ONE PER TARGET)---> Avoids class imbalance
# -----------------------------------------------
#Ensures random split with similar distribution of the outcome variable 
data_split_h1n1 <- initial_split(train_df, prop = 0.8, strata = h1n1_vaccine)
train_data_h1n1 <- training(data_split_h1n1)
eval_data_h1n1  <- testing(data_split_h1n1)

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
# 6. R E C I P E  –– consistent imputation + dummies
# -----------------------------------------------
# Keep the recipe basic — no interactions
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>%
  
  # Step 1: Impute + encode
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # Step 2: Interactions (after dummy encoding)
  step_interact(terms = ~ starts_with("doctor_recc_h1n1_"):starts_with("opinion_h1n1_vacc_effective_")) %>%
  step_interact(terms = ~ starts_with("doctor_recc_h1n1_"):starts_with("opinion_h1n1_risk_")) %>%
  step_interact(terms = ~ starts_with("opinion_h1n1_vacc_effective_"):starts_with("opinion_h1n1_risk_")) %>%
  step_interact(terms = ~ starts_with("doctor_recc_seasonal_"):starts_with("opinion_seas_vacc_effective_")) %>%
  step_interact(terms = ~ starts_with("opinion_h1n1_sick_from_vacc_"):starts_with("opinion_seas_sick_from_vacc_")) %>%
  
  # Step 3: Final cleanup
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())





seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  
  # Step 1: Impute + encode
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # Step 2: Interactions (after dummy encoding, using dummy names)
  step_interact(terms = ~ starts_with("opinion_seas_vacc_effective_"):starts_with("opinion_seas_risk_")) %>%
  step_interact(terms = ~ starts_with("opinion_seas_risk_"):starts_with("doctor_recc_seasonal_")) %>%
  step_interact(terms = ~ starts_with("opinion_seas_vacc_effective_"):starts_with("doctor_recc_seasonal_")) %>%
  step_interact(terms = ~ starts_with("opinion_h1n1_risk_"):starts_with("opinion_seas_vacc_effective_")) %>%
  step_interact(terms = ~ starts_with("opinion_seas_sick_from_vacc_"):starts_with("opinion_h1n1_sick_from_vacc_")) %>%
  
  # Step 3: Final cleanup
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())



  
  

# -----------------------------------------------
# 7. CREATE WORKFLOWS
# -----------------------------------------------
lr_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(log_spec)

lr_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(log_spec)

# -----------------------------------------------
# 8. TRAIN & EVALUATE ON SPLIT
# -----------------------------------------------
set.seed(6)

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

# 1. Pull out predictions (with class‐probabilities)
lr_h1n1_preds <- collect_predictions(lr_h1n1_dt_wkfl_fit)
lr_seas_preds <- collect_predictions(lr_seas_dt_wkfl_fit)

# 2. Compute ROC curve data
lr_roc_h1n1 <- roc_curve(lr_h1n1_preds, truth = h1n1_vaccine, .pred_1)
lr_roc_seas <- roc_curve(lr_seas_preds, truth = seasonal_vaccine, .pred_1)

# 2a. Plot separately  ROC curves
autoplot(lr_roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve")

autoplot(lr_roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve")

#2b.Calcualte the ROC AUC VALUES
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


# View performance metrics

# Some info from data camp course:
# A very high in-sample AUC like can be an indicator of overfitting. 
# It is also possible that the dataset is just very well structured, or the model might just be terrific
# To check which of these is true, we need to produce out-of-sample estimates of the AUC, and because 
# we don't want to touch the test set yet, we can produce these using cross-validation on the training set.

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

#For H1N1  Flu Vaccine
#We have used cross validation to evaluate the performance of your random forest  workflow. 
#Across the 10 cross validation folds, the average area under the ROC curve was ... . 
#The average sensitivity and specificity were ... and ..., respectively.

#For  Seasonal Flu Vaccine
#We have used cross validation to evaluate the performance of your random forest  workflow. 
#Across the 10 cross validation folds, the average area under the ROC curve was ... . 
#The average sensitivity and specificity were ... and ..., respectively.


#Use this to compare against other model types


# -----------------------------------------------
# 11.Hyperparameter tuning
# -----------------------------------------------
lr_dt_tune_model <- logistic_reg(
  penalty = tune(),   # let penalty be tuned
  mixture = tune()    # optionally tune mixture (0 = ridge, 1 = lasso)
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



#we went from range(-4,1) to this
penalty_range <- penalty(range = c(-4, 4))  # log scale
mixture_range <- mixture(range = c(0, 1))

log_grid <- grid_regular(penalty_range, mixture_range, levels = 5)

set.seed(123)
cv_folds_lr_h1n1 <- vfold_cv(train_data_h1n1, v = 10)
cv_folds_lr_seas <- vfold_cv(train_data_seas, v = 10)


# Hyperparameter tuning with grid search

# For speed
plan(multisession, workers = 4) 


# Hyperparameter tuning
lr_h1n1_dt_tuning <- lr_h1n1_tune_wkfl %>% 
  tune_grid(resamples = h1n1_folds,
            grid = log_grid,
            metrics = data_metrics)


lr_seas_dt_tuning <- lr_seas_tune_wkfl %>% 
  tune_grid(resamples = seasonal_folds,
            grid = log_grid,
            metrics = data_metrics)

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
#Tune random forest trained with entire training dataset
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
#library(yardstick)
#library(ggplot2)

# 1) Pull out predictions (with probabilities)
lr_aftr_tunning_h1n1_preds <- lr_h1n1_final_fit %>% 
  collect_predictions()
lr_aftr_tunning_seas_preds <- lr_seas_final_fit  %>%
  collect_predictions()

# 2) Compute ROC curve data
lr_aftr_tunning_roc_h1n1 <- roc_curve(lr_aftr_tunning_h1n1_preds, truth = h1n1_vaccine, .pred_1)
lr_aftr_tunning_roc_seas <- roc_curve(lr_aftr_tunning_seas_preds, truth = seasonal_vaccine, .pred_1)

# 3a) Plot separately
autoplot(lr_aftr_tunning_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (Random Forest)")
autoplot(lr_aftr_tunning_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (Random Forest)")

# -----------------------------------------------
# 17. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
lr_final_h1n1 <- fit(lr_final_h1n1_tune_wkfl, train_df)
lr_final_seas <- fit(lr_final_seas_tune_wkfl, train_df)

# Here I am checking for variable importance
vip::vip(lr_final_h1n1, num_features= 20)
vip::vip(lr_final_seas,  num_features= 20)


# vip::vi(lr_final_h1n1)%>%
#   arrange(desc(Importance)) %>%
#   slice_head(n = 5) %>%
#   pull(Variable)
# 
# 
# vip::vi(lr_final_seas)%>%
#   arrange(desc(Importance)) %>%
#   slice_head(n = 5) %>%
#   pull(Variable)


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
write_csv(submission_log_reg, "logistic_reg_base_model_predictions_submission-converted-to-cat.csv")

# Print confirmation message
#cat("Submission file created successfully with", nrow(submission), "predictions.\n")