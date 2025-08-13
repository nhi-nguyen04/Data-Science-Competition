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
library(stacks)
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
dim(train_df)

# -----------------------------------------------
# 4. CREATE TWO SEPARATE SPLITS (ONE PER TARGET)---> Avoids class imbalance
# -----------------------------------------------
#Ensures random split with similar distribution of the outcome variable 
set.seed(921)

data_split_h1n1 <- initial_split(train_df, prop = 0.8, strata = h1n1_vaccine)
train_data_h1n1 <- training(data_split_h1n1)
eval_data_h1n1  <- testing(data_split_h1n1)
set.seed(4513)

data_split_seas <- initial_split(train_df, prop = 0.8, strata = seasonal_vaccine)
train_data_seas <- training(data_split_seas)
eval_data_seas  <- testing(data_split_seas)
# -----------------------------------------------
# 5. SPECIFY BASE MODEL 
# -----------------------------------------------
xgb_model <- boost_tree() %>%
  # Set the mode
  set_mode("classification") %>%
  # Set the engine
  set_engine("xgboost")

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
  step_zv(all_predictors())
#not need for tree based model
 # step_normalize(all_numeric_predictors())





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
  step_zv(all_predictors())
#not need for tree based model
# step_normalize(all_numeric_predictors())


# -----------------------------------------------
# 7.WORKFLOWS
# -----------------------------------------------
xgb_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(xgb_model)

xgb_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(xgb_model)


# -----------------------------------------------
# 8.Train the workflow
# -----------------------------------------------
xgb_h1n1_dt_wkfl_fit <- xgb_wf_h1n1 %>%
  last_fit(split = data_split_h1n1)

xgb_seas_dt_wkfl_fit <- xgb_wf_seas %>%
  last_fit(split = data_split_seas)




# -----------------------------------------------
# 9.Calculate performance metrics on test data
# -----------------------------------------------
xgb_metrics_h1n1 <- xgb_h1n1_dt_wkfl_fit %>% 
  collect_metrics()

xgb_metrics_seas <- xgb_seas_dt_wkfl_fit %>% 
  collect_metrics()

# 1. Pull out predictions (with class‐probabilities)
xgb_h1n1_preds <- collect_predictions(xgb_h1n1_dt_wkfl_fit)
xgb_seas_preds <- collect_predictions(xgb_seas_dt_wkfl_fit)

# 2. Compute ROC curve data
xgb_roc_h1n1 <- roc_curve(xgb_h1n1_preds, truth = h1n1_vaccine, .pred_1)
xgb_roc_seas <- roc_curve(xgb_seas_preds, truth = seasonal_vaccine, .pred_1)

# 3a. Plot separately
autoplot(xgb_roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve (XGBoost)")

autoplot(xgb_roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve (XGBoost)")







#2b.Calcualte the ROC AUC VALUES
roc_auc(xgb_h1n1_preds, truth = h1n1_vaccine, .pred_1)

roc_auc(xgb_seas_preds, truth = seasonal_vaccine, .pred_1)


#confusion matrix

# Compute confusion matrix with Heatmap for counts
xgb_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

xgb_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "heatmap")

# Compute confusion matrix with mosaic plots for visualization of sensitivity and specitivity
xgb_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "mosaic")

xgb_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "mosaic")

#custom metric predictions 
custom_metrics <- metric_set(accuracy,sens,spec,roc_auc)

custom_metrics(xgb_h1n1_preds,truth = h1n1_vaccine, estimate = .pred_class,.pred_1)
custom_metrics(xgb_seas_preds,truth = seasonal_vaccine, estimate = .pred_class,.pred_1)


# -----------------------------------------------
# 10.Cross Validation
# -----------------------------------------------
# Cross-validation gives you a more robust estimate of your out-of-sample performance without 
# the statistical pitfalls - it assesses your model more profoundly.
 
# For speed

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


# Some info from data camp course:
# A very high in-sample AUC like can be an indicator of overfitting. 
# It is also possible that the dataset is just very well structured, or the model might just be terrific
# To check which of these is true, we need to produce out-of-sample estimates of the AUC, and because 
# we don't want to touch the test set yet, we can produce these using cross-validation on the training set.


# Fit resamples
xgb_h1n1_dt_rs <- xgb_wf_h1n1 %>%
  fit_resamples(resamples = h1n1_folds,
                metrics = data_metrics)

xgb_seasonal_dt_rs <- xgb_wf_seas %>%
  fit_resamples(resamples = seasonal_folds,
                metrics = data_metrics)


# View performance metrics

xgb_rs_metrics_h1n1 <- xgb_h1n1_dt_rs %>% 
  collect_metrics()

xgb_rs_metrics_seas <- xgb_seasonal_dt_rs %>% 
  collect_metrics()



# Detailed cross validation results
xgb_h1n1_dt_rs_results <- xgb_h1n1_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for xgboost
xgb_h1n1_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate),
            sd = sd(.estimate))


xgb_seasonal_dt_rs_results <- xgb_seasonal_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for xgboost
xgb_seasonal_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate),
            sd = sd(.estimate))


# -----------------------------------------------
# 11.Hyperparameter tuning
# -----------------------------------------------
plan(multisession, workers = 16) 

xgb_dt_tune_model <- 
  boost_tree(
    trees       = tune(),    # nrounds ∈ [1, 5000]
    tree_depth  = tune(),    # max_depth ∈ [1, 20]
    learn_rate  = tune(),    # eta ∈ [1e−4, 1], log-scale
    sample_size = tune()     # subsample ∈ [0.1, 1]
  ) %>%
  set_engine(
    "xgboost",nthread = 1
  ) %>%
  set_mode("classification")


xgb_dt_tune_model


# Create a tuning workflow
xgb_h1n1_tune_wkfl <- xgb_wf_h1n1 %>% 
  # Replace model
  update_model(xgb_dt_tune_model)

xgb_h1n1_tune_wkfl


xgb_seas_tune_wkfl <- xgb_wf_seas %>% 
  # Replace model
  update_model(xgb_dt_tune_model)

xgb_seas_tune_wkfl


xgb_params <- parameters(
  trees(range = c(1L, 5000L)),                                    # nrounds
  tree_depth(range = c(1L, 20L)),                                 # max_depth
  learn_rate(range = c(1e-4, 1), trans = scales::log10_trans()),  # eta
  sample_prop(range = c(0.1, 1))                                  # subsample
)
# 4) Finalize any data-dependent params (like mtry for RF; not strictly needed here,
#    but good practice if you ever tune something that depends on # predictors)
xgb_h1n1_params <- finalize(xgb_params, train_data_h1n1)
xgb_seas_params <- finalize(xgb_params, train_data_seas)



# Finalize parameter ranges for both
# xgb_h1n1_params  <- finalize(parameters(xgb_dt_tune_model), train_data_h1n1)
# xgb_seas_params  <- finalize(parameters(xgb_dt_tune_model), train_data_seas)

# Hyperparameter tuning with grid search




set.seed(214)
xgb_h1n1_grid <- grid_random(xgb_h1n1_params, size = 500)

set.seed(215)
xgb_seas_grid <- grid_random(xgb_seas_params, size = 500)


ctrl_grid <- control_stack_grid()      # for tune_grid()
ctrl_res <- control_stack_resamples()  # for fit_resamples()

# Hyperparameter tuning
xgb_h1n1_dt_tuning <- xgb_h1n1_tune_wkfl %>%
  tune_grid(resamples = h1n1_folds,
            grid = xgb_h1n1_grid,
            metrics = data_metrics,
            control = control_stack_grid())


xgb_seas_dt_tuning <- xgb_seas_tune_wkfl %>%
  tune_grid(resamples = seasonal_folds,
            grid = xgb_seas_grid,
            metrics = data_metrics,
            control = control_stack_grid())




# View results
xgb_h1n1_dt_tuning %>% 
  collect_metrics()


# View results
xgb_seas_dt_tuning %>% 
  collect_metrics()


# Collect detailed tuning results
xgb_h1n1_dt_tuning_results <- xgb_h1n1_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
xgb_h1n1_dt_tuning_results %>% 
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))


# Collect detailed tuning results
xgb_seas_dt_tuning_results <- xgb_seas_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
xgb_seas_dt_tuning_results %>% 
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))


# -----------------------------------------------
# 12.Selecting the best model
# -----------------------------------------------

# Display 5 best performing models
xgb_h1n1_dt_tuning %>% 
  show_best(metric = 'roc_auc', n = 5)

xgb_seas_dt_tuning %>% 
  show_best(metric = 'roc_auc', n = 5)



# Select based on best performance
xgb_best_h1n1_dt_model <- xgb_h1n1_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

xgb_best_h1n1_dt_model


xgb_best_seas_dt_model <- xgb_seas_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

xgb_best_seas_dt_model


# -----------------------------------------------
# 13.Finalize your workflow
# -----------------------------------------------
xgb_final_h1n1_tune_wkfl <- xgb_h1n1_tune_wkfl %>% 
  finalize_workflow(xgb_best_h1n1_dt_model)

xgb_final_h1n1_tune_wkfl


xgb_final_seas_tune_wkfl <- xgb_seas_tune_wkfl %>% 
  finalize_workflow(xgb_best_seas_dt_model)

xgb_final_seas_tune_wkfl


# -----------------------------------------------
# 14. LAST_FIT ON THE HELD-OUT SPLITS
# -----------------------------------------------
xgb_h1n1_final_fit <- xgb_final_h1n1_tune_wkfl %>% 
  last_fit(split = data_split_h1n1)

xgb_seas_final_fit <- xgb_final_seas_tune_wkfl %>% 
  last_fit(split = data_split_seas)


#-----------------------------------------------
# 15. COLLECT METRICS
# -----------------------------------------------
xgb_h1n1_final_fit %>% collect_metrics()
xgb_seas_final_fit  %>% collect_metrics()


# -----------------------------------------------
# 16. ROC CURVE VISUALIZATION (via last_fit results)
# -----------------------------------------------
#library(yardstick)
#library(ggplot2)

# 1) Pull out predictions (with probabilities)
xgb_aftr_tunning_h1n1_preds <- xgb_h1n1_final_fit %>% 
  collect_predictions()
xgb_aftr_tunning_seas_preds <- xgb_seas_final_fit  %>% 
  collect_predictions()

# 2) Compute ROC curve data
xgb_aftr_tunning_roc_h1n1 <- roc_curve(xgb_aftr_tunning_h1n1_preds, truth = h1n1_vaccine, .pred_1)
xgb_aftr_tunning_roc_seas <- roc_curve(xgb_aftr_tunning_seas_preds, truth = seasonal_vaccine, .pred_1)

# 3a) Plot separately
autoplot(xgb_aftr_tunning_roc_h1n1) +
  ggtitle("Final H1N1 Vaccine ROC Curve (XGBoost)")
autoplot(xgb_aftr_tunning_roc_seas) + 
  ggtitle("Final Seasonal Vaccine ROC Curve(XGBoost)")


# -----------------------------------------------
# 17. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
xgb_final_h1n1 <- fit(xgb_final_h1n1_tune_wkfl, train_df)
xgb_final_seas <- fit(xgb_final_seas_tune_wkfl, train_df)


# Here I am checking for variable importance
vip::vip(xgb_final_h1n1, num_features= 15)
vip::vip(xgb_final_seas,  num_features= 15)


# -----------------------------------------------
# 18. MAKE PREDICTIONS ON TEST DATA
# -----------------------------------------------
# Add missing columns to test data to match training structure
test_df_prepared <- test_df %>%
  mutate(
    h1n1_vaccine = factor(NA, levels = c(1, 0)),
    seasonal_vaccine = factor(NA, levels = c(1, 0)),
    strata = NA_character_)

test_pred_h1n1_xgboost <- predict(xgb_final_h1n1, test_df_prepared, type = "prob") %>%
  pull(.pred_1)
test_pred_seas_xgboost <- predict(xgb_final_seas, test_df_prepared, type = "prob") %>% 
  pull(.pred_1)

head(test_pred_h1n1_xgboost)
head(test_pred_seas_xgboost)


# -----------------------------------------------
# 19. CREATE SUBMISSION FILE
# -----------------------------------------------
submission_xgboost <- tibble(
  respondent_id = test_df$respondent_id,
  h1n1_vaccine = test_pred_h1n1_xgboost,
  seasonal_vaccine = test_pred_seas_xgboost
)


# -----------------------------------------------
# 20. SAVE SUBMISSION
# -----------------------------------------------
write_csv(submission_xgboost, "xgboost_predictions_submission.csv")
