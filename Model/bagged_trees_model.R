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
# 5. SPECIFY BASE MODEL (RPART TREE)
# -----------------------------------------------
bt_model <- bag_tree() %>%
  set_engine("rpart", times = 20) %>%
  set_mode("classification") 


# -----------------------------------------------
# 6. R E C I P E –– Optimized for H1N1 (Tree-based models)
# -----------------------------------------------
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>%
  
  # Step 1: Impute + encode
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  #step_unknown(all_nominal_predictors()) %>%
  #step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
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
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  #step_unknown(all_nominal_predictors()) %>%
  #step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
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
bt_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(bt_model)

bt_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(bt_model)


# -----------------------------------------------
# 8.Train the workflow
# -----------------------------------------------
# bt_h1n1_dt_wkfl_fit <- bt_wf_h1n1 %>% 
#   last_fit(split = data_split_h1n1)
# 
# 
# bt_seas_dt_wkfl_fit <- bt_wf_seas %>% 
#   last_fit(split = data_split_seas)
# 
# saveRDS(bt_h1n1_dt_wkfl_fit, "results/section8_bt_h1n1_dt_wkfl_fit.rds")
# saveRDS(bt_seas_dt_wkfl_fit, "results/section8_bt_seas_dt_wkfl_fit.rds")


bt_h1n1_dt_wkfl_fit     <- readRDS("Model/results/section8_bt_h1n1_dt_wkfl_fit.rds")
bt_seas_dt_wkfl_fit     <- readRDS("Model/results/section8_bt_seas_dt_wkfl_fit.rds")

# -----------------------------------------------
# 9.Model Evaluation -->Calculate performance metrics on test data(20% of the trainning data)
# -----------------------------------------------
bt_metrics_h1n1 <- bt_h1n1_dt_wkfl_fit %>% 
  collect_metrics()

bt_metrics_seas <- bt_seas_dt_wkfl_fit %>% 
  collect_metrics()


# 1. Pull out predictions (with class‐probabilities)
bt_h1n1_preds <- collect_predictions(bt_h1n1_dt_wkfl_fit)
bt_seas_preds <- collect_predictions(bt_seas_dt_wkfl_fit)

# 2. Compute ROC curve data
bt_roc_h1n1 <- roc_curve(bt_h1n1_preds, truth = h1n1_vaccine, .pred_1)
bt_roc_seas <- roc_curve(bt_seas_preds, truth = seasonal_vaccine, .pred_1)


# 2a. Plot separately  ROC curves
autoplot(bt_roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve")

autoplot(bt_roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve")

#2b.Calcualte the ROC AUC VALUES
roc_auc(bt_h1n1_preds, truth = h1n1_vaccine, .pred_1)

roc_auc(bt_seas_preds, truth = seasonal_vaccine, .pred_1)

#confusion matrix

# Compute confusion matrix with Heatmap for counts
bt_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

bt_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "heatmap")


# Compute confusion matrix with mosaic plots for visualization of sensitivity and specitivity
bt_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "mosaic")

bt_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "mosaic")


#custom metric predictions 
custom_metrics <- metric_set(accuracy,sens,spec,roc_auc)

custom_metrics(bt_h1n1_preds,truth = h1n1_vaccine, estimate = .pred_class,.pred_1)
custom_metrics(bt_seas_preds,truth = seasonal_vaccine, estimate = .pred_class,.pred_1)


# -----------------------------------------------
# 10.Cross Validation
# -----------------------------------------------
# Cross-validation gives you a more robust estimate of your out-of-sample performance without 
# the statistical pitfalls - it assesses your model more profoundly.

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
# bt_h1n1_dt_rs <- bt_wf_h1n1 %>% 
#   fit_resamples(resamples = h1n1_folds,
#                 metrics = data_metrics)
# 
# bt_seasonal_dt_rs <- bt_wf_seas %>% 
#   fit_resamples(resamples = seasonal_folds,
#                 metrics = data_metrics)
# 
# 
# saveRDS(bt_h1n1_dt_rs, "results/section10_bt_h1n1_dt_rs.rds")
# saveRDS(bt_seasonal_dt_rs, "results/section10_bt_seasonal_dt_rs.rds")


bt_h1n1_dt_rs           <- readRDS("Model/results/section10_bt_h1n1_dt_rs.rds")
bt_seasonal_dt_rs       <- readRDS("Model/results/section10_bt_seasonal_dt_rs.rds")


# View performance metrics

bt_rs_metrics_h1n1 <- bt_h1n1_dt_rs %>% 
  collect_metrics()

bt_rs_metrics_seas <- bt_seasonal_dt_rs %>% 
  collect_metrics()



# Detailed cross validation results
bt_h1n1_dt_rs_results <- bt_h1n1_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for decision tree
bt_rs_perf_h1n1 <- bt_h1n1_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))


bt_seasonal_dt_rs_results <- bt_seasonal_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for decision tree
bt_rs_perf_seas <- bt_seasonal_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))


# -----------------------------------------------
# 11.Hyperparameter tuning
# -----------------------------------------------
# Define the model with tunable parameters
bt_dt_tune_model <- bag_tree(cost_complexity = tune(),
                             tree_depth = tune(),
                             min_n = tune()) %>%
  set_engine("rpart", times = 20) %>%
  set_mode("classification") 



bt_dt_tune_model


# Create a tuning workflow
bt_h1n1_tune_wkfl <- bt_wf_h1n1 %>% 
  # Replace model
  update_model(bt_dt_tune_model)

bt_h1n1_tune_wkfl


bt_seas_tune_wkfl <- bt_wf_seas %>% 
  # Replace model
  update_model(bt_dt_tune_model)

bt_seas_tune_wkfl



# Finalize parameter ranges for both
bt_h1n1_params  <- finalize(parameters(bt_dt_tune_model), train_data_h1n1)
bt_seas_params  <- finalize(parameters(bt_dt_tune_model), train_data_seas)


# Hyperparameter tuning with grid search
set.seed(214)
# For speed
plan(multisession, workers = 4) 


set.seed(214)
bt_h1n1_grid <- grid_random(bt_h1n1_params, size = 100)

set.seed(215)
bt_seas_grid <- grid_random(bt_seas_params, size = 100)


ctrl_grid <- control_stack_grid()      # for tune_grid()
ctrl_res <- control_stack_resamples()  # for fit_resamples()

# Hyperparameter tuning
# bt_h1n1_dt_tuning <- bt_h1n1_tune_wkfl %>% 
#   tune_grid(resamples = h1n1_folds,
#             grid = bt_h1n1_grid,
#             metrics = data_metrics,
#             control = control_stack_grid())
# 
# 
# bt_seas_dt_tuning <- bt_seas_tune_wkfl %>% 
#   tune_grid(resamples = seasonal_folds,
#             grid = bt_seas_grid,
#             metrics = data_metrics,
#             control = control_stack_grid())
# 
# 
# saveRDS(bt_h1n1_dt_tuning, "results/section11_bt_h1n1_dt_tuning.rds")
# saveRDS(bt_seas_dt_tuning, "results/section11_bt_seas_dt_tuning.rds")


bt_h1n1_dt_tuning       <- readRDS("Model/results/section11_bt_h1n1_dt_tuning.rds")
bt_seas_dt_tuning       <- readRDS("Model/results/section11_bt_seas_dt_tuning.rds")


# View results
bt_h1n1_dt_tuning %>% 
  collect_metrics()


# View results
bt_seas_dt_tuning %>% 
  collect_metrics()



# Collect detailed tuning results
bt_h1n1_dt_tuning_results <- bt_h1n1_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
bt_h1n1_dt_tuning_results %>% 
  filter(.metric == "roc_auc") %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))



# Collect detailed tuning results
bt_seas_dt_tuning_results <- bt_seas_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
bt_seas_dt_tuning_results %>% 
  filter(.metric == "roc_auc") %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))


# -----------------------------------------------
# 12.Selecting the best model
# -----------------------------------------------

# Display 5 best performing models
bt_h1n1_dt_tuning %>% 
  show_best(metric = "roc_auc", n = 5)

bt_seas_dt_tuning %>% 
  show_best(metric = "roc_auc", n = 5)



# Select based on best performance
bt_best_h1n1_dt_model <- bt_h1n1_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = "roc_auc")

bt_best_h1n1_dt_model


bt_best_seas_dt_model <- bt_seas_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = "roc_auc")

bt_best_seas_dt_model


# -----------------------------------------------
# 13.Finalize your workflow
# -----------------------------------------------
bt_final_h1n1_tune_wkfl <- bt_h1n1_tune_wkfl %>% 
  finalize_workflow(bt_best_h1n1_dt_model)

bt_final_h1n1_tune_wkfl


bt_final_seas_tune_wkfl <- bt_seas_tune_wkfl %>% 
  finalize_workflow(bt_best_seas_dt_model)

bt_final_seas_tune_wkfl


# -----------------------------------------------
# 14. LAST_FIT ON THE HELD-OUT SPLITS
# -----------------------------------------------
#Here Training and test dataset are created
#recipe trained and applied
#Tune random forest trained with entire training dataset
bt_h1n1_final_fit <- 
  bt_final_h1n1_tune_wkfl %>% 
  last_fit(split = data_split_h1n1)

bt_seas_final_fit <- 
  bt_final_seas_tune_wkfl %>% 
  last_fit(split = data_split_seas)
#-----------------------------------------------
# 15. COLLECT METRICS
# -----------------------------------------------
#predictions and metrics are generated with test dataset
bt_h1n1_final_fit %>% collect_metrics()
bt_seas_final_fit  %>% collect_metrics()

# -----------------------------------------------
# 16. ROC CURVE VISUALIZATION (via last_fit results)
# -----------------------------------------------
#library(yardstick)
#library(ggplot2)

# 1) Pull out predictions (with probabilities)
bt_aftr_tunning_h1n1_preds <- bt_h1n1_final_fit %>% 
  collect_predictions()
bt_aftr_tunning_seas_preds <- bt_seas_final_fit  %>%
  collect_predictions()

# 2) Compute ROC curve data
bt_aftr_tunning_roc_h1n1 <- roc_curve(bt_aftr_tunning_h1n1_preds, truth = h1n1_vaccine, .pred_1)
bt_aftr_tunning_roc_seas <- roc_curve(bt_aftr_tunning_seas_preds, truth = seasonal_vaccine, .pred_1)

# 3a) Plot separately
autoplot(bt_aftr_tunning_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (Random Forest)")
autoplot(bt_aftr_tunning_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (Random Forest)")


# -----------------------------------------------
# 17. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
bt_final_h1n1 <- fit(bt_final_h1n1_tune_wkfl, train_df)
bt_final_seas <- fit(bt_final_seas_tune_wkfl, train_df)



# -----------------------------------------------
# 18. MAKE PREDICTIONS ON TEST DATA
# -----------------------------------------------
# Add missing columns to test data to match training structure
test_df_prepared <- test_df %>%
  mutate(
    h1n1_vaccine = factor(NA, levels = c(1, 0)),
    seasonal_vaccine = factor(NA, levels = c(1, 0)),
    strata = NA_character_)

test_pred_h1n1_bagged_tree <- predict(bt_final_h1n1, test_df_prepared, type = "prob") %>% 
  pull(.pred_1)
test_pred_seas_bagged_tree <- predict(bt_final_seas, test_df_prepared, type = "prob") %>% 
  pull(.pred_1)

head(test_pred_h1n1_bagged_tree)
head(test_pred_seas_bagged_tree)


# -----------------------------------------------
# 17. CREATE SUBMISSION FILE
# -----------------------------------------------
submission_bagged_tree <- tibble(
  respondent_id = test_df$respondent_id,
  h1n1_vaccine = test_pred_h1n1_bagged_tree,
  seasonal_vaccine = test_pred_seas_bagged_tree
)


# -----------------------------------------------
# 18. SAVE SUBMISSION
# -----------------------------------------------
write_csv(submission_bagged_tree, "bagged_tree_workflow.csv")
