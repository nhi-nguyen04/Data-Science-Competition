# -----------------------------------------------
# LightGBM Workflow
# -----------------------------------------------


# -----------------------------------------------
#0. Author: Vanilton Paulo + Nhi Nguyen
# -----------------------------------------------

# <!-- Workload distributions: -->
#   
#   <!--1.Nhi Nguyen: Section 1 - 6 -->
#   
#   <!-- 2.Vanilton Paulo: Section 7 - 20  -->

# -----------------------------------------------
# 1. SET UP ENVIRONMENT
# -----------------------------------------------
# Then load both packages
library(tidyverse)
library(tidymodels)
library(bonsai)
library(lightgbm)
library(tune)
library(baguette)
library(future)
library(vip)
library(skimr)
library(stacks)
library(finetune)


set.seed(6)
show_engines("boost_tree")
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
    h1n1_vaccine     = factor(h1n1_vaccine, levels = c(1, 0)),
    seasonal_vaccine = factor(seasonal_vaccine, levels = c(1, 0))
  )


ordinal_vars <- c(
  "h1n1_concern",                    
  "h1n1_knowledge",                  
  "opinion_h1n1_vacc_effective",     
  "opinion_h1n1_risk",               
  "opinion_h1n1_sick_from_vacc",     
  "opinion_seas_vacc_effective",     
  "opinion_seas_risk",               
  "opinion_seas_sick_from_vacc"      
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
# -----------------------------------------------
# 4. SPLIT DATA (PER TARGET)
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
lgbm_model <- boost_tree() %>%
  set_mode("classification") %>%
  set_engine("lightgbm", verbose = -1)

# -----------------------------------------------
# 6. RECIPES –– CREATION OF RECIPES
# -----------------------------------------------
# Define a preprocessing recipe for data preparation before modeling


# H1N1 Vaccine recipe 
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>%
  # Step 1: Impute + encode
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # Step 2: Interactions 
  step_interact(terms = ~ starts_with("doctor_recc_h1n1_"):starts_with("opinion_h1n1_vacc_effective_")) %>%
  step_interact(terms = ~ starts_with("doctor_recc_h1n1_"):starts_with("opinion_h1n1_risk_")) %>%
  step_interact(terms = ~ starts_with("opinion_h1n1_vacc_effective_"):starts_with("opinion_h1n1_risk_")) %>%
  step_interact(terms = ~ starts_with("doctor_recc_seasonal_"):starts_with("opinion_seas_vacc_effective_")) %>%
  step_interact(terms = ~ starts_with("opinion_h1n1_sick_from_vacc_"):starts_with("opinion_seas_sick_from_vacc_")) %>%
  # Step 3: Remove predictors with zero variance (no useful information)
  step_zv(all_predictors())


# Seasonal Vaccine recipe
seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  # Step 1: Impute + encode
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # Step 2: Interactions 
  step_interact(terms = ~ starts_with("opinion_seas_vacc_effective_"):starts_with("opinion_seas_risk_")) %>%
  step_interact(terms = ~ starts_with("opinion_seas_risk_"):starts_with("doctor_recc_seasonal_")) %>%
  step_interact(terms = ~ starts_with("opinion_seas_vacc_effective_"):starts_with("doctor_recc_seasonal_")) %>%
  step_interact(terms = ~ starts_with("opinion_h1n1_risk_"):starts_with("opinion_seas_vacc_effective_")) %>%
  step_interact(terms = ~ starts_with("opinion_seas_sick_from_vacc_"):starts_with("opinion_h1n1_sick_from_vacc_")) %>%
  # Step 3: Remove predictors with zero variance (no useful information)
  step_zv(all_predictors())

# -----------------------------------------------
# 7. WORKFLOWS
# -----------------------------------------------
lgbm_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(lgbm_model)

lgbm_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(lgbm_model)

# -----------------------------------------------
# 8. BASELINE TRAIN (last_fit)
# -----------------------------------------------
 lgbm_h1n1_dt_wkfl_fit <- lgbm_wf_h1n1 %>% last_fit(split = data_split_h1n1)
 lgbm_seas_dt_wkfl_fit <- lgbm_wf_seas %>% last_fit(split = data_split_seas)
# -----------------------------------------------
# 9. BASELINE METRICS & ROC
# -----------------------------------------------
lgbm_metrics_h1n1 <- lgbm_h1n1_dt_wkfl_fit %>% collect_metrics()

lgbm_metrics_seas <- lgbm_seas_dt_wkfl_fit %>% collect_metrics()

lgbm_h1n1_preds <- collect_predictions(lgbm_h1n1_dt_wkfl_fit)

lgbm_seas_preds <- collect_predictions(lgbm_seas_dt_wkfl_fit)




# 2. Compute ROC curve data
lgbm_roc_h1n1 <- roc_curve(lgbm_h1n1_preds, truth = h1n1_vaccine, .pred_1)
lgbm_roc_seas <- roc_curve(lgbm_seas_preds, truth = seasonal_vaccine, .pred_1)

#  Plot separately  ROC curves
autoplot(lgbm_roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve")

autoplot(lgbm_roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve")

#Calcualte the ROC AUC VALUES
roc_auc(lgbm_h1n1_preds, truth = h1n1_vaccine, .pred_1)

roc_auc(lgbm_seas_preds, truth = seasonal_vaccine, .pred_1)


#confusion matrix

# Compute confusion matrix with Heatmap for counts
lgbm_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

lgbm_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "heatmap")

# Compute confusion matrix with mosaic plots for visualization of sensitivity and specitivity
lgbm_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "mosaic")

lgbm_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "mosaic")

#custom metric predictions --> we added f1 score
custom_metrics <- metric_set(accuracy,sens,spec,roc_auc,f_meas)


custom_metrics(lgbm_h1n1_preds,truth = h1n1_vaccine, estimate = .pred_class,.pred_1)
custom_metrics(lgbm_seas_preds,truth = seasonal_vaccine, estimate = .pred_class,.pred_1)

# -----------------------------------------------
# 10. CROSS-VALIDATION
# -----------------------------------------------

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



lgbm_h1n1_dt_rs <- lgbm_wf_h1n1 %>%
     fit_resamples(resamples = h1n1_folds,
                   metrics = data_metrics)



lgbm_seasonal_dt_rs <- lgbm_wf_seas %>%
     fit_resamples(resamples = seasonal_folds,
                   metrics = data_metrics)

lgbm_rs_metrics_h1n1 <- lgbm_h1n1_dt_rs %>% 
  collect_metrics()

lgbm_rs_metrics_seas <- lgbm_seasonal_dt_rs %>% 
  collect_metrics()


# Detailed cross validation results
lgbm_h1n1_dt_rs_results <- lgbm_h1n1_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for decision tree
lgbm_h1n1_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate),
            sd = sd(.estimate))


lgbm_seasonal_dt_rs_results <- lgbm_seasonal_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for decision tree
lgbm_seasonal_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate),
            sd = sd(.estimate))

# -----------------------------------------------
# 11. HYPERPARAMETER TUNING
# -----------------------------------------------
lgbm_dt_tune_model <- boost_tree(
  trees       = tune(),
  tree_depth  = tune(),
  learn_rate  = tune(),
  sample_size = tune(),
  
) %>% set_mode("classification") %>% set_engine("lightgbm", verbose = -1)


lgbm_dt_tune_model

lgbm_h1n1_tune_wkfl <- lgbm_wf_h1n1 %>% update_model(lgbm_dt_tune_model)
lgbm_h1n1_tune_wkfl


lgbm_seas_tune_wkfl <- lgbm_wf_seas %>% update_model(lgbm_dt_tune_model)

lgbm_seas_tune_wkfl




lgbm_default_params <- parameters(lgbm_dt_tune_model)

lgbm_h1n1_params <- finalize(lgbm_default_params, train_data_h1n1)
lgbm_seas_params <- finalize(lgbm_default_params, train_data_seas)




set.seed(214)

lgbm_h1n1_grid <- grid_random(lgbm_h1n1_params, size = 500)

set.seed(215)
lgbm_seas_grid <- grid_random(lgbm_seas_params, size = 500)


ctrl_grid <- control_stack_grid()      # for tune_grid()
ctrl_res <- control_stack_resamples()  # for fit_resamples()


doParallel::registerDoParallel()

ctrl_race <- control_race(verbose_elim = TRUE)

lgbm_h1n1_dt_tuning <- lgbm_h1n1_tune_wkfl %>%
  tune_race_anova(
    resamples = h1n1_folds,
    grid = lgbm_h1n1_grid,
    metrics = data_metrics,
    control = ctrl_race
  )

lgbm_seas_dt_tuning <- lgbm_seas_tune_wkfl %>%
  tune_race_anova(
    resamples = seasonal_folds,
    grid = lgbm_seas_grid,
    metrics = data_metrics,
    control = ctrl_race
  )

# View results
lgbm_h1n1_dt_tuning %>% 
  collect_metrics()


# View results
lgbm_seas_dt_tuning %>% 
  collect_metrics()


# Collect detailed tuning results
lgbm_h1n1_dt_tuning_results <- lgbm_h1n1_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
lgbm_h1n1_dt_tuning_results %>% 
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))


# Collect detailed tuning results
lgbm_seas_dt_tuning_results <- lgbm_seas_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
lgbm_seas_dt_tuning_results %>% 
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))



# -----------------------------------------------
# 12.Selecting the best model
# -----------------------------------------------

# Display 5 best performing models
lgbm_h1n1_dt_tuning %>% 
  show_best(metric = 'roc_auc', n = 5)

lgbm_seas_dt_tuning %>% 
  show_best(metric = 'roc_auc', n = 5)



# Select based on best performance
lgbm_best_h1n1_dt_model <- lgbm_h1n1_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

lgbm_best_h1n1_dt_model


lgbm_best_seas_dt_model <- lgbm_seas_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

lgbm_best_seas_dt_model
# -----------------------------------------------
# 13.Finalize your workflow
# -----------------------------------------------
lgbm_final_h1n1_tune_wkfl <- lgbm_h1n1_tune_wkfl %>% 
  finalize_workflow(lgbm_best_h1n1_dt_model)

lgbm_final_h1n1_tune_wkfl


lgbm_final_seas_tune_wkfl <- lgbm_seas_tune_wkfl %>% 
  finalize_workflow(lgbm_best_seas_dt_model)

lgbm_final_seas_tune_wkfl
# -----------------------------------------------
# 14. LAST_FIT ON THE HELD-OUT SPLITS
# -----------------------------------------------
lgbm_h1n1_final_fit <- lgbm_final_h1n1_tune_wkfl %>% 
  last_fit(split = data_split_h1n1)

lgbm_seas_final_fit <- lgbm_final_seas_tune_wkfl %>% 
  last_fit(split = data_split_seas)

#-----------------------------------------------
# 15. COLLECT METRICS
# -----------------------------------------------
lgbm_h1n1_final_fit %>% collect_metrics()
lgbm_seas_final_fit  %>% collect_metrics()


# -----------------------------------------------
# 16. ROC CURVE VISUALIZATION (via last_fit results)
# -----------------------------------------------

# Pull out predictions (with probabilities)
lgbm_aftr_tunning_h1n1_preds <- lgbm_h1n1_final_fit %>% 
  collect_predictions()
lgbm_aftr_tunning_seas_preds <- lgbm_seas_final_fit  %>% 
  collect_predictions()

# Compute ROC curve data
lgbm_aftr_tunning_roc_h1n1 <- roc_curve(lgbm_aftr_tunning_h1n1_preds, truth = h1n1_vaccine, .pred_1)
lgbm_aftr_tunning_roc_seas <- roc_curve(lgbm_aftr_tunning_seas_preds, truth = seasonal_vaccine, .pred_1)

# Plot separately
autoplot(lgbm_aftr_tunning_roc_h1n1) +
  ggtitle("Final H1N1 Vaccine ROC Curve (XGBoost)")
autoplot(lgbm_aftr_tunning_roc_seas) + 
  ggtitle("Final Seasonal Vaccine ROC Curve(XGBoost)")


# -----------------------------------------------
# 17. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
lgbm_final_h1n1 <- fit(lgbm_final_h1n1_tune_wkfl, train_df)
lgbm_final_seas <- fit(lgbm_final_seas_tune_wkfl, train_df)


# Here I am checking for variable importance
vip::vip(lgbm_final_h1n1, num_features= 15)
vip::vip(lgbm_final_seas,  num_features= 15)


# -----------------------------------------------
# 18. MAKE PREDICTIONS ON TEST DATA
# -----------------------------------------------
# Add missing columns to test data to match training structure
test_df_prepared <- test_df %>%
  mutate(
    h1n1_vaccine = factor(NA, levels = c(1, 0)),
    seasonal_vaccine = factor(NA, levels = c(1, 0)),
    strata = NA_character_)

test_pred_h1n1_lgbm <- predict(lgbm_final_h1n1, test_df_prepared, type = "prob") %>%
  pull(.pred_1)
test_pred_seas_lgbm <- predict(lgbm_final_seas, test_df_prepared, type = "prob") %>% 
  pull(.pred_1)

head(test_pred_h1n1_lgbm)
head(test_pred_seas_lgbm)


# -----------------------------------------------
# 19. CREATE SUBMISSION FILE
# -----------------------------------------------
submission_lightgbm <- tibble(
  respondent_id = test_df$respondent_id,
  h1n1_vaccine = test_pred_h1n1_lgbm,
  seasonal_vaccine = test_pred_seas_lgbm
)


# -----------------------------------------------
# 20. SAVE SUBMISSION
# -----------------------------------------------
write_csv(submission_lightgbm, "lightGBM_predictions_submission.csv")
