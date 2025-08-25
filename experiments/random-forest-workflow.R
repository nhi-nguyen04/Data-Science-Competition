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
library(tidyverse)
library(tidymodels)
library(baguette)
library(tune)
library(future)
library(vip) 
library(skimr)
library(stacks)
library(kableExtra)


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
# 4. CREATE TWO SEPARATE SPLITS (ONE PER TARGET)---> Avoids class imbalance
# -----------------------------------------------
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
# 5. SPECIFY BASE MODEL (RPART TREE)
# -----------------------------------------------
rf_model <- rand_forest() %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")


# -----------------------------------------------
# 6. R E C I P E –– CREATION OF RECIPES
# -----------------------------------------------
# Define a preprocessing recipe for data preparation before modeling

#H1N1 Vaccine recipe 
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>%
  # Step 1: Impute 
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  # Step 2: Interactions 
  step_interact(terms = ~ starts_with("doctor_recc_h1n1_"):starts_with("opinion_h1n1_vacc_effective_")) %>%
  step_interact(terms = ~ starts_with("doctor_recc_h1n1_"):starts_with("opinion_h1n1_risk_")) %>%
  step_interact(terms = ~ starts_with("opinion_h1n1_vacc_effective_"):starts_with("opinion_h1n1_risk_")) %>%
  step_interact(terms = ~ starts_with("doctor_recc_seasonal_"):starts_with("opinion_seas_vacc_effective_")) %>%
  step_interact(terms = ~ starts_with("opinion_h1n1_sick_from_vacc_"):starts_with("opinion_seas_sick_from_vacc_")) %>%
  # Step 3: Remove predictors with zero variance (no useful information)
  step_zv(all_predictors())


#Seasonal Vaccine recipe
seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  # Step 1: Imputation 
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  # Step 2: Interactions 
  step_interact(terms = ~ starts_with("opinion_seas_vacc_effective_"):starts_with("opinion_seas_risk_")) %>%
  step_interact(terms = ~ starts_with("opinion_seas_risk_"):starts_with("doctor_recc_seasonal_")) %>%
  step_interact(terms = ~ starts_with("opinion_seas_vacc_effective_"):starts_with("doctor_recc_seasonal_")) %>%
  step_interact(terms = ~ starts_with("opinion_h1n1_risk_"):starts_with("opinion_seas_vacc_effective_")) %>%
  step_interact(terms = ~ starts_with("opinion_seas_sick_from_vacc_"):starts_with("opinion_h1n1_sick_from_vacc_")) %>%
  # Step 3: Remove predictors with zero variance (no useful information)
  step_zv(all_predictors())
# -----------------------------------------------
# 7.WORKFLOWS
# -----------------------------------------------
rf_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(rf_model)

rf_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(rf_model)


# -----------------------------------------------
# 8.Train the workflow 
# -----------------------------------------------
#Here Training and test sets are created
#recipe trained and applied

rf_h1n1_dt_wkfl_fit <- rf_wf_h1n1 %>%
  last_fit(split = data_split_h1n1)

rf_seas_dt_wkfl_fit <- rf_wf_seas %>%
  last_fit(split = data_split_seas)

# -----------------------------------------------
# 9.Model Evaluation -->Calculate performance metrics on test data(20% of the trainning data)
# -----------------------------------------------
#predictions and metrics are generated with test dataset
rf_metrics_h1n1 <- rf_h1n1_dt_wkfl_fit %>% 
  collect_metrics()

rf_metrics_seas <- rf_seas_dt_wkfl_fit %>% 
  collect_metrics()

# Pull out predictions (with class‐probabilities)
rf_h1n1_preds <- collect_predictions(rf_h1n1_dt_wkfl_fit)
rf_seas_preds <- collect_predictions(rf_seas_dt_wkfl_fit)

# Compute ROC curve data
rf_roc_h1n1 <- roc_curve(rf_h1n1_preds, truth = h1n1_vaccine, .pred_1)
rf_roc_seas <- roc_curve(rf_seas_preds, truth = seasonal_vaccine, .pred_1)

# Plot separately  ROC curves
autoplot(rf_roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve")

autoplot(rf_roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve")

# Calcualte the ROC AUC VALUES
roc_auc(rf_h1n1_preds, truth = h1n1_vaccine, .pred_1)

roc_auc(rf_seas_preds, truth = seasonal_vaccine, .pred_1)


#confusion matrix

# Compute confusion matrix with Heatmap for counts
 rf_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

rf_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "heatmap")

# Compute confusion matrix with mosaic plots for visualization of sensitivity and specitivity
rf_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "mosaic")

rf_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "mosaic")

#custom metric predictions --> we added f1 score
custom_metrics <- metric_set(accuracy,sens,spec,roc_auc,f_meas)


custom_metrics(rf_h1n1_preds,truth = h1n1_vaccine, estimate = .pred_class,.pred_1)
custom_metrics(rf_seas_preds,truth = seasonal_vaccine, estimate = .pred_class,.pred_1)

# -----------------------------------------------
# 10.Cross Validation--> Estimating performance with CV
# -----------------------------------------------
#Cross-validation gives you a more robust estimate of your out-of-sample performance without 
#the statistical pitfalls - it assesses your model more profoundly.


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

rf_h1n1_dt_rs <- rf_wf_h1n1 %>%
  fit_resamples(resamples = h1n1_folds,
                metrics = data_metrics)

rf_seasonal_dt_rs <- rf_wf_seas %>%
  fit_resamples(resamples = seasonal_folds,
                metrics = data_metrics)

# View performance metrics

rf_rs_metrics_h1n1 <- rf_h1n1_dt_rs %>% 
  collect_metrics()

rf_rs_metrics_seas <- rf_seasonal_dt_rs %>% 
  collect_metrics()


# Detailed cross validation results
rf_h1n1_dt_rs_results <- rf_h1n1_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for decision tree
rf_h1n1_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate),
            sd = sd(.estimate))


rf_seasonal_dt_rs_results <- rf_seasonal_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for decision tree
rf_seasonal_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate),
            sd = sd(.estimate))

# -----------------------------------------------
# 11.Hyperparameter tuning
# -----------------------------------------------
#  Declare the tunable model spec
rf_dt_tune_model <- 
  rand_forest(
    mtry  = tune(),      # will become mtry.ratio
    min_n = tune(),      # your existing tuning
    trees = tune()       # tunes num.trees
  ) %>%
  set_engine(
    "ranger",
    importance      = "impurity",
    sample.fraction = tune()  
  ) %>%
  set_mode("classification")

rf_dt_tune_model


# Create a tuning workflow
rf_h1n1_tune_wkfl <- rf_wf_h1n1 %>% 
  # Replace model
  update_model(rf_dt_tune_model)

rf_h1n1_tune_wkfl


rf_seas_tune_wkfl <- rf_wf_seas %>% 
  # Replace model
  update_model(rf_dt_tune_model)

rf_seas_tune_wkfl




#  Build the default parameter set
rf_params <- parameters(rf_dt_tune_model)

# Override only the ranges we care about
rf_params <- rf_params %>% update(
  mtry = mtry(range = c(0, 1)),          # mtry.ratio in [0,1]
  trees  = trees(range = c(1, 2000)),      # num.trees in [1,2000]
  sample.fraction = sample_prop(range = c(0.1, 1))  # sample.fraction in [0.1,1]
)

# 4) Finalize any data‐dependent settings
rf_h1n1_params <- finalize(rf_params, train_data_h1n1)
rf_seas_params <- finalize(rf_params, train_data_seas)



# Hyperparameter tuning with grid search

set.seed(214)
rf_h1n1_grid <- grid_random(rf_h1n1_params, size = 500)

set.seed(215)
rf_seas_grid <- grid_random(rf_seas_params, size = 500)


ctrl_grid <- control_stack_grid()      # for tune_grid()
ctrl_res <- control_stack_resamples()  # for fit_resamples()

# Hyperparameter tuning

rf_h1n1_dt_tuning <- rf_h1n1_tune_wkfl %>%
  tune_grid(resamples = h1n1_folds,
            grid = rf_h1n1_grid,
            metrics = data_metrics,
            control = control_stack_grid())


rf_seas_dt_tuning <- rf_seas_tune_wkfl %>%
  tune_grid(resamples = seasonal_folds,
            grid = rf_seas_grid,
            metrics = data_metrics,
            control = control_stack_grid())

# View results
rf_h1n1_dt_tuning %>% 
  collect_metrics()


# View results
rf_seas_dt_tuning %>% 
  collect_metrics()


# Collect detailed tuning results
rf_h1n1_dt_tuning_results <- rf_h1n1_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
rf_h1n1_dt_tuning_results %>% 
  filter(.metric == "roc_auc") %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))

# Collect detailed tuning results
rf_seas_dt_tuning_results <- rf_seas_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
rf_seas_dt_tuning_results %>% 
  filter(.metric == "roc_auc") %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))


# -----------------------------------------------
# 12.Selecting the best model
# -----------------------------------------------
# Display 5 best performing models
rf_h1n1_dt_tuning %>% 
  show_best(metric = "roc_auc", n = 5)

rf_seas_dt_tuning %>% 
  show_best(metric = "roc_auc", n = 5)


# Select based on best performance
rf_best_h1n1_dt_model <- rf_h1n1_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

rf_best_h1n1_dt_model


rf_best_seas_dt_model <- rf_seas_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

rf_best_seas_dt_model


# -----------------------------------------------
# 13.Finalize your workflow
# -----------------------------------------------
rf_final_h1n1_tune_wkfl <- rf_h1n1_tune_wkfl %>% 
  finalize_workflow(rf_best_h1n1_dt_model)

rf_final_h1n1_tune_wkfl


rf_final_seas_tune_wkfl <- rf_seas_tune_wkfl %>% 
  finalize_workflow(rf_best_seas_dt_model)

rf_final_seas_tune_wkfl


# -----------------------------------------------
# 14. LAST_FIT ON THE HELD-OUT SPLITS
# -----------------------------------------------
#Here Training and test dataset are created
#Recipe trained and applied
#Tune random forest trained with entire training set
rf_h1n1_final_fit <- 
  rf_final_h1n1_tune_wkfl %>% 
  last_fit(split = data_split_h1n1)

rf_seas_final_fit <- 
  rf_final_seas_tune_wkfl %>% 
  last_fit(split = data_split_seas)


#-----------------------------------------------
# 15. COLLECT METRICS
# -----------------------------------------------
#predictions and metrics are generated with test dataset
rf_h1n1_final_fit %>% collect_metrics()
rf_seas_final_fit  %>% collect_metrics()

# -----------------------------------------------
# 16. ROC CURVE VISUALIZATION (via last_fit results)
# -----------------------------------------------


# Pull out predictions (with probabilities)
rf_aftr_tunning_h1n1_preds <- rf_h1n1_final_fit %>% 
  collect_predictions()
rf_aftr_tunning_seas_preds <- rf_seas_final_fit  %>%
  collect_predictions()

# Compute ROC curve data
rf_aftr_tunning_roc_h1n1 <- roc_curve(rf_aftr_tunning_h1n1_preds, truth = h1n1_vaccine, .pred_1)
rf_aftr_tunning_roc_seas <- roc_curve(rf_aftr_tunning_seas_preds, truth = seasonal_vaccine, .pred_1)

# Plot separately
autoplot(rf_aftr_tunning_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (Random Forest)")
autoplot(rf_aftr_tunning_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (Random Forest)")
# -----------------------------------------------
# 17. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
rf_final_h1n1 <- fit(rf_final_h1n1_tune_wkfl, train_df)
rf_final_seas <- fit(rf_final_seas_tune_wkfl, train_df)


# Here I am checking for variable importance
vip::vip(rf_final_h1n1, num_features= 10)
vip::vip(rf_final_seas,  num_features= 10)

# -----------------------------------------------
# 18. MAKE PREDICTIONS ON TEST DATA
# -----------------------------------------------
# Add missing columns to test data to match training structure
test_df_prepared <- test_df %>%
  mutate(
    h1n1_vaccine = factor(NA, levels = c(1, 0)),
    seasonal_vaccine = factor(NA, levels = c(1, 0)),
    strata = NA_character_)

test_pred_h1n1_random_forest <- predict(rf_final_h1n1, test_df_prepared, type = "prob") %>% 
  pull(.pred_1)
test_pred_seas_random_forest <- predict(rf_final_seas, test_df_prepared, type = "prob") %>% 
  pull(.pred_1)

head(test_pred_h1n1_random_forest)
head(test_pred_seas_random_forest)


# -----------------------------------------------
# 19. CREATE SUBMISSION FILE
# -----------------------------------------------
submission_random_forest <- tibble(
  respondent_id = test_df$respondent_id,
  h1n1_vaccine = test_pred_h1n1_random_forest,
  seasonal_vaccine = test_pred_seas_random_forest)


# -----------------------------------------------
# 20. SAVE SUBMISSION
# -----------------------------------------------
#alredy available
#write_csv(submission_random_forest, "random_forest_predictions_submission.csv")
