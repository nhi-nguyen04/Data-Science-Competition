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


# -----------------------------------------------
# 3. DATA PREPARATION 
# -----------------------------------------------
train_df <- train_df %>%
  mutate(
    h1n1_vaccine     = factor(h1n1_vaccine, levels = c(1, 0)),
    seasonal_vaccine = factor(seasonal_vaccine, levels = c(1, 0))
  )

# IDENTIFY NUMERIC VS. CATEGORICAL BY TYPE
# (Rather than manually listing variable names)
# First, convert any integer‐coded categories to factor *if* they’re not already numeric
# For example: if 'age_group' was stored as integer 1:4 representing bins, do:
# train_df <- train_df %>% mutate(age_group = factor(age_group))
# After that, let tidymodels detect which are numeric vs. nominal:
numeric_vars     <- train_df %>% select(where(is.numeric))    %>% names()
categorical_vars <- train_df %>% select(where(is.character), where(is.factor)) %>% names()

# Remove the target + ID from those lists
numeric_vars     <- setdiff(numeric_vars,    c("respondent_id"))
categorical_vars <- setdiff(categorical_vars, c("respondent_id", "h1n1_vaccine", "seasonal_vaccine"))


# -----------------------------------------------
# 4. CREATE TWO SEPARATE SPLITS (ONE PER TARGET)
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
  step_rm(seasonal_vaccine) %>% # Remove target leakage
  
  # 1. Handle missing values first - Order matters!
  # Impute numericals (median is generally robust for skewed data)
  step_impute_median(all_numeric_predictors()) %>%
  # Impute nominals (creates a new level 'unknown' for NAs)
  step_unknown(all_nominal_predictors()) %>%
  
  # 2. Feature Engineering based on Importance Plot
  # These are highly influential. Let's create more specific interactions
  # beyond just pairwise based on the plot insights.
  
  # Stronger Interaction: Doctor's H1N1 Rec + Opinion H1N1 Vacc Effective + H1N1 Risk
  # This combines the top 3 most important features from the plot
  step_interact(terms = ~ doctor_recc_h1n1:opinion_h1n1_vacc_effective:opinion_h1n1_risk) %>%
  
  # General Pro-Vaccine Stance: Doctor's Seasonal Rec + Opinion Seasonal Vacc Effective
  # This captures a general inclination towards vaccines, as suggested by their importance for H1N1
  step_interact(terms = ~ doctor_recc_seasonal:opinion_seas_vacc_effective) %>%
  
  # Concern vs. Risk Perception (reinforcement or contradiction)
  # Captures if high concern translates to high perceived risk.
  step_interact(terms = ~ h1n1_concern:opinion_h1n1_risk) %>%
  
  # Adverse Opinion: If you think you'll get sick from the vaccine (H1N1 or Seasonal)
  # These were also highly important, but negatively.
  # We might want a combined "vaccine hesitancy" interaction.
  step_interact(terms = ~ opinion_h1n1_sick_from_vacc:opinion_seas_sick_from_vacc) %>%
  
  # 3. Create dummy variables for all nominal predictors AFTER imputations and interactions
  # This ensures interaction terms are created from original levels, then dummified.
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% # one_hot=TRUE is generally preferred for tree models
  
  # 4. Clean up
  step_zv(all_predictors()) %>% # Remove zero-variance predictors
  # Normalization for tree models is optional, but harmless. Keeping for consistency.
  step_normalize(all_numeric_predictors())

# Note: For tree-based models (especially Random Forest and XGBoost),
# feature selection based on importance can sometimes be beneficial,
# but it's often better to let the tree model handle it internally unless
# you have a very high-dimensional dataset or strong overfitting.
# Given your current features, letting the model prune is usually fine.



seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>% # Remove target leakage
  
  # 1. Handle missing values first
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # **CRUCIAL CHANGE: Create dummy variables BEFORE creating interactions**
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # 2. Feature Engineering - NOW create interactions with the DUMMY VARIABLES
  #    Note: 'age_group' will now be expanded into 'age_group_X65_.Years', etc.
  #    You will need to refer to the specific dummy variable columns for interaction,
  #    or simplify how you express the interaction.
  #    Let's assume the dummy variables are named like 'age_group_X65_.Years',
  #    'age_group_18_34.Years', etc., after step_dummy.
  
  # Core Seasonal Interaction: Effectiveness + Risk + Doctor Rec
  # Assuming these were also nominal and now dummified.
  # If 'opinion_seas_vacc_effective' is still the original column, it might require its own dummification if nominal.
  # But typically if it's already binary (0/1) or numeric, this is fine.
  # If they are nominal (like "Strongly Agree", "Disagree"), after step_dummy,
  # they become things like 'opinion_seas_vacc_effective_StronglyAgree'.
  # For simplicity, if these are original nominals, we'll interact them and let 'recipes' handle it.
  # If they become individual dummy columns like 'opinion_seas_vacc_effective_Yes', then we need specific interactions.
  
  # Let's assume for now that 'opinion_seas_vacc_effective', 'opinion_seas_risk', 'doctor_recc_seasonal'
  # are either numeric/binary OR 'step_interact' can handle them when they're still nominal
  # BUT age_group is the tricky one with many levels.
  
  # New approach to interactions after step_dummy:
  # Now you'd typically interact specific dummy variables.
  # However, if you use the original nominal name in step_interact *after* step_dummy,
  # 'recipes' will interact *all* the resulting dummy variables from that nominal. This is usually what you want.
  
  # Let's keep the `age_group` specific interaction and trust `recipes` to use its dummy versions.
  
  step_interact(terms = ~ opinion_seas_vacc_effective:opinion_seas_risk:doctor_recc_seasonal) %>%
  
  # Age Group X65+ specific interactions.
  # This syntax means: interact all dummy variables created from 'age_group' with 'opinion_seas_vacc_effective'
  #step_interact(terms = ~ age_group:opinion_seas_vacc_effective) %>%
  #step_interact(terms = ~ age_group:doctor_recc_seasonal) %>%
  
  # Cross-Vaccine Risk Perception + Seasonal Effectiveness
  step_interact(terms = ~ opinion_h1n1_risk:opinion_seas_vacc_effective) %>%
  
  # Overall Vaccine Hesitancy/Negative Experience/Belief
  step_interact(terms = ~ opinion_seas_sick_from_vacc:opinion_h1n1_sick_from_vacc) %>%
  
  # 3. Clean up
  step_zv(all_predictors()) %>% # Remove zero-variance predictors
  step_normalize(all_numeric_predictors()) # Optional for tree models, but harmless.
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
bt_h1n1_dt_wkfl_fit <- bt_wf_h1n1 %>% 
  last_fit(split = data_split_h1n1)


bt_seas_dt_wkfl_fit <- bt_wf_seas %>% 
  last_fit(split = data_split_seas)


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
bt_h1n1_dt_rs <- bt_wf_h1n1 %>% 
  fit_resamples(resamples = h1n1_folds,
                metrics = data_metrics)

bt_seasonal_dt_rs <- bt_wf_seas %>% 
  fit_resamples(resamples = seasonal_folds,
                metrics = data_metrics)


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
bt_h1n1_grid <- grid_random(bt_h1n1_params, size = 50)

set.seed(215)
bt_seas_grid <- grid_random(bt_seas_params, size = 50)

# Hyperparameter tuning
bt_h1n1_dt_tuning <- bt_h1n1_tune_wkfl %>% 
  tune_grid(resamples = h1n1_folds,
            grid = bt_h1n1_grid,
            metrics = data_metrics)


bt_seas_dt_tuning <- bt_seas_tune_wkfl %>% 
  tune_grid(resamples = seasonal_folds,
            grid = bt_seas_grid,
            metrics = data_metrics)


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
