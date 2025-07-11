#This started running from 4:40 pm to 8:15 pm (3 hours and 35 minutes)

# -----------------------------------------------
# 1. SET UP ENVIRONMENT
# -----------------------------------------------
library(tidyverse)
library(tidymodels)
library(baguette)
library(tune)
library(future)
library(vip) # for variable importance
library(skimr) # Better Overview of the variables
library(stacks)
#install.packages("mlr3tuningspaces")
#library(mlr3tuningspaces) # we need this package to look at which tunning space improve the model

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


# this gives us a break down of the variables in the dataset
#skim(train_df)

# IDENTIFY NUMERIC VS. CATEGORICAL BY TYPE
# (Rather than manually listing variable names)
# First, convert any integer‐coded categories to factor *if* they’re not already numeric
# For example: if 'age_group' was stored as integer 1:4 representing bins, do:
# train_df <- train_df %>% mutate(age_group = factor(age_group))
# After that, let tidymodels detect which are numeric vs. nominal:
numeric_vars     <- train_df %>% select(where(is.numeric))    %>% names()
numeric_vars
categorical_vars <- train_df %>% select(where(is.character), where(is.factor)) %>% names()
categorical_vars
# Remove the target + ID from those lists
numeric_vars     <- setdiff(numeric_vars,    c("respondent_id"))
numeric_vars
categorical_vars <- setdiff(categorical_vars, c("respondent_id", "h1n1_vaccine", "seasonal_vaccine"))
categorical_vars

# -----------------------------------------------
# 4. CREATE TWO SEPARATE SPLITS (ONE PER TARGET)
# -----------------------------------------------
data_split_h1n1 <- initial_split(train_df, prop = 0.8, strata = h1n1_vaccine)
train_data_h1n1 <- training(data_split_h1n1)
eval_data_h1n1  <- testing(data_split_h1n1)

data_split_seas <- initial_split(train_df, prop = 0.8, strata = seasonal_vaccine)
train_data_seas <- training(data_split_seas)
eval_data_seas  <- testing(data_split_seas)


# -----------------------------------------------
# 5. SPECIFY BASE MODEL (RPART TREE)
# -----------------------------------------------
xgb_model <- boost_tree() %>%
  # Set the mode
  set_mode("classification") %>%
  # Set the engine
  set_engine("xgboost")


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

# prepped_rec <- prep(h1n1_recipe, training = train_data_h1n1)
# 
# # 2. Extract the processed training set
# View(juice(prepped_rec))
# glimpse(juice(prepped_rec))





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
xgb_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(xgb_model)

xgb_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(xgb_model)


# -----------------------------------------------
# 8.Train the workflow
# -----------------------------------------------
# xgb_h1n1_dt_wkfl_fit <- xgb_wf_h1n1 %>% 
#   last_fit(split = data_split_h1n1)
# 
# xgb_seas_dt_wkfl_fit <- xgb_wf_seas %>% 
#   last_fit(split = data_split_seas)
# 
# 
# saveRDS(xgb_h1n1_dt_wkfl_fit, "results/section8_xgb_h1n1_dt_wkfl_fit.rds")
# saveRDS(xgb_seas_dt_wkfl_fit, "results/section8_xgb_seas_dt_wkfl_fit.rds")


xgb_h1n1_dt_wkfl_fit     <- readRDS("Model/results/section8_xgb_h1n1_dt_wkfl_fit.rds")
xgb_seas_dt_wkfl_fit     <- readRDS("Model/results/section8_xgb_seas_dt_wkfl_fit.rds")

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


# Some info from data camp course:
# A very high in-sample AUC like can be an indicator of overfitting. 
# It is also possible that the dataset is just very well structured, or the model might just be terrific
# To check which of these is true, we need to produce out-of-sample estimates of the AUC, and because 
# we don't want to touch the test set yet, we can produce these using cross-validation on the training set.


# Fit resamples
# xgb_h1n1_dt_rs <- xgb_wf_h1n1 %>% 
#   fit_resamples(resamples = h1n1_folds,
#                 metrics = data_metrics)
# 
# xgb_seasonal_dt_rs <- xgb_wf_seas %>% 
#   fit_resamples(resamples = seasonal_folds,
#                 metrics = data_metrics)
# 
# 
# 
# saveRDS(xgb_h1n1_dt_rs, "results/section10_xgb_h1n1_dt_rs.rds")
# saveRDS(xgb_seasonal_dt_rs, "results/section10_xgb_seasonal_dt_rs.rds")

xgb_h1n1_dt_rs           <- readRDS("Model/results/section10_xgb_h1n1_dt_rs.rds")
xgb_seasonal_dt_rs       <- readRDS("Model/results/section10_xgb_seasonal_dt_rs.rds")

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
xgb_dt_tune_model <- 
  boost_tree(
    trees       = tune(),    # nrounds ∈ [1, 5000]
    tree_depth  = tune(),    # max_depth ∈ [1, 20]
    learn_rate  = tune(),    # eta ∈ [1e−4, 1], log-scale
    sample_size = tune()     # subsample ∈ [0.1, 1]
  ) %>%
  set_engine(
    "xgboost",
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

# For speed
plan(multisession, workers = 4) 


set.seed(214)
xgb_h1n1_grid <- grid_random(xgb_h1n1_params, size = 100)

set.seed(215)
xgb_seas_grid <- grid_random(xgb_seas_params, size = 100)


ctrl_grid <- control_stack_grid()      # for tune_grid()
ctrl_res <- control_stack_resamples()  # for fit_resamples()

# Hyperparameter tuning
# xgb_h1n1_dt_tuning <- xgb_h1n1_tune_wkfl %>% 
#   tune_grid(resamples = h1n1_folds,
#             grid = xgb_h1n1_grid,
#             metrics = data_metrics,
#             control = control_stack_grid())
# 
# 
# xgb_seas_dt_tuning <- xgb_seas_tune_wkfl %>% 
#   tune_grid(resamples = seasonal_folds,
#             grid = xgb_seas_grid,
#             metrics = data_metrics,
#             control = control_stack_grid())
# 
# 
# 
# saveRDS(xgb_h1n1_dt_tuning, "results/section11_xgb_h1n1_dt_tuning.rds")
# saveRDS(xgb_seas_dt_tuning, "results/section11_xgb_seas_dt_tuning.rds")

xgb_h1n1_dt_tuning  <- readRDS("Model/results/section11_xgb_h1n1_dt_tuning.rds")
xgb_seas_dt_tuning  <- readRDS("Model/results/section11_xgb_seas_dt_tuning.rds")




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
write_csv(submission_xgboost, "xgboost-workflow-2.csv")
