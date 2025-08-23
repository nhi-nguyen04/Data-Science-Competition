# -----------------------------------------------
# 0. Author: Vanilton Paulo
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
# 4. CREATE TWO SEPARATE SPLITS (ONE PER TARGET)
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
# 5. SPECIFY BASE MODEL (RPART TREE)
# -----------------------------------------------
model <- boost_tree() %>%
  # Set the mode
  set_mode("classification") %>%
  # Set the engine
  set_engine("xgboost")

# -----------------------------------------------
#6. R E C I P E 
# -----------------------------------------------


# H1N1 recipe
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  # Remove the other target (seasonal) if it’s present
  step_rm(seasonal_vaccine) %>%
  # Impute all numeric predictors by median
  step_impute_median(all_numeric_predictors()) %>%
  # Create an "unknown" level for any missing factor
  step_unknown(all_nominal_predictors()) %>%
  # One‐hot encode all factors
  step_dummy(all_nominal_predictors()) %>%
  # Drops any predictors that have zero variance
  step_zv(all_predictors()) %>% 
  # Normalize numeric columns
  step_normalize(all_numeric_predictors())


#Seasonal recipe
seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# -----------------------------------------------
#7.WORKFLOWS
# -----------------------------------------------
wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(model)

wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(model)

# -----------------------------------------------
#8.Train the workflow
# -----------------------------------------------
h1n1_dt_wkfl_fit <- wf_h1n1 %>% 
  last_fit(split = data_split_h1n1)

seas_dt_wkfl_fit <- wf_seas %>% 
  last_fit(split = data_split_seas)

# -----------------------------------------------
#9.Calculate performance metrics on test data
# -----------------------------------------------
# Calculate performance metrics on test data
h1n1_dt_wkfl_fit %>% 
  collect_metrics()

seas_dt_wkfl_fit %>% 
  collect_metrics()

# Pull out predictions (with class‐probabilities)
h1n1_preds <- collect_predictions(h1n1_dt_wkfl_fit)
seas_preds <- collect_predictions(seas_dt_wkfl_fit)

# Compute ROC curve data
roc_h1n1 <- roc_curve(h1n1_preds, truth = h1n1_vaccine, .pred_1)
roc_seas <- roc_curve(seas_preds, truth = seasonal_vaccine, .pred_1)

# Plot separately
autoplot(roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve")

autoplot(roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve")
# -----------------------------------------------
#10.Cross Validation
# -----------------------------------------------
#Cross-validation gives you a more robust estimate of your out-of-sample performance without 
#the statistical pitfalls - it assesses your model more profoundly.

#For speed
plan(multisession, workers = 4) 

set.seed(290)
h1n1_folds <- vfold_cv(train_data_h1n1, v = 10,
                       strata = h1n1_vaccine)

h1n1_folds

set.seed(291)
seasonal_folds <- vfold_cv(train_data_seas, v = 10,
                           strata = seasonal_vaccine)

seasonal_folds

# Create custom metrics function
data_metrics <- metric_set(roc_auc, sens, spec,accuracy)


# Fit resamples

h1n1_dt_rs <- wf_h1n1 %>% 
  fit_resamples(resamples = h1n1_folds,
                metrics = data_metrics)

seasonal_dt_rs <- wf_seas %>% 
  fit_resamples(resamples = seasonal_folds,
                metrics = data_metrics)


# View performance metrics


h1n1_dt_rs %>% 
  collect_metrics()

seasonal_dt_rs %>% 
  collect_metrics()



# Detailed cross validation results
h1n1_dt_rs_results <- h1n1_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for xgboost
h1n1_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))


seasonal_dt_rs_results <- seasonal_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for xgboost
seasonal_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))


# -----------------------------------------------
#11.Hyperparameter tuning
# -----------------------------------------------

dt_tune_model <-boost_tree(
  learn_rate = tune(),
  tree_depth = tune(),
  trees = tune(), 
  sample_size = tune()
) %>%
  set_engine("xgboost", importance = "impurity") %>%
  set_mode("classification") 





dt_tune_model


# Create a tuning workflow
h1n1_tune_wkfl <- wf_h1n1 %>% 
  # Replace model
  update_model(dt_tune_model)

h1n1_tune_wkfl


seas_tune_wkfl <- wf_seas %>% 
  # Replace model
  update_model(dt_tune_model)

seas_tune_wkfl



# Finalize parameter ranges for both
h1n1_params <- finalize(parameters(dt_tune_model), train_data_h1n1)
seas_params <- finalize(parameters(dt_tune_model), train_data_seas)

# Hyperparameter tuning with grid search

#For speed
plan(multisession, workers = 4) 


set.seed(214)
h1n1_grid <- grid_random(h1n1_params, size = 100)

set.seed(215)
seas_grid <- grid_random(seas_params, size = 100)



# Hyperparameter tuning
h1n1_dt_tuning <- h1n1_tune_wkfl %>% 
  tune_grid(resamples = h1n1_folds,
            grid = h1n1_grid,
            metrics = data_metrics)


seas_dt_tuning <- seas_tune_wkfl %>% 
  tune_grid(resamples = seasonal_folds,
            grid = seas_grid,
            metrics = data_metrics)


# View results
h1n1_dt_tuning %>% 
  collect_metrics()


# View results
seas_dt_tuning %>% 
  collect_metrics()





# Collect detailed tuning results
h1n1_dt_tuning_results <- h1n1_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
h1n1_dt_tuning_results %>% 
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))



# Collect detailed tuning results
seas_dt_tuning_results <- seas_dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
seas_dt_tuning_results %>% 
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))

# -----------------------------------------------
#12.Selecting the best model
# -----------------------------------------------

# Display 5 best performing models
h1n1_dt_tuning %>% 
  show_best(metric = 'roc_auc', n = 5)

seas_dt_tuning %>% 
  show_best(metric = 'roc_auc', n = 5)



# Select based on best performance
best_h1n1_dt_model <- h1n1_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

best_h1n1_dt_model


best_seas_dt_model <- seas_dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

best_seas_dt_model


# -----------------------------------------------
#13.Finalize your workflow
# -----------------------------------------------
final_h1n1_tune_wkfl <- h1n1_tune_wkfl %>% 
  finalize_workflow(best_h1n1_dt_model)

final_h1n1_tune_wkfl


final_seas_tune_wkfl <- seas_tune_wkfl %>% 
  finalize_workflow(best_seas_dt_model)

final_seas_tune_wkfl

# -----------------------------------------------
# 14. LAST_FIT ON THE HELD-OUT SPLITS
# -----------------------------------------------
h1n1_final_fit <- 
  final_h1n1_tune_wkfl %>% 
  last_fit(split = data_split_h1n1)

seas_final_fit <- 
  final_seas_tune_wkfl %>% 
  last_fit(split = data_split_seas)


#-----------------------------------------------
# 15. COLLECT METRICS
# -----------------------------------------------

h1n1_final_fit %>% collect_metrics()
seas_final_fit  %>% collect_metrics()


# -----------------------------------------------
# 16. ROC CURVE VISUALIZATION (via last_fit results)
# -----------------------------------------------

# Pull out predictions (with probabilities)
h1n1_preds <- h1n1_final_fit %>% collect_predictions()
seas_preds <- seas_final_fit  %>% collect_predictions()

# Compute ROC curve data
roc_h1n1 <- roc_curve(h1n1_preds, truth = h1n1_vaccine, .pred_1)
roc_seas <- roc_curve(seas_preds, truth = seasonal_vaccine, .pred_1)

# Plot separately
autoplot(roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve")
autoplot(roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve")
# -----------------------------------------------
# 17. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
final_h1n1 <- fit(final_h1n1_tune_wkfl, train_df)
final_seas <- fit(final_seas_tune_wkfl, train_df)

#Here I am checking for variable importance
vip::vip(final_h1n1, num_features= 10)
vip::vip(final_seas,  num_features= 10)
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

test_pred_h1n1_xgboost <- predict(final_h1n1, test_df_prepared, type = "prob") %>% pull(.pred_1)
test_pred_seas_xgboost <- predict(final_seas, test_df_prepared, type = "prob") %>% pull(.pred_1)

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
write_csv(submission_xgboost, "xgboost_default-tunning-grid-100.csv")