# -----------------------------------------------
# 1. SET UP ENVIRONMENT
# -----------------------------------------------
library(tidyverse)
library(tidymodels)
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
    h1n1_vaccine     = factor(h1n1_vaccine, levels = c(0, 1)),
    seasonal_vaccine = factor(seasonal_vaccine, levels = c(0, 1))
  )

# 3 IDENTIFY NUMERIC VS. CATEGORICAL BY TYPE
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
data_split_h1n1 <- initial_split(train_df, prop = 0.67, strata = h1n1_vaccine)
train_data_h1n1 <- training(data_split_h1n1)
eval_data_h1n1  <- testing(data_split_h1n1)

data_split_seas <- initial_split(train_df, prop = 0.67, strata = seasonal_vaccine)
train_data_seas <- training(data_split_seas)
eval_data_seas  <- testing(data_split_seas)


# -----------------------------------------------
# 5. SPECIFY BASE MODEL (RPART TREE)
# -----------------------------------------------
model <- decision_tree(
) %>%
  set_engine("rpart") %>%
  set_mode("classification")


# -----------------------------------------------
#6. R E C I P E  –– consistent imputation + dummies
# -----------------------------------------------
#
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  # Remove the other target (seasonal) if it’s present
  #creates a specification of a recipe step that will remove selected variables.
  step_rm(seasonal_vaccine) %>%
  # Impute all numeric predictors by median:
  step_impute_median(all_numeric_predictors()) %>%
  # Create an "unknown" level for any missing factor
  step_unknown(all_nominal_predictors()) %>%
  # One‐hot encode all factors
  step_dummy(all_nominal_predictors()) %>%
  # Normalize numeric columns
  step_normalize(all_numeric_predictors())




seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
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
# Train the workflow
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


# -----------------------------------------------
#10.Cross Validation
# -----------------------------------------------
set.seed(290)
h1n1_folds <- vfold_cv(train_data_h1n1, v = 10,
                        strata = h1n1_vaccine)

h1n1_folds


seasonal_folds <- vfold_cv(train_data_seas, v = 10,
                       strata = seasonal_vaccine)

seasonal_folds

# Create custom metrics function
data_metrics <- metric_set(roc_auc, sens, spec)


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

# Explore model performance for decision tree
h1n1_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))


seasonal_dt_rs_results <- seasonal_dt_rs %>% 
  collect_metrics(summarize = FALSE)

# Explore model performance for decision tree
seasonal_dt_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))


# -----------------------------------------------
#11.Hyperparameter tuning
# -----------------------------------------------

dt_tune_model <- decision_tree(cost_complexity = tune(),
                               tree_depth = tune(),
                               min_n = tune()) %>% 
  # Specify engine
  set_engine("rpart") %>% 
  # Specify mode
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



# Hyperparameter tuning with grid search
set.seed(214)
dt_grid <- grid_random(parameters(dt_tune_model),
                       size = 5)

dt_grid


# Hyperparameter tuning
h1n1_dt_tuning <- h1n1_tune_wkfl %>% 
  tune_grid(resamples = h1n1_folds,
            grid = dt_grid,
            metrics = data_metrics)


seas_dt_tuning <- seas_tune_wkfl %>% 
  tune_grid(resamples = seasonal_folds,
            grid = dt_grid,
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
# 14. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------


final_h1n1 <- fit(final_h1n1_tune_wkfl, train_df)
final_seas <- fit(final_seas_tune_wkfl, train_df)


# -----------------------------------------------
# 15. MAKE PREDICTIONS ON TEST DATA
# -----------------------------------------------
# Add missing columns to test data to match training structure
test_df_prepared <- test_df %>%
  mutate(
    h1n1_vaccine = factor(NA, levels = c(0, 1)),
    seasonal_vaccine = factor(NA, levels = c(0, 1)),
    strata = NA_character_
  )

test_pred_h1n1 <- predict(final_h1n1, test_df_prepared, type = "prob") %>% pull(.pred_1)
test_pred_seas <- predict(final_seas, test_df_prepared, type = "prob") %>% pull(.pred_1)

head(test_pred_h1n1)
head(test_pred_seas)


# -----------------------------------------------
# 16. CREATE SUBMISSION FILE
# -----------------------------------------------
submission <- tibble(
  respondent_id = test_df$respondent_id,
  h1n1_vaccine = test_pred_h1n1,
  seasonal_vaccine = test_pred_seas
)

# -----------------------------------------------
# 17. SAVE SUBMISSION
# -----------------------------------------------
write_csv(submission, "finalized_dt_workflow.csv")
