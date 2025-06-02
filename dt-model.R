# -----------------------------------------------
# 1. SET UP ENVIRONMENT
# -----------------------------------------------
library(tidyverse)   # For data manipulation
library(tidymodels)  # Machine learning framework
set.seed(6)          # For reproducibility

# -----------------------------------------------
# 2. LOAD DATA
# -----------------------------------------------
train_features <- read_csv("Data/training_set_features.csv")
train_labels   <- read_csv("Data/training_set_labels.csv")
train_df       <- left_join(train_features, train_labels, by = "respondent_id")
test_df        <- read_csv("Data/test_set_features.csv")

# Ideally
nrow(train_df) == nrow(train_features)
nrow(train_df) == nrow(train_labels)

# -----------------------------------------------
# 3. DATA PREPARATION 
# -----------------------------------------------
train_df <- train_df %>%
  mutate(
    h1n1_vaccine     = factor(h1n1_vaccine, levels = c(0, 1)),
    seasonal_vaccine = factor(seasonal_vaccine, levels = c(0, 1))
  )

#check for numeric
train_df %>% select(all_of(numeric_cols)) %>% map_chr(~ class(.x))


numeric_cols <- c(
  'h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds',
  'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands',
  'behavioral_large_gatherings', 'behavioral_outside_home',
  'behavioral_touch_face', 'doctor_recc_h1n1', 'doctor_recc_seasonal',
  'chronic_med_condition', 'child_under_6_months', 'health_worker',
  'health_insurance', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
  'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
  'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'household_adults',
  'household_children'
)

# Stratify for splitting
train_df <- train_df %>%
  mutate(strata = paste(h1n1_vaccine, seasonal_vaccine))

data_split <- initial_split(train_df, prop = 0.67, strata = strata)
train_data <- training(data_split)
eval_data  <- testing(data_split)

# Identify categorical predictors
all_cols_in_train   <- names(train_data)
non_predictors      <- c("respondent_id", "h1n1_vaccine", "seasonal_vaccine", "strata")
potential_preds     <- setdiff(all_cols_in_train, non_predictors)
categorical_cols    <- setdiff(potential_preds, numeric_cols)



# -----------------------------------------------
# 4. MODEL SPECIFICATIONS
# -----------------------------------------------
dt_model <- decision_tree() %>% 
  # Specify the engine
  set_engine('rpart') %>% 
  # Specify the mode
  set_mode('classification')

# -----------------------------------------------
# 5. PREPROCESSING RECIPES
# -----------------------------------------------

# Fixed approach: Use step_rm() to remove unwanted columns
h1n1_recipe <- recipe(
  h1n1_vaccine ~ .,
  data = train_data
) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine, strata) %>%
  step_impute_knn(all_predictors())%>%
  #step_impute_median(all_of(numeric_cols)) %>%
  step_unknown(all_of(categorical_cols)) %>%
  step_dummy(all_of(categorical_cols)) %>%
  step_normalize(all_of(numeric_cols))

seas_recipe <- recipe(
  seasonal_vaccine ~ .,
  data = train_data
) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine, strata) %>%
  step_impute_median(all_of(numeric_cols)) %>%
  step_unknown(all_of(categorical_cols)) %>%
  step_dummy(all_of(categorical_cols)) %>%
  step_normalize(all_of(numeric_cols))

# -----------------------------------------------
# 6. CREATE WORKFLOWS
# -----------------------------------------------
wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(dt_model)

wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(dt_model)

# -----------------------------------------------
# 7. TRAIN & EVALUATE ON SPLIT
# -----------------------------------------------

# Train the workflow
wf_h1n1_fit <- wf_h1n1 %>% 
  last_fit(split = data_split)

# Calculate performance metrics on test data
wf_h1n1_fit %>% 
  collect_metrics()



wf_seas_fit <- wf_seas %>% 
  last_fit(split = data_split)

# Calculate performance metrics on test data
wf_seas_fit %>% 
  collect_metrics()

# -----------------------------------------------
# 7. CV
# -----------------------------------------------

# Create cross validation folds
set.seed(290)
h1n1_folds <- vfold_cv(train_data, v = 10,
                       strata = strata)

# Create custom metrics function
h1n1_metrics <- metric_set(roc_auc, sens, spec)

# Fit resamples
h1n1_rs <- wf_h1n1 %>% 
  fit_resamples(resamples = h1n1_folds,
                metrics = h1n1_metrics)

# View performance metrics
h1n1_rs_results <- h1n1_rs %>% 
  collect_metrics(summarize = FALSE)


# Explore model performance for decision tree
h1n1_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))



#########


seas_folds <- vfold_cv(train_data, v = 10,
                       strata = strata)

# Create custom metrics function
seas_metrics <- metric_set(roc_auc, sens, spec)

# Fit resamples
seas_rs <- wf_seas %>% 
  fit_resamples(resamples = seas_folds,
                metrics = seas_metrics)

# View performance metrics
seas_rs_results <- seas_rs %>% 
  collect_metrics(summarize = FALSE)


# Explore model performance for decision tree
seas_rs_results %>% 
  group_by(.metric) %>% 
  summarize(min = min(.estimate),
            median = median(.estimate),
            max = max(.estimate))


# -----------------------------------------------
# 7. Hypertuning
# -----------------------------------------------

dt_tune_model <- decision_tree(cost_complexity = tune(),
                               tree_depth = tune(),
                               min_n = tune()) %>% 
  # Specify engine
  set_engine('rpart') %>% 
  # Specify mode
  set_mode('classification')

# Create a tuning workflow
h1n1_tune_wkfl <- wf_h1n1 %>% 
  # Replace model
  update_model(dt_tune_model)


h1n1_tune_wkfl


set.seed(214)
dt_grid <- grid_random(parameters(dt_tune_model),
                       size = 5)

# Hyperparameter tuning
dt_tuning <- h1n1_tune_wkfl %>% 
  tune_grid(resamples = h1n1_folds,
            grid = dt_grid,
            metrics = h1n1_metrics)




# Collect detailed tuning results
dt_tuning_results <- dt_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
dt_tuning_results %>% 
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))





# Display 5 best performing models
dt_tuning %>% 
  show_best(metric = 'roc_auc', n = 5)

# Select based on best performance
best_dt_model <- dt_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

# Finalize your workflow
final_h1n1_wkfl <- h1n1_tune_wkfl %>% 
  finalize_workflow(best_dt_model)

final_h1n1_wkfl






#########


# Create a tuning workflow
seas_tune_wkfl <- wf_seas %>% 
  # Replace model
  update_model(dt_tune_model)


seas_tune_wkfl


set.seed(214)
dt_seas_grid <- grid_random(parameters(dt_tune_model),
                       size = 5)

# Hyperparameter tuning
dt_seas_tuning <- seas_tune_wkfl %>% 
  tune_grid(resamples = seas_folds,
            grid = dt_seas_grid,
            metrics = seas_metrics)




# Collect detailed tuning results
dt_seas_tuning_results <- dt_seas_tuning %>% 
  collect_metrics(summarize = FALSE)

# Explore detailed ROC AUC results for each fold
dt_seas_tuning_results %>% 
  filter(.metric == 'roc_auc') %>% 
  group_by(id) %>% 
  summarize(min_roc_auc = min(.estimate),
            median_roc_auc = median(.estimate),
            max_roc_auc = max(.estimate))





# Display 5 best performing models
dt_seas_tuning %>% 
  show_best(metric = 'roc_auc', n = 5)

# Select based on best performance
best_seas_dt_model <- dt_seas_tuning %>% 
  # Choose the best model based on roc_auc
  select_best(metric = 'roc_auc')

# Finalize your workflow
final_seas_wkfl <- seas_tune_wkfl %>% 
  finalize_workflow(best_seas_dt_model)

final_seas_wkfl



# -----------------------------------------------
# 8. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------


final_h1n1 <- fit(final_h1n1_wkfl, train_df)
final_seas <- fit(final_seas_wkfl, train_df)



# -----------------------------------------------
# 9. MAKE PREDICTIONS ON TEST DATA
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
# 10. CREATE SUBMISSION FILE
# -----------------------------------------------
submission <- tibble(
  respondent_id = test_df$respondent_id,
  h1n1_vaccine = test_pred_h1n1,
  seasonal_vaccine = test_pred_seas
)

# -----------------------------------------------
# 11. SAVE SUBMISSION
# -----------------------------------------------
write_csv(submission, "vaccine_predictions_submission-2.csv")

# Print confirmation message
cat("Submission file created successfully with", nrow(submission), "predictions.\n")