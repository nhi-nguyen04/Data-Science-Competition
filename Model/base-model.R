# Predicting H1N1 and Seasonal Flu Vaccine Uptake
# A tidymodels approach with ridge logistic regression

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

# -----------------------------------------------
# 3. DATA PREPARATION 
# -----------------------------------------------
train_df <- train_df %>%
  mutate(
    h1n1_vaccine     = factor(h1n1_vaccine, levels = c(0, 1)),
    seasonal_vaccine = factor(seasonal_vaccine, levels = c(0, 1))
  )

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
log_spec <- logistic_reg(
  penalty = 1,
  mixture = 0
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# -----------------------------------------------
# 5. PREPROCESSING RECIPES
# -----------------------------------------------

# Fixed approach: Use step_rm() to remove unwanted columns
h1n1_recipe <- recipe(
  h1n1_vaccine ~ .,
  data = train_data
) %>%
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
  # <- drops any predictors that have zero variance
  step_zv(all_predictors()) %>% 
  # Normalize numeric columns
  step_normalize(all_numeric_predictors())

seas_recipe <- recipe(
  seasonal_vaccine ~ .,
  data = train_data
) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# -----------------------------------------------
# 6. CREATE WORKFLOWS
# -----------------------------------------------
wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(log_spec)

wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(log_spec)

# -----------------------------------------------
# 7. TRAIN & EVALUATE ON SPLIT
# -----------------------------------------------
fit_h1n1 <- fit(wf_h1n1, train_data)
fit_seas <- fit(wf_seas, train_data)

pred_h1n1 <- predict(fit_h1n1, eval_data, type = "prob") %>% pull(.pred_1)
pred_seas <- predict(fit_seas, eval_data, type = "prob") %>% pull(.pred_1)

# -----------------------------------------------
# 8. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
final_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(log_spec)

final_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(log_spec)

final_h1n1 <- fit(final_wf_h1n1, train_df)
final_seas <- fit(final_wf_seas, train_df)

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

test_pred_h1n1_log_reg <- predict(final_h1n1, test_df_prepared, type = "prob") %>% pull(.pred_1)
test_pred_seas_log_reg <- predict(final_seas, test_df_prepared, type = "prob") %>% pull(.pred_1)

head(test_pred_h1n1_log_reg)
head(test_pred_seas_log_reg)

# -----------------------------------------------
# 10. CREATE SUBMISSION FILE
# -----------------------------------------------
submission_log_reg <- tibble(
  respondent_id = test_df$respondent_id,
  h1n1_vaccine = test_pred_h1n1_log_reg,
  seasonal_vaccine = test_pred_seas_log_reg
)

# -----------------------------------------------
# 11. SAVE SUBMISSION
# -----------------------------------------------
write_csv(submission_log_reg, "vaccine_predictions_submission.csv")

# Print confirmation message
cat("Submission file created successfully with", nrow(submission), "predictions.\n")