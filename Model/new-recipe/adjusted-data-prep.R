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
    across(all_of(binary_vars), ~ factor(.x, levels = c(0, 1), labels = c("No", "Yes")))
  )

# Apply same transformations to test data
test_df <- test_df %>%
  mutate(
    # Convert ordinal variables to ordered factors
    across(all_of(ordinal_vars), ~ factor(.x, ordered = TRUE)),
    
    # Convert nominal variables to factors
    across(all_of(nominal_vars), ~ factor(.x)),
    
    # Convert binary variables to factors with meaningful labels
    across(all_of(binary_vars), ~ factor(.x, levels = c(0, 1), labels = c("No", "Yes")))
  )
# -----------------------------------------------
# 4. CREATE TWO SEPARATE SPLITS (ONE PER TARGET)---> Avoids class imbalance
# -----------------------------------------------
#Ensures random split with similar distribution of the outcome variable 
data_split_h1n1 <- initial_split(train_df, prop = 0.8, strata = h1n1_vaccine)
data_split_h1n1
train_data_h1n1 <- training(data_split_h1n1)
eval_data_h1n1  <- testing(data_split_h1n1)

data_split_seas <- initial_split(train_df, prop = 0.8, strata = seasonal_vaccine)
data_split_seas
train_data_seas <- training(data_split_seas)
eval_data_seas  <- testing(data_split_seas)
# -----------------------------------------------
# 5. SPECIFY BASE MODEL (LOGISTIC REGRESSION)
# -----------------------------------------------
log_spec <- logistic_reg(
  penalty = 1,
  mixture = 0
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# -----------------------------------------------
# 6. R E C I P E  –– consistent imputation + dummies
# -----------------------------------------------
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  # Remove the other target (seasonal) if it’s present
  #creates a specification of a recipe step that will remove selected variables.
  step_rm(seasonal_vaccine) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # Impute all numeric predictors by median:
  step_impute_median(all_numeric_predictors()) %>%
  # Create an "unknown" level for any missing factor
  step_unknown(all_nominal_predictors()) %>%
  # One‐hot encode all factors
  step_dummy(all_nominal_predictors()) %>%
  # <- drops any predictors that have zero variance
  step_zv(all_predictors()) %>% 
  # Normalize numeric columns
  step_normalize(all_numeric_predictors()) # this migth be wrong???????????????????



seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# -----------------------------------------------
# 7. CREATE WORKFLOWS
# -----------------------------------------------
lr_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(log_spec)

lr_wf_h1n1

lr_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(log_spec)

# -----------------------------------------------
# 8. TRAIN & EVALUATE ON SPLIT
# -----------------------------------------------
lr_h1n1_dt_wkfl_fit <- lr_wf_h1n1 %>% 
  last_fit(split = data_split_h1n1)

lr_seas_dt_wkfl_fit <- lr_wf_seas %>% 
  last_fit(split = data_split_seas)


# -----------------------------------------------
# 9.Model Evaluation -->Calculate performance metrics on test data(20% of the trainning data)
# -----------------------------------------------
#predictions and metrics are generated with test dataset
lr_metrics_h1n1 <- lr_h1n1_dt_wkfl_fit %>% 
  collect_metrics()


lr_metrics_seas <- lr_seas_dt_wkfl_fit %>% 
  collect_metrics()

# 1. Pull out predictions (with class‐probabilities)
lr_h1n1_preds <- collect_predictions(lr_h1n1_dt_wkfl_fit)
lr_h1n1_preds
lr_seas_preds <- collect_predictions(lr_seas_dt_wkfl_fit)

# 2. Compute ROC curve data
lr_roc_h1n1 <- roc_curve(lr_h1n1_preds, truth = h1n1_vaccine, .pred_1)
lr_roc_h1n1
lr_roc_seas <- roc_curve(lr_seas_preds, truth = seasonal_vaccine, .pred_1)

# 2a. Plot separately  ROC curves
autoplot(lr_roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve")

autoplot(lr_roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve")

#2b.Calcualte the ROC AUC VALUES
roc_auc(lr_h1n1_preds, truth = h1n1_vaccine, .pred_1)

roc_auc(lr_seas_preds, truth = seasonal_vaccine, .pred_1)


#confusion matrix

# Compute confusion matrix with Heatmap for counts
lr_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

lr_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "heatmap")

# Compute confusion matrix with mosaic plots for visualization of sensitivity and specitivity
lr_h1n1_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "mosaic")

lr_seas_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class)%>%
  autoplot(type = "mosaic")

#custom metric predictions 
custom_metrics <- metric_set(accuracy,sens,spec,roc_auc)

custom_metrics(lr_h1n1_preds,truth = h1n1_vaccine, estimate = .pred_class,.pred_1)
custom_metrics(lr_seas_preds,truth = seasonal_vaccine, estimate = .pred_class,.pred_1)
