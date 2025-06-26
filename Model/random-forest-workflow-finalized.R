# This is a better version of random forest


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

glimpse(train_df)
skim(train_df)
#View(train_df)
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
rf_model <- rand_forest() %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")


# -----------------------------------------------
# 6. R E C I P E  –– consistent imputation + dummies
# -----------------------------------------------
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
  # <- drops any predictors that have zero variance
  step_zv(all_predictors()) %>% 
  # Normalize numeric columns
  step_normalize(all_numeric_predictors()) # this migth be wrong???????????????????

#the types collum migth show a problem
h1n1_recipe%>%
  summary()
tidy(h1n1_recipe, number = 4)


seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

tidy(seas_recipe, number = 4)

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
# 8.Train the workflow --> A lot goes behind the scene here
# -----------------------------------------------
#Here Training and test dataset are created
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

# 1. Pull out predictions (with class‐probabilities)
rf_h1n1_preds <- collect_predictions(rf_h1n1_dt_wkfl_fit)
rf_seas_preds <- collect_predictions(rf_seas_dt_wkfl_fit)

# 2. Compute ROC curve data
rf_roc_h1n1 <- roc_curve(rf_h1n1_preds, truth = h1n1_vaccine, .pred_1)
rf_roc_seas <- roc_curve(rf_seas_preds, truth = seasonal_vaccine, .pred_1)

# 2a. Plot separately  ROC curves
autoplot(rf_roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve")

autoplot(rf_roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve")

#2b.Calcualte the ROC AUC VALUES
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

#custom metric predictions 
custom_metrics <- metric_set(accuracy,sens,spec,roc_auc)

custom_metrics(rf_h1n1_preds,truth = h1n1_vaccine, estimate = .pred_class,.pred_1)
custom_metrics(rf_seas_preds,truth = seasonal_vaccine, estimate = .pred_class,.pred_1)

# -----------------------------------------------
# 10.Cross Validation--> Estimating performance with CV
# -----------------------------------------------
#Cross-validation gives you a more robust estimate of your out-of-sample performance without 
#the statistical pitfalls - it assesses your model more profoundly.

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
rf_h1n1_dt_rs <- rf_wf_h1n1 %>% 
  fit_resamples(resamples = h1n1_folds,
                metrics = data_metrics)

rf_seasonal_dt_rs <- rf_wf_seas %>% 
  fit_resamples(resamples = seasonal_folds,
                metrics = data_metrics)


# View performance metrics

# Some info from data camp course:
# A very high in-sample AUC like can be an indicator of overfitting. 
# It is also possible that the dataset is just very well structured, or the model might just be terrific
# To check which of these is true, we need to produce out-of-sample estimates of the AUC, and because 
# we don't want to touch the test set yet, we can produce these using cross-validation on the training set.

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

#For H1N1  Flu Vaccine
#We have used cross validation to evaluate the performance of your random forest  workflow. 
#Across the 10 cross validation folds, the average area under the ROC curve was ... . 
#The average sensitivity and specificity were ... and ..., respectively.

#For  Seasonal Flu Vaccine
#We have used cross validation to evaluate the performance of your random forest  workflow. 
#Across the 10 cross validation folds, the average area under the ROC curve was ... . 
#The average sensitivity and specificity were ... and ..., respectively.


#Use this to compare against other model types

# -----------------------------------------------
# 11.Hyperparameter tuning
# -----------------------------------------------
# it is adviced that trees should be between 500-1000
rf_dt_tune_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500) %>%
  set_engine("ranger", importance = "impurity") %>%
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


# Finalize parameter ranges for both
rf_h1n1_params  <- finalize(parameters(rf_dt_tune_model), train_data_h1n1)
rf_seas_params  <- finalize(parameters(rf_dt_tune_model), train_data_seas)

# Hyperparameter tuning with grid search

# For speed
plan(multisession, workers = 4) 


set.seed(214)
rf_h1n1_grid <- grid_random(rf_h1n1_params, size = 10)

set.seed(215)
rf_seas_grid <- grid_random(rf_seas_params, size = 10)


# Hyperparameter tuning
rf_h1n1_dt_tuning <- rf_h1n1_tune_wkfl %>% 
  tune_grid(resamples = h1n1_folds,
            grid = rf_h1n1_grid,
            metrics = data_metrics)


rf_seas_dt_tuning <- rf_seas_tune_wkfl %>% 
  tune_grid(resamples = seasonal_folds,
            grid = rf_seas_grid,
            metrics = data_metrics)

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
#recipe trained and applied
#Tune random forest trained with entire training dataset
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

#If section 15 and 11(last part) have similar scores it mean the modle will perform similarly in new datasets


# -----------------------------------------------
# 16. ROC CURVE VISUALIZATION (via last_fit results)
# -----------------------------------------------
#library(yardstick)
#library(ggplot2)

# 1) Pull out predictions (with probabilities)
rf_aftr_tunning_h1n1_preds <- rf_h1n1_final_fit %>% 
  collect_predictions()
rf_aftr_tunning_seas_preds <- rf_seas_final_fit  %>%
  collect_predictions()

# 2) Compute ROC curve data
rf_aftr_tunning_roc_h1n1 <- roc_curve(rf_aftr_tunning_h1n1_preds, truth = h1n1_vaccine, .pred_1)
rf_aftr_tunning_roc_seas <- roc_curve(rf_aftr_tunning_seas_preds, truth = seasonal_vaccine, .pred_1)

# 3a) Plot separately
autoplot(rf_aftr_tunning_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (Random Forest)")
autoplot(rf_aftr_tunning_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (Random Forest)")


# -----------------------------------------------
# 17. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
rf_final_h1n1 <- fit(rf_final_h1n1_tune_wkfl, train_df)
rf_final_seas <- fit(rf_final_seas_tune_wkfl, train_df)


# Here I am checking for variable importance
vip::vip(rf_final_h1n1, num_features= 15)
vip::vip(rf_final_seas,  num_features= 15)


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
write_csv(submission_random_forest, "random_forest_workflow-2.csv")
