# -----------------------------------------------
# LightGBM Workflow (tidymodels + bonsai)
# -----------------------------------------------

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

set.seed(6)
show_engines("boost_tree")




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
# 4. SPLIT DATA (PER TARGET)
# -----------------------------------------------
data_split_h1n1 <- initial_split(train_df, prop = 0.8, strata = h1n1_vaccine)
train_data_h1n1 <- training(data_split_h1n1)
eval_data_h1n1  <- testing(data_split_h1n1)

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
# 6. RECIPES
# -----------------------------------------------
# H1N1 Recipe
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_interact(terms = ~ doctor_recc_h1n1:opinion_h1n1_vacc_effective:opinion_h1n1_risk) %>%
  step_interact(terms = ~ doctor_recc_seasonal:opinion_seas_vacc_effective) %>%
  step_interact(terms = ~ h1n1_concern:opinion_h1n1_risk) %>%
  step_interact(terms = ~ opinion_h1n1_sick_from_vacc:opinion_seas_sick_from_vacc) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

# Seasonal Recipe
seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_interact(terms = ~ opinion_seas_vacc_effective:opinion_seas_risk:doctor_recc_seasonal) %>%
  step_interact(terms = ~ opinion_h1n1_risk:opinion_seas_vacc_effective) %>%
  step_interact(terms = ~ opinion_seas_sick_from_vacc:opinion_h1n1_sick_from_vacc) %>%
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

saveRDS(lgbm_h1n1_dt_wkfl_fit, "results/section8_lgbm_h1n1_dt_wkfl_fit.rds")
saveRDS(lgbm_seas_dt_wkfl_fit, "results/section8_lgbm_seas_dt_wkfl_fit.rds")
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

# 2a. Plot separately  ROC curves
autoplot(lgbm_roc_h1n1) + 
  ggtitle("H1N1 Vaccine ROC Curve")

autoplot(lgbm_roc_seas) + 
  ggtitle("Seasonal Vaccine ROC Curve")

#2b.Calcualte the ROC AUC VALUES
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



lgbm_h1n1_dt_rs <- lgbm_wf_h1n1 %>%
     fit_resamples(resamples = h1n1_folds,
                   metrics = data_metrics)
  
  
  
lgbm_seasonal_dt_rs <- lgbm_wf_seas %>%
     fit_resamples(resamples = seasonal_folds,
                   metrics = data_metrics)


saveRDS(lgbm_h1n1_dt_rs, "results/section10_lgbm_h1n1_dt_rs.rds")
saveRDS(lgbm_seasonal_dt_rs, "results/section10_lgbm_seasonal_dt_rs.rds")






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
  sample_size = tune()
) %>% set_mode("classification") %>% set_engine("lightgbm", verbose = -1)


lgbm_dt_tune_model

lgbm_h1n1_tune_wkfl <- lgbm_wf_h1n1 %>% update_model(lgbm_dt_tune_model)
lgbm_h1n1_tune_wkfl


lgbm_seas_tune_wkfl <- lgbm_wf_seas %>% update_model(lgbm_dt_tune_model)

lgbm_seas_tune_wkfl

lgbm_default_params <- parameters(lgbm_dt_tune_model)

# 4) Finalize any data-dependent params (like mtry for RF; not strictly needed here,
#    but good practice if you ever tune something that depends on # predictors)
lgbm_h1n1_params <- finalize(lgbm_default_params, train_data_h1n1)
lgbm_seas_params <- finalize(lgbm_default_params, train_data_seas)


# For speed
plan(multisession, workers = 4) 


set.seed(214)
#Increased grid random from 50 to 60 and about to test to server
lgbm_h1n1_grid <- grid_random(lgbm_h1n1_params, size = 50)

set.seed(215)
lgbm_seas_grid <- grid_random(lgbm_seas_params, size = 50)


ctrl_grid <- control_stack_grid()      # for tune_grid()
ctrl_res <- control_stack_resamples()  # for fit_resamples()







lgbm_h1n1_dt_tuning <- lgbm_h1n1_tune_wkfl %>%
  tune_grid(resamples = h1n1_folds,
            grid = lgbm_h1n1_grid,
            metrics = data_metrics,
            control = control_stack_grid())


lgbm_seas_dt_tuning <- lgbm_seas_tune_wkfl %>%
  tune_grid(resamples = seasonal_folds,
            grid = lgbm_seas_grid,
            metrics = data_metrics,
            control = control_stack_grid())


saveRDS(lgbm_h1n1_dt_tuning, "results/section11_lgbm_h1n1_dt_tuning.rds")
saveRDS(lgbm_seas_dt_tuning, "results/section11_lgbm_seas_dt_tuning.rds")





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
#library(yardstick)
#library(ggplot2)

# 1) Pull out predictions (with probabilities)
lgbm_aftr_tunning_h1n1_preds <- lgbm_h1n1_final_fit %>% 
  collect_predictions()
lgbm_aftr_tunning_seas_preds <- lgbm_seas_final_fit  %>% 
  collect_predictions()

# 2) Compute ROC curve data
lgbm_aftr_tunning_roc_h1n1 <- roc_curve(lgbm_aftr_tunning_h1n1_preds, truth = h1n1_vaccine, .pred_1)
lgbm_aftr_tunning_roc_seas <- roc_curve(lgbm_aftr_tunning_seas_preds, truth = seasonal_vaccine, .pred_1)

# 3a) Plot separately
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
