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
lgbm_h1n1_base <- lgbm_wf_h1n1 %>% last_fit(split = data_split_h1n1)
lgbm_seas_base <- lgbm_wf_seas %>% last_fit(split = data_split_seas)

saveRDS(lgbm_h1n1_base, "results/section8_lgbm_h1n1_dt_wkfl_fit.rds")
saveRDS(lgbm_seas_base, "results/section8_lgbm_seas_dt_wkfl_fit.rds")
# -----------------------------------------------
# 9. BASELINE METRICS & ROC
# -----------------------------------------------
lgbm_h1n1_base %>% collect_metrics()
lgbm_seas_base %>% collect_metrics()

h1n1_base_preds <- collect_predictions(lgbm_h1n1_base)
seas_base_preds <- collect_predictions(lgbm_seas_base)

roc_curve(h1n1_base_preds, truth = h1n1_vaccine, .pred_1) %>% autoplot() + ggtitle("H1N1 ROC (LightGBM)")
roc_curve(seas_base_preds, truth = seasonal_vaccine, .pred_1) %>% autoplot() + ggtitle("Seasonal ROC (LightGBM)")

custom_metrics <- metric_set(accuracy, sens, spec, roc_auc)
custom_metrics(h1n1_base_preds, truth = h1n1_vaccine, estimate = .pred_class, .pred_1)
custom_metrics(seas_base_preds, truth = seasonal_vaccine, estimate = .pred_class, .pred_1)

# -----------------------------------------------
# 10. CROSS-VALIDATION
# -----------------------------------------------
plan(multisession, workers = 4)
h1n1_folds <- vfold_cv(train_data_h1n1, v = 10, strata = h1n1_vaccine)
seasonal_folds <- vfold_cv(train_data_seas, v = 10, strata = seasonal_vaccine)

cv_metrics <- metric_set(accuracy, roc_auc, sens, spec)

lgbm_h1n1_rs <- fit_resamples(
  lgbm_wf_h1n1,
  resamples = h1n1_folds,
  metrics = cv_metrics
)
lgbm_seas_rs <- fit_resamples(
  lgbm_wf_seas,
  resamples = seasonal_folds,
  metrics = cv_metrics
)


saveRDS(lgbm_h1n1_rs, "results/section10_lgbm_h1n1_dt_rs.rds")
saveRDS(lgbm_seas_rs, "results/section10_lgbm_seasonal_dt_rs.rds")


lgbm_h1n1_rs %>% collect_metrics()
lgbm_seas_rs %>% collect_metrics()

# -----------------------------------------------
# 11. HYPERPARAMETER TUNING
# -----------------------------------------------
tune_model <- boost_tree(
  trees       = tune(),
  tree_depth  = tune(),
  learn_rate  = tune(),
  sample_size = tune()
) %>% set_mode("classification") %>% set_engine("lightgbm", verbose = -1)

lgbm_h1n1_tune_wf <- lgbm_wf_h1n1 %>% update_model(tune_model)
lgbm_seas_tune_wf <- lgbm_wf_seas %>% update_model(tune_model)

params <- parameters(
  trees(range = c(1L,5000L)),
  tree_depth(range = c(1L,20L)),
  learn_rate(range = c(1e-4,1), trans = scales::log10_trans()),
  sample_prop(range = c(0.1,1))
)
params_h1n1 <- finalize(params, train_data_h1n1)
params_seas <- finalize(params, train_data_seas)

set.seed(214)
grid_h1n1 <- grid_random(params_h1n1, size=100)
set.seed(215)
grid_seas <- grid_random(params_seas, size=100)

lgbm_h1n1_tuning <- tune_grid(
  lgbm_h1n1_tune_wf,
  resamples = h1n1_folds,
  grid      = grid_h1n1,
  metrics   = cv_metrics,
  control = control_stack_grid()
)
lgbm_seas_tuning <- tune_grid(
  lgbm_seas_tune_wf,
  resamples = seasonal_folds,
  grid      = grid_seas,
  metrics   = cv_metrics,
 control = control_stack_grid()
)


saveRDS(lgbm_h1n1_tuning, "results/section11_lgbm_h1n1_dt_tuning.rds")
saveRDS(lgbm_seas_tuning, "results/section11_lgbm_seas_dt_tuning.rds")


# -----------------------------------------------
# 12. SELECT BEST
# -----------------------------------------------
best_h1n1 <- select_best(lgbm_h1n1_tuning, "roc_auc")
best_seas <- select_best(lgbm_seas_tuning, "roc_auc")

# -----------------------------------------------
# 13. FINALIZE WORKFLOWS
# -----------------------------------------------
lgbm_h1n1_final_wf <- lgbm_h1n1_tune_wf %>% finalize_workflow(best_h1n1)
lgbm_seas_final_wf <- lgbm_seas_tune_wf %>% finalize_workflow(best_seas)

# -----------------------------------------------
# 14. LAST_FIT FINAL
# -----------------------------------------------
lgbm_h1n1_final_fit <- lgbm_h1n1_final_wf %>% last_fit(split = data_split_h1n1)
lgbm_seas_final_fit <- lgbm_seas_final_wf %>% last_fit(split = data_split_seas)

# -----------------------------------------------
# 15. FINAL METRICS
# -----------------------------------------------
lgbm_h1n1_final_fit %>% collect_metrics()
lgbm_seas_final_fit %>% collect_metrics()

# -----------------------------------------------
# 16. ROC FINAL PLOTS
# -----------------------------------------------
lgbm_aftr_tunning_h1n1_preds <- collect_predictions(lgbm_h1n1_final_fit)
lgbm_aftr_tunning_seas_preds <- collect_predictions(lgbm_seas_final_fit)

roc_curve(lgbm_aftr_tunning_h1n1_preds, truth=h1n1_vaccine,.pred_1) %>% autoplot() + ggtitle("Final H1N1 ROC")
roc_curve(lgbm_aftr_tunning_seas_preds, truth=seasonal_vaccine,.pred_1) %>% autoplot() + ggtitle("Final Seasonal ROC")

# -----------------------------------------------
# 17. TRAIN ON FULL DATA
# -----------------------------------------------
lgbm_full_h1n1 <- fit(lgbm_h1n1_final_wf, train_df)
lgbm_full_seas <- fit(lgbm_seas_final_wf, train_df)

vip(lgbm_full_h1n1, num_features=15)
vip(lgbm_full_seas, num_features=15)

# -----------------------------------------------
# 18. PREDICT ON TEST
# -----------------------------------------------
test_prep <- test_df %>%
  mutate(
    h1n1_vaccine=factor(NA,levels=c(1,0)),
    seasonal_vaccine=factor(NA,levels=c(1,0)),
    strata=NA_character_
  )

pred_h1n1 <- predict(lgbm_full_h1n1,test_prep,type="prob") %>% pull(.pred_1)
pred_seas <- predict(lgbm_full_seas,test_prep,type="prob") %>% pull(.pred_1)

# -----------------------------------------------
# 19. CREATE SUBMISSION
# -----------------------------------------------
submission <- tibble(
  respondent_id=train_df$respondent_id,
  h1n1_vaccine=pred_h1n1,
  seasonal_vaccine=pred_seas
)

# -----------------------------------------------
# 20. SAVE SUBMISSION
# -----------------------------------------------
write_csv(submission,"lightgbm-workflow.csv")
