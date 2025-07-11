# -----------------------------------------------
# CatBoost Workflow (mirroring original XGBoost pipeline)
# -----------------------------------------------

# -----------------------------------------------
# 1. SET UP ENVIRONMENT
# -----------------------------------------------
install.packages('remotes')
#remotes::install_url('https://github.com/catboost/catboost/releases/download/v1.2.8/catboost-R-darwin-universal2-1.2.8.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
install.packages(
  "/Users/vaniltonpaulo/Desktop/catboost-R-darwin-universal2-1.2.8.tgz",
  repos = NULL,
  type  = "binary"
)

install.packages(
  local_file,
  repos = NULL,
  type  = "source",
  INSTALL_opts = c("--no-multiarch", "--no-test-load")
)

library(tidyverse)
library(tidymodels)
library(bonsai)
library(parsnip)
show_engines("boost_tree")
library(catboost)
library(tune)
library(baguette)
library(future)
library(vip)
library(skimr)
library(stacks)
library(parsnip)


show_engines("boost_tree")
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

numeric_vars     <- train_df %>% select(where(is.numeric)) %>% names() %>% setdiff("respondent_id")
categorical_vars <- train_df %>% select(where(is.character), where(is.factor)) %>% names() %>% setdiff(c("respondent_id", "h1n1_vaccine", "seasonal_vaccine"))

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
# 5. SPECIFY BASE MODEL (CatBoost)
# -----------------------------------------------
cb_model <- parsnip::boost_tree(mode="classification") %>%
  set_engine(
    "catboost"
  )

# -----------------------------------------------
# 6. R E C I P E –– Optimized for H1N1 (no dummying or normalization)
# -----------------------------------------------
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_interact(terms = ~ doctor_recc_h1n1:opinion_h1n1_vacc_effective:opinion_h1n1_risk) %>%
  step_interact(terms = ~ doctor_recc_seasonal:opinion_seas_vacc_effective) %>%
  step_interact(terms = ~ h1n1_concern:opinion_h1n1_risk) %>%
  step_interact(terms = ~ opinion_h1n1_sick_from_vacc:opinion_seas_sick_from_vacc) %>%
  step_zv(all_predictors())


seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_interact(terms = ~ opinion_seas_vacc_effective:opinion_seas_risk:doctor_recc_seasonal) %>%
  step_interact(terms = ~ opinion_h1n1_risk:opinion_seas_vacc_effective) %>%
  step_interact(terms = ~ opinion_seas_sick_from_vacc:opinion_h1n1_sick_from_vacc) %>%
  step_zv(all_predictors())

# -----------------------------------------------
# 7. WORKFLOWS
# -----------------------------------------------
cb_wf_h1n1 <- workflow() %>%
  add_recipe(h1n1_recipe) %>%
  add_model(cb_model)
cb_wf_h1n1
cb_wf_seas <- workflow() %>%
  add_recipe(seas_recipe) %>%
  add_model(cb_model)

# -----------------------------------------------
# 8. TRAIN THE WORKFLOW (Baseline, without hyperparameter tuning)
# -----------------------------------------------
# Fit on training split and evaluate on held-out test
cb_h1n1_base_fit <- cb_wf_h1n1 %>% last_fit(split = data_split_h1n1)
cb_seas_base_fit <- cb_wf_seas %>% last_fit(split = data_split_seas)

# Optional: save baseline fits
saveRDS(cb_h1n1_base_fit, "Model/results/section8_cb_h1n1_base_fit.rds")
saveRDS(cb_seas_base_fit, "Model/results/section8_cb_seas_base_fit.rds")

# -----------------------------------------------
# 9. CALCULATE PERFORMANCE METRICS ON TEST DATA
# -----------------------------------------------
cb_h1n1_base_metrics <- cb_h1n1_base_fit %>% collect_metrics()
cb_seas_base_metrics <- cb_seas_base_fit %>% collect_metrics()

cb_h1n1_base_preds <- cb_h1n1_base_fit %>% collect_predictions()
cb_seas_base_preds <- cb_seas_base_fit %>% collect_predictions()

# ROC curves and AUC
cb_h1n1_base_roc <- roc_curve(cb_h1n1_base_preds, truth = h1n1_vaccine, .pred_1)
cb_seas_base_roc <- roc_curve(cb_seas_base_preds, truth = seasonal_vaccine, .pred_1)

autoplot(cb_h1n1_base_roc) + ggtitle("H1N1 CatBoost ROC (Baseline)")
autoplot(cb_seas_base_roc) + ggtitle("Seasonal CatBoost ROC (Baseline)")

# Confusion matrices
cb_h1n1_base_preds %>%
  conf_mat(truth = h1n1_vaccine, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

cb_seas_base_preds %>%
  conf_mat(truth = seasonal_vaccine, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

# Custom metrics
custom_metrics <- metric_set(accuracy, sens, spec, roc_auc)
custom_metrics(cb_h1n1_base_preds, truth = h1n1_vaccine, estimate = .pred_class, .pred_1)
custom_metrics(cb_seas_base_preds, truth = seasonal_vaccine, estimate = .pred_class, .pred_1)

# -----------------------------------------------
# 10. CROSS VALIDATION (Baseline)
# -----------------------------------------------
plan(multisession, workers = 4)
set.seed(290)
h1n1_folds <- vfold_cv(train_data_h1n1, v = 10, strata = h1n1_vaccine)
seasonal_folds <- vfold_cv(train_data_seas, v = 10, strata = seasonal_vaccine)

cb_h1n1_base_rs <- fit_resamples(
  cb_wf_h1n1,
  resamples = h1n1_folds,
  metrics   = custom_metrics
)
cb_seas_base_rs <- fit_resamples(
  cb_wf_seas,
  resamples = seasonal_folds,
  metrics   = custom_metrics
)


saveRDS(cb_h1n1_base_rs, "results/section10_cb_h1n1_dt_rs.rds")
saveRDS(cb_seas_base_rs, "results/section10_cb_seasonal_dt_rs.rds")

cb_h1n1_base_rs %>% collect_metrics()
cb_seas_base_rs  %>% collect_metrics()

# -----------------------------------------------
# 11. HYPERPARAMETER TUNING
# -----------------------------------------------
cb_tune_model <- boost_tree(
  trees       = tune(),
  tree_depth  = tune(),
  learn_rate  = tune(),
  sample_size = tune()
) %>%
  set_mode("classification") %>%
  set_engine("catboost", loss_function = "Logloss", verbose = 0)

cb_h1n1_tune_wf <- cb_wf_h1n1 %>% update_model(cb_tune_model)
cb_seas_tune_wf <- cb_wf_seas %>% update_model(cb_tune_model)

cb_params <- parameters(
  trees(range = c(1L, 5000L)),
  tree_depth(range = c(1L, 20L)),
  learn_rate(range = c(1e-4, 1), trans = scales::log10_trans()),
  sample_prop(range = c(0.1, 1))
)
cb_params_h1n1 <- finalize(cb_params, train_data_h1n1)
cb_params_seas <- finalize(cb_params, train_data_seas)

plan(multisession, workers = 4)
set.seed(214)
cb_grid_h1n1 <- grid_random(cb_params_h1n1, size = 100)
set.seed(215)
cb_grid_seas <- grid_random(cb_params_seas, size = 100)

cb_h1n1_tuning <- tune_grid(
  cb_h1n1_tune_wf,
  resamples = h1n1_folds,
  grid      = cb_grid_h1n1,
  metrics   = custom_metrics,
 control = control_stack_grid()
)
cb_seas_tuning <- tune_grid(
  cb_seas_tune_wf,
  resamples = seasonal_folds,
  grid      = cb_grid_seas,
  metrics   = custom_metrics,
control = control_stack_grid()
)

 saveRDS(cb_h1n1_tuning, "results/section11_cb_h1n1_dt_tuning.rds")
 saveRDS(cb_seas_tuning, "results/section11_cb_seas_dt_tuning.rds")

# -----------------------------------------------
# 12. Selecting the best model
# -----------------------------------------------
best_h1n1 <- cb_h1n1_tuning %>% select_best(metric = "roc_auc")
best_seas <- cb_seas_tuning %>% select_best(metric = "roc_auc")

# -----------------------------------------------
# 13. Finalize your workflow
# -----------------------------------------------
cb_h1n1_final_wf <- cb_h1n1_tune_wf %>% finalize_workflow(best_h1n1)
cb_seas_final_wf <- cb_seas_tune_wf %>% finalize_workflow(best_seas)

# -----------------------------------------------
# 14. LAST_FIT ON THE HELD-OUT SPLITS
# -----------------------------------------------
cb_h1n1_final_fit <- cb_h1n1_final_wf %>% last_fit(split = data_split_h1n1)
cb_seas_final_fit <- cb_seas_final_wf %>% last_fit(split = data_split_seas)

# -----------------------------------------------
# 15. COLLECT METRICS
# -----------------------------------------------
cb_h1n1_final_fit %>% collect_metrics()
cb_seas_final_fit  %>% collect_metrics()

# -----------------------------------------------
# 16. ROC CURVE VISUALIZATION (via last_fit results)
# -----------------------------------------------
cb_h1n1_final_preds <- cb_h1n1_final_fit %>% collect_predictions()
cb_seas_final_preds <- cb_seas_final_fit %>% collect_predictions()

roc_curve(cb_h1n1_final_preds, truth = h1n1_vaccine, .pred_1) %>% autoplot() + ggtitle("Final H1N1 CatBoost ROC")
roc_curve(cb_seas_final_preds, truth = seasonal_vaccine, .pred_1) %>% autoplot() + ggtitle("Final Seasonal CatBoost ROC")

# -----------------------------------------------
# 17. TRAIN FINAL MODELS ON FULL TRAINING DATA
# -----------------------------------------------
cb_full_h1n1 <- fit(cb_h1n1_final_wf, train_df)
cb_full_seas <- fit(cb_seas_final_wf,  train_df)

vip(cb_full_h1n1, num_features = 15)
vip(cb_full_seas,  num_features = 15)

# -----------------------------------------------
# 18. MAKE PREDICTIONS ON TEST DATA
# -----------------------------------------------
test_df_prepped <- test_df %>%
  mutate(
    h1n1_vaccine     = factor(NA, levels = c(1, 0)),
    seasonal_vaccine = factor(NA, levels = c(1, 0)),
    strata           = NA_character_
  )

pred_h1n1_cb <- predict(cb_full_h1n1, test_df_prepped, type = "prob") %>% pull(.pred_1)
pred_seas_cb <- predict(cb_full_seas,  test_df_prepped, type = "prob") %>% pull(.pred_1)

# -----------------------------------------------
# 19. CREATE SUBMISSION FILE
# -----------------------------------------------
submission_cb <- tibble(
  respondent_id    = test_df$respondent_id,
  h1n1_vaccine     = pred_h1n1_cb,
  seasonal_vaccine = pred_seas_cb
)

# -----------------------------------------------
# 20. SAVE SUBMISSION
# -----------------------------------------------
write_csv(submission_cb, "catboost-workflow.csv")
