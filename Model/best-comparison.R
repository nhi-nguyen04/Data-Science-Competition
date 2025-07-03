# ───────────────1.Pre Hypertuning────────────────────────────────

# Create a custom metric set for comprehensive evaluation
multi_metrics <- metric_set(roc_auc, accuracy, sens, spec)

#######################
#For H1N1  Flu Vaccine#
#######################
preds_stacked_pre_tune_h1n1 <- bind_rows(
  lr_h1n1_preds %>% mutate(model = "logistic_regression"),
  dt_h1n1_preds %>% mutate(model = "decision_trees"),
  bt_h1n1_preds %>% mutate(model = "bagged_trees"),
  rf_h1n1_preds %>% mutate(model = "random_forest"),
  xgb_h1n1_preds %>% mutate(model = "xgboost")
)

# Calculate comprehensive performance metrics
performance_results_pre_tune_h1n1 <- preds_stacked_pre_tune_h1n1 %>%
  group_by(model) %>%
  multi_metrics(truth = h1n1_vaccine, estimate = .pred_class, .pred_1) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(roc_auc))

performance_results_pre_tune_h1n1

# ROC curves
roc_data_h1n1 <- preds_stacked_pre_tune_h1n1 %>%
  group_by(model) %>%
  roc_curve(truth = h1n1_vaccine, .pred_1)

roc_data_h1n1 %>%
  autoplot() + 
  ggtitle("H1N1 Vaccine: ROC Curves (Pre-Tuning)")

##########################
#For Seasonal Flu Vaccine#
##########################
preds_stacked_pre_tune_seas <- bind_rows(
  lr_seas_preds %>% mutate(model = "logistic_regression"),
  dt_seas_preds %>% mutate(model = "decision_trees"),
  bt_seas_preds %>% mutate(model = "bagged_trees"),
  rf_seas_preds %>% mutate(model = "random_forest"),
  xgb_seas_preds %>% mutate(model = "xgboost")
)

# Calculate comprehensive performance metrics
performance_results_pre_tune_seas <- preds_stacked_pre_tune_seas %>%
  group_by(model) %>%
  multi_metrics(truth = seasonal_vaccine, estimate = .pred_class, .pred_1) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(roc_auc))

performance_results_pre_tune_seas

# ROC curves
roc_data_seas <- preds_stacked_pre_tune_seas %>%
  group_by(model) %>%
  roc_curve(truth = seasonal_vaccine, .pred_1)

roc_data_seas %>%
  autoplot() + 
  ggtitle("Seasonal Vaccine: ROC Curves (Pre-Tuning)")

# ────────────────────────────────1.Post Hypertuning ────────────────────────────────

#######################
#For H1N1  Flu Vaccine#
#######################
preds_stacked_post_tune_h1n1 <- bind_rows(
  lr_aftr_tunning_h1n1_preds %>% mutate(model = "logistic_regression"),
  dt_aftr_tunning_h1n1_preds %>% mutate(model = "decision_trees"),
  bt_aftr_tunning_h1n1_preds %>% mutate(model = "bagged_trees"),
  rf_aftr_tunning_h1n1_preds %>% mutate(model = "random_forest"),
  xgb_aftr_tunning_h1n1_preds %>% mutate(model = "xgboost")
)

# Calculate comprehensive performance metrics
performance_results_post_tune_h1n1 <- preds_stacked_post_tune_h1n1 %>%
  group_by(model) %>%
  multi_metrics(truth = h1n1_vaccine, estimate = .pred_class, .pred_1) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(roc_auc))

performance_results_post_tune_h1n1

# ROC curves
roc_data_post_h1n1 <- preds_stacked_post_tune_h1n1 %>%
  group_by(model) %>%
  roc_curve(truth = h1n1_vaccine, .pred_1)

roc_data_post_h1n1 %>%
  autoplot() + 
  ggtitle("H1N1 Vaccine: ROC Curves (Post-Tuning)")

##########################
#For Seasonal Flu Vaccine#
##########################
preds_stacked_post_tune_seas <- bind_rows(
  lr_aftr_tunning_seas_preds %>% mutate(model = "logistic_regression"),
  dt_aftr_tunning_seas_preds %>% mutate(model = "decision_trees"),
  bt_aftr_tunning_seas_preds %>% mutate(model = "bagged_trees"),
  rf_aftr_tunning_seas_preds %>% mutate(model = "random_forest"),
  xgb_aftr_tunning_seas_preds %>% mutate(model = "xgboost")
)

# Calculate comprehensive performance metrics
performance_results_post_tune_seas <- preds_stacked_post_tune_seas %>%
  group_by(model) %>%
  multi_metrics(truth = seasonal_vaccine, estimate = .pred_class, .pred_1) %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate) %>%
  arrange(desc(roc_auc))

performance_results_post_tune_seas

# ROC curves
roc_data_post_seas <- preds_stacked_post_tune_seas %>%
  group_by(model) %>%
  roc_curve(truth = seasonal_vaccine, .pred_1)

roc_data_post_seas %>%
  autoplot() + 
  ggtitle("Seasonal Vaccine: ROC Curves (Post-Tuning)")

# ─────────────────────────────── COMPARISON SUMMARY ────────────────────────────────

# Compare Pre vs Post Tuning Performance
cat("H1N1 VACCINE PERFORMANCE COMPARISON\n")
cat("===================================\n")
print("Pre-Tuning:")
print(performance_results_pre_tune_h1n1)
cat("\nPost-Tuning:")
print(performance_results_post_tune_h1n1)

cat("\n\nSEASONAL VACCINE PERFORMANCE COMPARISON\n")
cat("======================================\n")
print("Pre-Tuning:")
print(performance_results_pre_tune_seas)
cat("\nPost-Tuning:")
print(performance_results_post_tune_seas)