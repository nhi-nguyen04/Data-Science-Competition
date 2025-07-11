#install.packages("stacks")
library(stacks)
library(yardstick)



# 
# ctrl_grid <- control_stack_grid()      # for tune_grid()
# ctrl_res <- control_stack_resamples()  # for fit_resamples()
# 
# 
# rf_h1n1_stack <- rf_h1n1_tune_wkfl %>% 
#   tune_grid(resamples = h1n1_folds,
#             grid = rf_h1n1_grid,
#             metrics = data_metrics,
#             control = control_stack_grid())
# 
# 
# rf_seas_dt_stack <- rf_seas_tune_wkfl %>% 
#   tune_grid(resamples = seasonal_folds,
#             grid = rf_seas_grid,
#             metrics = data_metrics,
#             control = control_stack_grid())
# 
# 
# xgb_h1n1_dt_stack <- xgb_h1n1_tune_wkfl %>% 
#   tune_grid(resamples = h1n1_folds,
#             grid = xgb_h1n1_grid,
#             metrics = data_metrics,
#             control = control_stack_grid())
# 
# 
# xgb_seas_dt_stack <- xgb_seas_tune_wkfl %>% 
#   tune_grid(resamples = seasonal_folds,
#             grid = xgb_seas_grid,
#             metrics = data_metrics,
#             control = control_stack_grid())

lr_h1n1_stack_rs <- lr_final_h1n1_tune_wkfl %>%
  fit_resamples(resamples = h1n1_folds, metrics = data_metrics, control = control_stack_grid())

lr_seas_stack_rs <- lr_final_seas_tune_wkfl %>%
  fit_resamples(resamples = seasonal_folds, metrics = data_metrics, control = control_stack_grid())





# ——— 1. STACK H1N1 TUNED RESULTS ———
h1n1_stack <- stacks() %>%
 # add_candidates(xgb_h1n1_dt_tuning) %>%
  add_candidates(rf_h1n1_dt_tuning) %>%
  add_candidates(lr_h1n1_stack_rs) %>%
  blend_predictions() %>%
  fit_members()

# Evaluate on H1N1 hold-out:
h1n1_eval_preds <- predict(h1n1_stack, eval_data_h1n1, type = "prob") %>%
  bind_cols(eval_data_h1n1)

h1n1_eval_preds %>%
  roc_auc(truth = h1n1_vaccine, .pred_1)




# ——— 2. STACK SEASONAL TUNED RESULTS ———
seas_stack <- stacks() %>%
 # add_candidates(xgb_seas_dt_tuning) %>%
  add_candidates(rf_seas_dt_tuning) %>%
  add_candidates(lr_seas_stack_rs) %>%
  blend_predictions() %>%
  fit_members()

# Evaluate on Seasonal hold-out:
seas_eval_preds <- predict(seas_stack, eval_data_seas, type = "prob") %>%
  bind_cols(eval_data_seas)

seas_eval_preds %>%
  roc_auc(truth = seasonal_vaccine, .pred_1)

# ——— 3. FINAL TEST SET PREDICTIONS & SUBMISSION ———
final_preds_h1n1_stack <- predict(h1n1_stack, test_df_prepared, type = "prob") %>% pull(.pred_1)
final_preds_seas_stack <- predict(seas_stack, test_df_prepared, type = "prob") %>% pull(.pred_1)

submission_stacked <- tibble(
  respondent_id     = test_df$respondent_id,
  h1n1_vaccine      = final_preds_h1n1_stack,
  seasonal_vaccine  = final_preds_seas_stack
)

write_csv(submission_stacked, "stacked_trees_xgb_lr_rf_submission.csv")
