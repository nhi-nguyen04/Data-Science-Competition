# ───────────────1.Pre Hypertuning────────────────────────────────

# We are  performing a model comparison across all types of models in this course:
# decision trees, bagged trees, random forests, and gradient boosting.

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

# Then calculate AUC for each model and ranking them
auc_results_pre_tune_h1n1 <- preds_stacked_pre_tune_h1n1 %>%
  group_by(model) %>%
  roc_auc(truth = h1n1_vaccine, .pred_1)%>%
  arrange(desc(.estimate))

auc_results_pre_tune_h1n1


roc_data <- preds_stacked_pre_tune_h1n1 %>%
  group_by(model) %>%
  roc_curve(truth = h1n1_vaccine, .pred_1)

# Plot the ROC curves
roc_data %>%
  autoplot()+ 
  ggtitle("ROC curves of models")



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

# Then calculate AUC for each model and ranking them
auc_results_pre_tune_seas <- preds_stacked_pre_tune_seas %>%
  group_by(model) %>%
  roc_auc(truth = seasonal_vaccine, .pred_1)%>%
  arrange(desc(.estimate))

auc_results_pre_tune_seas

roc_data <- preds_stacked_pre_tune_seas %>%
  group_by(model) %>%
  roc_curve(truth = seasonal_vaccine, .pred_1)

# Plot the ROC curves
roc_data %>%
  autoplot()+ 
  ggtitle("ROC curves of models")





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

# Then calculate AUC for each model and ranking them
auc_results_post_tune_h1n1 <- preds_stacked_post_tune_h1n1 %>%
  group_by(model) %>%
  roc_auc(truth = h1n1_vaccine, .pred_1)%>%
  arrange(desc(.estimate))

auc_results_post_tune_h1n1

roc_data <- preds_stacked_post_tune_h1n1 %>%
  group_by(model) %>%
  roc_curve(truth = h1n1_vaccine, .pred_1)

# Plot the ROC curves
roc_data %>%
  autoplot()+ 
  ggtitle("ROC curves of models")





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

# Then calculate AUC for each model and ranking them
auc_results_post_tune_seas <- preds_stacked_post_tune_seas %>%
  group_by(model) %>%
  roc_auc(truth = seasonal_vaccine, .pred_1)%>%
  arrange(desc(.estimate))

auc_results_post_tune_seas


roc_data <- preds_stacked_post_tune_seas %>%
  group_by(model) %>%
  roc_curve(truth = seasonal_vaccine, .pred_1)

# Plot the ROC curves
roc_data %>%
  autoplot()+ 
  ggtitle("ROC curves of models")




#The best models so far I used a basic model formula
# -----------------------------------------------
# 19. CREATE SUBMISSION FILE
# -----------------------------------------------
submission_best_model <- tibble(
  respondent_id = test_df$respondent_id,
  h1n1_vaccine = test_pred_h1n1_xgboost,
  seasonal_vaccine = test_pred_seas_xgboost)


# -----------------------------------------------
# 20. SAVE SUBMISSION
# -----------------------------------------------
write_csv(submission_random_forest, "xgboost-best-model.csv")
