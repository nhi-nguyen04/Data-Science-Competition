# Load required packages
library(tidymodels)
library(vip)
library(probably)
library(car)
library(ggplot2)
library(probably)

# 1. Update recipe with interaction terms for logistic regression
# Create Interaction Terms using domain knowledge via formula
interactions_h1n1 <- as.formula("~
  h1n1_concern:behavioral_avoidance +
  h1n1_concern:behavioral_face_mask +
  h1n1_knowledge:behavioral_antiviral_meds +
  doctor_recc_h1n1:opinion_h1n1_vacc_effective +
  doctor_recc_seasonal:opinion_seas_vacc_effective +
  health_worker:opinion_h1n1_risk +
  health_worker:opinion_seas_risk +
  income_poverty:health_insurance +
  education:health_insurance +
  child_under_6_months:chronic_med_condition +
  child_under_6_months:behavioral_touch_face +
  opinion_h1n1_vacc_effective:opinion_h1n1_sick_from_vacc")

# Create Interaction Terms using domain knowledge via formula
interactions_seasonal <- as.formula("~ 
  doctor_recc_seasonal:opinion_seas_vacc_effective +
  doctor_recc_seasonal:opinion_seas_risk +
  doctor_recc_seasonal:opinion_seas_sick_from_vacc +
  health_worker:opinion_seas_risk +
  health_worker:opinion_seas_vacc_effective +
  health_insurance:education +
  income_poverty:health_insurance +
  chronic_med_condition:opinion_seas_risk +
  child_under_6_months:behavioral_touch_face +
  behavioral_face_mask:opinion_seas_vacc_effective +
  opinion_seas_risk:opinion_seas_vacc_effective +
  opinion_seas_vacc_effective:opinion_seas_sick_from_vacc")

# 2. Collect predictions from each model
lr_h1n1_preds <- lr_h1n1_preds %>%
  mutate(model = "Logistic Regression")
bt_h1n1_preds <- bt_h1n1_preds %>%
  mutate(model = "Bagged Trees")
rf_h1n1_preds  <- rf_h1n1_preds %>% 
  mutate(model = "Random Forest")
xgb_h1n1_preds <- xgb_h1n1_preds %>%
  mutate(model = "XGBoost")

lr_seas_preds <- lr_h1n1_seas %>%
  mutate(model = "Logistic Regression")
bt_seas_preds <- bt_seas_preds %>%
  mutate(model = "Bagged Trees")
rf_seas_preds  <- rf_seas_preds %>%
  mutate(model = "Random Forest")
xgb_seas_preds <- xgb_seas_preds %>%
  mutate(model = "XGBoost")


# Combine predictions (add , xgb_h1n1_preds when done)
all_preds_h1n1 <- bind_rows(bt_h1n1_preds, rf_h1n1_preds, xgb_h1n1_preds) %>%
  mutate(.pred_class = if_else(.pred_1 > 0.5, "Yes", "No") %>% factor(levels = levels(h1n1_vaccine)))

all_preds_seas <- bind_rows(bt_seas_preds, rf_seas_preds, xgb_seas_preds) %>%
  mutate(.pred_class = if_else(.pred_1 > 0.5, "Yes", "No") %>% factor(levels = levels(seasonal_vaccine)))

# 3. Model diagnostics

## 3.1 Lift Charts
# H1N1 lift curves
lift_h1n1_bt <- lift_curve(bt_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>%
  mutate(Model = "Bagged Trees - H1N1")
lift_h1n1_rf <- lift_curve(rf_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>%
  mutate(Model = "Random Forest - H1N1")
lift_h1n1_xgb <- lift_curve(xgb_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>%
  mutate(Model = "XGBoost - H1N1")

# Seasonal lift curves
lift_seas_bt <- lift_curve(bt_seas_preds, truth = seasonal_vaccine, .pred_1) %>%
  mutate(Model = "Bagged Trees - Seasonal")
lift_seas_rf <- lift_curve(rf_seas_preds, truth = seasonal_vaccine, .pred_1) %>%
  mutate(Model = "Random Forest - Seasonal")
lift_seas_xgb <- lift_curve(xgb_seas_preds, truth = seasonal_vaccine, .pred_1) %>%
  mutate(Model = "XGBoost - Seasonal")

# Combine
combined_lift_h1n1 <- bind_rows(lift_h1n1_bt, lift_h1n1_rf, lift_h1n1_xgb)
combined_lift_seas <- bind_rows(lift_seas_bt, lift_seas_rf, lift_seas_xgb)

# Plot
ggplot(combined_lift_h1n1, aes(x = .percent_tested, y = .lift, color = Model)) +
  geom_line(size = 1) +
  labs(
    title = "Lift Chart - H1N1 Vaccine Models",
    x = "% Samples Tested",
    y = "Lift") +
  theme_minimal()

ggplot(combined_lift_seas, aes(x = .percent_tested, y = .lift, color = Model)) +
  geom_line(size = 1) +
  labs(
    title = "Lift Chart - Seasonal Vaccine Models",
    x = "% Samples Tested",
    y = "Lift") +
  theme_minimal()

## 3.2 Calibration plots
# bt_h1n1_preds %>%
#   calibrate(truth = h1n1_vaccine, .pred_1, n_bins = 10) %>%
#   autoplot() +
#   ggtitle("Calibration Plot - Bagged Trees")
# 
# rf_h1n1_preds %>%
#   calibrate(truth = h1n1_vaccine, .pred_1, n_bins = 10) %>%
#   autoplot() +
#   ggtitle("Calibration Plot - Random Forest")
# 
# xgb_h1n1_preds %>%
#   calibrate(truth = h1n1_vaccine, .pred_1, n_bins = 10) %>%
#   autoplot() +
#   ggtitle("Calibration Plot - XGBoost")
# 
# 
# bt_seas_preds %>%
#   calibrate(truth = seasonal_vaccine, .pred_1, n_bins = 10) %>%
#   autoplot() +
#   ggtitle("Calibration Plot - Bagged Trees")
# 
# rf_seas_preds %>%
#   calibrate(truth = seasonal_vaccine, .pred_1, n_bins = 10) %>%
#   autoplot() +
#   ggtitle("Calibration Plot - Random Forest")
# 
# xgb_seas_preds %>%
#   calibrate(truth = seasonal_vaccine, .pred_1, n_bins = 10) %>%
#   autoplot() +
# ggtitle("Calibration Plot - XGBoost")


## 3.4 Confusion matrices
bt_conf_h1n1  <- conf_mat(bt_h1n1_preds,  truth = h1n1_vaccine, estimate = .pred_class)
rf_conf_h1n1  <- conf_mat(rf_h1n1_preds,  truth = h1n1_vaccine, estimate = .pred_class)
xgb_conf_h1n1 <- conf_mat(xgb_h1n1_preds, truth = h1n1_vaccine, estimate = .pred_class)

bt_conf_seas  <- conf_mat(bt_seas_preds,  truth = seasonal_vaccine, estimate = .pred_class)
rf_conf_seas  <- conf_mat(rf_seas_preds,  truth = seasonal_vaccine, estimate = .pred_class)
xgb_conf_seas <- conf_mat(xgb_seas_preds, truth = seasonal_vaccine, estimate = .pred_class)

## 3.5 Gain curves
gain_h1n1_bt  <- gain_curve(bt_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>% 
  mutate(Model = "Bagged Trees")
gain_h1n1_rf  <- gain_curve(rf_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>%
  mutate(Model = "Random Forest")
gain_h1n1_xgb <- gain_curve(xgb_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>%
  mutate(Model = "XGBoost")

gain_combined_h1n1 <- bind_rows(gain_h1n1_bt, gain_h1n1_rf, gain_h1n1_xgb)

ggplot(gain_combined_h1n1, aes(x = .percent_tested, y = .percent_found, color = Model)) +
  geom_line(size = 1) +
  labs(title = "Gain Curve: H1N1 Vaccine Models", x = "% Samples Tested", y = "% Positives Found") +
  theme_minimal()

gain_seas_bt  <- gain_curve(bt_seas_preds, truth = seasonal_vaccine, .pred_1) %>%
  mutate(Model = "Bagged Trees")
gain_seas_rf  <- gain_curve(rf_seas_preds, truth = seasonal_vaccine, .pred_1) %>% 
  mutate(Model = "Random Forest")
gain_seas_xgb <- gain_curve(xgb_seas_preds, truth = seasonal_vaccine, .pred_1) %>%
  mutate(Model = "XGBoost")

gain_combined_seas <- bind_rows(gain_seas_bt, gain_seas_rf, gain_seas_xgb)

ggplot(gain_combined_seas, aes(x = .percent_tested, y = .percent_found, color = Model)) +
  geom_line(size = 1) +
  labs(title = "Gain Curve: Seasonal Vaccine Models", x = "% Samples Tested", y = "% Positives Found") +
  theme_minimal()


######### ROC and Precision-Recall
# Bagged Trees
autoplot(bt_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (Bagged Trees)")
autoplot(bt_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (Bagged Trees)")

# Random Forest
autoplot(rf_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (Random Forest)")
autoplot(rf_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (Random Forest)")

# XGBoost
autoplot(xgb_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (XGBoost)")
autoplot(xgb_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (XGBoost)")

# Comparing all in one plot
bt_roc_h1n1$Model <- "Bagged Trees - H1N1"
bt_roc_seas$Model <- "Bagged Trees - Seasonal"
rf_roc_h1n1$Model <- "Random Forest - H1N1"
rf_roc_seas$Model <- "Random Forest - Seasonal"
xgb_roc_h1n1$Model <- "XGBoost - H1N1"
xgb_roc_seas$Model <- "XGBoost - Seasonal"


combined_roc_h1n1 <- bind_rows(bt_roc_h1n1, rf_roc_h1n1, xgb_roc_h1n1)
combined_roc_seas <- bind_rows(bt_roc_seas, rf_roc_seas, xgb_roc_seas)

ggplot(combined_roc_h1n1, aes(x = 1 - specificity, y = sensitivity, color = Model)) +
  geom_line(size = 1) +
  geom_abline(linetype = "dashed", color = "#6E6E6E") +
  labs(
    title = "ROC Curves for H1N1 Vaccine Models",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_minimal()

ggplot(combined_roc_seas, aes(x = 1 - specificity, y = sensitivity, color = Model)) +
  geom_line(size = 1) +
  geom_abline(linetype = "dashed", color = "#6E6E6E") +
  labs(
    title = "ROC Curves for Seasonal Vaccine Models",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_minimal()


# Precision-Recall
pr_h1n1_bt  <- pr_curve(bt_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>% 
  mutate(Model = "Bagged Trees")
pr_h1n1_rf  <- pr_curve(rf_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>%
  mutate(Model = "Random Forest")
pr_h1n1_xgb <- pr_curve(xgb_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>%
  mutate(Model = "XGBoost")

combined_pr_h1n1 <- bind_rows(pr_h1n1_bt, pr_h1n1_rf, pr_h1n1_xgb)

ggplot(combined_pr_h1n1, aes(x = recall, y = precision, color = Model)) +
  geom_line(size = 1) +
  labs(title = "Precision-Recall Curve: H1N1 Vaccine Models", x = "Recall", y = "Precision") +
  theme_minimal()

pr_seas_bt  <- pr_curve(bt_seas_preds, truth = seasonal_vaccine, .pred_1) %>% 
  mutate(Model = "Bagged Trees")
pr_seas_rf  <- pr_curve(rf_seas_preds, truth = seasonal_vaccine, .pred_1) %>% 
  mutate(Model = "Random Forest")
pr_seas_xgb <- pr_curve(xgb_seas_preds, truth = seasonal_vaccine, .pred_1) %>% 
  mutate(Model = "XGBoost")

combined_pr_seas <- bind_rows(pr_seas_bt, pr_seas_rf, pr_seas_xgb)

ggplot(combined_pr_seas, aes(x = recall, y = precision, color = Model)) +
  geom_line(size = 1) +
  labs(title = "Precision-Recall Curve: Seasonal Vaccine Models", x = "Recall", y = "Precision") +
  theme_minimal()

## 3.6 Performance metrics
bt_metrics_h1n1$model <- "Bagged Trees"
rf_metrics_h1n1$model <- "Random Forest"
xgb_metrics_h1n1$model <- "XGBoost"

bt_metrics_seas$model <- "Bagged Trees"
rf_metrics_seas$model <- "Random Forest"
xgb_metrics_seas$model <- "XGBoost"

all_metrics_h1n1 <- bind_rows(bt_metrics_h1n1, rf_metrics_h1n1, xgb_metrics_h1n1) %>%
  select(.metric, .estimate, model) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
all_metrics_h1n1

all_metrics_seas <- bind_rows(bt_metrics_seas, rf_metrics_seas, xgb_metrics_seas) %>%
  select(.metric, .estimate, model) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
all_metrics_seas


bt_rs_metrics_h1n1$model <- "Bagged Trees"
rf_rs_metrics_h1n1$model <- "Random Forest"
xgb_rs_metrics_h1n1$model <- "XGBoost"

bt_rs_metrics_seas$model <- "Bagged Trees"
rf_rs_metrics_seas$model <- "Random Forest"
xgb_rs_metrics_seas$model <- "XGBoost"

all_metrics_rs_h1n1 <- bind_rows(bt_rs_metrics_h1n1, rf_rs_metrics_h1n1, xgb_rs_metrics_h1n1)  #%>%
 # select(.metric, .estimate, model) %>%
 # pivot_wider(names_from = .metric, values_from = .estimate)
all_metrics_rs_h1n1

all_metrics_rs_seas <- bind_rows(bt_rs_metrics_seas, rf_rs_metrics_seas, xgb_rs_metrics_seas) #%>%
 # select(.metric, .estimate, model) %>%
 # pivot_wider(names_from = .metric, values_from = .estimate)
all_metrics_rs_seas
