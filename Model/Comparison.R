# Load required packages
library(tidymodels)
library(vip)
library(probably)
library(car)
library(ggplot2)
library(rms)

# 1. Update recipe with interaction terms for logistic regression
# Create Interaction Terms using domain knowledge via formula
interactions_h1n1 <- ~
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
  opinion_h1n1_vacc_effective:opinion_h1n1_sick_from_vacc

# Create Interaction Terms using domain knowledge via formula
interactions_seasonal <- ~ 
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
  opinion_seas_vacc_effective:opinion_seas_sick_from_vacc

# 2. Collect predictions from each model
bt_h1n1_preds <- bt_h1n1_preds %>%
  mutate(model = "Bagged Trees")
rf_h1n1_preds  <- rf_h1n1_preds %>% 
  mutate(model = "Random Forest")
xgb_h1n1_preds <- xgb_h1n1_preds %>%
  mutate(model = "XGBoost")

bt_seas_preds <- bt_seas_preds %>%
  mutate(model = "Bagged Trees")
rf_seas_preds  <- rf_seas_preds %>%
  mutate(model = "Random Forest")
xgb_seas_preds <- xgb_seas_preds %>%
  mutate(model = "XGBoost")


# Combine predictions (add , xgb_h1n1_preds when done)
all_preds_h1n1 <- bind_rows(bt_h1n1_preds, rf_h1n1_preds) %>%
  mutate(.pred_class = if_else(.pred_1 > 0.5, "Yes", "No") %>% factor(levels = levels(h1n1_vaccine)))

all_preds_seas <- bind_rows(bt_seas_preds, rf_seas_preds) %>%
  mutate(.pred_class = if_else(.pred_1 > 0.5, "Yes", "No") %>% factor(levels = levels(seasonal_vaccine)))

# 3. Model diagnostics

## 3.1 Variable Importance in model code

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
#   ggtitle("Calibration Plot - XGBoost")

## 3.4 Confusion matrices
bt_conf_h1n1  <- conf_mat(bt_h1n1_preds,  truth = h1n1_vaccine, estimate = .pred_class)
rf_conf_h1n1  <- conf_mat(rf_h1n1_preds,  truth = h1n1_vaccine, estimate = .pred_class)
xgb_conf_h1n1 <- conf_mat(xgb_h1n1_preds, truth = h1n1_vaccine, estimate = .pred_class)

bt_conf_seas  <- conf_mat(bt_seas_preds,  truth = seasonal_vaccine, estimate = .pred_class)
rf_conf_seas  <- conf_mat(rf_seas_preds,  truth = seasonal_vaccine, estimate = .pred_class)
xgb_conf_seas <- conf_mat(xgb_seas_preds, truth = seasonal_vaccine, estimate = .pred_class)

## 3.5 Gain curves
gain_curve(bt_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>% 
  autoplot() + 
  ggtitle("Gain Curve - Bagged Trees")

gain_curve(rf_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>% 
  autoplot() + 
  ggtitle("Gain Curve - Random Forest")

gain_curve(xgb_h1n1_preds, truth = h1n1_vaccine, .pred_1) %>% 
  autoplot() + 
  ggtitle("Gain Curve - XGBoost")


gain_curve(bt_seas_preds, truth = seasonal_vaccine, .pred_1) %>% 
  autoplot() + 
  ggtitle("Gain Curve - Bagged Trees")

gain_curve(rf_seas_preds, truth = seasonal_vaccine, .pred_1) %>% 
  autoplot() + 
  ggtitle("Gain Curve - Random Forest")

gain_curve(xgb_seas_preds, truth = seasonal_vaccine, .pred_1) %>% 
  autoplot() + 
  ggtitle("Gain Curve - XGBoost")


######### ROC
# Bagged Trees
autoplot(bt_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (Bagged Trees)")
autoplot(bt_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (Bagged Trees)")

# Random Forest
autoplot(rf_roc_h1n1) + ggtitle("Final H1N1 Vaccine ROC Curve (Random Forest)")
autoplot(rf_roc_seas)  + ggtitle("Final Seasonal Vaccine ROC Curve (Random Forest)")


## 3.6 Performance metrics
bt_metrics_h1n1
rf_metrics_h1n1
xgb_metrics_h1n1

bt_metrics_seas
rf_metrics_seas
xgb_metrics_seas

bt_rs_metrics_h1n1
rf_rs_metrics_h1n1
xgb_rs_metrics_h1n1

bt_rs_metrics_seas
rf_rs_metrics_h1n1
xgb_rs_metrics_seas