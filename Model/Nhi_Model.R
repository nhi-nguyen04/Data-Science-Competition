# STEP 4: Create a Baseline Model

# Load in required packages
library(caret)
library(glmnet)

# Set seed for reproducibility
set.seed(123)

# Create 5-fold cross-validation for h1n1_vaccine and seasonal_vaccine
folds_h1n1 <- createFolds(data_train$h1n1_vaccine,
                          k = 5, 
                          list = TRUE, 
                          returnTrain = TRUE)

folds_seasonal <- createFolds(data_train$seasonal_vaccine,
                              k = 5, 
                              list = TRUE, 
                              returnTrain = TRUE)

results <- list()

# Model for h1n1_vaccine
for (i in 1:5) {
  
  train_idx_h1n1 <- folds_h1n1[[i]]
  train_data_h1n1 <- data_train[train_idx_h1n1, ]
  val_data_h1n1 <- data_train[-train_idx_h1n1, ]
  
  # Impute missing data in training data
  imputed_train_h1n1 <- train_data_h1n1 %>%
    mutate(across(where(is.character), ~ replace_na(.x, mode(., na.rm = TRUE))),
           across(where(is.numeric), ~ replace_na(.x, median(., na.rm = TRUE))))
  
  # Store training medians and modes for test imputation
  train_medians_h1n1 <- sapply(train_data_h1n1[sapply(train_data_h1n1, is.numeric)], median, na.rm = TRUE)
  train_modes_h1n1   <- modes_df(train_data_h1n1)
  
  # Change data types to factor
  imputed_train_h1n1 <- imputed_train_h1n1 %>%
    mutate(across(2:last_col(), factor))
  
  # Create Interaction Terms using domain knowledge via formula
  formula_h1n1 <- h1n1_vaccine ~ . +
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
  
  # Prepare Design Matrix for glmnet
  x_train_h1n1 <- model.matrix(formula_h1n1, data = imputed_train_h1n1)[, -1]
  y_train_h1n1 <- imputed_train_h1n1$h1n1_vaccine
  
  # LASSO with CV to choose best lambda
  cv_fit_h1n1 <- cv.glmnet(x_train_h1n1, y_train_h1n1, alpha = 1, family = "binomial")
  
  best_lambda_h1n1 <- cv_fit_h1n1$lambda.min
  lasso_model_h1n1 <- glmnet(x_train_h1n1, y_train_h1n1, alpha = 1, lambda = best_lambda_h1n1, family = "binomial")
  
  # Apply same interaction to validation data and use median and mode from training set to impute
  # Avoid data leakage (models don't handle NAs)
  
  imputed_val_h1n1 <- impute_with_training_stats(df = val_data_h1n1, 
                                            medians = train_medians_h1n1, 
                                            modes = train_modes_h1n1)
  
  # Change datatype
  imputed_val_h1n1 <- imputed_val_h1n1 %>%
    mutate(across(2:last_col(), factor))
  
  x_val_h1n1 <- model.matrix(formula_h1n1, data = imputed_val_h1n1)[, -1]
  y_val_h1n1 <- imputed_val_h1n1$h1n1_vaccine
  
  # Predict and Store Results (Note: we want probabilities later)
  pred_prob_h1n1 <- predict(lasso_model_h1n1, newx = x_val_h1n1, type = "response")
  pred_class_h1n1 <- ifelse(pred_prob_h1n1 > 0.5, 1, 0)
  
  accuracy_h1n1 <- mean(pred_class_h1n1 == y_val_h1n1)
  results[[i]] <- list(model = lasso_model_h1n1, accuracy = accuracy_h1n1, lambda = best_lambda_h1n1)
}

# Summary of CV Accuracies (h1n1_vaccine)
mean(sapply(results, function(x) x$accuracy))

# Predictions h1n1_vaccine
# Prepare data_test

# Use the imputation function to fill in missing values in test set
data_test_imputed_h1n1 <- impute_with_training_stats(data_test, train_medians_h1n1, train_modes_h1n1)

# Ensure correct data types for consistency
data_test_imputed_h1n1 <- data_test_imputed_h1n1 %>%
  mutate(across(2:last_col(), factor))

# Create h1n1_vaccine and seasonal_vaccine for predictions
data_test_imputed_h1n1$h1n1_vaccine <- 0
data_test_imputed_h1n1$seasonal_vaccine <- 0

# Create design matrix for test data using same formula
x_test_h1n1 <- model.matrix(formula_h1n1, data = data_test_imputed_h1n1)[, -1]

# Get the final model (e.g. from fold 5)
final_model_h1n1 <- results[[5]]$model

# Predict probabilities
pred_prob_h1n1 <- predict(final_model_h1n1, newx = x_test_h1n1, type = "response")

# View predicted probabilities and labels
head(pred_prob_h1n1)

# Save in dataframe
results_df <- data.frame(respondent_id = data_test$respondent_id,
                         h1n1_vaccine = pred_prob_h1n1)
results_df <- results_df %>%
  rename("h1n1_vaccine" = "s0")

# Model for seasonal_vaccine
for (i in 1:5) {
  
  train_idx_seasonal <- folds_seasonal[[i]]
  train_data_seasonal <- data_train[train_idx_seasonal, ]
  val_data_seasonal <- data_train[-train_idx_seasonal, ]
  
  # Impute missing data in training data
  imputed_train_seasonal <- train_data_seasonal %>%
    mutate(across(where(is.character), ~ replace_na(.x, mode(., na.rm = TRUE))),
           across(where(is.numeric), ~ replace_na(.x, median(., na.rm = TRUE))))
  
  # Store training medians and modes for test imputation
  train_medians_seasonal <- sapply(train_data_seasonal[sapply(train_data_seasonal, is.numeric)], median, na.rm = TRUE)
  train_modes_seasonal   <- modes_df(train_data_seasonal)
  
  # Change data types to factor
  imputed_train_seasonal <- imputed_train_seasonal %>%
    mutate(across(2:last_col(), factor))
  
  # Create Interaction Terms using domain knowledge via formula
  formula_seasonal <- seasonal_vaccine ~ . +
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
  
  # Prepare Design Matrix for glmnet
  x_train_seasonal <- model.matrix(formula_seasonal, data = imputed_train_seasonal)[, -1]
  y_train_seasonal <- imputed_train_seasonal$seasonal_vaccine
  
  # LASSO with CV to choose best lambda
  cv_fit_seasonal <- cv.glmnet(x_train_seasonal, y_train_seasonal, alpha = 1, family = "binomial")
  
  best_lambda_seasonal <- cv_fit_seasonal$lambda.min
  lasso_model_seasonal <- glmnet(x_train_seasonal, y_train_seasonal, alpha = 1, lambda = best_lambda_seasonal, family = "binomial")
  
  # Apply same interaction to validation data and use median and mode from training set to impute
  # Avoid data leakage (models don't handle NAs)
  
  imputed_val_seasonal <- impute_with_training_stats(df = val_data_seasonal, 
                                            medians = train_medians_seasonal, 
                                            modes = train_modes_seasonal)
  
  # Change datatype
  imputed_val_seasonal <- imputed_val_seasonal %>%
    mutate(across(2:last_col(), factor))
  
  x_val_seasonal <- model.matrix(formula_seasonal, data = imputed_val_seasonal)[, -1]
  y_val_seasonal <- imputed_val_seasonal$seasonal_vaccine
  
  # Predict and Store Results (Note: we want probabilities later)
  pred_prob_seasonal <- predict(lasso_model_seasonal, newx = x_val_seasonal, type = "response")
  pred_class_seasonal <- ifelse(pred_prob_seasonal > 0.5, 1, 0)
  
  accuracy_seasonal <- mean(pred_class_seasonal == y_val_seasonal)
  results[[i]] <- list(model = lasso_model_seasonal, accuracy = accuracy_seasonal, lambda = best_lambda_seasonal)
}

# Summary of CV Accuracies (seasonal_vaccine)
mean(sapply(results, function(x) x$accuracy))

# Use the imputation function to fill in missing values in test set
data_test_imputed_seasonal <- impute_with_training_stats(data_test, train_medians_seasonal, train_modes_seasonal)

# Ensure correct data types for consistency
data_test_imputed_seasonal <- data_test_imputed_seasonal %>%
  mutate(across(2:last_col(), factor))

# Create h1n1_vaccine and seasonal_vaccine for predictions
data_test_imputed_seasonal$h1n1_vaccine <- 0
data_test_imputed_seasonal$seasonal_vaccine <- 0

# Create design matrix for test data using same formula
x_test_seasonal <- model.matrix(formula_seasonal, data = data_test_imputed_seasonal)[, -1]

# Get the final model (e.g. from fold 5)
final_model_seasonal <- results[[5]]$model

# Predict probabilities
pred_prob_seasonal <- predict(final_model_seasonal, newx = x_test_seasonal, type = "response")

# View predicted probabilities and labels
head(pred_prob_seasonal)

# Save in dataframe
results_df2 <- data.frame(respondent_id = data_test$respondent_id,
                          seasonal_vaccine = pred_prob_seasonal)
results_df2 <- results_df2 %>%
  rename("seasonal_vaccine" = "s0")

results <- results_df %>%
  inner_join(results_df2, by = "respondent_id") %>%
  mutate(across(1:3, as.double))

# Save
write.csv(results,"./data/filename.csv", row.names = FALSE)


# STEP 5: Analyse Misclassifications
library(pROC)
library(MLmetrics)

# Create Confusion Matrix
conf_matrix_h1n1 <- confusionMatrix(factor(pred_class_h1n1), factor(y_val_h1n1), positive = "1")
conf_matrix_seasonal <- confusionMatrix(factor(pred_class_seasonal), factor(y_val_seasonal), positive = "1")

# Accuracy
acc_h1n1 <- conf_matrix_h1n1$overall["Accuracy"]
acc_seasonal <- conf_matrix_seasonal$overall["Accuracy"]

# Misclassification Rate
misclass_rate_h1n1 <- 1 - acc_h1n1
misclass_rate_seasonal <- 1 - acc_seasonal

# Precision, Recall, F1
precision_h1n1 <- conf_matrix_h1n1$byClass["Precision"]
recall_h1n1 <- conf_matrix_h1n1$byClass["Recall"]
f1_h1n1 <- conf_matrix_h1n1$byClass["F1"]

precision_seasonal <- conf_matrix_seasonal$byClass["Precision"]
recall_seasonal <- conf_matrix_seasonal$byClass["Recall"]
f1_seasonal <- conf_matrix_seasonal$byClass["F1"]

# # AUC
# roc_obj_h1n1 <- roc(y_val_h1n1, as.numeric(pred_prob_h1n1))
# auc_h1n1 <- auc(roc_obj_h1n1)
# 
# roc_obj_seasonal <- roc(y_val_seasonal, as.numeric(pred_prob_seasonal))
# auc_seasonal <- auc(roc_obj_seasonal)

# # Log Loss (Cross Entropy)
# log_loss_h1n1 <- LogLoss(pred_prob_h1n1, y_val_h1n1)
# log_loss_seasonal <- LogLoss(pred_prob_seasonal, y_val_seasonal)

# Print all
print(list(
  Accuracy = acc_h1n1,
  MisclassificationRate = misclass_rate_h1n1,
  Precision = precision_h1n1,
  Recall = recall_h1n1,
  F1 = f1_h1n1
 # AUC = auc_h1n1,
 # LogLoss = log_loss_h1n1
))

print(list(
  Accuracy = acc_seasonal,
  MisclassificationRate = misclass_rate_seasonal,
  Precision = precision_seasonal,
  Recall = recall_seasonal,
  F1 = f1_seasonal
 # AUC = auc_seasonal,
 # LogLoss = log_loss_seasonal
))


