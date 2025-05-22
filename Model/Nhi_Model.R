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
  
  train_idx <- folds_h1n1[[i]]
  train_data <- data_train[train_idx, ]
  val_data <- data_train[-train_idx, ]
  
  # Impute missing data in training data
  imputed_train <- train_data %>%
    mutate(across(where(is.character), ~ replace_na(.x, mode(., na.rm = TRUE))),
           across(where(is.numeric), ~ replace_na(.x, median(., na.rm = TRUE))))
  
  # Store training medians and modes for test imputation
  train_medians <- sapply(train_data[sapply(train_data, is.numeric)], median, na.rm = TRUE)
  train_modes   <- modes_df(train_data)
  
  # Change data types to factor
  imputed_train <- imputed_train %>%
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
  x_train <- model.matrix(formula_h1n1, data = imputed_train)[, -1]
  y_train <- imputed_train$h1n1_vaccine
  
  # LASSO with CV to choose best lambda
  cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
  
  best_lambda <- cv_fit$lambda.min
  lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda, family = "binomial")
  
  # Apply same interaction to validation data and use median and mode from training set to impute
  # Avoid data leakage (models don't handle NAs)
  
  imputed_val <- impute_with_training_stats(df = val_data, 
                                            medians = train_medians, 
                                            modes = train_modes)
  
  # Change datatype
  imputed_val <- imputed_val %>%
    mutate(across(2:last_col(), factor))
  
  x_val <- model.matrix(formula_h1n1, data = imputed_val)[, -1]
  y_val <- imputed_val$h1n1_vaccine
  
  # Predict and Store Results (Note: we want probabilities later)
  pred_prob <- predict(lasso_model, newx = x_val, type = "response")
  pred_class <- ifelse(pred_prob > 0.5, 1, 0)
  
  accuracy <- mean(pred_class == y_val)
  results[[i]] <- list(model = lasso_model, accuracy = accuracy, lambda = best_lambda)
}

# Summary of CV Accuracies (h1n1_vaccine)
mean(sapply(results, function(x) x$accuracy))

# Predictions h1n1_vaccine
# Prepare data_test

# Use the imputation function to fill in missing values in test set
data_test_imputed <- impute_with_training_stats(data_test, train_medians, train_modes)

# Ensure correct data types for consistency
data_test_imputed <- data_test_imputed %>%
  mutate(across(2:last_col(), factor))

# Create h1n1_vaccine and seasonal_vaccine for predictions
data_test_imputed$h1n1_vaccine <- 0
data_test_imputed$seasonal_vaccine <- 0

# Create design matrix for test data using same formula
x_test <- model.matrix(formula_h1n1, data = data_test_imputed)[, -1]

# Get the final model (e.g. from fold 5)
final_model <- results[[5]]$model

# Predict probabilities
pred_prob <- predict(final_model, newx = x_test, type = "response")

# View predicted probabilities and labels
head(pred_prob)
head(pred_class)

# Save in dataframe
results_df <- data.frame(respondent_id = data_test$respondent_id,
                         h1n1_vaccine = pred_prob)
results_df <- results_df %>%
  rename("h1n1_vaccine" = "s0")

# Model for seasonal_vaccine
for (i in 1:5) {
  
  train_idx <- folds_seasonal[[i]]
  train_data <- data_train[train_idx, ]
  val_data <- data_train[-train_idx, ]
  
  # Impute missing data in training data
  imputed_train <- train_data %>%
    mutate(across(where(is.character), ~ replace_na(.x, mode(., na.rm = TRUE))),
           across(where(is.numeric), ~ replace_na(.x, median(., na.rm = TRUE))))
  
  # Store training medians and modes for test imputation
  train_medians <- sapply(train_data[sapply(train_data, is.numeric)], median, na.rm = TRUE)
  train_modes   <- modes_df(train_data)
  
  # Change data types to factor
  imputed_train <- imputed_train %>%
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
  x_train <- model.matrix(formula_seasonal, data = imputed_train)[, -1]
  y_train <- imputed_train$seasonal_vaccine
  
  # LASSO with CV to choose best lambda
  cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
  
  best_lambda <- cv_fit$lambda.min
  lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda, family = "binomial")
  
  # Apply same interaction to validation data and use median and mode from training set to impute
  # Avoid data leakage (models don't handle NAs)
  
  imputed_val <- impute_with_training_stats(df = val_data, 
                                            medians = train_medians, 
                                            modes = train_modes)
  
  # Change datatype
  imputed_val <- imputed_val %>%
    mutate(across(2:last_col(), factor))
  
  x_val <- model.matrix(formula_seasonal, data = imputed_val)[, -1]
  y_val <- imputed_val$seasonal_vaccine
  
  # Predict and Store Results (Note: we want probabilities later)
  pred_prob <- predict(lasso_model, newx = x_val, type = "response")
  pred_class <- ifelse(pred_prob > 0.5, 1, 0)
  
  accuracy <- mean(pred_class == y_val)
  results[[i]] <- list(model = lasso_model, accuracy = accuracy, lambda = best_lambda)
}

# Summary of CV Accuracies (seasonal_vaccine)
mean(sapply(results, function(x) x$accuracy))

# Create design matrix for test data using same formula
x_test <- model.matrix(formula_seasonal, data = data_test_imputed)[, -1]

# Get the final model (e.g. from fold 5)
final_model <- results[[5]]$model

# Predict probabilities
pred_prob <- predict(final_model, newx = x_test, type = "response")

# View predicted probabilities and labels
head(pred_prob)

# Save in dataframe
results_df2 <- data.frame(respondent_id = data_test$respondent_id,
                          seasonal_vaccine = pred_prob)
results_df2 <- results_df2 %>%
  rename("seasonal_vaccine" = "s0")

results <- results_df %>%
  inner_join(results_df2, by = "respondent_id") %>%
  mutate(across(1:3, as.double))

# Save
write.csv(results,"./data/filename.csv", row.names = FALSE)