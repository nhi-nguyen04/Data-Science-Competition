# -----------------------------------------------
# 1.Pre Hypertuning
# -----------------------------------------------
# We are  performing a model comparison across all types of models in this course:
# decision trees, bagged trees, random forests, and gradient boosting.

#######################
#For H1N1  Flu Vaccine#
#######################


#Combine Preddictions
bind_cols(rf_h1n1_preds,rf_h1n1_preds,rf_h1n1_preds,rf_h1n1_preds,rf_h1n1_preds)




# Calculate the AUC for each model
auc_tree   <- roc_auc(preds_combined, truth = still_customer, estimate = preds_tree)
auc_bagged <- roc_auc(preds_combined, truth = still_customer, estimate = preds_bagging)
auc_forest <- roc_auc(preds_combined, truth = still_customer, estimate = preds_forest)
auc_boost  <- roc_auc(preds_combined, truth = still_customer, estimate = preds_boosting)

# Combine AUCs into one tibble
combined <- bind_rows(decision_tree = auc_tree,
                      bagged_tree = auc_bagged,
                      random_forest = auc_forest,
                      boosted_tree = auc_boost,
                      .id = "model")

combined




# Reshape the predictions into long format
predictions_long <- tidyr::pivot_longer(preds_combined,
                                        cols = starts_with("preds_"),
                                        names_to = "model",
                                        values_to = "predictions")

predictions_long %>% 
  # Group by model
  group_by(model) %>% 
  # Calculate values for every cutoff
  roc_curve(truth = still_customer, 
            estimate = predictions) %>%
  # Create a plot from the calculated data
  autoplot()





##########################
#For Seasonal Flu Vaccine#
##########################





# -----------------------------------------------
# 1.Post Hypertuning
# -----------------------------------------------
