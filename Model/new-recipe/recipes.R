# ──────────── Basic Recipe ────────────────────

# -----------------------------------------------
# 6. R E C I P E  –– consistent imputation + dummies
# -----------------------------------------------
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  # Remove the other target (seasonal) if it’s present
  #creates a specification of a recipe step that will remove selected variables.
  step_rm(seasonal_vaccine) %>%
  # Impute all numeric predictors by median:
  step_impute_median(all_numeric_predictors()) %>%
  # Create an "unknown" level for any missing factor
  step_unknown(all_nominal_predictors()) %>%
  # One‐hot encode all factors
  step_dummy(all_nominal_predictors()) %>%
  # <- drops any predictors that have zero variance
  step_zv(all_predictors()) %>% 
  # Normalize numeric columns
  step_normalize(all_numeric_predictors()) # this migth be wrong???????????????????



seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())




# ──────────── Advanced  Recipe ────────────────────

h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>% # Remove target leakage
  
  # 1. Handle missing values first - Order matters!
  # Impute numericals (median is generally robust for skewed data)
  step_impute_median(all_numeric_predictors()) %>%
  # Impute nominals (creates a new level 'unknown' for NAs)
  step_unknown(all_nominal_predictors()) %>%
  
  # 2. Feature Engineering based on Importance Plot
  # These are highly influential. Let's create more specific interactions
  # beyond just pairwise based on the plot insights.
  
  # Stronger Interaction: Doctor's H1N1 Rec + Opinion H1N1 Vacc Effective + H1N1 Risk
  # This combines the top 3 most important features from the plot
  step_interact(terms = ~ doctor_recc_h1n1:opinion_h1n1_vacc_effective:opinion_h1n1_risk) %>%
  
  # General Pro-Vaccine Stance: Doctor's Seasonal Rec + Opinion Seasonal Vacc Effective
  # This captures a general inclination towards vaccines, as suggested by their importance for H1N1
  step_interact(terms = ~ doctor_recc_seasonal:opinion_seas_vacc_effective) %>%
  
  # Concern vs. Risk Perception (reinforcement or contradiction)
  # Captures if high concern translates to high perceived risk.
  step_interact(terms = ~ h1n1_concern:opinion_h1n1_risk) %>%
  
  # Adverse Opinion: If you think you'll get sick from the vaccine (H1N1 or Seasonal)
  # These were also highly important, but negatively.
  # We might want a combined "vaccine hesitancy" interaction.
  step_interact(terms = ~ opinion_h1n1_sick_from_vacc:opinion_seas_sick_from_vacc) %>%
  
  # 3. Create dummy variables for all nominal predictors AFTER imputations and interactions
  # This ensures interaction terms are created from original levels, then dummified.
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% # one_hot=TRUE is generally preferred for tree models
  
  # 4. Clean up
  step_zv(all_predictors()) %>% # Remove zero-variance predictors
  # Normalization for tree models is optional, but harmless. Keeping for consistency.
  step_normalize(all_numeric_predictors())

# Note: For tree-based models (especially Random Forest and XGBoost),
# feature selection based on importance can sometimes be beneficial,
# but it's often better to let the tree model handle it internally unless
# you have a very high-dimensional dataset or strong overfitting.
# Given your current features, letting the model prune is usually fine.



seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>% # Remove target leakage
  
  # 1. Handle missing values first
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # **CRUCIAL CHANGE: Create dummy variables BEFORE creating interactions**
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # 2. Feature Engineering - NOW create interactions with the DUMMY VARIABLES
  #    Note: 'age_group' will now be expanded into 'age_group_X65_.Years', etc.
  #    You will need to refer to the specific dummy variable columns for interaction,
  #    or simplify how you express the interaction.
  #    Let's assume the dummy variables are named like 'age_group_X65_.Years',
  #    'age_group_18_34.Years', etc., after step_dummy.
  
  # Core Seasonal Interaction: Effectiveness + Risk + Doctor Rec
  # Assuming these were also nominal and now dummified.
  # If 'opinion_seas_vacc_effective' is still the original column, it might require its own dummification if nominal.
  # But typically if it's already binary (0/1) or numeric, this is fine.
  # If they are nominal (like "Strongly Agree", "Disagree"), after step_dummy,
  # they become things like 'opinion_seas_vacc_effective_StronglyAgree'.
  # For simplicity, if these are original nominals, we'll interact them and let 'recipes' handle it.
  # If they become individual dummy columns like 'opinion_seas_vacc_effective_Yes', then we need specific interactions.
  
  # Let's assume for now that 'opinion_seas_vacc_effective', 'opinion_seas_risk', 'doctor_recc_seasonal'
  # are either numeric/binary OR 'step_interact' can handle them when they're still nominal
  # BUT age_group is the tricky one with many levels.
  
  # New approach to interactions after step_dummy:
  # Now you'd typically interact specific dummy variables.
  # However, if you use the original nominal name in step_interact *after* step_dummy,
  # 'recipes' will interact *all* the resulting dummy variables from that nominal. This is usually what you want.
  
  # Let's keep the `age_group` specific interaction and trust `recipes` to use its dummy versions.
  
  step_interact(terms = ~ opinion_seas_vacc_effective:opinion_seas_risk:doctor_recc_seasonal) %>%
  
  # Age Group X65+ specific interactions.
  # This syntax means: interact all dummy variables created from 'age_group' with 'opinion_seas_vacc_effective'
  #step_interact(terms = ~ age_group:opinion_seas_vacc_effective) %>%
  #step_interact(terms = ~ age_group:doctor_recc_seasonal) %>%
  
  # Cross-Vaccine Risk Perception + Seasonal Effectiveness
  step_interact(terms = ~ opinion_h1n1_risk:opinion_seas_vacc_effective) %>%
  
  # Overall Vaccine Hesitancy/Negative Experience/Belief
  step_interact(terms = ~ opinion_seas_sick_from_vacc:opinion_h1n1_sick_from_vacc) %>%
  
  # 3. Clean up
  step_zv(all_predictors()) %>% # Remove zero-variance predictors
  step_normalize(all_numeric_predictors()) # Optional for tree models, but harmless.