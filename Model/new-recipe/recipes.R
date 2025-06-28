h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>%
  
  # 1) impute & mark unknown
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # 2) **now** build interactions on the *untouched* columns
  step_interact(
    terms = ~ 
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
  ) %>%
  
  # 3) then dummy‐encode all your factors
  step_dummy(all_nominal_predictors()) %>%
  
  # 4) drop zero‐variances and normalize
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  step_interact(
    terms = ~ 
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
  ) %>%
  
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())




#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################



# H1N1 recipe: dummies first, then interactions
h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>%
  
  # 1) impute & mark unknown
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # 2) one-hot encode all factors
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # inspect dummy names to plug into step_interact()
  # tidy(h1n1_recipe, number = 3)
  
  # 3) interact on the dummy columns
  step_interact(
    terms = ~
      # e.g. h1n1_concern is numeric, so you can still do:
      h1n1_concern:behavioral_avoidance +
      h1n1_concern:behavioral_face_mask +
      # for factor f with levels A,B,C dummy-encoded to f_A, f_B, f_C:
      # interact specific levels, e.g. poverty & insurance:
      income_poverty_BelowPoverty:health_insurance_Yes +
      income_poverty_AtOrAbovePoverty:health_insurance_Yes +
      # and so on for your other categorical interactions...
      opinion_h1n1_vacc_effective:opinion_h1n1_sick_from_vacc
  ) %>%
  
  # 4) drop any zero-variance cols and normalize
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Seasonal recipe: same pattern
seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  
  # tidy(seas_recipe, number = 3)
  
  step_interact(
    terms = ~
      h1n1_concern:behavioral_avoidance +
      h1n1_concern:behavioral_face_mask +
      # example categorical interaction:
      education_HighSchool:health_insurance_Yes +
      child_under_6_months_Yes:behavioral_touch_face +
      opinion_h1n1_vacc_effective:opinion_h1n1_sick_from_vacc
  ) %>%
  
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())











#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################


library(recipes)

h1n1_recipe <- recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  # 1) Don’t use respondent_id as a predictor
  update_role(respondent_id, new_role = "ID") %>%
  # 2) Remove the seasonal_vaccine column (second target)
  step_rm(seasonal_vaccine) %>%
  # 3) Drop predictors that vary almost not at all
  step_nzv(all_predictors()) %>%
  # 4) Impute numeric NAs with the median
  step_impute_median(all_numeric_predictors()) %>%
  # 5) Impute categorical NAs with the most frequent level
  step_impute_mode(all_nominal_predictors()) %>%
  # 6) Map our ordered scales to integers
  step_integer(
    h1n1_concern, h1n1_knowledge,
    opinion_h1n1_vacc_effective, opinion_h1n1_risk, opinion_h1n1_sick_from_vacc,
    opinion_seas_vacc_effective, opinion_seas_risk, opinion_seas_sick_from_vacc,
    age_group, education, income_poverty, employment_status, marital_status,
    ordered = TRUE
  ) %>%
  # 7) Tag any new (unseen) factor levels in test data as “unknown”
  step_unknown(all_nominal_predictors()) %>%
  # 8) Turn every remaining nominal predictor into 0/1 dummies
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # 9) Center & scale all numeric predictors
  step_normalize(all_numeric_predictors())





#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################





library(recipes)
library(recipeselectors)

h1n1_recipe_fs <-
  recipe(h1n1_vaccine ~ ., data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(seasonal_vaccine) %>%
  step_nzv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_integer(
    h1n1_concern, h1n1_knowledge,
    opinion_h1n1_vacc_effective, opinion_h1n1_risk,
    opinion_h1n1_sick_from_vacc, ordered = TRUE
  ) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # 1) create _all_ pairwise numeric × numeric interactions
  step_interact(terms = ~ all_numeric_predictors():all_numeric_predictors()) %>%  # :contentReference[oaicite:0]{index=0}
  # 2) filter by Information Gain: keep only top 30 interactions
  step_select_infgain(
    all_predictors(), 
    outcome = "h1n1_vaccine", 
    top_p = 30
  )             