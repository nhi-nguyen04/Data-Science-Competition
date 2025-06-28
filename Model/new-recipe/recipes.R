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

#the types collum migth show a problem
h1n1_recipe%>%
  summary()
tidy(h1n1_recipe, number = 4)


seas_recipe <- recipe(seasonal_vaccine ~ ., data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

tidy(seas_recipe, number = 4)



# ──────────── Advanced  Recipe ────────────────────

glimpse(train_df)
# -----------------------------------------------
# 6. R E C I P E  –– consistent imputation + dummies
# -----------------------------------------------
h1n1_recipe <- recipe(h1n1_vaccine ~  doctor_recc_h1n1+
                        opinion_h1n1_vacc_effective+
                        opinion_h1n1_risk+
                        employment_industry+
                        doctor_recc_seasonal+
                        health_worker+
                        age_group+
                        opinion_seas_risk+
                        education, data = train_data_h1n1) %>%
  update_role(respondent_id, new_role = "ID") %>%
  # Remove the other target (seasonal) if it’s present
  #creates a specification of a recipe step that will remove selected variables.
  step_rm(seasonal_vaccine) %>%
  # 1. Handle missing values FIRST
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # 2. Create dummy variables
  step_dummy(all_nominal_predictors()) %>%
  
  # 3. Add just a few key interactions
  # — Clinical × Attitude — 
  step_interact(terms = ~ doctor_recc_h1n1:opinion_h1n1_risk) %>%
  step_interact(terms = ~ doctor_recc_h1n1:opinion_h1n1_vacc_effective) %>%
  
  # — Clinical × Access —
  step_interact(terms = ~ doctor_recc_h1n1:health_insurance) %>%
  
  # — Attitude × Concern —
  step_interact(terms = ~ opinion_h1n1_risk:h1n1_concern) %>%
  step_interact(terms = ~ opinion_h1n1_vacc_effective:h1n1_concern) %>%
  
  # — Cross‐vaccine Attitudes —
  step_interact(terms = ~ opinion_seas_risk:opinion_h1n1_risk) %>%
  step_interact(terms = ~ doctor_recc_seasonal:opinion_h1n1_vacc_effective) %>%
  
  # — Occupational/Knowledge Synergy —
  step_interact(terms = ~ health_worker:h1n1_knowledge) %>%
  
  # — Risk vs. Side‐Effect Worry —
  step_interact(terms = ~ opinion_h1n1_risk:opinion_h1n1_sick_from_vacc) %>%
  
  # 4. Remove zero variance AFTER interactions
  step_zv(all_predictors()) %>%
  
  # 5. Normalize last
  step_normalize(all_numeric_predictors())



seas_recipe <- recipe(seasonal_vaccine ~ opinion_seas_risk+
                        age_group+
                        doctor_recc_seasonal+
                        opinion_h1n1_vacc_effective+
                        opinion_seas_sick_from_vacc+
                        employment_industry+
                        health_worker+
                        education, data = train_data_seas) %>%
  update_role(respondent_id, new_role = "ID") %>%
  step_rm(h1n1_vaccine) %>%
  # 1. Handle missing values FIRST
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  
  # 2. Create dummy variables
  step_dummy(all_nominal_predictors()) %>%
  
  # 3. Add just a few key interactions
  step_interact(terms = ~ opinion_seas_vacc_effective:opinion_seas_risk) %>%
  step_interact(terms = ~ opinion_seas_vacc_effective:doctor_recc_seasonal) %>%
  #step_interact(terms = ~ opinion_seas_vacc_effective:age_group) %>%
  step_interact(terms = ~ opinion_seas_risk:doctor_recc_seasonal) %>%
  #step_interact(terms = ~ opinion_seas_risk:age_group) %>%
  #step_interact(terms = ~ doctor_recc_seasonal:age_group) %>%
  
  # 4. Remove zero variance AFTER interactions
  step_zv(all_predictors()) %>%
  
  # 5. Normalize last
  step_normalize(all_numeric_predictors())




# ──────────── Possible Interactions ────────────────────


# — one interaction per step! —
# step_interact(terms = ~ doctor_recc_h1n1: h1n1_concern) %>%
#   step_interact(terms = ~ doctor_recc_h1n1: health_insurance) %>%
#   step_interact(terms = ~ chronic_med_condition: opinion_h1n1_risk) %>%
#   step_interact(terms = ~ age_group: child_under_6_months) %>%
#   step_interact(terms = ~ education: h1n1_knowledge) %>%
#   step_interact(terms = ~ income_poverty: behavioral_face_mask) %>%
#   step_interact(terms = ~ race: opinion_h1n1_vacc_effective) %>%
#   step_interact(terms = ~ health_worker: behavioral_wash_hands) %>%
#   step_interact(terms = ~ employment_status: behavioral_outside_home) %>%
#   step_interact(terms = ~ h1n1_knowledge: opinion_h1n1_sick_from_vacc) %>%






# ──────────── Removal of these Interactions for h1n1 ────────────────────


# Remove these “near‐zero” importance features:
c(
  "household_adults",      # tiny bar
  "household_children",
  grep("^employment_", names(vi_h1n1), value = TRUE),  # both industry & occupation codes
  "age_group_55-64.Years", # that specific bin is very low
  "race_White",            # single race dummy with no signal
  "chronic_med_condition",
  grep("age_group_", names(train_df), value = TRUE)[-1],  # all other age bins
  "education_College.Graduate",
  "child_under_6_months",
  "income_poverty_<=.75.000", # (and all other income bins)
  "hhs_geo_region",         # region codes
  grep("^marital_status", names(train_df), value = TRUE),
  grep("^rent_or_own", names(train_df), value = TRUE),
  grep("^behavioral_", names(train_df), value = TRUE),   # all avoidance/hand‐washing/mask‐wearing flags
  grep("^census_msa", names(train_df), value = TRUE)
)





# ──────────── Removal of these Interactions for seasonal ────────────────────



to_remove_season <- c(
  # Very low‐importance demographics & household
  "household_children",    # tiny bar
  "household_adults",
  "age_group_55 - 64 Years",
  "race White",
  "sex Male",
  "education College Graduate",
  "education Some College",
  "marital_status Not Married",
  "rent_or_own Rent",
  
  # Employment & income
  "employment_status Not in Labor Force",
  "employment_industry",   # code strings, no signal
  "employment_occupation", # code strings, no signal
  "income_poverty Below Poverty",
  "income_poverty <= $75,000, Above Poverty",
  
  # Behavioral flags
  "behavioral_wash_hands",
  "behavioral_outside_home",
  "behavioral_avoidance",
  "behavioral_large_gatherings",
  
  # Geographics
  "census_msa MSA, Not Principle City",
  "census_msa Non-MSA",
  "hhs_geo_region",        # region codes
  
  # Medical history
  "chronic_med_condition",
  "child_under_6_months"
)