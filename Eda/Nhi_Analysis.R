# Load in Packages

# STEP 1: Does imputation with median/mode make sense?
# Package to check if NAs are MCAR
library(naniar)

# Perform a test for MCAR
mcar_test(data_train)

# p > 0.05 → Fail to reject the null hypothesis → Data is MCAR 
# p ≤ 0.05 → Reject the null hypothesis → Data is MAR or MNAR (follows a pattern)
# p-value here: 0.996 -> MCAR and median/mode imputation works
# To avoid data leakage: apply on training set only

# STEP 2: Outlier detection and what to do with them (categorical and numerical)?

# Function that detects "categorical" outliers based on a threshold.
# Parameters:
# df: data frame that should be analysed
# cat_threshold: the threshold, where a variable is considered to be an outlier

categorical_outliers <- function(df, cat_threshold = 0.01) {
  
  # Select all categorical columns
  cat_cols <- df[sapply(df, function(x) is.character(x))]
  
  # Compute the proportions of the values in categorical columns and compare with threshold
  # Frequencies < threshold are considered as outliers
  lapply(cat_cols, function(col) {
    freq <- prop.table(table(col))
    rare <- names(freq[freq < cat_threshold])
    list(rare_levels = rare, count = length(rare))
  })
}


# Function that detects "numeriical" outliers based on an IQR multiplier.
# Parameters:
# df: data frame that should be analysed
# iqr_multiplier: the factor used to identify an outlier

numeric_outliers <- function(df, iqr_multiplier = 1.5) {
  
  # Select all numeric columns
  num_cols <- df[sapply(df, is.numeric)]
  
  # Compute the quantiles and IQR and calculate the bounds
  # Values outside of the bound are considered as outliers
  lapply(num_cols, function(col) {
    Q1 <- quantile(col, 0.25, na.rm = TRUE)
    Q3 <- quantile(col, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    bounds <- c(lower = Q1 - iqr_multiplier * IQR, upper = Q3 + iqr_multiplier * IQR)
    # Find indices of values that lie outside the bounds
    out_idx <- which(col < bounds["lower"] | col > bounds["upper"])
    list(outlier_count = length(out_idx), bounds = bounds)
  })
}

# Apply the functions on the data:
categorical_outliers(df = data_train, cat_threshold = 0.01)

numeric_outliers(df = data_train, iqr_multiplier = 1.5)


# Little amount of outliers
# The outliers are still important and real. so leaving them in is better than removing them.

# STEP 3: Feature Engineering
# Analyse correlation between features using heatmaps

# TODO: heatmap

# Creating interactions terms based on domain knowledge:

# Attitude and Behavior:
# h1n1_concern × behavioral_avoidance
# h1n1_concern × behavioral_face_mask
# h1n1_knowledge × behavioral_antiviral_meds
# ->  People concerned or knowledgeable about H1N1 might be more likely to adopt 
# protective behaviors

# Medical Advice × Trust
# doctor_recc_h1n1 × opinion_h1n1_vacc_effective
# doctor_recc_seasonal × opinion_seas_vacc_effective
# If people trust their doctor and believe the vaccine is effective, 
# they are more likely to get vaccinated
# 
# Healthcare Role × Risk
# health_worker × opinion_h1n1_risk
# health_worker × opinion_seas_risk
# Healthcare workers may perceive risk differently and their behavior may differ
# 
# Socioeconomic Status × Insurance
# income_poverty × health_insurance
# education × health_insurance
# Poor or less-educated people without insurance may face more vaccine access issues
# 
# Children & Chronic Illness
# child_under_6_months × chronic_med_condition
# child_under_6_months × behavioral_touch_face
# Living with vulnerable people might motivate stronger preventive action.
# 
# Opinion Conflicts
# opinion_h1n1_vacc_effective × opinion_h1n1_sick_from_vacc
# People who think the vaccine is effective but fear side effects might be 
# conflicted affecting uptake.
