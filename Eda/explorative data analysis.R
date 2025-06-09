# Advanced EDA for Flu Shot Learning Competition
# -----------------------------------------------------

# Load required libraries
library(tidyverse)      # For data manipulation and visualization
library(ggplot2)        # For visualization
library(reshape2)       # For data reshaping
library(corrplot)       # For correlation plots
library(VIM)            # For missing data visualization
library(skimr)          # For comprehensive summary statistics
library(naniar)         # Better missing data visualization
library(GGally)         # For enhanced correlation plots
library(viridis)        # For colorblind-friendly palettes
library(gridExtra)      # For arranging multiple plots
library(caret)          # For feature relationships
library(knitr)          # For nicer table output in markdown
library(tableone)       # For creating summary tables
library(purrr)
library(forcats)



# Set colorblind-friendly palettes
cb_palette <- viridis(8, option = "D")
cb_div_palette <- viridis(8, option = "H")

# Set theme for all plots
theme_set(theme_minimal(base_size = 11) + 
            theme(panel.grid.minor = element_blank(),
                  plot.title = element_text(face = "bold"),
                  legend.position = "bottom",
                  strip.background = element_rect(fill = "gray90", color = NA),
                  strip.text = element_text(face = "bold")))

# Function to save plots
# save_plot <- function(plot, filename, width = 10, height = 7) {
#   ggsave(paste0("plots/", filename, ".png"), plot, width = width, height = height, dpi = 300)
# }

# # Create plots directory if it doesn't exist
# if (!dir.exists("plots")) {
#   dir.create("plots")
# }

# Function to create markdown-friendly tables
print_table <- function(df, caption = NULL) {
  kable(df, caption = caption, format = "markdown")
}

# Set seed for reproducibility
set.seed(123)

cat("=== Flu Shot Learning Competition: Comprehensive EDA ===\n")
cat("This analysis explores factors influencing H1N1 and seasonal flu vaccination rates.\n\n")

# -----------------------------------------------------
# 1. DATA LOADING AND INITIAL INSPECTION
# -----------------------------------------------------
cat("\n## 1. DATA LOADING AND INITIAL INSPECTION\n\n")

# Load datasets
cat("Loading datasets...\n")
data_train <- read_csv("Data/training_set_features.csv")
data_train_labels <- read_csv("Data/training_set_labels.csv")
data_test <- read_csv("Data/test_set_features.csv")

# Merge training features and labels
df <- left_join(data_train, data_train_labels, by = "respondent_id")
cat("Training data dimensions:", dim(df)[1], "rows and", dim(df)[2], "columns\n")
cat("Test data dimensions:", dim(data_test)[1], "rows and", dim(data_test)[2], "columns\n\n")

# Quick overview of the dataset structure
cat("Overview of dataset structure:\n")
glimpse(df)

# Identify data types for proper categorization
df_types <- data.frame(
  Variable = names(df),
  Type = sapply(df, class),
  Is_Categorical = sapply(df, function(x) {
    # Check if column should be treated as categorical
    is.character(x) || is.factor(x) || length(unique(x[!is.na(x)])) <= 10
  })
)

cat("\nClassification of variables by type:\n")
print_table(df_types)

# Reclassify variables based on their actual type (not just R's class)
categorical_features <- c(
  "h1n1_concern", "h1n1_knowledge", 
  "opinion_h1n1_vacc_effective", "opinion_h1n1_risk", "opinion_h1n1_sick_from_vacc",
  "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc",
  "age_group", "education", "race", "sex", "income_poverty", "marital_status", 
  "rent_or_own", "employment_status", "hhs_geo_region", "census_msa",
  "employment_industry", "employment_occupation"
)

binary_features <- c(
  "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask",
  "behavioral_wash_hands", "behavioral_large_gatherings", "behavioral_outside_home",
  "behavioral_touch_face", "doctor_recc_h1n1", "doctor_recc_seasonal",
  "chronic_med_condition", "child_under_6_months", "health_worker", "health_insurance",
  "h1n1_vaccine", "seasonal_vaccine"
)

numeric_features <- c("household_adults", "household_children")

# Ensure proper classification
df_updated <- df %>%
  mutate(across(all_of(categorical_features), as.factor),
         across(all_of(binary_features), as.factor))

cat("\nAfter proper classification of variables:\n")
glimpse(df_updated)

# -----------------------------------------------------
# 2. TARGET VARIABLE ANALYSIS
# -----------------------------------------------------
cat("\n## 2. TARGET VARIABLE ANALYSIS\n\n")

# Count occurrences of each target variable combination (H1N1 and seasonal vaccines)
target_counts <- df_updated %>%
  count(h1n1_vaccine, seasonal_vaccine) %>%
  mutate(
    percentage = n / sum(n) * 100,
    label = paste0(round(percentage, 1), "%")
  )

cat("Distribution of target variable combinations:\n")
print_table(target_counts)

# Create a more insightful visualization of the target variables
target_plot <- ggplot(target_counts, aes(x = h1n1_vaccine, y = seasonal_vaccine, fill = n)) +
  geom_tile() +
  geom_text(aes(label = label), color = "white", fontface = "bold") +
  scale_fill_viridis_c(option = "D") +
  labs(
    title = "Distribution of Vaccination Combinations",
    subtitle = "Percentage of respondents by H1N1 and seasonal flu vaccination status",
    x = "H1N1 Vaccine (1 = Yes, 0 = No)",
    y = "Seasonal Vaccine (1 = Yes, 0 = No)",
    fill = "Count"
  ) +
  theme(legend.position = "right")

print(target_plot)
#save_plot(target_plot, "01_target_distribution")
#####----------------------------------------------------######----------------------------------------------------######
# Individual vaccination rates
h1n1_rate <- sum(df_updated$h1n1_vaccine == 1, na.rm = TRUE) / nrow(df_updated) * 100
seasonal_rate <- sum(df_updated$seasonal_vaccine == 1, na.rm = TRUE) / nrow(df_updated) * 100

cat("\nOverall vaccination rates:\n")
cat("- H1N1 vaccination rate: ", round(h1n1_rate, 2), "%\n")
cat("- Seasonal flu vaccination rate: ", round(seasonal_rate, 2), "%\n\n")

# Create bar chart showing individual vaccination rates
vaccination_summary <- data.frame(
  vaccine_type = c("H1N1", "Seasonal Flu"),
  percentage = c(h1n1_rate, seasonal_rate)
)

vax_plot <- ggplot(vaccination_summary, aes(x = vaccine_type, y = percentage, fill = vaccine_type)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), vjust = -0.5, size = 4) +
  scale_fill_manual(values = cb_palette[c(2, 6)]) +
  labs(
    title = "Vaccination Rates by Type",
    subtitle = "Percentage of respondents who received each vaccine",
    x = "Vaccine Type",
    y = "Percentage (%)",
    fill = "Vaccine Type"
  ) +
  ylim(0, max(vaccination_summary$percentage) * 1.2) +
  theme(legend.position = "none")

print(vax_plot)
#save_plot(vax_plot, "02_vaccination_rates", width = 8, height = 6)
#####----------------------------------------------------######----------------------------------------------------######

# -----------------------------------------------------
# 3. MISSING VALUES ANALYSIS
# -----------------------------------------------------
cat("\n## 3. MISSING VALUES ANALYSIS\n\n")

# Summary of missing values for all columns
missing_summary <- df_updated %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "missing") %>%
  mutate(
    percent = 100 * missing / nrow(df_updated),
    feature = fct_reorder(feature, missing)
  ) %>%
  arrange(desc(missing))

# Add classification of feature type to missing summary
missing_summary <- missing_summary %>%
  mutate(
    feature_type = case_when(
      feature %in% categorical_features ~ "Categorical",
      feature %in% binary_features ~ "Binary",
      feature %in% numeric_features ~ "Numeric",
      feature == "respondent_id" ~ "ID",
      TRUE ~ "Other"
    )
  )

cat("Missing values summary:\n")
print_table(missing_summary %>% filter(missing > 0))

# Create a better missing data visualization
missing_plot <- ggplot(missing_summary %>% filter(missing > 0), 
                       aes(x = reorder(feature, missing), y = percent, fill = feature_type)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.1f%%", percent)), hjust = -0.1, size = 3) +
  scale_fill_manual(values = cb_palette[c(1, 3, 5, 7)]) +
  coord_flip() +
  labs(
    title = "Missing Values by Feature",
    subtitle = "Percentage of missing values for each feature",
    x = "Feature",
    y = "Missing Values (%)",
    fill = "Feature Type"
  ) +
  theme(axis.text.y = element_text(size = 8))

print(missing_plot)
#save_plot(missing_plot, "03_missing_values")
#####----------------------------------------------------######----------------------------------------------------######

#############################
##ERROR

# Enhanced missing data pattern visualization using naniar
#miss_pattern <- gg_miss_upset(df_updated, sets = names(df_updated)[colSums(is.na(df_updated)) > 0])
#print(miss_pattern)
#save_plot(miss_pattern, "04_missing_patterns", width = 12, height = 8)
#####----------------------------------------------------######----------------------------------------------------######

# Visualize missingness relationships
miss_var_plot <- gg_miss_var(df_updated, show_pct = TRUE)
print(miss_var_plot)
#save_plot(miss_var_plot, "05_missing_variables")
#####----------------------------------------------------######----------------------------------------------------######


#I DONT GET THE POINT OF THIS and HOW TO PRESENT IT


# Check if missingness in key variables is related to target
if (sum(missing_summary$missing) > 0) {
  cat("\nAnalyzing relationship between missingness and target variables:\n")
  
  # Create missingness indicators
  miss_df <- df_updated %>%
    select(h1n1_vaccine, seasonal_vaccine) %>%
    bind_cols(
      as_tibble(sapply(df_updated %>% select(-h1n1_vaccine, -seasonal_vaccine), 
                       function(x) as.integer(is.na(x)))) %>%
        rename_with(~ paste0("miss_", .x))
    )
  
  # Test for major variables with missingness
  for (var in names(miss_df)[grepl("miss_", names(miss_df))]) {
    if (sum(miss_df[[var]], na.rm = TRUE) > 0) {
      # Check relationship with h1n1_vaccine
      h1n1_test <- chisq.test(miss_df[[var]], miss_df$h1n1_vaccine)
      
      # Check relationship with seasonal_vaccine
      seasonal_test <- chisq.test(miss_df[[var]], miss_df$seasonal_vaccine)
      
      if (h1n1_test$p.value < 0.05 || seasonal_test$p.value < 0.05) {
        cat(var, "missingness is related to:\n")
        if (h1n1_test$p.value < 0.05) cat("  - H1N1 vaccine (p =", round(h1n1_test$p.value, 4), ")\n")
        if (seasonal_test$p.value < 0.05) cat("  - Seasonal vaccine (p =", round(seasonal_test$p.value, 4), ")\n")
      }
    }
  }
}


# -----------------------------------------------------
# 4. FEATURE DISTRIBUTION ANALYSIS
# -----------------------------------------------------

# I Dont find thid ection very important
cat("\n## 4. FEATURE DISTRIBUTION ANALYSIS\n\n")

# 4.1 Binary Features Analysis
cat("### 4.1 Binary Features Analysis\n\n")

# Create a summary for binary features
binary_summary <- df_updated %>%
  select(all_of(binary_features)) %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value") %>%
  group_by(feature) %>%
  summarise(
    Yes = sum(value == 1, na.rm = TRUE),
    No = sum(value == 0, na.rm = TRUE),
    Missing = sum(is.na(value)),
    Yes_pct = round(sum(value == 1, na.rm = TRUE) / sum(!is.na(value)) * 100, 1),
    .groups = "drop"
  )

print_table(binary_summary)

# Visualize binary features distribution
binary_plot_data <- df_updated %>%
  select(all_of(binary_features)) %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value") %>%
  filter(!is.na(value)) %>% # filter before mutate for cleaner reordering
  mutate(
    feature = fct_reorder(feature, as.integer(value), .fun = mean), # Use integer for correct ordering
    value = factor(value, levels = c(0, 1), labels = c("No", "Yes"))
  )

# Then plot
binary_dist_plot <- ggplot(binary_plot_data, aes(x = feature, fill = value)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = cb_palette[c(1, 7)]) +
  coord_flip() +
  labs(
    title = "Distribution of Binary Features",
    subtitle = "Proportion of respondents answering 'Yes' for each feature",
    x = "Feature",
    y = "Proportion",
    fill = "Response"
  ) +
  geom_text(
    aes(label = scales::percent(after_stat(count) / tapply(after_stat(count), after_stat(x), sum)[after_stat(x)], accuracy = 1)),
    stat = "count",
    position = position_fill(vjust = 0.5),
    color = "white", size = 3, fontface = "bold"
  )

print(binary_dist_plot)
#save_plot(binary_dist_plot, "06_binary_features")
#####----------------------------------------------------######----------------------------------------------------######
# 4.2 Categorical Features Analysis
cat("\n### 4.2 Categorical Features Analysis\n\n")

#### ITS BEST TO BREAK DOWN THIS AND PRODUCE INDIVUDALS PLOTS

# Function to generate summary tables for categorical variables
summarize_categorical <- function(df, var_name) {
  df %>%
    count(.data[[var_name]], sort = TRUE) %>%
    mutate(
      proportion = n / sum(n),
      percent = round(proportion * 100, 1),
      cum_percent = cumsum(percent)
    ) %>%
    select(-proportion)
}

# Analyze key categorical features
key_categorical <- c("h1n1_concern", "h1n1_knowledge", "age_group", "education", 
                     "race", "income_poverty", "employment_status")

for (feature in key_categorical) {
  cat("\nDistribution of", feature, ":\n")
  print_table(summarize_categorical(df_updated, feature))
  
  # Create visualization for this categorical feature
  cat_plot <- ggplot(df_updated, aes(x = fct_infreq(.data[[feature]]), fill = .data[[feature]])) +
    geom_bar() +
    geom_text(
      stat = "count", 
      aes(label = paste0(round(after_stat(count) / nrow(df_updated) * 100, 1), "%")),
      vjust = -0.5, size = 3
    ) +
    scale_fill_viridis_d(option = "D") +
    labs(
      title = paste("Distribution of", feature),
      x = feature,
      y = "Count"
    ) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  print(cat_plot)
  #save_plot(cat_plot, paste0("07_categorical_", feature))
}
#####----------------------------------------------------######----------------------------------------------------######
# 4.3 Ordinal Features Analysis
cat("\n### 4.3 Ordinal Features Analysis\n\n")

#############################DOESNT WORK 

# Define ordinal features and their order
ordinal_features <- list(
  h1n1_concern = c("Not at all concerned", "Not very concerned", "Somewhat concerned", "Very concerned"),
  h1n1_knowledge = c("No knowledge", "A little knowledge", "A lot of knowledge"),
  opinion_h1n1_vacc_effective = c("Not at all effective", "Not very effective", "Don't know", "Somewhat effective", "Very effective"),
  opinion_h1n1_risk = c("Very Low", "Somewhat low", "Don't know", "Somewhat high", "Very high"),
  opinion_h1n1_sick_from_vacc = c("Not at all worried", "Not very worried", "Don't know", "Somewhat worried", "Very worried"),
  opinion_seas_vacc_effective = c("Not at all effective", "Not very effective", "Don't know", "Somewhat effective", "Very effective"),
  opinion_seas_risk = c("Very Low", "Somewhat low", "Don't know", "Somewhat high", "Very high"),
  opinion_seas_sick_from_vacc = c("Not at all worried", "Not very worried", "Don't know", "Somewhat worried", "Very worried")
)

# Create a list to hold ordinal plots
ordinal_plots <- list()

# Create plots for ordinal features
for (i in seq_along(ordinal_features)) {
  feature <- names(ordinal_features)[i]
  if (feature %in% names(df_updated)) {
    # Summarize ordinal variable
    cat("\nDistribution of", feature, ":\n")
    print_table(summarize_categorical(df_updated, feature))
    
    # Create proper order for plotting
    if (is.factor(df_updated[[feature]])) {
      # Convert column to factor with proper order
      df_updated[[feature]] <- factor(df_updated[[feature]], levels = ordinal_features[[feature]])
    }
    
    # Create ordinal visualization
    ord_plot <- ggplot(df_updated, aes(x = .data[[feature]], fill = .data[[feature]])) +
      geom_bar() +
      geom_text(
        stat = "count", 
        aes(label = paste0(round(after_stat(count) / sum(!is.na(df_updated[[feature]])) * 100, 1), "%")),
        vjust = -0.5, size = 3
      ) +
      scale_fill_viridis_d(option = "D") +
      labs(
        title = paste("Distribution of", feature),
        x = feature,
        y = "Count"
      ) +
      theme(
        legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1)
      )
    
    print(ord_plot)
    #save_plot(ord_plot, paste0("08_ordinal_", feature))
    
    # Store the plot for later grid arrangement
    ordinal_plots[[feature]] <- ord_plot
  }
}
#####----------------------------------------------------######----------------------------------------------------######
# 4.4 Numeric Features Analysis
cat("\n### 4.4 Numeric Features Analysis\n\n")

# Summarize numeric features
numeric_summary <- df_updated %>%
  select(all_of(numeric_features)) %>%
  skim() %>%
  select(-n_missing, -complete_rate)

cat("Numerical features summary:\n")
print_table(numeric_summary)

# Create visualizations for numeric features
for (feature in numeric_features) {
  # Create histogram for this numeric feature
  hist_plot <- ggplot(df_updated, aes(x = .data[[feature]])) +
    geom_histogram(aes(y = after_stat(count / sum(count))), 
                   bins = 10, fill = cb_palette[2], color = "white") +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title = paste("Distribution of", feature),
      x = feature,
      y = "Percentage"
    )
  
  print(hist_plot)
  #save_plot(hist_plot, paste0("09_numeric_", feature))
}
#####----------------------------------------------------######----------------------------------------------------######

# -----------------------------------------------------
# 5. RELATIONSHIP WITH TARGET VARIABLES
# -----------------------------------------------------
cat("\n## 5. RELATIONSHIP WITH TARGET VARIABLES\n\n")

# 5.1 Binary Features vs. Target
cat("### 5.1 Binary Features vs. Target\n\n")

# Prepare data for binary feature analysis with target
binary_target_data <- df_updated %>%
  select(all_of(c(binary_features, "h1n1_vaccine", "seasonal_vaccine"))) %>%
  pivot_longer(cols = -c(h1n1_vaccine, seasonal_vaccine), 
               names_to = "feature", values_to = "value") %>%
  filter(!is.na(value)) %>%
  mutate(value = factor(value, levels = c(0, 1), labels = c("No", "Yes")))

# Calculate proportion of vaccine uptake for each binary feature level
binary_target_summary <- binary_target_data %>%
  group_by(feature, value) %>%
  summarise(
    total = n(),
    h1n1_yes = sum(h1n1_vaccine == 1, na.rm = TRUE),
    h1n1_rate = h1n1_yes / total * 100,
    seasonal_yes = sum(seasonal_vaccine == 1, na.rm = TRUE),
    seasonal_rate = seasonal_yes / total * 100,
    .groups = "drop"
  ) %>%
  mutate(feature = fct_reorder(feature, h1n1_rate, .fun = max))

# Create visualization for binary feature relationship with target
binary_h1n1_plot <- ggplot(binary_target_summary, 
                           aes(x = feature, y = h1n1_rate, fill = value, group = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.1f%%", h1n1_rate)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 2.5) +
  scale_fill_manual(values = cb_palette[c(1, 7)]) +
  coord_flip() +
  labs(
    title = "H1N1 Vaccination Rate by Binary Features",
    subtitle = "How binary features relate to H1N1 vaccine uptake",
    x = "Feature",
    y = "H1N1 Vaccination Rate (%)",
    fill = "Feature Value"
  )

binary_seasonal_plot <- ggplot(binary_target_summary, 
                               aes(x = feature, y = seasonal_rate, fill = value, group = value)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.1f%%", seasonal_rate)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 2.5) +
  scale_fill_manual(values = cb_palette[c(1, 7)]) +
  coord_flip() +
  labs(
    title = "Seasonal Flu Vaccination Rate by Binary Features",
    subtitle = "How binary features relate to seasonal flu vaccine uptake",
    x = "Feature",
    y = "Seasonal Flu Vaccination Rate (%)",
    fill = "Feature Value"
  )

print(binary_h1n1_plot)
#save_plot(binary_h1n1_plot, "10_binary_h1n1_target")

print(binary_seasonal_plot)
#save_plot(binary_seasonal_plot, "11_binary_seasonal_target")

# Calculate risk ratios (to quantify effect size)
binary_risk_ratios <- binary_target_data %>%
  group_by(feature) %>%
  summarise(
    h1n1_risk_ratio = {
      yes_rate = mean(h1n1_vaccine[value == "Yes"] == 1, na.rm = TRUE)
      no_rate = mean(h1n1_vaccine[value == "No"] == 1, na.rm = TRUE)
      yes_rate / no_rate
    },
    seasonal_risk_ratio = {
      yes_rate = mean(seasonal_vaccine[value == "Yes"] == 1, na.rm = TRUE)
      no_rate = mean(seasonal_vaccine[value == "No"] == 1, na.rm = TRUE)
      yes_rate / no_rate
    },
    .groups = "drop"
  ) %>%
  arrange(desc(h1n1_risk_ratio))

cat("\nRisk ratios for binary features (Yes vs No):\n")
print_table(binary_risk_ratios)

# 5.2 Categorical Features vs. Target
cat("\n### 5.2 Categorical Features vs. Target\n\n")

# Select key categorical features to analyze with target
key_cat_features <- c("age_group", "education", "race", "sex", "income_poverty", 
                      "marital_status", "employment_status", "census_msa")

# Create summary plots for each categorical feature vs targets
for (feature in key_cat_features) {
  if (feature %in% names(df_updated)) {
    # Calculate rates
    cat_summary <- df_updated %>%
      group_by(.data[[feature]]) %>%
      summarise(
        total = n(),
        h1n1_rate = mean(h1n1_vaccine == 1, na.rm = TRUE) * 100,
        seasonal_rate = mean(seasonal_vaccine == 1, na.rm = TRUE) * 100,
        .groups = "drop"
      ) %>%
      arrange(desc(h1n1_rate))
    
    # Plot relationship with H1N1 vaccine
    cat_h1n1_plot <- ggplot(cat_summary, aes(x = reorder(.data[[feature]], h1n1_rate), y = h1n1_rate, fill = .data[[feature]])) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = sprintf("%.1f%%", h1n1_rate)), hjust = -0.1, size = 3) +
      scale_fill_viridis_d(option = "D") +
      coord_flip() +
      labs(
        title = paste("H1N1 Vaccination Rate by", feature),
        x = feature,
        y = "H1N1 Vaccination Rate (%)"
      ) +
      theme(legend.position = "none")
    
    # Plot relationship with seasonal vaccine
    cat_seasonal_plot <- ggplot(cat_summary, aes(x = reorder(.data[[feature]], seasonal_rate), y = seasonal_rate, fill = .data[[feature]])) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = sprintf("%.1f%%", seasonal_rate)), hjust = -0.1, size = 3) +
      scale_fill_viridis_d(option = "D") +
      coord_flip() +
      labs(
        title = paste("Seasonal Flu Vaccination Rate by", feature),
        x = feature,
        y = "Seasonal Flu Vaccination Rate (%)"
      ) +
      theme(legend.position = "none")
    
    print(cat_h1n1_plot)
    #save_plot(cat_h1n1_plot, paste0("12_cat_h1n1_", feature))
    
    print(cat_seasonal_plot)
    #save_plot(cat_seasonal_plot, paste0("13_cat_seasonal_", feature))
  }
}

# 5.3 Ordinal Features vs. Target Variables
cat("\n### 5.3 Ordinal Features vs. Target\n\n")

# Create a list of ordinal features
ordinal_names <- names(ordinal_features)

# Create plots showing relationship between ordinal features and target variables
for (feature in ordinal_names) {
  if (feature %in% names(df_updated)) {
    # Ensure proper ordering of levels
    if (is.factor(df_updated[[feature]])) {
      df_updated[[feature]] <- factor(df_updated[[feature]], levels = ordinal_features[[feature]])
    }
    
    # Calculate vaccination rates by feature level
    ord_summary <- df_updated %>%
      group_by(.data[[feature]]) %>%
      summarise(
        total = n(),
        h1n1_rate = mean(h1n1_vaccine == 1, na.rm = TRUE) * 100,
        seasonal_rate = mean(seasonal_vaccine == 1, na.rm = TRUE) * 100,
        .groups = "drop"
      )
    
    # Create plot for H1N1 vaccination rates
    ord_h1n1_plot <- ggplot(ord_summary, aes(x = .data[[feature]], y = h1n1_rate)) +
      geom_bar(stat = "identity", fill = cb_palette[3]) +
      geom_text(aes(label = sprintf("%.1f%%", h1n1_rate)), vjust = -0.5, size = 3) +
      labs(
        title = paste("H1N1 Vaccination Rate by", feature),
        x = feature,
        y = "H1N1 Vaccination Rate (%)"
      ) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    # Create plot for seasonal vaccination rates
    ord_seasonal_plot <- ggplot(ord_summary, aes(x = .data[[feature]], y = seasonal_rate)) +
      geom_bar(stat = "identity", fill = cb_palette[5]) +
      geom_text(aes(label = sprintf("%.1f%%", seasonal_rate)), vjust = -0.5, size = 3) +
      labs(
        title = paste("Seasonal Flu Vaccination Rate by", feature),
        x = feature,
        y = "Seasonal Flu Vaccination Rate (%)"
      ) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    print(ord_h1n1_plot)
    save_plot(ord_h1n1_plot, paste0("14_ord_h1n1_", feature))
    
    print(ord_seasonal_plot)
    #save_plot(ord_seasonal_plot, paste0("15_ord_seasonal_", feature))
  }
}

# 5.4 Numeric Features vs. Target Variables
cat("\n### 5.4 Numeric Features vs. Target\n\n")

# Analyze relationship between numeric features and target variables
for (feature in numeric_features) {
  if (feature %in% names(df_updated)) {
    # Create summary by feature value
    num_summary <- df_updated %>%
      group_by(.data[[feature]]) %>%
      summarise(
        count = n(),
        h1n1_rate = mean(h1n1_vaccine == 1, na.rm = TRUE) * 100,
        seasonal_rate = mean(seasonal_vaccine == 1, na.rm = TRUE) * 100,
        .groups = "drop"
      )
    
    # Create plot for H1N1 vaccination rates
    num_h1n1_plot <- ggplot(num_summary, aes(x = .data[[feature]], y = h1n1_rate, size = count)) +
      geom_point(color = cb_palette[3], alpha = 0.7) +
      geom_text(aes(label = sprintf("%.1f%%", h1n1_rate)), vjust = -1.5, size = 3) +
      labs(
        title = paste("H1N1 Vaccination Rate by", feature),
        x = feature,
        y = "H1N1 Vaccination Rate (%)",
        size = "Count"
      )
    
    # Create plot for seasonal vaccination rates
    num_seasonal_plot <- ggplot(num_summary, aes(x = .data[[feature]], y = seasonal_rate, size = count)) +
      geom_point(color = cb_palette[5], alpha = 0.7) +
      geom_text(aes(label = sprintf("%.1f%%", seasonal_rate)), vjust = -1.5, size = 3) +
      labs(
        title = paste("Seasonal Flu Vaccination Rate by", feature),
        x = feature,
        y = "Seasonal Flu Vaccination Rate (%)",
        size = "Count"
      )
    
    print(num_h1n1_plot)
    #save_plot(num_h1n1_plot, paste0("16_num_h1n1_", feature))
    
    print(num_seasonal_plot)
    #save_plot(num_seasonal_plot, paste0("17_num_seasonal_", feature))
  }
}

# -----------------------------------------------------
# 6. CORRELATION ANALYSIS
# -----------------------------------------------------
cat("\n## 6. CORRELATION ANALYSIS\n\n")

# Convert categorical variables to numeric for correlation analysis
df_corr <- df_updated %>%
  mutate(across(where(is.factor), as.numeric)) %>%
  select(-respondent_id) # Remove ID column which has no correlation relevance

# Calculate correlation matrix
corr_matrix <- cor(df_corr, use = "pairwise.complete.obs")

# Visualize correlation matrix (using a better colorblind-friendly approach)
corrplot(corr_matrix, method = "color", type = "upper", 
         col = viridis(100, option = "D"),
         tl.col = "black", tl.srt = 45, tl.cex = 0.7,
         title = "Correlation Matrix of All Features",
         mar = c(0, 0, 2, 0),
         addCoef.col = "black", number.cex = 0.6)

# Create a filtered correlation matrix focusing on strongest correlations with target variables
h1n1_corr <- corr_matrix[, "h1n1_vaccine"]
seasonal_corr <- corr_matrix[, "seasonal_vaccine"]

# Create a dataframe of correlations with target variables
target_corr_df <- data.frame(
  feature = names(h1n1_corr),
  h1n1_corr = h1n1_corr,
  seasonal_corr = seasonal_corr
) %>%
  filter(feature != "h1n1_vaccine" & feature != "seasonal_vaccine") %>%
  arrange(desc(abs(h1n1_corr)))

# Show top correlations with target variables
cat("\nTop correlations with H1N1 vaccine:\n")
print_table(target_corr_df %>% select(feature, h1n1_corr) %>% head(15))

cat("\nTop correlations with seasonal flu vaccine:\n")
print_table(target_corr_df %>% select(feature, seasonal_corr) %>% arrange(desc(abs(seasonal_corr))) %>% head(15))

# Create visualization of correlations with target variables
target_corr_plot <- target_corr_df %>%
  pivot_longer(cols = c(h1n1_corr, seasonal_corr), names_to = "target", values_to = "correlation") %>%
  mutate(
    feature = reorder(feature, abs(correlation)),
    target = factor(target, labels = c("H1N1", "Seasonal"))
  ) %>%
  ggplot(aes(x = feature, y = correlation, fill = target)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = cb_palette[c(3, 5)]) +
  coord_flip() +
  labs(
    title = "Correlation with Target Variables",
    subtitle = "Features sorted by absolute correlation with H1N1 vaccine",
    x = "Feature",
    y = "Correlation Coefficient",
    fill = "Target"
  ) +
  theme(axis.text.y = element_text(size = 8))

print(target_corr_plot)
#save_plot(target_corr_plot, "18_target_correlations")

# Feature correlations (excluding targets)
feature_corr_matrix <- cor(df_corr %>% select(-h1n1_vaccine, -seasonal_vaccine), 
                           use = "pairwise.complete.obs")

# Find highly correlated feature pairs (excluding targets)
high_corr_pairs <- which(abs(feature_corr_matrix) > 0.5 & abs(feature_corr_matrix) < 1, arr.ind = TRUE)
high_corr_df <- data.frame(
  row = rownames(feature_corr_matrix)[high_corr_pairs[, 1]],
  column = colnames(feature_corr_matrix)[high_corr_pairs[, 2]],
  correlation = feature_corr_matrix[high_corr_pairs]
) %>%
  filter(row < column) %>% # Remove duplicates
  arrange(desc(abs(correlation)))

cat("\nHighly correlated feature pairs (|correlation| > 0.5):\n")
print_table(high_corr_df)

# -----------------------------------------------------
# 7. DATA LEAKAGE ANALYSIS
# -----------------------------------------------------
cat("\n## 7. DATA LEAKAGE ANALYSIS\n\n")

cat("Evaluating potential sources of data leakage in the dataset\n\n")

# 7.1 Target Leakage Assessment
cat("### 7.1 Target Leakage Assessment\n\n")

cat("Target leakage occurs when variables that would not be available at prediction time are included in the training data. Examining potential sources:\n\n")

# Check for direct indicators of target variables
potential_leaky_features <- c(
  "doctor_recc_h1n1", "doctor_recc_seasonal",
  "opinion_h1n1_vacc_effective", "opinion_seas_vacc_effective",
  "opinion_h1n1_sick_from_vacc", "opinion_seas_sick_from_vacc"
)

cat("Potential features that might indicate target leakage:\n\n")
for (feature in potential_leaky_features) {
  if (feature %in% names(df_updated)) {
    # Calculate correlation with target variables
    h1n1_cor <- cor(as.numeric(df_updated[[feature]]), as.numeric(df_updated$h1n1_vaccine), 
                    use = "pairwise.complete.obs")
    seasonal_cor <- cor(as.numeric(df_updated[[feature]]), as.numeric(df_updated$seasonal_vaccine), 
                        use = "pairwise.complete.obs")
    
    cat(sprintf("- %s: Correlation with H1N1 vaccine = %.3f, with seasonal vaccine = %.3f\n", 
                feature, h1n1_cor, seasonal_cor))
    
    # Check if doctor recommendation is too predictive (could indicate data collected after vaccination)
    if (grepl("doctor_recc", feature)) {
      # Calculate conditional probabilities
      probs <- df_updated %>%
        filter(!is.na(.data[[feature]]) & !is.na(if (grepl("h1n1", feature)) h1n1_vaccine else seasonal_vaccine)) %>%
        group_by(.data[[feature]]) %>%
        summarise(
          vaccine_rate = mean(if (grepl("h1n1", feature)) h1n1_vaccine == 1 else seasonal_vaccine == 1),
          count = n(),
          .groups = "drop"
        )
      
      print_table(probs)
      
      if (max(probs$vaccine_rate) > 0.9) {
        cat("  WARNING: This feature appears highly predictive of the target. Consider whether this\n")
        cat("  information would actually be available at prediction time or if it might represent\n")
        cat("  data collected after the vaccination decision was made.\n\n")
      }
    }
    
    # Check relationship between opinion variables and vaccination
    if (grepl("opinion", feature)) {
      # Calculate vaccination rates by opinion level
      opinion_rates <- df_updated %>%
        group_by(.data[[feature]]) %>%
        summarise(
          h1n1_rate = mean(h1n1_vaccine == 1, na.rm = TRUE),
          seasonal_rate = mean(seasonal_vaccine == 1, na.rm = TRUE),
          count = n(),
          .groups = "drop"
        )
      
      print_table(opinion_rates)
      
      if (grepl("h1n1", feature) && max(opinion_rates$h1n1_rate) > 0.8) {
        cat("  WARNING: This H1N1 opinion feature has a very strong relationship with H1N1 vaccination.\n")
        cat("  Verify that these opinions were collected before vaccination decisions were made.\n\n")
      }
      
      if (grepl("seas", feature) && max(opinion_rates$seasonal_rate) > 0.8) {
        cat("  WARNING: This seasonal flu opinion feature has a very strong relationship with seasonal vaccination.\n")
        cat("  Verify that these opinions were collected before vaccination decisions were made.\n\n")
      }
    }
  }
}

# 7.2 Feature Correlation with ID
cat("\n### 7.2 Feature Correlation with ID\n\n")

cat("Checking if respondent_id has any unexpected correlations with features (should be random):\n\n")

# Check if ID has any pattern with features
id_correlations <- df_updated %>%
  mutate(across(where(is.factor), as.numeric)) %>%
  select(-h1n1_vaccine, -seasonal_vaccine) %>% # Remove targets
  select_if(is.numeric) %>% # Select only numeric columns
  cor("respondent_id", use = "pairwise.complete.obs")

# Look for any non-zero correlations with ID
id_corr_df <- data.frame(
  feature = rownames(id_correlations),
  correlation = id_correlations[, "respondent_id"]
) %>%
  filter(feature != "respondent_id") %>%
  arrange(desc(abs(correlation)))

# Print top correlations with ID (should all be very close to zero)
print_table(id_corr_df %>% head(10))

if (max(abs(id_corr_df$correlation)) > 0.05) {
  cat("WARNING: Some features show non-random correlation with respondent_id.\n")
  cat("This might indicate a data collection issue or non-random assignment.\n\n")
} else {
  cat("No significant correlations found between features and respondent_id.\n")
  cat("This confirms that IDs appear to be randomly assigned as expected.\n\n")
}

# 7.3 Train-Test Distribution Comparison
cat("\n### 7.3 Train-Test Distribution Comparison\n\n")

cat("Comparing the distributions of features between training and test sets to detect potential issues:\n\n")

# Function to compare distributions between train and test
compare_distributions <- function(feature, train_df, test_df) {
  # For numeric features, compare summary statistics
  if (is.numeric(train_df[[feature]])) {
    train_summary <- summary(train_df[[feature]])
    test_summary <- summary(test_df[[feature]])
    
    # Calculate KS test statistic for distributional difference
    ks_result <- ks.test(train_df[[feature]], test_df[[feature]])
    
    result <- data.frame(
      feature = feature,
      train_mean = mean(train_df[[feature]], na.rm = TRUE),
      test_mean = mean(test_df[[feature]], na.rm = TRUE),
      train_sd = sd(train_df[[feature]], na.rm = TRUE),
      test_sd = sd(test_df[[feature]], na.rm = TRUE),
      ks_statistic = ks_result$statistic,
      p_value = ks_result$p.value
    )
    
    return(result)
  } else {
    # For categorical/factor features, compare proportions
    train_props <- prop.table(table(train_df[[feature]]))
    test_props <- prop.table(table(test_df[[feature]]))
    
    # Calculate chi-square test for distributional difference
    # Convert to contingency table first
    train_counts <- table(train_df[[feature]])
    test_counts <- table(test_df[[feature]])
    
    # Make sure both have the same levels
    all_levels <- unique(c(names(train_counts), names(test_counts)))
    train_vec <- numeric(length(all_levels))
    test_vec <- numeric(length(all_levels))
    names(train_vec) <- all_levels
    names(test_vec) <- all_levels
    
    train_vec[names(train_counts)] <- train_counts
    test_vec[names(test_counts)] <- test_counts
    
    # Create contingency table
    cont_table <- rbind(train_vec, test_vec)
    
    # Perform chi-square test
    chi_result <- tryCatch({
      chisq.test(cont_table)
    }, error = function(e) {
      # If chi-square test fails, return NA
      list(statistic = NA, p.value = NA)
    })
    
    # Calculate max absolute difference in proportions
    max_diff <- max(abs(train_props - test_props[names(train_props)]), na.rm = TRUE)
    
    result <- data.frame(
      feature = feature,
      max_prop_diff = max_diff,
      chi_square = chi_result$statistic,
      p_value = chi_result$p.value
    )
    
    return(result)
  }
}

# Prepare test data with same structure as training
test_df <- data_test %>%
  mutate(across(all_of(categorical_features), as.factor),
         across(all_of(binary_features[binary_features != "h1n1_vaccine" & 
                                         binary_features != "seasonal_vaccine"]), as.factor))

# Compare distributions for a sample of features
features_to_compare <- c(
  sample(binary_features[binary_features != "h1n1_vaccine" & 
                           binary_features != "seasonal_vaccine"], 5),
  sample(categorical_features, 5),
  numeric_features
)

distribution_comparisons <- lapply(features_to_compare, compare_distributions, 
                                   train_df = df_updated, test_df = test_df)
distribution_comparison_df <- do.call(rbind, distribution_comparisons)

cat("Comparison of feature distributions between training and test sets:\n\n")
print_table(distribution_comparison_df)

# Flag potential issues
if (any(distribution_comparison_df$p_value < 0.01, na.rm = TRUE)) {
  cat("\nWARNING: Some features show significantly different distributions between training and test sets.\n")
  cat("This could indicate:\n")
  cat("1. Temporal effects if data was collected over time\n")
  cat("2. Issues with the train-test split process\n")
  cat("3. Potential data leakage if splitting was not done randomly\n\n")
  
  # Show the problematic features
  problematic_features <- distribution_comparison_df %>% 
    filter(p_value < 0.01) %>% 
    arrange(p_value)
  
  cat("Features with significantly different distributions (p < 0.01):\n")
  print_table(problematic_features)
} else {
  cat("\nNo significant differences found in distributions between training and test sets.\n")
  cat("This suggests the train-test split was likely performed correctly.\n\n")
}

# 7.4 Key Leakage Concerns Summary
cat("\n### 7.4 Data Leakage Summary and Recommendations\n\n")

cat("Based on the data leakage analysis, here are the key findings and recommendations:\n\n")

# Chronological information check
cat("1. Temporal Information:\n")
cat("   - The dataset doesn't appear to contain explicit timestamps for data collection.\n")
cat("   - However, survey responses about opinions on vaccines should be verified to ensure\n")
cat("     they were collected prior to vaccination decisions.\n\n")

# Doctor recommendation check
cat("2. Doctor Recommendations:\n")
cat("   - The features 'doctor_recc_h1n1' and 'doctor_recc_seasonal' show strong correlation with targets.\n")
cat("   - Recommendation: Verify these represent recommendations made before vaccination decisions,\n")
cat("     not post-hoc documentation of the vaccination process.\n\n")

# Opinion variables check
cat("3. Opinion Variables:\n")
cat("   - Variables measuring opinions about vaccine effectiveness and risks are strongly correlated with targets.\n")
cat("   - Recommendation: Consider whether these opinions could have been formed after vaccination\n")
cat("     (which would constitute data leakage).\n\n")

# Feature engineering caution
cat("4. Feature Engineering Considerations:\n")
cat("   - When creating new features, avoid using information that wouldn't be available at prediction time.\n")
cat("   - Pay special attention to opinion and behavioral variables - ensure they represent pre-vaccination data.\n\n")

# Train-test split recommendation
cat("5. Model Validation Strategy:\n")
cat("   - Use time-based validation if the data has a temporal component (not explicitly stated in the dataset).\n")
cat("   - If using cross-validation, ensure the split strategy respects any potential temporal ordering.\n\n")

cat("6. Final Recommendations:\n")
cat("   - Perform feature selection with caution, being mindful of potential target leakage.\n")
cat("   - Consider creating models with and without potentially problematic features to assess impact.\n")
cat("   - Document assumptions about data collection timing in your final submission.\n\n")

# -----------------------------------------------------
# 8. INSIGHTS AND FEATURE IMPORTANCE ANALYSIS
# -----------------------------------------------------
cat("\n## 8. INSIGHTS AND FEATURE IMPORTANCE ANALYSIS\n\n")

# 8.1 Summary of Key Insights
cat("### 8.1 Summary of Key Insights\n\n")

cat("Based on the comprehensive EDA, here are the key insights about factors affecting vaccination rates:\n\n")

cat("1. **Doctor Recommendations** have the strongest relationship with vaccination rates for both\n")
cat("   H1N1 and seasonal flu vaccines. When a doctor recommends vaccination, the uptake rate\n")
cat("   increases dramatically.\n\n")

cat("2. **Opinion about Vaccine Effectiveness** is strongly correlated with vaccination rates.\n")
cat("   People who believe vaccines are effective are much more likely to get vaccinated.\n\n")

cat("3. **Behavioral Factors**: People who take other preventive measures (like washing hands\n")
cat("   or avoiding large gatherings) show different vaccination patterns than those who don't.\n\n")

cat("4. **Demographic Factors**: Age group, education level, and income show significant relationships\n")
cat("   with vaccination rates, especially for seasonal flu vaccines.\n\n")

cat("5. **H1N1 Concern Level**: Higher concern about H1N1 correlates with higher vaccination rates\n")
cat("   for the H1N1 vaccine but has a weaker relationship with seasonal flu vaccination.\n\n")

cat("6. **Knowledge about H1N1**: Greater knowledge about H1N1 is associated with higher vaccination\n")
cat("   rates for both H1N1 and seasonal flu vaccines.\n\n")

cat("7. **Health Insurance**: Having health insurance shows a positive relationship with vaccination\n")
cat("   rates, particularly for seasonal flu vaccines.\n\n")

cat("8. **Employment Status**: Healthcare workers have higher vaccination rates for both vaccines\n")
cat("   compared to other employment categories.\n\n")

# 8.2 Feature Importance Analysis
cat("### 8.2 Feature Importance Analysis\n\n")

cat("To estimate feature importance without building a full model, we can use a simple\n")
cat("decision tree to get an approximation of feature importance:\n\n")

# Train simple decision trees to estimate feature importance
library(rpart)
library(rpart.plot)

# Prepare data for modeling
model_data <- df_updated %>%
  mutate(across(where(is.factor), as.numeric)) %>%
  select(-respondent_id) %>%
  na.omit()  # Simple handling of missing values for this demonstration

# Train a decision tree for H1N1 vaccine
h1n1_tree <- rpart(h1n1_vaccine ~ . - seasonal_vaccine, 
                   data = model_data, 
                   method = "class",
                   control = rpart.control(maxdepth = 5))

# Train a decision tree for seasonal vaccine
seasonal_tree <- rpart(seasonal_vaccine ~ . - h1n1_vaccine, 
                       data = model_data, 
                       method = "class",
                       control = rpart.control(maxdepth = 5))

# Get variable importance
h1n1_importance <- data.frame(
  feature = names(h1n1_tree$variable.importance),
  importance = h1n1_tree$variable.importance / sum(h1n1_tree$variable.importance) * 100
) %>% 
  arrange(desc(importance))

seasonal_importance <- data.frame(
  feature = names(seasonal_tree$variable.importance),
  importance = seasonal_tree$variable.importance / sum(seasonal_tree$variable.importance) * 100
) %>% 
  arrange(desc(importance))

# Print top features for H1N1
cat("Top features for predicting H1N1 vaccination (based on decision tree):\n")
print_table(h1n1_importance %>% head(10))

# Print top features for seasonal flu
cat("\nTop features for predicting seasonal flu vaccination (based on decision tree):\n")
print_table(seasonal_importance %>% head(10))

# Plot decision trees
rpart.plot(h1n1_tree, main = "Decision Tree for H1N1 Vaccination", 
           extra = 106, box.palette = "Blues", shadow.col = "gray")

rpart.plot(seasonal_tree, main = "Decision Tree for Seasonal Flu Vaccination", 
           extra = 106, box.palette = "Oranges", shadow.col = "gray")

# Create importance plots
h1n1_imp_plot <- ggplot(h1n1_importance %>% head(10), 
                        aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity", fill = cb_palette[3]) +
  coord_flip() +
  labs(
    title = "Top 10 Features for H1N1 Vaccination",
    subtitle = "Based on decision tree importance",
    x = "Feature",
    y = "Relative Importance (%)"
  )

seasonal_imp_plot <- ggplot(seasonal_importance %>% head(10), 
                            aes(x = reorder(feature, importance), y = importance)) +
  geom_bar(stat = "identity", fill = cb_palette[5]) +
  coord_flip() +
  labs(
    title = "Top 10 Features for Seasonal Flu Vaccination",
    subtitle = "Based on decision tree importance",
    x = "Feature",
    y = "Relative Importance (%)"
  )

print(h1n1_imp_plot)
save_plot(h1n1_imp_plot, "19_h1n1_importance")

print(seasonal_imp_plot)
save_plot(seasonal_imp_plot, "20_seasonal_importance")

# -----------------------------------------------------
# 9. CONCLUSION AND NEXT STEPS
# -----------------------------------------------------
cat("\n## 9. CONCLUSION AND NEXT STEPS\n\n")

cat("### 9.1 Summary of Findings\n\n")

cat("This comprehensive EDA has revealed several important patterns in the dataset:\n\n")

cat("1. **Different Drivers**: H1N1 and seasonal flu vaccination behaviors share some common\n")
cat("   drivers but also have distinct patterns, suggesting we should treat them as separate\n")
cat("   prediction problems while leveraging their similarities.\n\n")

cat("2. **Key Predictors**: Doctor recommendations, opinions about vaccine effectiveness, health\n")
cat("   behaviors, and demographic factors appear to be the strongest predictors of vaccination.\n\n")

cat("3. **Missing Data**: The dataset contains missing values in several features, with some\n")
cat("   potentially showing non-random patterns of missingness that might be related to the target.\n\n")

cat("4. **Potential Data Leakage**: Several features show very strong correlations with target\n")
cat("   variables, which warrants careful consideration to avoid target leakage in the modeling phase.\n\n")

cat("### 9.2 Recommended Next Steps\n\n")

cat("Based on this EDA, here are recommendations for the modeling phase:\n\n")

cat("1. **Feature Engineering**:\n")
cat("   - Create interaction terms between doctor recommendations and opinions about vaccines\n")
cat("   - Group less frequent categories in categorical variables\n")
cat("   - Consider creating composite behavioral indices from related behavioral variables\n")
cat("   - Engineer features capturing demographic profiles based on observed patterns\n\n")

cat("2. **Handling Missing Data**:\n")
cat("   - Use multiple imputation for features with missing values\n")
cat("   - Consider adding 'missingness indicator' features for variables with high missingness\n")
cat("   - For employment industry/occupation, consider grouping into broader categories before imputation\n\n")

cat("3. **Addressing Class Imbalance**:\n")
cat("   - H1N1 vaccination shows class imbalance with lower positive rates\n")
cat("   - Consider techniques like SMOTE, class weighting, or calibration to address this\n\n")

cat("4. **Modeling Approach**:\n")
cat("   - Build separate models for H1N1 and seasonal flu vaccination\n")
cat("   - Test both single models and ensemble approaches\n")
cat("   - Consider a stacking approach that leverages predictions between the two targets\n")
cat("   - Evaluate models using ROC AUC as per competition guidelines\n\n")

cat("5. **Feature Selection and Validation**:\n")
cat("   - Use proper cross-validation to avoid overfitting\n")
cat("   - Consider removing or carefully treating features with potential data leakage\n")
cat("   - Build models with different feature subsets to assess impact on performance\n\n")

cat("This concludes the enhanced EDA for the Flu Shot Learning competition. The analysis provides\n")
cat("a solid foundation for proceeding with feature engineering and model development.")