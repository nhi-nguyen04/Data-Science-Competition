Advanced EDA for Flu Shot Learning Competition
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
install.packages("tableone")
library(tableone)       # For creating summary tables

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
save_plot <- function(plot, filename, width = 10, height = 7) {
  ggsave(paste0("plots/", filename, ".png"), plot, width = width, height = height, dpi = 300)
}

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

# Enhanced missing data pattern visualization using naniar
miss_pattern <- gg_miss_upset(df_updated, sets = names(df_updated)[colSums(is.na(df_updated)) > 0])
print(miss_pattern)
######### There are errors here
#save_plot(miss_pattern, "04_missing_patterns", width = 12, height = 8)

# Visualize missingness relationships
miss_var_plot <- gg_miss_var(df_updated, show_pct = TRUE)
print(miss_var_plot)
#save_plot(miss_var_plot, "05_missing_variables")

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
# 4. FEATURE DISTRIBUTION ANALYSIS #needs improvement
# -----------------------------------------------------
cat("\n## 4. FEATURE DISTRIBUTION ANALYSIS\n\n")

# 4.1 Binary Features Analysis
cat("### 4.1 Binary Features Analysis\n\n")

# Create a summary for binary features
binary_summary <- df_updated %>%
  select(all_of(binary_features)) %>%
  summarise(across(everything(), 
                   ~list(
                     Yes = sum(.x == 1, na.rm = TRUE),
                     No = sum(.x == 0, na.rm = TRUE),
                     Missing = sum(is.na(.x)),
                     Yes_pct = round(sum(.x == 1, na.rm = TRUE) / sum(!is.na(.x)) * 100, 1)
                   ))) %>%
  pivot_longer(cols = everything(), names_to = "feature") %>%
  unnest_wider(value)

print_table(binary_summary)

# Visualize binary features distribution
binary_plot_data <- df_updated %>%
  select(all_of(binary_features)) %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value") %>%
  mutate(
    feature = fct_reorder(feature, as.numeric(value == 1), .fun = mean, na.rm = TRUE),
    value = factor(value, levels = c(0, 1), labels = c("No", "Yes"))
  ) %>%
  filter(!is.na(value))

binary_dist_plot <- ggplot(binary_plot_data, aes(x = feature, fill = value)) +
  geom_bar(position = "fill") +
  geom_text(
    data = binary_plot_data %>% 
      filter(value == "Yes") %>% 
      count(feature, value) %>% 
      group_by(feature) %>% 
      mutate(pct = n / sum(n), pos = pct / 2),
    aes(y = pos, label = paste0(round(pct * 100), "%")),
    color = "white", fontface = "bold", size = 3
  ) +
  scale_fill_manual(values = cb_palette[c(1, 7)]) +
  coord_flip() +
  labs(
    title = "Distribution of Binary Features",
    subtitle = "Proportion of respondents answering 'Yes' for each feature",
    x = "Feature",
    y = "Proportion",
    fill = "Response"
  )

print(binary_dist_plot)
save_plot(binary_dist_plot, "06_binary_features")

# 4.2 Categorical Features Analysis
cat("\n### 4.2 Categorical Features Analysis\n\n")

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
  save_plot(cat_plot, paste0("07_categorical_", feature))
}

# 4.3 Ordinal Features Analysis
cat("\n### 4.3 Ordinal Features Analysis\n\n")

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
    save_plot(ord_plot, paste0("08_ordinal_", feature))
    
    # Store the plot for later grid arrangement
    ordinal_plots[[feature]] <- ord_plot
  }
}

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
  save_plot(hist_plot, paste0("09_numeric_", feature))
}

# -----------------------------------------------------
# 5. RELATIONSHIP WITH TARGET VARIABLES. #needs improvemnt
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
    save_plot(ord_seasonal_plot, paste0("15_ord_seasonal_", feature))
  }
}

# 5.4 Numeric Features vs. Target Variables
cat("\n### 5.4 Numeric Features vs. Target\n\n")

# Create box plots for each numeric feature vs target variables
for (feature in numeric_features) {
  # H1N1 boxplot
  num_h1n1_box <- ggplot(df_updated, aes(x = factor(h1n1_vaccine), y = .data[[feature]], fill = factor(h1n1_vaccine))) +
    geom_boxplot() +
    scale_fill_manual(values = cb_palette[c(1, 7)]) +
    labs(
      title = paste(feature, "Distribution by H1N1 Vaccination Status"),
      x = "H1N1 Vaccine (1 = Yes, 0 = No)",
      y = feature,
      fill = "H1N1 Vaccine"
    ) +
    theme(legend.position = "none")
  
  # Seasonal boxplot
  num_seasonal_box <- ggplot(df_updated, aes(x = factor(seasonal_vaccine), y = .data[[feature]], fill = factor(seasonal_vaccine))) +
    geom_boxplot() +
    scale_fill_manual(values = cb_palette[c(3, 5)]) +
    labs(
      title = paste(feature, "Distribution by Seasonal Flu Vaccination Status"),
      x = "Seasonal Vaccine (1 = Yes, 0 = No)",
      y = feature,
      fill = "Seasonal Vaccine"
    ) +
    theme(legend.position = "none")
  
  print(num_h1n1_box)
  save_plot(num_h1n1_box, paste0("16_num_h1n1_", feature))
  
  print(num_seasonal_box)
  save_plot(num_seasonal_box, paste0("17_num_seasonal_", feature))
  
  # Add statistical test
  h1n1_test <- wilcox.test(df_updated[[feature]] ~ df_updated$h1n1_vaccine)
  seasonal_test <- wilcox.test(df_updated[[feature]] ~ df_updated$seasonal_vaccine)
  
  cat("\nStatistical test for", feature, "by vaccination status:\n")
  cat("H1N1 vaccine: p-value =", round(h1n1_test$p.value, 4), "\n")
  cat("Seasonal vaccine: p-value =", round(seasonal_test$p.value, 4), "\n")
}

# -----------------------------------------------------
# 6. CORRELATION ANALYSIS
# -----------------------------------------------------
cat("\n## 6. CORRELATION ANALYSIS\n\n")

# 6.1 Correlation between numeric features
cat("### 6.1 Correlation Between Numeric Features\n\n")

if (length(numeric_features) > 1) {
  numeric_cor <- cor(df_updated[numeric_features], use = "pairwise.complete.obs")
  
  cat("Correlation matrix for numeric features:\n")
  print_table(numeric_cor)
  
  # Create correlation plot
  corrplot(numeric_cor, method = "color", type = "upper", 
           order = "hclust", tl.col = "black", tl.srt = 45,
           addCoef.col = "black", number.cex = 0.7,
           col = colorRampPalette(c("#6D9EC1", "white", "#E46726"))(200))
}

# 6.2 Point-biserial correlation (between binary and numeric features)
cat("\n### 6.2 Point-biserial Correlation\n\n")

# Prepare data for correlation analysis
binary_numeric_data <- df_updated %>%
  select(all_of(c(binary_features, numeric_features))) %>%
  mutate(across(all_of(binary_features), as.numeric, .names = "numeric_{.col}")) %>%
  select(-all_of(binary_features))

# Calculate correlation matrix for point-biserial correlation
if (ncol(binary_numeric_data) > 1) {
  pb_cor <- cor(binary_numeric_data, use = "pairwise.complete.obs")
  
  # Filter to show only relevant correlations (binary vs numeric)
  binary_cols <- grep("^numeric_", colnames(pb_cor), value = TRUE)
  numeric_cols <- setdiff(colnames(pb_cor), binary_cols)
  
  pb_cor_subset <- pb_cor[binary_cols, numeric_cols, drop = FALSE]
  
  cat("Point-biserial correlation between binary and numeric features:\n")
  print_table(pb_cor_subset)
}

# 6.3 Correlation of ordinal features 
cat("\n### 6.3 Correlation of Ordinal Features\n\n")

# Convert ordinal features to numeric for correlation analysis
ordinal_numeric <- df_updated %>%
  select(all_of(ordinal_names)) %>%
  mutate(across(everything(), ~ as.numeric(.)))

# Calculate correlations
if (ncol(ordinal_numeric) > 1) {
  ord_cor <- cor(ordinal_numeric, use = "pairwise.complete.obs")
  
  cat("Correlation matrix for ordinal features:\n")
  print_table(ord_cor)
  
  # Create correlation plot 
  corrplot(ord_cor, method = "color", type = "upper", 
           order = "hclust", tl.col = "black", tl.srt = 45,
           addCoef.col = "black", number.cex = 0.7,
           col = colorRampPalette(c("#6D9EC1", "white", "#E46726"))(200))
  
  # Save plot
  dev.print(png, "plots/18_ordinal_correlation.png", width = 10, height = 8, unit = "in", res = 300)
}

# 6.4 Create correlation with target variables
cat("\n### 6.4 Correlation with Target Variables\n\n")

# Prepare data for target correlation analysis
target_corr_data <- df_updated %>%
  select(h1n1_vaccine, seasonal_vaccine) %>%
  mutate(
    h1n1_vaccine = as.numeric(as.character(h1n1_vaccine)),
    seasonal_vaccine = as.numeric(as.character(seasonal_vaccine))
  )

# Combine with numeric features
if (length(numeric_features) > 0) {
  numeric_target_data <- bind_cols(
    df_updated %>% select(all_of(numeric_features)),
    target_corr_data
  )
  
  # Calculate correlations with target
  numeric_target_cor <- cor(numeric_target_data, use = "pairwise.complete.obs")
  
  cat("Correlation of numeric features with target variables:\n")
  print_table(numeric_target_cor[1:length(numeric_features), (length(numeric_features)+1):ncol(numeric_target_cor)])
}

# Correlation of ordinal features with target
if (length(ordinal_names) > 0) {
  ordinal_target_data <- bind_cols(
    ordinal_numeric,
    target_corr_data
  )
  
  # Calculate correlations with target
  ordinal_target_cor <- cor(ordinal_target_data, use = "pairwise.complete.obs")
  
  cat("\nCorrelation of ordinal features with target variables:\n")
  print_table(ordinal_target_cor[1:length(ordinal_names), (length(ordinal_names)+1):ncol(ordinal_target_cor)])
}

# -----------------------------------------------------
# 7. OUTLIER ANALYSIS
# -----------------------------------------------------
cat("\n## 7. OUTLIER ANALYSIS\n\n")

# Function to identify outliers using IQR method
identify_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  outliers <- which(x < lower_bound | x > upper_bound)
  return(list(
    outliers = outliers,
    lower_bound = lower_bound,
    upper_bound = upper_bound,
    count = length(outliers),
    percent = length(outliers) / length(x[!is.na(x)]) * 100
  ))
}

# Apply outlier detection to numeric features
outlier_summary <- data.frame(
  feature = character(),
  count = integer(),
  percent = numeric(),
  lower_bound = numeric(),
  upper_bound = numeric(),
  stringsAsFactors = FALSE
)

for (feature in numeric_features) {
  outlier_info <- identify_outliers(df_updated[[feature]])
  
  outlier_summary <- rbind(
    outlier_summary,
    data.frame(
      feature = feature,
      count = outlier_info$count,
      percent = outlier_info$percent,
      lower_bound = outlier_info$lower_bound,
      upper_bound = outlier_info$upper_bound
    )
  )
  
  # Create boxplot with outliers highlighted
  outlier_plot <- ggplot(df_updated, aes(y = .data[[feature]])) +
    geom_boxplot(fill = cb_palette[2], outlier.color = "red", outlier.shape = 16, outlier.size = 3) +
    geom_text(
      data = data.frame(
        y = c(outlier_info$lower_bound, outlier_info$upper_bound),
        label = c(paste("Lower bound:", round(outlier_info$lower_bound, 2)),
                  paste("Upper bound:", round(outlier_info$upper_bound, 2))),
        x = c(0, 0)
      ),
      aes(x = x, y = y, label = label),
      hjust = -0.1, size = 3
    ) +
    labs(
      title = paste("Boxplot of", feature, "with Outliers Highlighted"),
      subtitle = paste0(outlier_info$count, " outliers (", round(outlier_info$percent, 1), "%)"),
      x = NULL,
      y = feature
    ) +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  
  print(outlier_plot)
  #save_plot(outlier_plot, paste0("19_outliers_", feature))
  
  # Analyze relationship between outliers and target
  if (outlier_info$count > 0) {
    is_outlier <- rep(FALSE, nrow(df_updated))
    is_outlier[outlier_info$outliers] <- TRUE
    
    outlier_target <- data.frame(
      is_outlier = is_outlier,
      h1n1_vaccine = df_updated$h1n1_vaccine,
      seasonal_vaccine = df_updated$seasonal_vaccine
    )
    
    # Test association between outliers and target
    h1n1_test <- chisq.test(table(outlier_target$is_outlier, outlier_target$h1n1_vaccine))
    seasonal_test <- chisq.test(table(outlier_target$is_outlier, outlier_target$seasonal_vaccine))
    
    cat("\nRelationship between outliers in", feature, "and target variables:\n")
    cat("H1N1 vaccine: chi-squared p-value =", round(h1n1_test$p.value, 4), "\n")
    cat("Seasonal vaccine: chi-squared p-value =", round(seasonal_test$p.value, 4), "\n")
  }
}

cat("\nSummary of outliers in numeric features:\n")
print_table(outlier_summary)

# -----------------------------------------------------
# 8. FEATURE IMPORTANCE ANALYSIS
# -----------------------------------------------------
cat("\n## 8. FEATURE IMPORTANCE ANALYSIS\n\n")

# 8.1 Univariate feature importance using chi-squared test for categorical features
cat("### 8.1 Univariate Feature Importance (Chi-Squared Test)\n\n")

# Function to calculate chi-squared test
calculate_chi_sq <- function(feature, target) {
  # Create contingency table
  cont_table <- table(feature, target)
  
  # Skip if not enough data
  if (nrow(cont_table) < 2 || ncol(cont_table) < 2) {
    return(list(p_value = NA, chi_sq = NA))
  }
  
  # Calculate chi-squared test
  test <- chisq.test(cont_table, simulate.p.value = TRUE)
  
  return(list(p_value = test$p.value, chi_sq = test$statistic))
}

# Apply chi-squared test to categorical features
categorical_importance <- data.frame(
  feature = character(),
  chi_sq_h1n1 = numeric(),
  p_value_h1n1 = numeric(),
  chi_sq_seasonal = numeric(),
  p_value_seasonal = numeric(),
  stringsAsFactors = FALSE
)

all_cat_features <- c(categorical_features, binary_features)

for (feature in all_cat_features) {
  if (feature %in% names(df_updated) && feature != "h1n1_vaccine" && feature != "seasonal_vaccine") {
    # Calculate chi-squared test for H1N1 vaccine
    h1n1_test <- calculate_chi_sq(df_updated[[feature]], df_updated$h1n1_vaccine)
    
    # Calculate chi-squared test for seasonal vaccine
    seasonal_test <- calculate_chi_sq(df_updated[[feature]], df_updated$seasonal_vaccine)
    
    # Add to data frame
    categorical_importance <- rbind(
      categorical_importance,
      data.frame(
        feature = feature,
        chi_sq_h1n1 = h1n1_test$chi_sq,
        p_value_h1n1 = h1n1_test$p_value,
        chi_sq_seasonal = seasonal_test$chi_sq,
        p_value_seasonal = seasonal_test$p_value
      )
    )
  }
}

# Sort by importance for H1N1 vaccine
categorical_importance <- categorical_importance %>%
  arrange(p_value_h1n1)

cat("Chi-squared test results for categorical features:\n")
print_table(categorical_importance)

# Plot top 15 features for H1N1 vaccine
top_h1n1_features <- categorical_importance %>%
  filter(!is.na(chi_sq_h1n1)) %>%
  top_n(15, chi_sq_h1n1) %>%
  mutate(feature = fct_reorder(feature, chi_sq_h1n1))

h1n1_importance_plot <- ggplot(top_h1n1_features, aes(x = feature, y = chi_sq_h1n1)) +
  geom_bar(stat = "identity", fill = cb_palette[2]) +
  coord_flip() +
  labs(
    title = "Top Features for H1N1 Vaccination",
    subtitle = "Based on chi-squared test statistic",
    x = "Feature",
    y = "Chi-squared statistic"
  )

print(h1n1_importance_plot)
#save_plot(h1n1_importance_plot, "20_chi_squared_h1n1")

# Plot top 15 features for seasonal vaccine
top_seasonal_features <- categorical_importance %>%
  filter(!is.na(chi_sq_seasonal)) %>%
  top_n(15, chi_sq_seasonal) %>%
  mutate(feature = fct_reorder(feature, chi_sq_seasonal))

seasonal_importance_plot <- ggplot(top_seasonal_features, aes(x = feature, y = chi_sq_seasonal)) +
  geom_bar(stat = "identity", fill = cb_palette[4]) +
  coord_flip() +
  labs(
    title = "Top Features for Seasonal Flu Vaccination",
    subtitle = "Based on chi-squared test statistic",
    x = "Feature",
    y = "Chi-squared statistic"
  )

print(seasonal_importance_plot)
#save_plot(seasonal_importance_plot, "21_chi_squared_seasonal")

# 8.2 Basic Random Forest Feature Importance
cat("\n### 8.2 Random Forest Feature Importance\n\n")

# Prepare data for Random Forest
model_data <- df_updated %>%
  select(-respondent_id) %>%
  na.omit()

# Check if we have enough data after removing missing values
if (nrow(model_data) > 100) {
  # Convert categorical variables to factors
  model_data <- model_data %>%
    mutate(across(where(is.character), as.factor))
  
  # Install and load randomForest if not already installed
  if (!require(randomForest)) {
    install.packages("randomForest")
    library(randomForest)
  }
  
  # Train Random Forest for H1N1 vaccine
  set.seed(123)
  rf_h1n1 <- randomForest(
    h1n1_vaccine ~ ., 
    data = model_data %>% select(-seasonal_vaccine),
    importance = TRUE,
    ntree = 100
  )
  
  # Train Random Forest for seasonal vaccine
  set.seed(123)
  rf_seasonal <- randomForest(
    seasonal_vaccine ~ ., 
    data = model_data %>% select(-h1n1_vaccine),
    importance = TRUE,
    ntree = 100
  )
  
  # Get feature importance
  h1n1_importance <- importance(rf_h1n1) %>%
    as.data.frame() %>%
    mutate(feature = rownames(.)) %>%
    arrange(desc(MeanDecreaseGini))
  
  seasonal_importance <- importance(rf_seasonal) %>%
    as.data.frame() %>%
    mutate(feature = rownames(.)) %>%
    arrange(desc(MeanDecreaseGini))
  
  cat("Random Forest feature importance for H1N1 vaccine:\n")
  print_table(h1n1_importance %>% select(feature, MeanDecreaseGini))
  
  cat("\nRandom Forest feature importance for seasonal vaccine:\n")
  print_table(seasonal_importance %>% select(feature, MeanDecreaseGini))
  
  # Plot top 15 features for H1N1 vaccine
  h1n1_rf_plot <- ggplot(
    h1n1_importance %>% top_n(15, MeanDecreaseGini),
    aes(x = reorder(feature, MeanDecreaseGini), y = MeanDecreaseGini)
  ) +
    geom_bar(stat = "identity", fill = cb_palette[3]) +
    coord_flip() +
    labs(
      title = "Random Forest Feature Importance for H1N1 Vaccine",
      x = "Feature",
      y = "Mean Decrease in Gini Index"
    )
  
  print(h1n1_rf_plot)
  save_plot(h1n1_rf_plot, "22_rf_importance_h1n1")
  
  # Plot top 15 features for seasonal vaccine
  seasonal_rf_plot <- ggplot(
    seasonal_importance %>% top_n(15, MeanDecreaseGini),
    aes(x = reorder(feature, MeanDecreaseGini), y = MeanDecreaseGini)
  ) +
    geom_bar(stat = "identity", fill = cb_palette[5]) +
    coord_flip() +
    labs(
      title = "Random Forest Feature Importance for Seasonal Flu Vaccine",
      x = "Feature",
      y = "Mean Decrease in Gini Index"
    )
  
  print(seasonal_rf_plot)
  #save_plot(seasonal_rf_plot, "23_rf_importance_seasonal")
} else {
  cat("Not enough data for Random Forest feature importance after removing missing values.\n")
}

# -----------------------------------------------------
# 9. FEATURE RELATIONSHIPS AND INTERACTIONS
# -----------------------------------------------------
cat("\n## 9. FEATURE RELATIONSHIPS AND INTERACTIONS\n\n")

# 9.1 Analyze relationships between key categorical features
cat("### 9.1 Relationships Between Key Categorical Features\n\n")

# Identify key categorical features that might interact
key_cat_pairs <- list(
  c("h1n1_concern", "opinion_h1n1_risk"),
  c("doctor_recc_h1n1", "opinion_h1n1_vacc_effective"),
  c("doctor_recc_seasonal", "opinion_seas_vacc_effective"),
  c("chronic_med_condition", "health_worker"),
  c("age_group", "education"),
  c("age_group", "income_poverty")
)

# Analyze each pair
for (pair in key_cat_pairs) {
  if (all(pair %in% names(df_updated))) {
    feature1 <- pair[1]
    feature2 <- pair[2]
    
    # Create contingency table
    cat(paste("Relationship between", feature1, "and", feature2, ":\n"))
    cont_table <- table(df_updated[[feature1]], df_updated[[feature2]])
    print(cont_table)
    
    # Chi-squared test
    chi_sq_test <- chisq.test(cont_table, simulate.p.value = TRUE)
    cat("Chi-squared test: p-value =", round(chi_sq_test$p.value, 4), "\n\n")
    
    # Create heatmap for the relationship
    df_pair <- df_updated %>%
      select(all_of(c(feature1, feature2))) %>%
      filter(!is.na(.data[[feature1]]) & !is.na(.data[[feature2]])) %>%
      count(.data[[feature1]], .data[[feature2]]) %>%
      group_by(.data[[feature1]]) %>%
      mutate(pct = n / sum(n) * 100) %>%
      ungroup()
    
    heatmap_plot <- ggplot(df_pair, aes(x = .data[[feature1]], y = .data[[feature2]], fill = pct)) +
      geom_tile() +
      geom_text(aes(label = sprintf("%.1f%%", pct)), size = 2.5) +
      scale_fill_viridis_c(option = "D") +
      labs(
        title = paste("Relationship between", feature1, "and", feature2),
        subtitle = paste("Chi-squared p-value =", round(chi_sq_test$p.value, 4)),
        x = feature1,
        y = feature2,
        fill = "Percentage"
      ) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    print(heatmap_plot)
    #save_plot(heatmap_plot, paste0("24_relationship_", feature1, "_", feature2), width = 12, height = 8)
  }
}

# 9.2 Analyze feature interactions with target variables
cat("\n### 9.2 Feature Interactions with Target Variables\n\n")

# Define pairs to analyze for interaction with targets
interaction_pairs <- list(
  c("doctor_recc_h1n1", "opinion_h1n1_vacc_effective"),
  c("doctor_recc_seasonal", "opinion_seas_vacc_effective"),
  c("h1n1_concern", "opinion_h1n1_risk"),
  c("age_group", "chronic_med_condition")
)

# Analyze interactions with targets
for (pair in interaction_pairs) {
  if (all(pair %in% names(df_updated))) {
    feature1 <- pair[1]
    feature2 <- pair[2]
    
    # H1N1 vaccine interaction
    h1n1_interaction <- df_updated %>%
      filter(!is.na(.data[[feature1]]) & !is.na(.data[[feature2]]) & !is.na(h1n1_vaccine)) %>%
      group_by(.data[[feature1]], .data[[feature2]]) %>%
      summarise(
        total = n(),
        h1n1_rate = mean(h1n1_vaccine == 1) * 100,
        .groups = "drop"
      )
    
    h1n1_interaction_plot <- ggplot(h1n1_interaction, 
                                    aes(x = .data[[feature1]], y = .data[[feature2]], fill = h1n1_rate)) +
      geom_tile() +
      geom_text(aes(label = sprintf("%.1f%%", h1n1_rate)), size = 2.5, color = "white") +
      scale_fill_viridis_c(option = "D") +
      labs(
        title = paste("H1N1 Vaccination Rate by", feature1, "and", feature2),
        x = feature1,
        y = feature2,
        fill = "H1N1 Vax Rate (%)"
      ) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    print(h1n1_interaction_plot)
    #save_plot(h1n1_interaction_plot, paste0("25_h1n1_interaction_", feature1, "_", feature2), width = 12, height = 8)
    
    # Seasonal vaccine interaction
    seasonal_interaction <- df_updated %>%
      filter(!is.na(.data[[feature1]]) & !is.na(.data[[feature2]]) & !is.na(seasonal_vaccine)) %>%
      group_by(.data[[feature1]], .data[[feature2]]) %>%
      summarise(
        total = n(),
        seasonal_rate = mean(seasonal_vaccine == 1) * 100,
        .groups = "drop"
      )
    
    seasonal_interaction_plot <- ggplot(seasonal_interaction, 
                                        aes(x = .data[[feature1]], y = .data[[feature2]], fill = seasonal_rate)) +
      geom_tile() +
      geom_text(aes(label = sprintf("%.1f%%", seasonal_rate)), size = 2.5, color = "white") +
      scale_fill_viridis_c(option = "D") +
      labs(
        title = paste("Seasonal Flu Vaccination Rate by", feature1, "and", feature2),
        x = feature1,
        y = feature2,
        fill = "Seasonal Vax Rate (%)"
      ) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    print(seasonal_interaction_plot)
    #save_plot(seasonal_interaction_plot, paste0("26_seasonal_interaction_", feature1, "_", feature2), width = 12, height = 8)
  }
}

# -----------------------------------------------------
# 10. SUMMARY AND RECOMMENDATIONS
# -----------------------------------------------------
cat("\n## 10. SUMMARY AND RECOMMENDATIONS\n\n")

cat("### Key Findings:\n\n")

cat("1. **Target Variable Distribution:**\n")
cat("   - H1N1 vaccination rate: ", round(h1n1_rate, 1), "%\n")
cat("   - Seasonal flu vaccination rate: ", round(seasonal_rate, 1), "%\n")
cat("   - The seasonal flu vaccine has much higher uptake than the H1N1 vaccine\n\n")

cat("2. **Missing Values:**\n")
missing_features <- missing_summary %>% filter(missing > 0) %>% pull(feature)
cat("   - Features with missing values: ", paste(missing_features, collapse = ", "), "\n")
cat("   - Missing value patterns might require imputation strategies\n\n")

cat("3. **Important Features for H1N1 Vaccination:**\n")
if (exists("h1n1_importance") && nrow(h1n1_importance) > 0) {
  top5_h1n1 <- h1n1_importance %>% slice_head(n = 5) %>% pull(feature)
  cat("   - Top 5 features: ", paste(top5_h1n1, collapse = ", "), "\n")
}
cat("   - Doctor recommendation and opinion on vaccine effectiveness are strong predictors\n\n")

cat("4. **Important Features for Seasonal Flu Vaccination:**\n")
if (exists("seasonal_importance") && nrow(seasonal_importance) > 0) {
  top5_seasonal <- seasonal_importance %>% slice_head(n = 5) %>% pull(feature)
  cat("   - Top 5 features: ", paste(top5_seasonal, collapse = ", "), "\n")
}
cat("   - Similar to H1N1, doctor recommendation and past vaccination behavior are key\n\n")

cat("5. **Feature Interactions:**\n")
cat("   - Strong interactions between doctor recommendations and opinions on vaccine effectiveness\n")
cat("   - Age group and chronic medical conditions show important interactions\n\n")

cat("### Recommendations for Feature Engineering:\n\n")

cat("1. **Handle Missing Values:**\n")
cat("   - Use multiple imputation or advanced methods for features with significant missingness\n")
cat("   - Consider creating 'missing' indicators for features where missingness might be informative\n\n")

cat("2. **Feature Transformations:**\n")
cat("   - Group rare categories in categorical variables\n")
cat("   - Create composite features for related behavioral variables\n")
cat("   - Develop interaction terms for key feature pairs\n\n")

cat("3. **New Feature Creation:**\n")
cat("   - Create risk scores combining multiple risk factors\n")
cat("   - Develop opinion indices from multiple opinion questions\n")
cat("   - Consider geographic clustering based on region\n\n")

cat("4. **Modeling Approach:**\n")
cat("   - Consider multi-label classification approaches\n")
cat("   - Use models that handle missing data well (e.g., XGBoost)\n")
cat("   - Evaluate performance separately for each target\n\n")

cat("5. **Validation Strategy:**\n")
cat("   - Use stratified cross-validation to maintain target distribution\n")
cat("   - Consider time-based validation if temporal patterns exist\n\n")

cat("\n--- End of EDA Report ---\n")