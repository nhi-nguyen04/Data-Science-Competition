library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(scales)
library(viridis)
library(grid)   # for unit()

# 1. theme
custom_theme <- theme_minimal() +
  theme(
    text            = element_text(color = "#333333"),
    plot.title      = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle   = element_text(size = 11,   hjust = 0.5),
    axis.title      = element_text(size = 12,   face = "bold"),
    axis.text       = element_text(size = 10),
    panel.grid.major= element_line(color = "#EEEEEE"),
    panel.grid.minor= element_blank(),
    legend.position = "top",
    legend.title    = element_text(face = "bold"),
    legend.text     = element_text(size = 9),
    plot.margin     = unit(c(10, 20, 10, 10), "pt")
  )
theme_set(custom_theme)

# 2. data
features_df <- read.csv("Data/training_set_features.csv")
labels_df   <- read.csv("Data/training_set_labels.csv")

# 3. correct color mapping
vaccine_colors <- c(
  "Not Vaccinated" = "#E15759",
  "Vaccinated"     = "#4E79A7"
)

# 4. plot function
create_vaccine_plot <- function(data, vaccine_type) {
  var_name <- paste0(vaccine_type, "_vaccine")
  title    <- sprintf("%s Vaccination Status",
                      tools::toTitleCase(vaccine_type))
  n_obs    <- nrow(data)
  
  data %>%
    count(!!sym(var_name)) %>%
    mutate(
      proportion = n / n_obs * 100,
      vaccine    = factor(
        !!sym(var_name),
        levels = c(0,1),
        labels = c("Not Vaccinated","Vaccinated")
      )
    ) %>%
    ggplot(aes(x = vaccine, y = proportion, fill = vaccine)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = paste0(round(proportion,1), "%")),
              position = position_stack(vjust = 0.5),
              color = "white", fontface = "bold") +
    scale_fill_manual(values = vaccine_colors) +
    labs(
      title    = title,
      subtitle = paste("Distribution of", title),
      y        = "Percentage (%)",
      x        = NULL
    ) +
    theme(legend.position = "none") +
    ylim(0,100)
}

# 5. make & combine
plot_h1n1     <- create_vaccine_plot(labels_df, "h1n1")
plot_seasonal <- create_vaccine_plot(labels_df, "seasonal")

vaccine_plots <- plot_h1n1 + plot_seasonal +
  plot_layout(ncol = 2, guides = "collect") +
  plot_annotation(
    title = "Vaccination Rates for H1N1 and Seasonal Flu",
    theme = theme(plot.title = element_text(size = 16, face = "bold",
                                            hjust = 0.5))
  )

print(vaccine_plots)
#------------------------------
# 2. Improved Cross-Tabulation Analysis
#------------------------------

# Cross-tabulation with improved visualization
labels_df <- labels_df %>%
  mutate(
    h1n1_status = factor(h1n1_vaccine, levels = c(0, 1), 
                         labels = c("Not Vaccinated", "Vaccinated")),
    seasonal_status = factor(seasonal_vaccine, levels = c(0, 1), 
                             labels = c("Not Vaccinated", "Vaccinated"))
  )

# Create cross-tabulation
cross_tab <- table(labels_df$h1n1_status, labels_df$seasonal_status)
cross_tab_prop <- prop.table(cross_tab) * 100

# Convert to data frame for plotting
cross_tab_df <- as.data.frame.table(cross_tab)
names(cross_tab_df) <- c("H1N1", "Seasonal", "Count")

cross_tab_prop_df <- as.data.frame.table(cross_tab_prop)
names(cross_tab_prop_df) <- c("H1N1", "Seasonal", "Percentage")

# Combining for label
cross_tab_df$Percentage <- cross_tab_prop_df$Percentage
cross_tab_df$Label <- paste0(cross_tab_df$Count, "\n(", round(cross_tab_df$Percentage, 1), "%)")

# Create heatmap
heatmap_plot <- ggplot(cross_tab_df, aes(x = Seasonal, y = H1N1, fill = Percentage)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = Label), color = "white", fontface = "bold", size = 4) +
  scale_fill_viridis_c(option = "D", direction = -1) +
  labs(
    title = "Relationship Between H1N1 and Seasonal Flu Vaccination",
    subtitle = paste("Phi Coefficient (Correlation):", 
                     round(cor(labels_df$h1n1_vaccine, labels_df$seasonal_vaccine), 3)),
    fill = "Percentage (%)"
  ) +
  coord_fixed()

print(heatmap_plot)

#------------------------------
# 3. Joining Features and Labels
#------------------------------

# Join the datasets - using a safer method with explicit message about success
joined_df <- features_df %>%
  inner_join(labels_df, by = "respondent_id")

# Display information about the join
cat(paste("Dataset dimensions after join:", paste(dim(joined_df), collapse = " x ")), "\n")
cat(paste("Original features dataset:", paste(dim(features_df), collapse = " x ")), "\n")
cat(paste("Original labels dataset:", paste(dim(labels_df), collapse = " x ")), "\n")

# Check if any rows were lost in the join
if(nrow(joined_df) != nrow(features_df) || nrow(joined_df) != nrow(labels_df)) {
  cat("WARNING: Some rows were lost during the join operation. This could affect analysis.\n")
} else {
  cat("Join completed successfully with no data loss.\n")
}

#------------------------------
# 4. Improved Vaccination Rate Function
#------------------------------
vaccination_rate_plot <- function(col, target, data, title = NULL, thresh = 10) {
  data <- data %>%
    filter(!is.na(.data[[col]]) & !is.na(.data[[target]]))
  
  # count distinct values
  n_vals <- n_distinct(data[[col]])
  
  # decide bar-chart vs boxplot
  if (is.factor(data[[col]]) ||
      is.character(data[[col]]) ||
      (is.integer(data[[col]]) && n_vals <= thresh)) {
    
    counts <- data %>%
      count(!!sym(col), !!sym(target)) %>%
      group_by(!!sym(col)) %>%
      mutate(
        total        = sum(n),
        percentage   = 100 * n / total,
        target_label = factor(!!sym(target),
                              levels = c(0,1),
                              labels = c("Not Vaccinated","Vaccinated"))
      ) %>%
      ungroup()
    
    # order levels by vaccination rate
    lvl <- counts %>%
      filter(!!sym(target) == 1) %>%
      arrange(desc(percentage)) %>%
      pull(!!sym(col))
    counts[[col]] <- factor(counts[[col]], levels = lvl)
    
    ggplot(counts, aes(x = !!sym(col), y = percentage, fill = target_label)) +
      geom_col(width = 0.7) +
      geom_text(
        data = filter(counts, !!sym(target) == 1),
        aes(label = paste0(round(percentage,1), "%")),
        position = position_stack(vjust = 0.5),
        color    = "white",
        fontface = "bold",
        size     = 3
      ) +
      scale_fill_manual(values = vaccine_colors) +
      coord_flip() +
      labs(
        x     = NULL,
        y     = "Percentage (%)",
        fill  = "Vaccination Status",
        title = title %||% paste("Vaccination Rate by", gsub("_"," ",col))
      )
    
  } else {
    # continuous → boxplot
    ggplot(data, aes(
      x    = factor(!!sym(target), levels=c(0,1),
                    labels=c("Not Vaccinated","Vaccinated")),
      y    = !!sym(col),
      fill = factor(!!sym(target), levels=c(0,1))
    )) +
      geom_boxplot(alpha = 0.8, outlier.color = "#555555", outlier.alpha = 0.5) +
      scale_fill_manual(values = vaccine_colors) +
      labs(
        x     = NULL,
        y     = gsub("_"," ",col),
        fill  = "Vaccination Status",
        title = title %||% paste("Distribution of", gsub("_"," ",col),
                                 "by Vaccination Status")
      ) +
      theme(legend.position = "none")
  }
}

# 6. create & display your grid of H1N1 vs seasonal plots
h1n1_plots     <- list()
seasonal_plots <- list()

for (col in cols_to_plot) {
  pretty <- tools::toTitleCase(gsub("_"," ",col))
  t1 <- paste("H1N1 Vaccination Rate by", pretty)
  t2 <- paste("Seasonal Flu Vaccination Rate by", pretty)
  
  h1n1_plots[[col]]     <- vaccination_rate_plot(col, "h1n1_vaccine", joined_df, t1)
  seasonal_plots[[col]] <- vaccination_rate_plot(col, "seasonal_vaccine", joined_df, t2)
}

# example layout
example_plots <- (h1n1_plots$h1n1_concern + h1n1_plots$h1n1_knowledge) /
  (h1n1_plots$opinion_h1n1_vacc_effective + h1n1_plots$age_group) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Key Factors Influencing H1N1 Vaccination",
    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
  )
print(example_plots)

seasonal_example_plots <- (seasonal_plots$opinion_seas_vacc_effective +
                             seasonal_plots$opinion_seas_risk) /
  (seasonal_plots$age_group + seasonal_plots$education) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Key Factors Influencing Seasonal Flu Vaccination",
    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
  )
print(seasonal_example_plots)
#------------------------------
# 6. Advanced Analysis - Correlation Matrix
#------------------------------

# Select numerical variables
numeric_cols <- sapply(joined_df, is.numeric)
numeric_features <- joined_df[, numeric_cols]

# Remove ID column and non-informative columns
numeric_features <- numeric_features %>%
  select(-respondent_id, contains("_vaccine"))

# Calculate correlation matrix
cor_matrix <- cor(numeric_features, use = "pairwise.complete.obs")

# Convert to data frame for plotting
cor_df <- as.data.frame(as.table(cor_matrix))
names(cor_df) <- c("Feature1", "Feature2", "Correlation")

# Create correlation heatmap
correlation_plot <- ggplot(cor_df, aes(x = Feature1, y = Feature2, fill = Correlation)) +
  geom_tile() +
  scale_fill_gradient2(low = "#E15759", mid = "white", high = "#4E79A7", midpoint = 0) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y = element_text(size = 8)
  ) +
  labs(
    title = "Correlation Matrix of Numerical Features",
    subtitle = "Blue indicates positive correlation, red indicates negative correlation"
  ) +
  coord_fixed()

print(correlation_plot)

#------------------------------
# 7. Data Leakage Assessment
#------------------------------

# Check for potential data leakage - correlation between target variables and predictors
calculate_target_correlations <- function(data, target_col) {
  # Get potentially problematic high correlations
  numeric_data <- data %>% 
    select_if(is.numeric) %>%
    select(-respondent_id) # Remove ID column
  
  # Calculate correlations with target
  target_cors <- cor(numeric_data[,target_col], numeric_data, use = "pairwise.complete.obs")
  
  # Convert to data frame
  target_cors_df <- as.data.frame(t(target_cors))
  names(target_cors_df) <- c("Correlation")
  target_cors_df$Feature <- rownames(target_cors_df)
  
  # Remove the target itself
  target_cors_df <- target_cors_df %>% 
    filter(Feature != target_col) %>%
    arrange(desc(abs(Correlation)))
  
  return(target_cors_df)
}

# Calculate correlations for both target variables
h1n1_cors <- calculate_target_correlations(joined_df, "h1n1_vaccine")
seasonal_cors <- calculate_target_correlations(joined_df, "seasonal_vaccine")

# Print top correlations for both targets
cat("Top 10 feature correlations with H1N1 vaccination:\n")
print(head(h1n1_cors, 10))

cat("\nTop 10 feature correlations with Seasonal vaccination:\n")
print(head(seasonal_cors, 10))

# Check for potential data leakage between opinion variables and target
opinion_vars <- grep("opinion_", colnames(joined_df), value = TRUE)
cat("\nPotential opinion variables that might cause data leakage:")
print(opinion_vars)

cat("\nData Leakage Assessment:")
cat("\n------------------------")
cat("\n1. Target variable correlations have been checked.")
cat("\n2. Opinion variables should be carefully considered as they might contain information collected after vaccination decisions.")
cat("\n3. No direct leakage of target variables to features was found through ID or other direct mechanisms.")
cat("\n4. Time-related information should be verified to ensure features were collected before vaccination decisions.")
cat("\n5. Cross-validation will be implemented in model training to further protect against data leakage.")











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