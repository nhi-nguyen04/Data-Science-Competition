# -----------------------------------------------
# 1. SET UP ENVIRONMENT
# -----------------------------------------------
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(scales)
library(viridis)
library(grid)   
library(tableone)
library(knitr)
library(tidyverse) 
library(naniar)
library(skimr) #Better Overview of the variables

# theme set up for now
# probably going to change in the future
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

# correct color mapping
vaccine_colors <- c(
  "Not Vaccinated" = "#F1C45F",
  "Vaccinated"     = "#1B80BB"
)

# -----------------------------------------------
# 2. LOAD DATA
# -----------------------------------------------
features_df <- read.csv("Data/training_set_features.csv")
labels_df   <- read.csv("Data/training_set_labels.csv")

# -----------------------------------------------
# 3. QUICK DATA INSPECTION
# -----------------------------------------------
lapply(features_df, unique)

# the labels don't have to be processed
lapply(labels_df, unique)

# Rename labels
features_df <- features_df %>%
  rename("Respondend ID" = respondent_id ,
         "H1N1 Concern" = h1n1_concern,
         "H1N1 Knowledge" = h1n1_knowledge,
         "Antiviral Medication" = behavioral_antiviral_meds,
         "Avoidance" = behavioral_avoidance,
         "Face Mask" = behavioral_face_mask,
         "Wash Hands" = behavioral_wash_hands,
         "Large Gatherings" = behavioral_large_gatherings,
         "Outside Home" = behavioral_outside_home,
         "Touch Face" = behavioral_touch_face,
         "Doctor Recommendation H1N1" = doctor_recc_h1n1,
         "Doctor Recommendation Seasonal" = doctor_recc_seasonal,
         "Chronical Medical Condition" = chronic_med_condition,
         "Child under 6 Months" = child_under_6_months,
         "Health Worker" = health_worker,
         "Health Insurance" = health_insurance,
         "Opinion H1N1 Effect" = opinion_h1n1_vacc_effective,
         "Opinion H1N1 Risk" = opinion_h1n1_risk,
         "Opinion H1N1 sick from Vaccine" = opinion_h1n1_sick_from_vacc,
         "Opinion Seasonal Effect" = opinion_seas_vacc_effective,
         "Opinion Seasonal Risk" = opinion_seas_risk,
         "Opinion Seasonal sick from Vaccine" = opinion_seas_sick_from_vacc,
         "Age Group" = age_group,
         "Education Level" = education,
         "Race" = race,
         "Sex" = sex,
         "Income Level" = income_poverty,
         "Marital Status" = marital_status,
         "Housing Situation" = rent_or_own,
         "Employment Status" = employment_status,
         "Geographical Region" = hhs_geo_region,
         "Metropolitan Statistical Areas" = census_msa,
         "Number of other Adults in Household" = household_adults,
         "Number of Children in Household" = household_children,
         "Working Industry" = employment_industry,
         "Type of Occupation" = employment_occupation)

# missing 
vis_miss(features_df, warn_large_data = FALSE) +
  coord_flip() +
  theme(axis.text.x = element_text(angle = 0, hjust = 1),
        axis.text.y = element_text(size = 8)) +
  ggtitle("Missing Data by Variable") +
  labs(x = "Variables")

gg_miss_var(features_df) +
  scale_x_discrete(labels = c(
    "respondent_id" = "Respondend ID",
    "h1n1_concern" = "H1N1 Concern",
    "h1n1_knowledge" = "H1N1 Knowledge",
    "behavioral_antiviral_meds" = "Antiviral Medication",
    "behavioral_avoidance" = "Avoidance",
    "behavioral_face_mask" = "Face Mask",
    "behavioral_wash_hands" = "Wash Hands",
    "behavioral_large_gatherings" = "Large Gatherings",
    "behavioral_outside_home" = "Outside Home",
    "behavioral_touch_face" = "Touch Face",
    "doctor_recc_h1n1" = "Doctor Recommendation H1N1",
    "doctor_recc_seasonal" = "Doctor Recommendation Seasonal",
    "chronic_med_condition" = "Chronical Medical Condition",
    "child_under_6_months" = "Child under 6 Months",
    "health_worker" = "Health Worker",
    "health_insurance" = "Health Insurance",
    "opinion_h1n1_vacc_effective" = "Opinion H1N1 Effect",
    "opinion_h1n1_risk" = "Opinion H1N1 Risk",
    "opinion_h1n1_sick_from_vacc" = "Opinion H1N1 sick from Vaccine",
    "opinion_seas_vacc_effective" = "Opinion Seasonal Effect",
    "opinion_seas_risk" = "Opinion Seasonal Risk",
    "opinion_seas_sick_from_vacc" = "Opinion Seasonal sick from Vaccine",
    "age_group" = "Age Group",
    "education" = "Education Level",
    "race" = "Race",
    "sex" = "Sex",
    "income_poverty" = "Income Level",
    "marital_status" = "Marital Status",
    "rent_or_own" = "Housing Situation",
    "employment_status" = "Employment Status",
    "hhs_geo_region" = "Geographical Region",
    "census_msa" = "Metropolitan Statistical Areas",
    "household_adults" = "Number of other Adults in Household",
    "household_children" = "Number of Children in Household",
    "employment_industry" = "Working Industry",
    "employment_occupation" = "Type of Occupation"
  )) +
  labs(x = "Variable", y = "Number of Missing") +
  ggtitle("Missing Data by Variable") +
  theme(axis.text.y = element_text(size = 7))

#Only columns with missing
features_df %>%
  select(where(~ any(is.na(.)))) %>%
  vis_miss()

# -----------------------------------------------
# 4. PLOTS DESIGN -> VACCINATION RATE
# -----------------------------------------------

# plot function
create_vaccine_plot <- function(data, vaccine_type) {
  var_name <- paste0(vaccine_type, "_vaccine")
  title    <- sprintf("%s Vaccination Rates",
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
    scale_fill_manual(values = vaccine_colors) +
    labs(
#      title    = title,
      y        = "Percentage (%)",
      x        = NULL
    ) +
    theme(legend.position = "none") +
    ylim(0,100)
}

# -----------------------------------------------
# 5. VACCINATION RATE PLOT + VISUALIZATION
# -----------------------------------------------

# create the plots  & combine them
plot_h1n1 <- create_vaccine_plot(labels_df, "h1n1") +
  ggtitle("H1N1 Vaccination Rates")
plot_seasonal <- create_vaccine_plot(labels_df, "seasonal") +
  ylab(NULL) +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank()) +
  ggtitle("Seasonal Vaccination Rates")

vaccine_plots <- plot_h1n1 + plot_seasonal +
  plot_layout(ncol = 2, guides = "collect") +
  plot_annotation(
    title = "Vaccination Rates for H1N1 and Seasonal Flu",
    theme = theme(plot.title = element_text(size = 15, face = "bold",
                                            hjust = 0.5))
  )

vaccine_plots


# -----------------------------------------------
# 6. OBSERVATION OF VACCINATION RATE PLOT 
# -----------------------------------------------
# Observation
# A bit over half of the individuals received the seasonal flu vaccine, while only about 21% received the H1N1 flu vaccine.
# 
# From a class distribution perspective, the seasonal flu vaccine target has balanced classes, whereas the H1N1 flu vaccine target shows a moderate class imbalance.
# 
# In machine learning, a class refers to a category or label — in this case:
#   
#   Got the vaccine = one class
# 
#   Didn't get the vaccine = the other class
# 
# Balanced Classes (Seasonal Flu Vaccine)
# Since just over 50% of people received the seasonal flu vaccine and the rest did not, the dataset has a fairly even distribution between the two classes. This balance means the model will have a roughly equal number of examples for each outcome, making it easier to learn both cases effectively.
# 
# Moderately Imbalanced Classes (H1N1 Vaccine)
# In contrast, only 21% of individuals received the H1N1 vaccine, while 79% did not. This leads to an imbalance, with one class (those who didn't get vaccinated) heavily outnumbering the other. Such imbalances can make it more difficult for the model to accurately learn and predict the minority class.
# 
# Why It Matters
# Balanced data allows models to learn both outcomes more fairly and reliably. When classes are imbalanced, the model may default to predicting the majority class, resulting in poor performance on the less common — but potentially more important — cases.

#------------------------------
# 7.  Cross-Tabulation Analysis
#------------------------------
#Are the two target variables independent? Let's take a look.

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

# Create heatmap we need
heatmap_plot <- ggplot(cross_tab_df, aes(x = Seasonal, y = H1N1, fill = Percentage)) +
  geom_tile(color = "white", size = 0.5, lwd = 1.5, linetype = 1) +
  geom_text(aes(label = Label), color = "white", fontface = "bold", linewidth = 4) +
  scale_fill_viridis_c(option = "D", direction = -1) +
  labs(
    title = "Relationship Between H1N1 and Seasonal Flu Vaccination",
    subtitle = paste("Phi Coefficient (Correlation):", 
                     round(cor(labels_df$h1n1_vaccine, labels_df$seasonal_vaccine), 3)),
    fill = "Percentage (%)"
  ) +
  coord_fixed() +
  scale_fill_gradient(low = "gray", high = "#6E7FC2")

#------------------------------
# 8.  CROSS-TABULATION ANALYSIS VISUALIZATION
#------------------------------

heatmap_plot

#------------------------------
# 9.  CROSS-TABULATION ANALYSIS OBSERVATION
#------------------------------
# 
# #The Phi coefficient (ϕ) is a statistical measure used to determine the association between two binary variables — variables that only take on two possible values.
# Using the Phi coefficient between h1n1_vaccine and seasonal_vaccine helps us:
#   
#   Understand the relationship between the two targets.
# 
# Choose better modeling strategies.
# 
# Interpret results more effectively.
# 
# So 0.377 is not weak, but also not very strong — it's in the middle, meaning:
# 
# There's a meaningful pattern.
# 
# But there’s still plenty of variation — many people may have taken only one vaccine or neither.

#------------------------------
#10. JOINING FEATURES AND LABELS + config
#------------------------------

# Reload Data
features_df <- read.csv("Data/training_set_features.csv")

joined_df <- features_df %>%
  inner_join(labels_df, by = "respondent_id") %>%
  mutate(
    # 1a) “Real” multi-level factors
    age_group = factor(
      age_group,
      levels = c(
        "18 - 34 Years",
        "35 - 44 Years",
        "45 - 54 Years",
        "55 - 64 Years",
        "65+ Years"
      ),
      ordered = TRUE
    ),
    education = factor(
      education,
      levels = c("< 12 Years", "12 Years", "Some College", "College Graduate", ""),
      ordered = TRUE
    ),
    income_poverty = factor(
      income_poverty,
      levels = c("Below Poverty", "<= $75,000, Above Poverty", "> $75,000", ""),
      ordered = TRUE
    ),
    
    # 1b) All your 1→5 “opinion_*” items
    across(
      starts_with("opinion_"),
      ~ factor(.x, levels = 1:5, ordered = TRUE)
    ),
    
    # 1c) The two concern/knowledge items (0→5)
    h1n1_concern   = factor(h1n1_concern,   levels = 0:5, ordered = TRUE),
    h1n1_knowledge = factor(h1n1_knowledge, levels = 0:5, ordered = TRUE),
    
    # 1d) All true binaries (0→1)
    across(
      c(
        behavioral_outside_home, behavioral_touch_face,
        behavioral_antiviral_meds, behavioral_avoidance,
        behavioral_face_mask,     behavioral_wash_hands,
        behavioral_large_gatherings,
        doctor_recc_h1n1, doctor_recc_seasonal,
        chronic_med_condition, child_under_6_months,
        health_worker, health_insurance
      ),
      ~ factor(.x, levels = 0:1)
    ),
    
    # 1e) Household counts as small-integer ordinals
    household_adults   = factor(household_adults,
                                levels = sort(na.omit(unique(household_adults))),
                                ordered = TRUE),
    household_children = factor(household_children,
                                levels = sort(na.omit(unique(household_children))),
                                ordered = TRUE)
  )

#Some info of the df
# Display information about the join
cat(paste("Dataset dimensions after join:", paste(dim(joined_df), collapse = " x ")), "\n")
cat(paste("Original features dataset:", paste(dim(features_df), collapse = " x ")), "\n")
cat(paste("Original labels dataset:", paste(dim(labels_df), collapse = " x ")), "\n")

# A check to see if any rows were lost in the join
if(nrow(joined_df) != nrow(features_df) || nrow(joined_df) != nrow(labels_df)) {
  cat("WARNING: Some rows were lost during the join operation. This could affect analysis.\n")
} else {
  cat("Join completed successfully with no data loss.\n")
}
#Nice

#------------------------------
# 11. MORE PLOTS ON VACCINATION RATE FOR DIFFERENT FEATURES
#------------------------------
vaccination_rate_plot <- function(col, target, data, title = NULL, thresh = 10) {
  # Filter out NA + empty values
  data <- data %>%
    filter(
      !is.na(.data[[col]]),
      !is.na(.data[[target]])
    )
  
  # For factor/character columns, also filter out empty strings
  if (is.factor(data[[col]]) || is.character(data[[col]])) {
    data <- data %>% filter(as.character(.data[[col]]) != "")
  }
  
  # Count distinct values actually present in the data
  n_vals <- n_distinct(data[[col]])
  
  # Get the actual values present in the data
  actual_values <- sort(unique(data[[col]]))
  
  # Discrete if factor/char or small integer
  if (is.factor(data[[col]]) ||
      is.character(data[[col]]) ||
      (is.numeric(data[[col]]) && n_vals <= thresh)) {
    
    # For ordered factors like h1n1_concern and h1n1_knowledge,
    # we need to make sure we preserve the actual levels present in the data
    if (is.factor(data[[col]]) && is.ordered(data[[col]])) {
      # Get the actual levels present in the data
      present_levels <- levels(data[[col]])[levels(data[[col]]) %in% actual_values]
      
      # Create cross-tabulation of counts
      counts <- data %>%
        count(!!sym(col), !!sym(target)) %>%
        # No complete() here as we only want actual present levels
        group_by(!!sym(col)) %>%
        mutate(
          total = sum(n),
          percentage = 100 * n / total,
          target_label = factor(
            !!sym(target),
            levels = c(0, 1),
            labels = c("Not Vaccinated", "Vaccinated")
          )
        ) %>%
        ungroup()
    } else {
      # For non-ordered factors or other variables
      counts <- data %>%
        count(!!sym(col), !!sym(target)) %>%
        group_by(!!sym(col)) %>%
        mutate(
          total = sum(n),
          percentage = 100 * n / total,
          target_label = factor(
            !!sym(target),
            levels = c(0, 1),
            labels = c("Not Vaccinated", "Vaccinated")
          )
        ) %>%
        ungroup()
      
      # For unordered factors, sort by vaccination rate
      if (!is.ordered(data[[col]])) {
        lvl <- counts %>%
          filter(!!sym(target) == 1) %>%
          arrange(desc(percentage)) %>%
          pull(!!sym(col))
        
        counts[[col]] <- factor(counts[[col]], levels = lvl)
      }
    }
    
    ggplot(counts, aes(
      y = !!sym(col),       # category on y
      x = percentage,       # percent on x
      fill = target_label
    )) +
      geom_col(width = 0.7) +
      # geom_text(
      #   data = filter(counts, !!sym(target) == 1),
      #   aes(label = paste0(round(percentage, 1), "%")),
      #   position = position_stack(vjust = 0.5),
      #   color = "white",
      #   fontface = "bold",
      #   size = 3
      # ) +
      scale_fill_manual(values = vaccine_colors) +
      # Only show levels that exist in the data
      scale_y_discrete(drop = TRUE) +
      labs(
        x = "Percentage (%)",
        y = NULL,
        fill = "Vaccination Status",
        title = title %||% paste("Vaccination Rate by", gsub("_", " ", col))
      )
    
  } else {
    # Continuous → boxplot branch
    ggplot(data, aes(
      x = factor(!!sym(target), levels = c(0, 1),
                 labels = c("Not Vaccinated", "Vaccinated")),
      y = !!sym(col),
      fill = factor(!!sym(target), levels = c(0, 1))
    )) +
      geom_boxplot(
        alpha = 0.8,
        outlier.color = "#555555",
        outlier.alpha = 0.5
      ) +
      scale_fill_manual(values = vaccine_colors) +
      labs(
        x = NULL,
        y = gsub("_", " ", col),
        fill = "Vaccination Status",
        title = title %||% paste(
          "Distribution of", gsub("_", " ", col), "by Vaccination Status"
        )
      ) +
      theme(legend.position = "none")
  }
}

# 6. create & display your grid of H1N1 vs seasonal plots
h1n1_plots     <- list()
seasonal_plots <- list()

cols_to_plot <- c(
  "h1n1_concern",
  "h1n1_knowledge",
  "opinion_h1n1_vacc_effective",
  "age_group",
  "opinion_seas_risk",
  "education"
)

#Better naming of variable here
for (col in cols_to_plot) {
  pretty <- tools::toTitleCase(gsub("_"," ",col))
  t1 <- paste("H1N1 Vaccination Rate by", pretty)
  t2 <- paste("Seasonal Flu Vaccination Rate by", pretty)
  
  h1n1_plots[[col]]     <- vaccination_rate_plot(col, "h1n1_vaccine", joined_df, t1)
  seasonal_plots[[col]] <- vaccination_rate_plot(col, "seasonal_vaccine", joined_df, t2)
}

# Here you can just pick a feature you are interested in looking
#since h1n1_plots is list
#up to you man
h1n1_list_plots <- (h1n1_plots$h1n1_concern + ggtitle("H1N1 Vaccination Rate by H1N1 Concern") +
                      h1n1_plots$h1n1_knowledge + ggtitle("H1N1 Vaccination Rate by H1N1 Concern")) /
  (h1n1_plots$opinion_h1n1_vacc_effective + ggtitle("H1N1 Vaccination Rate by Opinion on Effectiveness") +
     h1n1_plots$age_group ) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Key Factors Influencing H1N1 Vaccination",
    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
  )

h1n1_list_plots

seasonal_list_plots <- (seasonal_plots$h1n1_concern + ggtitle("Seasonal Vaccination Rate by H1N1 Concern") +
                             seasonal_plots$opinion_seas_risk + ggtitle("Seasonal Vaccination Rate by Opinion on Risk")) /
  (seasonal_plots$age_group + seasonal_plots$education) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Key Factors Influencing Seasonal Flu Vaccination",
    theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5))
  )
seasonal_list_plots

#------------------------------
# 12. DEEPER CHECK ON MISSSIGNESS
#------------------------------
data_train <- read_csv("Data/training_set_features.csv")
data_train_labels <- read_csv("Data/training_set_labels.csv")

# Merge training features and labels
df <- left_join(data_train, data_train_labels, by = "respondent_id")

skim(df)

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
df_types

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


# -----------------------------------------------------
# 12 - a) -. MISSING VALUES ANALYSIS
# -----------------------------------------------------
cb_palette <- c(
  "#E69F00",  # orange
  "#56B4E9",  # sky blue
  "#009E73",  # bluish green
  "#F0E442",  # yellow
  "#0072B2",  # blue
  "#D55E00",  # vermillion
  "#CC79A7"   # reddish purple
)

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
missing_summary %>% filter(missing > 0)

# Create a better missing data visualization
missing_plot <- ggplot(missing_summary %>% filter(missing > 0), 
                       aes(x = reorder(feature, missing), y = percent, fill = feature_type)) +
  geom_bar(stat = "identity") +
#  geom_text(aes(label = sprintf("%.1f%%", percent)), hjust = -0.1, size = 3) +
  scale_fill_manual(values = cb_palette[c(1, 3, 7)]) +
  coord_flip() +
  labs(
    title = "Missing Values by Feature",
    subtitle = "Percentage of missing values for each feature",
    x = "Feature",
    y = "Missing Values (%)",
    fill = "Feature Type"
  ) +
  theme(axis.text.y = element_text(size = 8)) +
  scale_x_discrete(labels = c(
    "respondent_id" = "Respondend ID",
    "h1n1_concern" = "H1N1 Concern",
    "h1n1_knowledge" = "H1N1 Knowledge",
    "behavioral_antiviral_meds" = "Antiviral Medication",
    "behavioral_avoidance" = "Avoidance",
    "behavioral_face_mask" = "Face Mask",
    "behavioral_wash_hands" = "Wash Hands",
    "behavioral_large_gatherings" = "Large Gatherings",
    "behavioral_outside_home" = "Outside Home",
    "behavioral_touch_face" = "Touch Face",
    "doctor_recc_h1n1" = "Doctor Recommendation H1N1",
    "doctor_recc_seasonal" = "Doctor Recommendation Seasonal",
    "chronic_med_condition" = "Chronical Medical Condition",
    "child_under_6_months" = "Child under 6 Months",
    "health_worker" = "Health Worker",
    "health_insurance" = "Health Insurance",
    "opinion_h1n1_vacc_effective" = "Opinion H1N1 Effect",
    "opinion_h1n1_risk" = "Opinion H1N1 Risk",
    "opinion_h1n1_sick_from_vacc" = "Opinion H1N1 sick from Vaccine",
    "opinion_seas_vacc_effective" = "Opinion Seasonal Effect",
    "opinion_seas_risk" = "Opinion Seasonal Risk",
    "opinion_seas_sick_from_vacc" = "Opinion Seasonal sick from Vaccine",
    "age_group" = "Age Group",
    "education" = "Education Level",
    "race" = "Race",
    "sex" = "Sex",
    "income_poverty" = "Income Level",
    "marital_status" = "Marital Status",
    "rent_or_own" = "Housing Situation",
    "employment_status" = "Employment Status",
    "hhs_geo_region" = "Geographical Region",
    "census_msa" = "Metropolitan Statistical Areas",
    "household_adults" = "Number of other Adults in Household",
    "household_children" = "Number of Children in Household",
    "employment_industry" = "Working Industry",
    "employment_occupation" = "Type of Occupation"
  ))

missing_plot
