# 🧬 Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines

This repository contains code and documentation for the [DrivenData Flu Shot Learning competition](https://www.drivendata.org/competitions/66/flu-shot-learning/). The goal of the competition is to predict whether individuals received the H1N1 and seasonal flu vaccines during the 2009 flu season, based on demographic and behavioral features.

## 🧠 Competition Objective

In 2009, the H1N1 flu pandemic highlighted the challenge of vaccine uptake. This competition aims to develop machine learning models that can predict:

- Whether a person received the **H1N1 vaccine**
- Whether a person received the **seasonal flu vaccine**

These predictions can inform public health outreach and vaccination strategies in future scenarios.

## 🗃️ Dataset

The dataset is provided by the U.S. Centers for Disease Control and Prevention (CDC) and includes:

- **Training data (`training_set_features.csv` & `training_set_labels.csv`)**
- **Test data (`test_set_features.csv`)**
- **Sample submission (`submission_format.csv`)**

Each row represents an individual, and features include:

- Demographics (age group, education, income)
- Employment and household info
- Behavioral indicators (doctor visits, prior flu shot behavior)
- Health-related beliefs and concerns

## 🏁 Evaluation Metric

The competition uses **Area Under the ROC Curve (AUC-ROC)** as the evaluation metric, computed separately for each vaccine prediction and averaged.

## 📁 Repository Structure
```
├── Data/
│ ├── data/
│ ├── submission_format.csv
│ ├── test_set_features.csv
│ ├── training_set_features.csv
│ └── training_set_labels.csv
├── Eda/
│ ├── Basic_Analysis.Rmd
│ └── Paulo_Analysis.R
├── Model/
│ └── Model.Rmd
├── .Rhistory
├── Flu.Rproj
└── README.md
```

## 📓 Logbook:

17/05/2025

We held a discord  meeting at 10:00am. All  members attended. We discussed the dataset in general and decided that first step 
would be to take a deeper look at the individual variables and preparations.
A Google doc was set up in order for the members to collaborate on writing down ideas, tasks etc. 
Going forward, each member will take a look at their task  and we'll meet again on Monday (19/05/2025) to discuss the progress.
