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



## 📂 Project Structure & Navigation Guide

This repository is organized into several modules to keep the workflow clear and reproducible. Here’s a quick guide to help you navigate:

### 🔹 `/Data/`
- Contains the datasets used for modeling and analysis.  


### 🔹 `/eda/`
- Contains **Exploratory Data Analysis (EDA)** work.  
- Includes two R Markdown (`.Rmd`) files that walk through data cleaning, visualization, and key insights.

### 🔹 `/models/`
- Holds our **most promising models**.  
- Includes three R Rmd files with finalized model workflows.

### 🔹 `/report/`
- Contains the **final project report** in PDF format.  
- Summarizes methodology, results, and conclusions.

### 🔹 `/presentation/`
- Contains the project **presentation slides** used for presenting findings.

### 🔹 `/experiments/`
- Contains all experimental `.R` scripts.  
- These are raw experiments and tests, useful if you want to **reproduce or rerun** experiments on a server.

### 🔹 `/submission/`
- Tracks all submissions made during the project.  
- Includes structured logs and files submitted to the competition platform.

---

### 🚀 How to Use This Repo
1. **Start with** `/eda/` → to understand the dataset and initial findings.  
2. **Check `/models/`** → to see the main modeling approaches we tried.  
3. **Look at `/experiments/`** → if you want to dive deeper into  attempts or reproduce tuning runs on a server.  
4. **Read `/report/`** → for a polished summary of the entire project.  
5. **Open `/presentation/`** → for a concise overview of results.  
6. **Explore `/submission/`** → to acess our submissions files.  

---

This structure ensures that the project is **transparent, reproducible, and easy to follow** for both collaborators and external reviewers.  

## 📁 Repository Structure
```
├── Class notes/
│ ├── Best Practices (1).pdf
├── Data/
│ ├── data/
│ ├── submission_format.csv
│ ├── test_set_features.csv
│ ├── training_set_features.csv
│ └── training_set_labels.csv
├── Eda/
│ ├── Basic_Analysis.Rmd
│ └── best_exploration.R
│ └── explorative data analysis.R
├── Model/
│ └── Model.Rmd
│ └── ensemble-stack.R
│ └── testing-Model-Comparison
│ └── best-comparison.R
│ └── LightGBM-Workflow.R
│ └── catboost.R
│ └── testing.R
│ └── Comparison.R
│ └── base-model.R
│ └── bagged_trees_model.R
│ └── refined-dt.R
│ └── xgboost-model.R
│ └── random-forest-workflow-finalized.R
│ └── new-recipe/
│ └── ── recipes.R
│ └── ── bs-model.R
│ └── results/
│ └── ── xxx.rds
├── Submissions/
│ └── bagged_tree_workflow.csv
│ └── finalized_dt_workflow.csv
│ └── random_forest_workflow-2.csv
│ └── random_forest_workflow.csv
│ └── vaccine_predictions_submission-2.csv
│ └── vaccine_predictions_submission.csv
│ └── xgboost-workflow.csv
├── .DS_Store
├── .Rhistory
├── .gitignore
├── Flu.Rproj
└── README.md
```

## 📓 Logbook:

17/05/2025

We held a discord  meeting at 10:00am. All  members attended. We discussed the dataset in general and decided that first step 
would be to take a deeper look at the individual variables and preparations.
A Google doc was set up in order for the members to collaborate on writing down ideas, tasks etc. 
Going forward, each member will take a look at their task  and we'll meet again on Monday (19/05/2025) to discuss the progress.

19/05/2025

We held a discord  meeting at 10:00am. All  members attended. We discussed the following topics and presented our own approaches
- EDA: Distributions, missing values, outliers, and correlations 
- Cleaning & preprocessing
- Feature engineering
- Data leakage awareness
  
Going forward, each member will take a look will begin to merge their approach to the main branch.Our next task will be the creation of a baseline model  and we'll meet again on Friday (23/05/2025) to discuss the progress.

23/05/2025

We held a discord meeting at 1:40pm. All members attended. We discussed both of our baseline models and their scores based on the competetion website.
Both models achieved a score over 0.80. 
We looked at the models Accuracy, Precision, Recall, F1 Score and more, to find out where the model needs improvements.

Going forward, both members will try to improve the model with the better score.

13/06/2025

We held a discord meeting at 2:15pm. All members attended. We discussed our further plans.
Looking at the issues, we assigned each member equal tasks to do and finish before the 22nd.
After finishing all tasks, we will look at the models again.
In the next meeting with the professor, we will share our results and ask for feedback.

04/07/2025

Today we are going to optimize tunning spaces together.


23/07/2025
We held a meeting on Discord  at 3:00 PM with full attendance. During the meeting, we discussed our presentation, outlined our plan for presenting our findings, and made the final touches.


24/07/205

Presentation day.
Presentation day went well. 
Our fellow students gave us positive feedback, and our supervisor didn’t have any negative comments.
We decided to take a week and a half off to focus on exams.


06/08/205

We held a discord meeting at 3:00 pm. All members attended.
We went over the general feedback and made changes based on it. We also discussed what still needs to be done for the project submission and split up the workload.
We also consider improvement to our best model lightGBM.



## Submission Log

| Date/Time       | Description of Approach                    | H1N1 Pre-Tuning ROC | Seasonal Pre-Tuning ROC | H1N1 Post-Tuning ROC | Seasonal Post-Tuning ROC | Public Leaderboard Score | Team Member(s) |
|-----------------|--------------------------------------------|---------------------|-------------------------|----------------------|--------------------------|--------------------------|----------------|
| 2025-05-22 09:53| Logistic regression baseline(grid search tuning + default params)    | 0.804                    | 0.831                        | 0.823                      | 0.854                       | 0.8349                         | Nhi       |
| 2025-06-02 11:01| Decision trees (grid search tuning + default params)| 0.691               | 0.757                   | 0.804                | 0.827                    | 0.8097                       | Vanilton       |
| 2025-06-02 19:51| Bagged trees (grid search tuning + default params)         | 0.791               | 0.827                   | 0.822                | 0.848                    | 0.8328                   | Vanilton       |
| 2025-08-19 17:20| +XGBoost (baseline, default params)         | 0.745               | 0.772                   | 0.801                | 0.815                    | 0.759                    | Vanilton       |
| 2025-08-20 16:05| +XGBoost (grid + feature selection)         | 0.753               | 0.781                   | 0.822                | 0.838                    | 0.774                    | Vanilton, Nhi  |
| 2025-07-18 18:17| random forest (grid search tuning + mlr3 tunning params)  | 0.826               | 0.853                   | 0.828                | 0.854                    | 0.8393                    | Vanilton, Nhi       |
| 2025-07-20 15:49| LightGBM (changed data prep + feature eng(recipe)+ANOVA racing for hyperparameter tuning)   | 0.861               | 0.860                   | 0.864                | 0.862                    | 0.8623                    | Vanilton, Nhi  |
| 2025-07-23 09:38| Log regression baseline(grid search tuning + mlr3 tunning params)| 0.839               | 0.846                   | 0.854                | 0.859                    | 0.8542                    | Vanilton, Nhi            |
| 2025-08-04 15:50| LightGBM (changed seed for splitting data)   | 0.872               | 0.865                   | 0.875                | 0.867                    | 0.8625                    | Vanilton, Nhi  |


## 🏆 Leaderboard Ranks

| Date/Time       | Model                        | Rank |
|-----------------|------------------------------|------|
| 2025-05-22 09:53| Logistic Regression (baseline model)| 1364 |
| 2025-07-23 09:38| Logistic Regression (baseline model)|  836 |
| 2025-07-20 15:49| LightGBM                     |  256 |
| 2025-08-04 15:50| LightGBM                     |  230 |
