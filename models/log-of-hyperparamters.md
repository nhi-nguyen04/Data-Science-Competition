This is a log of the parameters of all models and discarded experiments

# Workflow tuning


### Logistic Regression  Model

#### H1N1
Logistic Regression Model Specification (classification)

Main Arguments:
  penalty = 0.01
  
  mixture = 0.25

Computational engine: glmnet 


#### Seasonal
Logistic Regression Model Specification (classification)

Main Arguments:
  penalty = 0.01
  
  mixture = 0.5

Computational engine: glmnet 


### Random Forest Model

#### H1N1
Random Forest Model Specification (classification)

Main Arguments:
  mtry = 0
  
  trees = 857
  
  min_n = 40

Engine-Specific Arguments:
  importance = impurity
  
  sample.fraction = 0.843189760390669

#### Seasonal

Random Forest Model Specification (classification)

Main Arguments:
  mtry = 0
  
  trees = 1700
  
  min_n = 37

Engine-Specific Arguments:
  importance = impurity
  
  sample.fraction = 0.934276362112723

Computational engine: ranger 


### LightGBM Model


#### H1N1

Boosted Tree Model Specification (classification)

Main Arguments:
  trees = 1196
  
  tree_depth = 12
  
  learn_rate = 0.00824667780223134
  
  sample_size = 0.339640083117411

Engine-Specific Arguments:
  verbose = -1

Computational engine: lightgbm 


#### Seasonal
Boosted Tree Model Specification (classification)

Main Arguments:
  trees = 563
  
  tree_depth = 9
  
  learn_rate = 0.0128343052330767
  
  sample_size = 0.440978640690446

Engine-Specific Arguments:
  verbose = -1

Computational engine: lightgbm 


# Default tuning

### Bagged Decision Tree Model

#### H1N1
Bagged Decision Tree Model Specification (classification)

Main Arguments:
  cost_complexity = 3.14057139233517e-08
  
  tree_depth = 13
  
  min_n = 28

Engine-Specific Arguments:
  times = 20

Computational engine: rpart 

#### Seasonal

Bagged Decision Tree Model Specification (classification)

Main Arguments:
  cost_complexity = 3.82301573030166e-05
  
  tree_depth = 12
  
  min_n = 30

Engine-Specific Arguments:
  times = 20

Computational engine: rpart



### Decision Tree Model

#### H1N1

Decision Tree Model Specification (classification)

Main Arguments:
  cost_complexity = 1.56960964162831e-09
  
  tree_depth = 14
  
  min_n = 31

Computational engine: rpart 

#### Seasonal

Decision Tree Model Specification (classification)

Main Arguments:
  cost_complexity = 1.37747457204029e-05
  
  tree_depth = 10
  
  min_n = 32



### Random Forest Model

#### H1N1

Random Forest Model Specification (classification)

Main Arguments:
  mtry = 14
  
  trees = 1877
  
  min_n = 39

Engine-Specific Arguments:
  importance = impurity

Computational engine: ranger 

#### Seasonal


Random Forest Model Specification (classification)

Main Arguments:
  mtry = 11
  
  trees = 1438
  
  min_n = 36

Engine-Specific Arguments:
  importance = impurity

Computational engine: ranger 



### Logistic Regression  Model

#### H1N1

Logistic Regression Model Specification (classification)

Main Arguments:
  penalty = 0.000868511373751352
  
  mixture = 0.457142857142857

Computational engine: glmnet 

#### Seasonal

Logistic Regression Model Specification (classification)

Main Arguments:
  penalty = 0.00138949549437314
  
  mixture = 0.612244897959184

Computational engine: glmnet 