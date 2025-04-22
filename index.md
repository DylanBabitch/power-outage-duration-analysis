---
layout: default
title: Power Outage Duration Analysis
---

# Can you predict power outage duration?

### Dylan Babitch  
dbabitch@umich.edu

*Welcome to my project site…*

## The Dataset
The Dataset used for this analysis was from the Purdue University Laboratory for Advancing Sustainable Critical Infrastructure. It tracked major power outages in the United States from January 2000 to July 2016. Major outages are those with at least 50,000 customers impacted and/or with a load loss greater 300MW. It also tracked various pieces of information about the location the outage occured, information about energy prices and usage, and much more. This dataset provides crucial information for power companies and consumers that can help them predict future outages and their severity. The question that I will be trying to answer using this dataset is can you predict the length of a power outage?

The dataset has a total of 1534 rows, however some were removed throughout analysis due to them lacking crucial information needed to predict outage duration.

### The columns of the dataset are as follows[^1]:


| Column Name | Description |
| ----------- | ----------- |
| YEAR | Year the outage occured |
| MONTH | Month the outage occured |
| U.S._STATE | US State where the outage occured |
| POSTALCODE | The postal code of where the outage occured |
| NERC.REGION | The North American Electric Reliability Corporation's region that the outage occured |
| CLIMATE.REGION | The type of climate the outage occured, based on classifications from the National Centers for Environmental Information |
| ANOMALY.LEVEL | The oceanic El Niño/La Niña index |
| CLIMATE.CATEGORY | The climate episode corresponding to the year the outage took place |
| OUTAGE.START.DATE | The date the outage began |
| OUTAGE.START.TIME | The time the outage began |
| OUTAGE.RESTORATION.DATE | The date the outage was resolved |
| OUTAGE.RESTORATION.TIME | The time the outage was resolved |
| CAUSE.CATEGORY | The general category of event that caused the outage |
| CAUSE.CATEGORY.DETAIL | The specific category of the event that caused the outage |
| HURRICANE.NAMES | The name of the hurricane that caused the outage, if applicable |
| OUTAGE.DURATION | The length of the outage in minutes |
| DEMAND.LOSS.MW | The amount of peak demand lost during the outage in Megawatts |
| CUSTOMERS.AFFECTED | The number of customers impacted by the outage |
| POPULATION | The population of the state the outage occured in |
| POPPCT_URBAN | The percent of people who live in an urban area in the state the outage occured |

[^1]: There are more columns in the dataset that were ignored because they have little to no relevance to the analysis.

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
I began cleaning the data by creating two new columns, OUTAGE.START and OUTAGE.RESTORATION, that contained Pandas datetime objects that combined the OUTAGE.START.DATE and OUTAGE.START.TIME column into one (and the same for restoration).

I then removed all the outages that lasted less than 5 minutes in total. This was an arbitrary choice I made, however I felt like it is a good cutout between not removing too many rows and also not looking at outages that have very little impact because they didn't last long enough. After removing these rows, I then divided the remaining rows in the OUTAGE.DURATION column by 60 to convert them into hours. 

Following this I made the CAUSE.CATEGORY.DETAIL all in title case and then removed all the detailed causes with less than 3 occurances because they made the predictions more difficult.

### Univariate Analysis

### Bivariate Analysis

### Interesting Aggregates

### Imputation


## The Prediction Problem
The problem I will be trying to predict the duration of a power outage. This problem will involve regression because the duration of an outage is a continuous, numerical value. The variable I will be predicting is outage duration in minutes (the cleaned OUTAGE.DURATION column).

I will be using mean squared error to evaluate the model's performance because it is a good indicator of how off the predictions the model makes are from the actual duration of the outages. I chose it over other metrics like mean absolute error because I feel as though the impact of a bad predicition increases more quadratically rather than linearly. This is because a very far off prediction can greatly impact how much a consumer or company cares about a specific outage, causing them to over or under estimate how much they should prepare. 

## Baseline Model

I first started by creating my own custom Imputer that fills in the NA values of DEMAND.LOSS.MW that takes the mean for the other rows with the same CAUSE.CATEGORY

```python
    from sklearn.base import BaseEstimator, TransformerMixin

    class CategoryMeanImputer(BaseEstimator, TransformerMixin):
        def __init__(self, category_col, value_col):
            self.category_col = category_col
            self.value_col = value_col

        def fit(self, X, y=None):
            self.category_means_ = (
                X
                .groupby(self.category_col)[self.value_col]
                .mean()
                .to_dict()
            )
            self.global_mean_ = X[self.value_col].mean()
            return self

        def transform(self, X):
            X = X.copy()
            X[self.value_col] = X.apply(
                lambda row: (
                    self.category_means_.get(row[self.category_col], self.global_mean_)
                    if pd.isna(row[self.value_col])
                    else row[self.value_col]
                ), axis=1
            )
            return X
…

Then I created my model: 

```python
    from sklearn.pipeline import Pipeline
    from sklearn.compose import  ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error


    pipeline = Pipeline([
        ('imputer', CategoryMeanImputer('CAUSE.CATEGORY', 'DEMAND.LOSS.MW')),
        ('features', ColumnTransformer([
            ('detail_ohe', OneHotEncoder(handle_unknown='ignore'), ['CAUSE.CATEGORY.DETAIL']),
            ('mw', 'passthrough', ['DEMAND.LOSS.MW'])
        ], remainder='drop')), #I only want to use cause.category to impute demand.loss.mw and not in the final model
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train,y_train)
…


## Final Model

After tweaking the columns used and other factors, I arrived at my final model:

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.model_selection import GridSearchCV

qt = QuantileTransformer(
    n_quantiles=100,
    output_distribution="normal",
    random_state=98
)

pipeline = Pipeline([
    ('imputer', CategoryMeanImputer('CAUSE.CATEGORY', 'DEMAND.LOSS.MW')),
    ('features', ColumnTransformer([
        ('detail_ohe', OneHotEncoder(handle_unknown='ignore'), ['CAUSE.CATEGORY.DETAIL', 'CLIMATE.REGION']),
        ("quantile_loss", qt, ["DEMAND.LOSS.MW", 'POPPCT_URBAN'])
    ], remainder='drop')), #I only want to use cause.category to impute demand.loss.mw and not in the final model
    ('regressor', Ridge())
])



gridSearch = GridSearchCV(
    pipeline,
    param_grid = {'regressor__solver': ["auto","lsqr","sparse_cg","sag"], 
                  'regressor__max_iter': [100, 1000, 5000, 10000, 50000], 
                  'regressor__alpha': [0, 0.001, 0.01, 0.1, 1, 10]},
    scoring='neg_mean_squared_error'
)

gridSearch.fit(X_train,y_train)
…