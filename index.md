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