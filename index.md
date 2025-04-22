---
layout: default
title: Power Outage Duration Analysis
---

# Can you predict power outage duration?

### Dylan Babitch  
dbabitch@umich.edu

*Welcome to my project site…*

## The Dataset
The Dataset used for this analysis was from the Purdue University Laboratory for Advancing Sustainable Critical Infrastructure. It tracked major power outages in the United States from January 2000 to July 2016. Major outages are those with at least 50,000 customers impacted and/or with a load loss greater 300MW. It also tracked various pieces of information about the location the outage occured, information about energy prices and usage, and much more.

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
| POPPCT_UC | The percent of people who live in a rural area in the state the outage occured |

[^1]: There are more columns in the dataset that were ignored because they have little to no relevance to the analysis.