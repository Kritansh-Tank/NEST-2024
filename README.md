# NEST-2024

## Problem Statement

Predicting Actual Enrollment Duration of clinical studies with explainability.

## Business context

While designing and writing a clinical trial protocol document, multiple empirical, scientific and medical references are used. Like any other hypothesis testing, it is helpful in gaining insights from similar historical research / experiments / clinical studies from the past to minimize chances of failure and improve predictability, quality & execution. Authors of protocol documents would benefit from accurately identifying the right criteria for patients and accelerating patient recruitment for the clinical trial. Time taken for actual enrollment prediction during protocol development will provide valuable insights into whether the criteria should be more restrictive or broad, given past clinical trials data (e.g., avoid overly restrictive criteria that hinder recruitment, but also avoid criteria so broad that they fail to adequately test drug efficacy). These insights can help avoid errors and improve the speed and efficiency of the clinical trial design, which remains a challenge for pharma sector. Delays in designing, errors etc. have a cascading impact on timelines for clinical trials. 

## Problem statement from technical lens

Predicting the actual enrolment duration (in months) of Completed "interventional" studies based on features (structured and unstructured) such as enrollment, study design, criteria, facility, country etc. Additionally, providing explainability that answers the reason for the prediction.

## Dataset

A smaller subset of 450,000 clinical trials data will be provided from clinicaltrials.gov (publicly available).

## Technical solution requirement

Utilizing AI & deep learning methods, obtain a solution such that the prediction of enrollment time taken based on Disease/Condition is backed by explainability (positively or negatively) affecting the magnitude. Causal inference approach will be given more points.

## Input (for example)

Condition, Phases, Facility location, Enrollment, Criteria, Study design, Study title, Intervention, etc. (various other features can be included)

## Output (for example)

30 months (predicted) with explainability – which features that positively and negatively influence the prediction.(Explainability: show the weightage of each feature on impact to make prediction)

## Metrics for Evaluation

Response variable: Time taken for Enrollment
RMSE (root mean squared error), R squared and adjusted R squared.
Symmetric Mean Absolute Percent Error (SMAPE)

## Additional evaluation

Exploratory data analysis, Data preparation/cleaning, feature engineering, modeling, evaluation using mentioned metrics and the model explainability (It is crucial that the model's explainability aligns with domain understanding of clinical trials); weighted more on model explainability. Important to identify the features used by the team.
