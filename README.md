# Clinical Trial Enrollment Duration Prediction

## Overview
This notebook provides a complete pipeline for predicting enrollment duration in clinical trials. It has been optimized for running in the Kaggle environment with special attention to memory efficiency.

## Problem Statement
The goal is to predict how long patient enrollment will take for clinical trials based on trial characteristics. This information helps trial designers make better decisions about eligibility criteria, recruitment strategies, and overall trial design.

## Features
- **Memory-efficient processing**: Handles large clinical trial datasets without running out of memory
- **Comprehensive EDA**: Generates visualizations to understand relationships in the data
- **Multiple models**: Trains and compares RandomForest, GradientBoosting, XGBoost, and LightGBM
- **Model explainability**: Uses SHAP values to explain model predictions
- **Interactive visualizations**: All visualizations display directly in the notebook

## How to Use

### Step 1: Upload Data
Upload your clinical trials dataset to Kaggle. The expected format is a CSV file with various clinical trial characteristics, including at least one column related to enrollment (containing "enroll" in the column name).

### Step 2: Update File Path
In the main code section at the bottom of the notebook, update the file path to point to your dataset:

```python
# Replace with your dataset path
file_path = '../input/your-dataset-folder/your-file.csv'
```

### Step 3: Run the Notebook
Execute all cells to run the complete pipeline. The process includes:
1. Data loading and initial exploration
2. Data preprocessing with memory-efficient options
3. Feature engineering
4. Model training and evaluation
5. Model explanation with SHAP

### Memory Efficiency Options
The parameter `memory_efficient=True` enables several optimizations:
- Dropping columns with >90% missing values
- Limiting categorical features to those with <50 unique values
- Using only the top 10 categories for high-cardinality features
- Limiting one-hot encoding with `max_categories=10`

If your dataset is small or you have access to a high-memory Kaggle instance, you can set `memory_efficient=False` for more comprehensive modeling.

## Expected Outputs
The notebook generates:
- EDA visualizations showing feature distributions and relationships
- Model performance metrics (RMSE, R², SMAPE)
- Model comparison plots
- SHAP feature importance visualizations
- Dependence plots for the most influential features

## Requirements
All required libraries are included in the standard Kaggle environment:
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- xgboost, lightgbm
- shap

## Tips for Better Results
- Try setting `memory_efficient=False` if your dataset is small
- Adjust the `missing_threshold` parameter in the preprocessing function if needed
- Experiment with different models and parameters
- Check the dependence plots to understand how specific features affect predictions 
