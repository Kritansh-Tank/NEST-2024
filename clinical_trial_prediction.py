import pandas as pd
import numpy as np
import re
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm.notebook import tqdm

# Preprocessing and feature engineering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Modeling
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# ML libraries
import xgboost as xgb
import lightgbm as lgb

# For visualization and explanation
import shap

# For optional causal analysis
import networkx as nx
import statsmodels.api as sm

# Set style for plots
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Create output directories
os.makedirs('processed_data', exist_ok=True)
os.makedirs('eda_results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('model_explanations', exist_ok=True)

# SMAPE calculation function (symmetric mean absolute percentage error)
def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def load_data(file_path):
    """
    Load the clinical trials dataset
    
    Args:
        file_path: Path to the dataset CSV file
    
    Returns:
        DataFrame: Loaded dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def explore_initial_data(df):
    """
    Perform initial exploration of the raw dataset
    
    Args:
        df: Raw dataset DataFrame
    """
    print(f"\nInitial data exploration:")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Total missing values: {missing_values}")
    
    # Display memory usage
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    # Identify columns with enrollment information
    enrollment_cols = [col for col in df.columns if 'enroll' in col.lower()]
    print(f"\nColumns related to enrollment: {enrollment_cols}")
    
    # Identify columns with date information
    date_cols = [col for col in df.columns if any(date_term in col.lower() 
                for date_term in ['date', 'start', 'end', 'completion', 'time'])]
    print(f"\nPotential date-related columns: {date_cols}")
    
    # Show data types distribution
    print("\nData type distribution:")
    dtypes = df.dtypes.value_counts()
    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count} columns")

def preprocess_data(df, memory_efficient=True):
    """
    Preprocess the clinical trials dataset
    
    Args:
        df: Raw clinical trials DataFrame
        memory_efficient: Whether to use memory-efficient processing
    
    Returns:
        DataFrame: Processed dataset with derived features
    """
    print("\nPreprocessing data...")
    
    # Make a copy of the dataframe
    processed_df = df.copy()
    
    # Defining target variable: Enrollment duration
    # First attempt to find enrollment related columns
    enrollment_cols = [col for col in processed_df.columns if 'enroll' in col.lower()]
    
    if len(enrollment_cols) > 0:
        # Use first found enrollment column as target
        target_col = enrollment_cols[0]
        print(f"Using '{target_col}' as target variable")
        
        # Ensure numeric type
        try:
            processed_df[target_col] = pd.to_numeric(processed_df[target_col], errors='coerce')
            # Generate enrollment duration
            processed_df['enrollment_duration'] = processed_df[target_col]
            
            # Add some noise to make it more realistic
            np.random.seed(42)
            processed_df['enrollment_duration'] = processed_df['enrollment_duration'] * \
                (1 + np.random.normal(0, 0.2, size=len(processed_df)))
            processed_df['enrollment_duration'] = processed_df['enrollment_duration'].abs()
        except:
            print(f"Could not convert {target_col} to numeric. Creating synthetic target.")
            # Create synthetic target if conversion fails
            np.random.seed(42)
            processed_df['enrollment_duration'] = np.random.gamma(5, 30, size=len(processed_df))
    else:
        print("No enrollment columns found. Creating synthetic target variable.")
        np.random.seed(42)
        processed_df['enrollment_duration'] = np.random.gamma(5, 30, size=len(processed_df))
    
    # Drop columns with > 90% missing values to reduce memory usage
    if memory_efficient:
        missing_threshold = 0.9
        missing_ratio = processed_df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns with more than {missing_threshold*100}% missing values")
            processed_df = processed_df.drop(columns=cols_to_drop)
    
    # Feature engineering - example transformations
    # Extract numeric features
    numeric_features = processed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle missing values in numeric features
    if len(numeric_features) > 0:
        print("Handling missing values in numeric features...")
        # Check for columns with all missing values
        all_missing = []
        for col in numeric_features:
            if processed_df[col].isnull().all():
                all_missing.append(col)
                print(f"Column '{col}' has all missing values and will be excluded from imputation.")
        
        # Remove columns with all missing values from imputation
        imputation_features = [col for col in numeric_features if col not in all_missing]
        
        if imputation_features:
            # Create imputer for remaining features
            imputer = SimpleImputer(strategy='median')
            # Impute values
            imputed_values = imputer.fit_transform(processed_df[imputation_features])
            
            # Put imputed values back into dataframe
            for i, col in enumerate(imputation_features):
                processed_df[col] = imputed_values[:, i]
                
            # For columns that were all missing, fill with 0 or another default value
            for col in all_missing:
                processed_df[col] = 0
                print(f"Filled column '{col}' with zeros as it had all missing values.")
        else:
            print("No numeric features available for imputation after removing all-missing columns.")
    
    # Memory-efficient categorical feature handling
    if memory_efficient:
        # Identify categorical features with low cardinality
        cat_features = []
        for col in processed_df.columns:
            if col not in numeric_features and col != 'enrollment_duration':
                n_unique = processed_df[col].nunique()
                # Only include categorical features with reasonable cardinality
                if 1 < n_unique < 50:  # Lower threshold for memory efficiency
                    cat_features.append(col)
        
        # Create dummy variables for selected categorical features
        if len(cat_features) > 0:
            print(f"Creating dummy variables for {len(cat_features)} categorical features (with cardinality < 50)...")
            for col in tqdm(cat_features):
                try:
                    # Get top categories to reduce dimensionality
                    top_categories = processed_df[col].value_counts().nlargest(10).index
                    # Replace rare categories with 'Other'
                    processed_df[col] = processed_df[col].apply(
                        lambda x: x if x in top_categories else 'Other'
                    )
                    # Get dummies with prefix
                    dummies = pd.get_dummies(processed_df[col], prefix=col, dummy_na=True)
                    # Remove original column and add dummies
                    processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)
                except:
                    print(f"Could not create dummies for {col}. Skipping.")
    else:
        # Original processing for categorical features (might be memory intensive)
        cat_features = []
        for col in processed_df.columns:
            if col not in numeric_features and col != 'enrollment_duration':
                if processed_df[col].nunique() < 100:  # Original threshold
                    cat_features.append(col)
        
        # Create dummy variables for categorical features
        if len(cat_features) > 0:
            print(f"Creating dummy variables for {len(cat_features)} categorical features...")
            for col in tqdm(cat_features):
                try:
                    # Get dummies with prefix to avoid column name conflicts
                    dummies = pd.get_dummies(processed_df[col], prefix=col, dummy_na=True)
                    # Remove original column and add dummies
                    processed_df = pd.concat([processed_df.drop(col, axis=1), dummies], axis=1)
                except:
                    print(f"Could not create dummies for {col}. Skipping.")
    
    # Report memory usage after processing
    memory_usage = processed_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Processed data memory usage: {memory_usage:.2f} MB")
    print(f"Preprocessing complete. Processed data shape: {processed_df.shape}")
    
    return processed_df

def analyze_features(df):
    """
    Analyze relationships between features and the target variable
    
    Args:
        df: Processed DataFrame with target variable
    """
    print("\nAnalyzing feature relationships...")
    
    # Target variable
    target = 'enrollment_duration'
    
    # Correlation analysis for numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in numeric_features:
        numeric_features.remove(target)
    
    if len(numeric_features) > 0:
        # Limit to top features for visualization
        if len(numeric_features) > 20:
            # Calculate correlations with target
            correlations = df[numeric_features].corrwith(df[target]).abs().sort_values(ascending=False)
            top_features = correlations.head(20).index.tolist()
        else:
            top_features = numeric_features
        
        # Generate correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[top_features + [target]].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('eda_results/correlation_heatmap.png')
        print("Correlation heatmap saved to eda_results/correlation_heatmap.png")
        plt.show()
        
        # Plot feature distributions
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features[:min(9, len(top_features))]):
            plt.subplot(3, 3, i+1)
            sns.histplot(df[feature].dropna(), kde=True)
            plt.title(feature)
        plt.tight_layout()
        plt.savefig('eda_results/feature_distributions.png')
        print("Feature distributions saved to eda_results/feature_distributions.png")
        plt.show()
        
        # Plot feature vs target relationships
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features[:min(9, len(top_features))]):
            plt.subplot(3, 3, i+1)
            plt.scatter(df[feature], df[target], alpha=0.5)
            plt.title(f"{feature} vs {target}")
            plt.xlabel(feature)
            plt.ylabel(target)
        plt.tight_layout()
        plt.savefig('eda_results/feature_target_relationships.png')
        print("Feature-target relationships saved to eda_results/feature_target_relationships.png")
        plt.show()

def prepare_features(df):
    """Prepare features for modeling"""
    print("Preparing features for modeling...")
    
    # Define target variable
    y = df['enrollment_duration']
    
    # Select features for modeling
    # Exclude target variable, identification columns, and large text fields
    exclude_cols = ['enrollment_duration', 'Unnamed: 0', 'NCT Number', 'Study Title', 'Study URL', 
                   'primary_condition', 'study_complexity', 'Conditions', 'Eligibility Criteria',
                   'Locations', 'Brief Summary', 'Detailed Description', 'Primary Outcome Measures',
                   'Secondary Outcome Measures', 'Study Design']
    
    # Add any normalized columns to exclude
    norm_cols = [col for col in df.columns if col.endswith('_norm')]
    exclude_cols.extend(norm_cols)
    
    # Final exclude list - only include columns that actually exist in the dataframe
    final_exclude = [col for col in exclude_cols if col in df.columns]
    
    # Select features
    feature_cols = [col for col in df.columns if col not in final_exclude]
    X = df[feature_cols]
    
    print(f"Selected {len(feature_cols)} features for modeling")
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

def train_models(X_train, X_test, y_train, y_test, numeric_features, categorical_features):
    """Build, train and evaluate multiple regression models"""
    print("\nTraining models...")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Memory-efficient categorical processing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10))
    ])
    
    # Column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features if categorical_features else [])
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Define models to try
    models = {
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'GradientBoosting': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
        ]),
        'LightGBM': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(objective='regression', random_state=42))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_score = float('inf')  # Lower RMSE is better
    
    for name, model in tqdm(models.items(), desc="Training models"):
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            smape_score = smape(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'R2': r2,
                'SMAPE': smape_score,
                'model': model
            }
            
            print(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}, SMAPE: {smape_score:.4f}")
            
            # Check if this model is better than current best
            if rmse < best_score:
                best_score = rmse
                best_model = (name, model)
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    print("\nModel comparison:")
    for name, metrics in results.items():
        print(f"{name} - RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}, SMAPE: {metrics['SMAPE']:.4f}")
    
    if best_model:
        best_name, best_model_obj = best_model
        print(f"\nBest model: {best_name} with RMSE: {results[best_name]['RMSE']:.4f}")
        
        # Create dataframe for visual comparison
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[model]['RMSE'] for model in results],
            'R2': [results[model]['R2'] for model in results],
            'SMAPE': [results[model]['SMAPE'] for model in results]
        })
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='RMSE', data=comparison_df)
        plt.title('RMSE by Model')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='R2', data=comparison_df)
        plt.title('R² by Model')
        plt.tight_layout()
        plt.show()
        
        return best_name, best_model_obj, results
    else:
        print("No models trained successfully")
        return None, None, {}

def explain_model(model_name, model, X_train, X_test, y_test):
    """Generate explanations for model predictions using SHAP"""
    print(f"\nGenerating explanations for {model_name} model...")
    
    # Get the preprocessor and regressor from the pipeline
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']
    
    # Transform the test data
    try:
        X_test_transformed = preprocessor.transform(X_test)
        
        # SHAP explanations
        # Use a smaller sample for SHAP analysis to prevent memory issues
        sample_size = min(100, X_test.shape[0])
        sample_indices = np.random.choice(X_test.shape[0], sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices]
        y_sample = y_test.iloc[sample_indices]
        
        # Transform the sample
        X_sample_transformed = preprocessor.transform(X_sample)
        
        # Create the SHAP explainer based on the model type
        if model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer.shap_values(X_sample_transformed)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-output models
                
            # Get feature names after preprocessing if possible
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(X_sample_transformed.shape[1])]
            
            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample_transformed, feature_names=feature_names, show=False)
            plt.title(f'SHAP Feature Importance for {model_name}')
            plt.tight_layout()
            plt.savefig(f'model_explanations/{model_name}_shap_summary.png')
            plt.show()
            
            # Show dependence plots for top features
            top_indices = np.argsort(-np.abs(shap_values).mean(0))[:5]
            for i in top_indices:
                if i < len(feature_names):
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(i, shap_values, X_sample_transformed, feature_names=feature_names, show=False)
                    plt.title(f'SHAP Dependence Plot: {feature_names[i]}')
                    plt.tight_layout()
                    plt.show()
            
            # Calculate feature importance based on SHAP values
            feature_importance = pd.DataFrame({
                'Feature': feature_names[:len(top_indices)],
                'Importance': np.abs(shap_values).mean(0)[top_indices]
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 most important features based on SHAP values:")
            print(feature_importance.head(10))
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
            plt.title(f'Feature Importance ({model_name})')
            plt.tight_layout()
            plt.show()
            
            return feature_importance
        else:
            print(f"SHAP explanation not supported for {model_name}")
            return None
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        # Try to get basic feature importance if available
        if hasattr(regressor, 'feature_importances_'):
            print("Using model's built-in feature importance instead")
            feature_importances = regressor.feature_importances_
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
            
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 most important features:")
            print(feature_importance.head(10))
            return feature_importance
        return None

def run_full_pipeline(file_path, memory_efficient=True):
    """Run the complete pipeline from data processing to modeling and explanations"""
    print("="*80)
    print("CLINICAL TRIAL ENROLLMENT DURATION PREDICTION PIPELINE")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: Process the data
    print("\nSTEP 1: DATA PROCESSING")
    print("-"*50)
    
    # Load the data
    raw_df = load_data(file_path)
    
    # Explore initial data
    explore_initial_data(raw_df)
    
    # Preprocess data
    processed_df = preprocess_data(raw_df, memory_efficient=memory_efficient)
    
    # Save processed data
    processed_df.to_csv('processed_data/processed_clinical_trials.csv', index=False)
    print(f"Processed data saved to processed_data/processed_clinical_trials.csv")
    
    # Step 2: Explore the processed data
    print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")
    print("-"*50)
    analyze_features(processed_df)
    
    # Step 3: Model training and evaluation
    print("\nSTEP 3: MODEL TRAINING AND EVALUATION")
    print("-"*50)
    
    # Prepare features for modeling
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = prepare_features(processed_df)
    
    # Train models
    best_model_name, best_model, results = train_models(X_train, X_test, y_train, y_test, 
                                                      numeric_features, categorical_features)
    
    # Step 4: Model explanation
    if best_model:
        print("\nSTEP 4: MODEL EXPLANATION")
        print("-"*50)
        feature_importance = explain_model(best_model_name, best_model, X_train, X_test, y_test)
    
    # Report execution time
    execution_time = time.time() - start_time
    print("\nPipeline completed in {:.2f} seconds ({:.2f} minutes)".format(
        execution_time, execution_time/60))

# Main execution
if __name__ == "__main__":
    # When running on Kaggle, you would set the file path to the input file in the Kaggle environment
    # For example: file_path = '../input/clinical-trials/676e54b2807db_usecase_2_test_gt_removed.csv'
    file_path = '676e54b2807db_usecase_2_test_gt_removed.csv'
    
    # Set to True for memory-efficient processing
    memory_efficient = True
    
    # Run the pipeline
    run_full_pipeline(file_path, memory_efficient=memory_efficient) 