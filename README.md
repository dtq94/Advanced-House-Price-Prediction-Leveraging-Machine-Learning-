# Advanced Realestate Predictive Analysis: Leveraging Machine Learning

## Objective
The objective of this project is to predict sale prices for homes based on various characteristics such as location, total area, number of rooms, amenities, etc.

## Tech Stack
- Language: Python
- Libraries: scikit-learn, pandas, NumPy, matplotlib, seaborn, xgboost, TensorFlow

## Approach

### 1. Data Cleaning
- Import the required libraries and read the dataset.
- Preliminary exploration.
- Check for outliers and remove them.
- Drop redundant feature columns.
- Handle missing values.
- Regularize the categorical columns.
- Save the cleaned data.

### 2. Data Analysis
- Import the required libraries and read the cleaned dataset.
- Convert binary columns to dummy variables.
- Perform feature engineering.
- Conduct univariate and bivariate analysis.
- Check for correlation.
- Perform feature selection.
- Scale the data.
- Save the final updated dataset.

### 3. Model Building
- Prepare the data.
- Perform train-test split.
- Implement various regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regressor
  - Elastic Net
  - Random Forest Regressor
  - XGBoost Regressor
  - K-Nearest Neighbours Regressor
  - Support Vector Regressor

### 4. Model Validation
- Evaluate models using:
  - Mean Squared Error
  - R2 score
- Plot residuals.

### 5. Grid Search and Cross-Validation
- Perform grid search and cross-validation for each regressor.

### 6. Fitting and Predictions
- Fit the model and make predictions on the test data.

### 7. Feature Importance
- Check for feature importance.

### 8. Model Comparisons

### 9. MLP (Multi-Layer Perceptron) Models
- Implement MLP Regression with scikit-learn.
- Implement Regression with TensorFlow.

## Project Structure
```
|-- InputFiles
    -- Realestate_data.csv
|-- SourceFolder
    |-- ML_Pipeline
        -- model_evaluation.py
        -- data_cleaning.py
        -- fea_eng.py
        -- feat_sel.py
        -- mlp_model_evulation.py
        -- models.py
        -- mlp_model.py
        -- utils.py
        -- xgboost.py
        -- preprocessing.py
    |-- Engine.py
