Modular Preprocessing & Feature Engineering Pipeline
Project Overview
This repository contains a modular, configuration-driven preprocessing framework built using Python, Scikit-learn, and pandas. The goal of this project was to move away from hard-coded transformation scripts and instead create a reusable pipeline that ensures reproducibility and prevents data leakage during the machine learning lifecycle.

The architecture leverages Scikit-learn’s Pipeline and ColumnTransformer to handle complex datasets with diverse data types. By separating the transformation logic from the dataset schema, the pipeline can be reconfigured for different projects simply by updating a configuration dictionary.

Key Features
Automated Missing Value Imputation: Uses strategy-based imputation (mean/median) for numeric features and constant-value filling for categorical data.

Robust Scaling: Supports multiple scaling techniques, including StandardScaler and RobustScaler, to normalize numeric distributions.

Advanced Encoding: Implements OneHotEncoder with handle_unknown='ignore' and explicit missing-category tracking to ensure stability in production environments.

Extensible Design: The modular factory pattern allows for easy integration of custom transformers or new feature engineering steps.

Technologies Used
Python 3.8+

Scikit-learn: For core pipeline architecture.

Pandas & NumPy: For efficient data manipulation and array handling.
