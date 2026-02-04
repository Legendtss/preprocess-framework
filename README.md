### Modular Preprocessing & Feature Engineering Pipeline

# Overview
This project implements a robust, configuration-driven preprocessing pipeline using Python, scikit-learn, and pandas. It is designed to transform raw datasets into machine-learning-ready formats while ensuring reproducibility and preventing data leakage.

By using a factory pattern to generate the pipeline, the transformation logic is decoupled from the dataset schema, allowing for easy extension and updates.

# Key Features
Modular Architecture: Built using ColumnTransformer to apply specific logic to numeric and categorical features independently.

Dynamic Configuration: All transformation steps, including scaling and imputation strategies, are managed via a single config object.

Missing Value Handling: Implements automated imputation for both numerical (mean/median) and categorical data (constant filling).

Feature Transformation: Includes standardized scaling for numerical values and One-Hot Encoding for categorical variables.

Production Ready: Uses handle_unknown='ignore' in encoding steps to ensure the pipeline does not crash on unseen data categories.

# Project Structure
pipeline_factory.py: The core module containing the function to build the Scikit-learn pipeline.

run_pipeline.py: An implementation script demonstrating the pipeline on a sample customer dataset.

requirements.txt: List of necessary libraries (pandas, scikit-learn, numpy).

Configuration Example
The pipeline is entirely driven by a dictionary schema, making it highly flexible:

Python

pipeline_cfg = {
    'numeric': {
        'cols': ['tenure_months', 'monthly_spend'],
        'impute_strategy': 'mean',
        'scaling': 'standard'
    },
    'categorical': {
        'cols': ['region', 'plan_type']
    }
} 
# Usage
Clone the repository and ensure you have the requirements installed.

Define your schema in the configuration dictionary.

Execute the pipeline:

Python

from pipeline_factory import create_preprocessing_pipeline

## Initialize
pipe = create_preprocessing_pipeline(pipeline_cfg)

## Fit and transform your raw data
processed_data = pipe.fit_transform(raw_df)
Technologies Used
Python 3.8+

Pandas: For data manipulation and DataFrame handling.

Scikit-Learn: For the Pipeline and ColumnTransformer architecture.

NumPy: For efficient numerical operations.
