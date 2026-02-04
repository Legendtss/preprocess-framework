import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler

def create_preprocessing_pipeline(config: dict):
    """
    Builds a Scikit-learn Pipeline based on a configuration schema.
    Uses 'cols' to define feature sets for transformation.
    """
    if not config:
        raise ValueError("No configuration provided.")

    transformations = []

    # Numeric Pipeline: Imputation + Scaling
    if 'numeric' in config:
        n_cfg = config['numeric']
        n_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy=n_cfg.get('impute_strategy', 'median'))),
            ('scaler', StandardScaler() if n_cfg.get('scaling') == 'standard' else RobustScaler())
        ])
        transformations.append(('num_proc', n_pipe, n_cfg['cols']))

    # Categorical Pipeline: Imputation + One-Hot Encoding
    if 'categorical' in config:
        c_cfg = config['categorical']
        c_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformations.append(('cat_proc', c_pipe, c_cfg['cols']))

    # Combine using ColumnTransformer. 'remainder' keeps non-specified cols like IDs.
    processor = ColumnTransformer(transformers=transformations, remainder='passthrough')
    
    return Pipeline(steps=[('preprocessor', processor)])