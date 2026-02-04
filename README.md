Modular Preprocessing Pipeline
This is a Scikit-Learn based implementation of a feature engineering pipeline. The goal was to build a system that can be reconfigured for different datasets without touching the core transformation logic.

Key Design Choices
Config-Driven: I used a dictionary-based configuration so the pipeline can be updated via a JSON/YAML file in a production environment.
Leakage Prevention: By wrapping everything in a Pipeline object, we ensure that scaling parameters are calculated only on the training data.
Robust Categoricals: Set handle_unknown='ignore' in the encoder to prevent the system from crashing if the test set contains a category we haven't seen before.
How to Run
Install dependencies: pip install pandas scikit-learn numpy
Run the example script: python run_pipeline.py
Project Structure
pipeline_factory.py: Contains the logic for building the ColumnTransformer.
run_pipeline.py: A sample implementation using a mock customer dataset.
