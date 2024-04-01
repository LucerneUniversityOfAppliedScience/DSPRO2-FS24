import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load the dataset
    data = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

    # Prepare data
    X, y = prepare_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the pipeline
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    evaluate_model(pipeline, X_test, y_test)

    # Save the trained pipeline
    save_pipeline(pipeline)

def prepare_data(data):
    # Clean up column names
    data.columns = data.columns.str.strip()

    # Select features and target
    X = data[data.columns.difference(['stroke'])]
    y = data['stroke']

    return X, y

def create_data_preprocessor():
    # Select categorical and numerical columns
    numerical_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    # Preprocessing pipeline for numerical features
    numerical_processor = SimpleImputer(strategy="mean") # Fill missing values with mean

    # Preprocessing pipeline for categorical features
    categorical_processor = Pipeline(steps=[
        ('ordinal_encoder', OrdinalEncoder(dtype=np.int64,
                                           handle_unknown='use_encoded_value',
                                           unknown_value=-1)),
    ])

    # Combine the numerical and categorical processors using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical_processor', categorical_processor, categorical_columns),
            ('passthrough', numerical_processor, numerical_columns),
        ],
        remainder='drop', verbose_feature_names_out=False)
    return preprocessor

def create_pipeline():
    # Create a pipeline with preprocessing and model
    preprocessor = create_data_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', DecisionTreeClassifier(random_state=42))
    ])
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

def save_pipeline(pipeline):
    # Create the 'trained_model' folder if it doesn't exist
    if not os.path.exists('trained_model'):
        os.makedirs('trained_model')

    # Save the trained pipeline to the 'trained_model' folder
    pipeline_filename = os.path.join('trained_model', 'stroke_prediction_pipeline.pkl')
    joblib.dump(pipeline, pipeline_filename)
    print(f"Trained pipeline saved as {pipeline_filename}")

if __name__ == "__main__":
    main()
