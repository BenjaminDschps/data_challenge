import rampwf as rw
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, ShuffleSplit

problem_title = 'Regression challenge on OBS_VALUE'

# Define Predictions as a regression task
Predictions = rw.prediction_types.make_regression()

# Define workflow
workflow = rw.workflows.Estimator()

# Define score types
score_types = [
    rw.score_types.RMSE(name='rmse', precision=4),
]

# Cross-validation strategy
def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)

# Function to load and process data
def load_data(path='.', file='DS_RP_EMPLOI_LR_PRINC.csv', test_size=0.2, random_state=42):
    path = Path(path) / "data"
    df = pd.read_csv(path / file, sep=";")  # Set correct separator

    # Clean column names
    df.columns = df.columns.str.replace('"', '').str.strip()

    # Convert OBS_VALUE to float (handling comma as decimal separator)
    df['OBS_VALUE'] = df['OBS_VALUE'].str.replace(',', '.').astype(float)

    # Remove rows where EMPSTA_ENQ == "_T"
    df = df[df['EMPSTA_ENQ'] != "_T"]
    
    # Remove rows where SEX is different from "_T" using .query()
    df = df.query("SEX != '_T'")

    # Pivot values of EMPSTA_ENQ into new columns with corresponding OBS_VALUE
    empsta_pivot = df.pivot_table(values='OBS_VALUE', index=['GEO', 'TIME_PERIOD'], columns='EMPSTA_ENQ', aggfunc='first')

    # Rename columns with meaningful names
    empsta_pivot.columns = [f"EMPSTA_ENQ_{col}" for col in empsta_pivot.columns]

    # Merge back to the original dataframe (only where EMPSTA_ENQ == 2)
    df = df[df['EMPSTA_ENQ'] == '2']
    df = df.merge(empsta_pivot, on=['GEO', 'TIME_PERIOD'], how='left')

    # Drop the original EMPSTA_ENQ and OBS_VALUE columns
    df = df.drop(columns=['EMPSTA_ENQ', 'OBS_VALUE'])

    # Split target variable
    y = df['EMPSTA_ENQ_2']  # This is the new target column
    X_df = df.drop(columns=['EMPSTA_ENQ_2'])  # Drop target from features

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# Get training data
def get_train_data(path='.'):
    X_train, _, y_train, _ = load_data(path)
    return X_train, y_train

# Get test data
def get_test_data(path='.'):
    _, X_test, _, y_test = load_data(path)
    return X_test, y_test
