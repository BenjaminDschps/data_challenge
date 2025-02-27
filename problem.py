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
def load_data(path='.', file='DS_RP_EMPLOI_LR_PRINC.csv', diplome_file='DS_RP_DIPLOMES.csv', satisfaction_file='DS_SRCV_SATISFACTION.csv', test_size=0.2, random_state=42):
    path = Path(path) / "data"
    df = pd.read_csv(path / file, sep=";")  # Load employment data
    df_diplomes = pd.read_csv(path / diplome_file, sep=";")  # Load diploma data
    df_satisfaction = pd.read_csv(path / satisfaction_file, sep=";")  # Load satisfaction data

    # Clean column names
    df.columns = df.columns.str.replace('"', '').str.strip()
    df_diplomes.columns = df_diplomes.columns.str.replace('"', '').str.strip()
    df_satisfaction.columns = df_satisfaction.columns.str.replace('"', '').str.strip()

    # Convert OBS_VALUE to float (handling comma as decimal separator)
    df['OBS_VALUE'] = df['OBS_VALUE'].str.replace(',', '.').astype(float)
    df_diplomes['OBS_VALUE'] = df_diplomes['OBS_VALUE'].str.replace(',', '.').astype(float)
    df_satisfaction['OBS_VALUE'] = df_satisfaction['OBS_VALUE'].str.replace(',', '.').astype(float)

    # Keep only rows where EMPSTA_ENQ == "2"
    df = df[df['EMPSTA_ENQ'] == "2"]
    
    # Remove rows where SEX is different from "_T" using .query()
    df = df.query("SEX != '_T'")
    
    # Drop EMPSTA_ENQ column as it's now redundant
    df = df.drop(columns=['EMPSTA_ENQ'])
    
    # Process diploma data
    df_diplomes = df_diplomes[df_diplomes['SEX'] != '_T']  # Remove total SEX category
    df_diplomes_pivot = df_diplomes.pivot_table(values='OBS_VALUE', index=['GEO', 'TIME_PERIOD', 'SEX'], columns='EDUC', aggfunc='first')
    df_diplomes_pivot.columns = [f"EDUC_{col}" for col in df_diplomes_pivot.columns]
    df_diplomes_pivot.reset_index(inplace=True)

    # Merge employment data with diploma data
    df = df.merge(df_diplomes_pivot, on=['GEO', 'TIME_PERIOD', 'SEX'], how='left')

    # Process satisfaction data (keeping only relevant columns)
    df_satisfaction = df_satisfaction[['TIME_PERIOD', 'SRCV_SATISFNOTE', 'OBS_VALUE']]
    df_satisfaction_pivot = df_satisfaction.pivot_table(values='OBS_VALUE', index=['TIME_PERIOD'], columns='SRCV_SATISFNOTE', aggfunc='first')
    df_satisfaction_pivot.columns = [f"SATISF_{col}" for col in df_satisfaction_pivot.columns]
    df_satisfaction_pivot.reset_index(inplace=True)

    # Merge satisfaction data
    df = df.merge(df_satisfaction_pivot, on=['TIME_PERIOD'], how='left')

    # Split target variable
    y = df['OBS_VALUE']  # This is the target column
    X_df = df.drop(columns=['OBS_VALUE'])  # Drop target from features

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
