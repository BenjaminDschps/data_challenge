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
def load_data(
    path=".",
    file="Demandeurs_emploi_ moins_25.csv",
    formation_file="Formation_demandeur_emploi.csv",
    job_file="Offre_emploi.csv",
    manpower_file="Besoins_main_oeuvre.csv",
    recruitment_file="Recrutement_difficile.csv",
    test_size=0.2,
    random_state=42
):
    path = Path(path) / "data"
    
    # Charger les fichiers CSV
    df = pd.read_csv(path / file, sep=";", encoding="utf-8")  
    df_courses = pd.read_csv(path / formation_file, sep=";", encoding="utf-8")  
    df_job = pd.read_csv(path / job_file, sep=";", encoding="utf-8")  
    df_manpower = pd.read_csv(path / manpower_file, sep=";", encoding="utf-8")  
    df_recruitment = pd.read_csv(path / recruitment_file, sep=";", encoding="utf-8")  

    # Nettoyage des noms de colonnes pour éviter les erreurs
    df.columns = df.columns.str.replace('"', '').str.strip()
    df_courses.columns = df_courses.columns.str.replace('"', '').str.strip()
    df_job.columns = df_job.columns.str.replace('"', '').str.strip()

    # ======= TRAITEMENT DES DONNÉES PRINCIPALES =======
    # Renommer la colonne des mois pour plus de clarté
    df.rename(columns={'Mois': 'TIME_PERIOD'}, inplace=True)

    # Extraire et convertir l'année
    df['TIME_PERIOD'] = df['TIME_PERIOD'].str.extract(r'(\d{2})$')[0]
    df = df.dropna(subset=['TIME_PERIOD'])
    df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(int)
    df['TIME_PERIOD'] = df['TIME_PERIOD'].apply(lambda x: x + 1900 if x >= 90 else x + 2000)

    # Conversion en format long
    df = df.melt(id_vars=["TIME_PERIOD"], var_name="GEO", value_name="OBS_VALUE")
    df["GEO"] = df["GEO"].str.extract(r'(\d{2})$')
    df["OBS_VALUE"] = df["OBS_VALUE"].str.replace(r"[^\d.]", "", regex=True)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna()
    df = df.groupby(["TIME_PERIOD", "GEO"], as_index=False)["OBS_VALUE"].mean()

    # ======= TRAITEMENT DES FORMATIONS =======
    df_courses.rename(columns={'Mois': 'TIME_PERIOD'}, inplace=True)
    df_courses['TIME_PERIOD'] = df_courses['TIME_PERIOD'].str.extract(r'(\d{2})$')[0]
    df_courses = df_courses.dropna(subset=['TIME_PERIOD'])
    df_courses['TIME_PERIOD'] = df_courses['TIME_PERIOD'].astype(int)
    df_courses['TIME_PERIOD'] = df_courses['TIME_PERIOD'].apply(lambda x: x + 1900 if x >= 90 else x + 2000)

    df_long = df_courses.melt(id_vars=["TIME_PERIOD"], var_name="GEO", value_name="number_courses")
    df_long["GEO"] = df_long["GEO"].str.extract(r'(\d{2})$')
    df_long["number_courses"] = df_long["number_courses"].str.replace(r"[^\d.]", "", regex=True)
    df_long["number_courses"] = pd.to_numeric(df_long["number_courses"], errors="coerce")
    df_formation = df_long.groupby(["TIME_PERIOD", "GEO"], as_index=False)["number_courses"].mean()

    df = df.merge(df_formation, on=['GEO', 'TIME_PERIOD'], how='inner')

    # ======= TRAITEMENT DES OFFRES D'EMPLOI =======
    df_job.rename(columns={'Trimestre': 'TIME_PERIOD'}, inplace=True)
    df_job['TIME_PERIOD'] = df_job['TIME_PERIOD'].str.extract(r'(\d{2})$')[0]
    df_job = df_job.dropna(subset=['TIME_PERIOD'])
    df_job['TIME_PERIOD'] = df_job['TIME_PERIOD'].astype(int)
    df_job['TIME_PERIOD'] = df_job['TIME_PERIOD'].apply(lambda x: x + 1900 if x >= 90 else x + 2000)

    df_long = df_job.melt(id_vars=["TIME_PERIOD"], var_name="GEO", value_name="job_offer")
    df_long["GEO"] = df_long["GEO"].str.extract(r'(\d{2})$')
    df_long["job_offer"] = df_long["job_offer"].str.replace(r"[^\d.]", "", regex=True)
    df_long["job_offer"] = pd.to_numeric(df_long["job_offer"], errors="coerce")
    df_job_offer = df_long.groupby(["TIME_PERIOD", "GEO"], as_index=False)["job_offer"].mean()

    df = df.merge(df_job_offer, on=['GEO', 'TIME_PERIOD'], how='inner')

    # ======= TRAITEMENT DES NOUVEAUX DATASETS =======
    def process_transposed_data(df, value_name):
        """ Fonction pour traiter uniquement les fichiers à transposer """
        df = df.T.reset_index()
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

        df.rename(columns={'Département': 'TIME_PERIOD'}, inplace=True)

        df['TIME_PERIOD'] = df['TIME_PERIOD'].str.extract(r'(\d{2})$')[0]
        df = df.dropna(subset=['TIME_PERIOD'])
        df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(int)
        df['TIME_PERIOD'] = df['TIME_PERIOD'].apply(lambda x: x + 1900 if x >= 90 else x + 2000)

        df_long = df.melt(id_vars=["TIME_PERIOD"], var_name="GEO", value_name=value_name)
        df_long["GEO"] = df_long["GEO"].str.extract(r'(\d{2})$')
        df_long[value_name] = df_long[value_name].str.replace(r"[^\d.]", "", regex=True)
        df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")

        return df_long.groupby(["TIME_PERIOD", "GEO"], as_index=False)[value_name].mean()

    df_manpower = process_transposed_data(df_manpower, "need_for_manpower")
    df_recruitment = process_transposed_data(df_recruitment, "difficult_recruitment")

    # Fusionner les nouvelles features
    df = df.merge(df_manpower, on=['GEO', 'TIME_PERIOD'], how='inner')
    df = df.merge(df_recruitment, on=['GEO', 'TIME_PERIOD'], how='inner')

    # Charger le fichier CSV 
    path = Path('.') / "data"
    df_out = pd.read_csv(path / "Sortie_liste_France_Travail.csv", sep=";", encoding="utf-8")  # Load data


    # Renommer la colonne des trimestres pour plus de clarté
    df_out.rename(columns={'Trimestre': 'TIME_PERIOD'}, inplace=True)
    df_out.rename(columns={"Nombre de demandeurs d'emploi sortis": 'out_of_list'}, inplace=True)

    # Extraire l'année et convertir correctement
    df_out['TIME_PERIOD'] = df_out['TIME_PERIOD'].str.extract(r'(\d{2})$')[0]
    df_out = df_out.dropna(subset=['TIME_PERIOD'])  # Suppression des NaN
    df_out['TIME_PERIOD'] = df_out['TIME_PERIOD'].astype(int)  # Conversion en entier
    df_out['TIME_PERIOD'] = df_out['TIME_PERIOD'].apply(lambda x: x + 1900 if x >= 90 else x + 2000)


    # Convertir OBS_VALUE en numérique (au cas où il y a des espaces ou erreurs)
    df_out["out_of_list"] = df_out["out_of_list"].str.replace(r"[^\d.]", "", regex=True)  # Supprime espaces & caractères spéciaux
    df_out["out_of_list"] = pd.to_numeric(df_out["out_of_list"], errors="coerce")

    # Calculer la moyenne annuelle par département
    df_final = df_out.groupby(["TIME_PERIOD"], as_index=False)["out_of_list"].sum()
    df = df.merge(df_final, on="TIME_PERIOD", how="inner")

    # Charger le fichier CSV 
    path = Path('.') / "data"
    df_entry = pd.read_csv(path / "Entres_listes_France_Travail.csv", sep=";", encoding="utf-8")  # Load data


    # Renommer la colonne des trimestres pour plus de clarté
    df_entry.rename(columns={'Trimestre': 'TIME_PERIOD'}, inplace=True)
    df_entry.rename(columns={"Nombre de demandeurs d'emploi entrés": 'entry_on_list'}, inplace=True)

    # Extraire l'année et convertir correctement
    df_entry['TIME_PERIOD'] = df_entry['TIME_PERIOD'].str.extract(r'(\d{2})$')[0]
    df_entry = df_entry.dropna(subset=['TIME_PERIOD'])  # Suppression des NaN
    df_entry['TIME_PERIOD'] = df_entry['TIME_PERIOD'].astype(int)  # Conversion en entier
    df_entry['TIME_PERIOD'] = df_entry['TIME_PERIOD'].apply(lambda x: x + 1900 if x >= 90 else x + 2000)


    # Convertir OBS_VALUE en numérique (au cas où il y a des espaces ou erreurs)
    df_entry["entry_on_list"] = df_entry["entry_on_list"].str.replace(r"[^\d.]", "", regex=True)  # Supprime espaces & caractères spéciaux
    df_entry["entry_on_list"] = pd.to_numeric(df_entry["entry_on_list"], errors="coerce")

    # Calculer la moyenne annuelle par département
    df_final = df_entry.groupby(["TIME_PERIOD"], as_index=False)["entry_on_list"].sum()
    df = df.merge(df_final, on="TIME_PERIOD", how="inner")

    # Séparer les features et la cible
    y = df['OBS_VALUE']  # Variable cible
    X_df = df.drop(columns=['OBS_VALUE'])  # Features

    # Division en ensembles d'entraînement et de test
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
