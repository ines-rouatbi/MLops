import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def prepare_data(train_path="churn.csv", test_path="churn.csv"):
    """
    Charge les fichiers Train et Test, applique encodage et normalisation.
    """
    print("📊 Chargement des données...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Vérification des types de données
    print("Types des colonnes du dataset d'entraînement:")
    print(train_df.dtypes)

    # Encodage des variables catégorielles
    train_df = pd.get_dummies(train_df, drop_first=True)
    test_df = pd.get_dummies(test_df, drop_first=True)

    # Alignement des colonnes des datasets
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    # Séparation des features et de la cible
    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    # Remplacement des valeurs NaN par la moyenne pour les données numériques
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    # Normalisation des données numériques
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"📊 Données préparées : {X_train.shape[0]} échantillons d'entraînement et {X_test.shape[0]} échantillons de test.")
    
    # Vérification que les données sont retournées
    print("Données d'entraînement (X_train) :", X_train.shape)
    print("Données de test (X_test) :", X_test.shape)

    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Entraîne un modèle Random Forest.
    """
    print("🚀 Entraînement du modèle avec", X_train.shape[0], "échantillons...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("🚀 Entraînement du modèle terminé.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle.
    """
    y_pred = model.predict(X_test)
    print(f"Précision: {accuracy_score(y_test, y_pred)}")
    print(f"Rapport de classification:\n{classification_report(y_test, y_pred)}")
    print(f"Matrice de confusion:\n{confusion_matrix(y_test, y_pred)}")

def save_model(model, filename="model.pkl"):
    """
    Sauvegarde le modèle entraîné dans un fichier.
    """
    joblib.dump(model, filename)

