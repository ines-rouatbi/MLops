import argparse
import joblib
from model_pipeline import prepare_data, train_model, evaluate_model, save_model

# Initialiser globalement le modèle et les données
model = None
X_train, X_test, y_train, y_test, scaler = None, None, None, None, None

# Fonction pour sauvegarder les données préparées
def save_data(X_train, X_test, y_train, y_test, scaler):
    print("💾 Sauvegarde des données préparées...")
    joblib.dump((X_train, X_test, y_train, y_test, scaler), 'prepared_data.pkl')
    print("💾 Données sauvegardées avec succès.")

# Fonction pour charger les données préparées
def load_data():
    print("📥 Chargement des données préparées...")
    try:
        data = joblib.load('prepared_data.pkl')
        print("📥 Données chargées avec succès.")
        return data
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données : {e}")
        return None

# Fonction pour sauvegarder le modèle
def save_trained_model(model):
    print("💾 Sauvegarde du modèle...")
    joblib.dump(model, 'trained_model.pkl')
    print("💾 Modèle sauvegardé avec succès.")

# Fonction pour charger le modèle
def load_trained_model():
    print("📥 Chargement du modèle entraîné...")
    try:
        model = joblib.load('trained_model.pkl')
        print("📥 Modèle chargé avec succès.")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

def main():
    global model, X_train, X_test, y_train, y_test, scaler

    parser = argparse.ArgumentParser(description="Pipeline de machine learning")
    parser.add_argument('--prepare_data', action='store_true', help="Préparer les données")
    parser.add_argument('--train', action='store_true', help="Entraîner le modèle")
    parser.add_argument('--validate', action='store_true', help="Valider le modèle")

    args = parser.parse_args()

    # Étape 1: Préparation des données
    if args.prepare_data:
        print("📊 Chargement des données...")
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        save_data(X_train, X_test, y_train, y_test, scaler)  # Sauvegarde des données
        print(f"📊 Données préparées : {X_train.shape[0]} échantillons d'entraînement et {X_test.shape[0]} échantillons de test.")
        print("✅ Données d'entraînement prêtes pour l'entraînement.")

    # Étape 2: Entraînement du modèle
    if args.train:
        print("🚀 Entraînement du modèle...")
        data = load_data()  # Charger les données préparées
        if data is not None:
            X_train, X_test, y_train, y_test, scaler = data
            model = train_model(X_train, y_train)
            save_trained_model(model)  # Sauvegarder le modèle après l'entraînement
            print("🚀 Modèle entraîné avec succès.")
        else:
            print("❌ Erreur : Les données d'entraînement ne sont pas disponibles.")

    # Étape 3: Validation du modèle
    if args.validate:
        print("📈 Validation du modèle...")
        model = load_trained_model()  # Charger le modèle
        if model is not None:
            # Recharger également les données de test
            data = load_data()
            if data is not None:
                X_train, X_test, y_train, y_test, scaler = data  # Charger les données de test
                evaluate_model(model, X_test, y_test)  # Valider le modèle
                print("📈 Validation du modèle terminée avec succès.")
            else:
                print("❌ Erreur : Les données de test ne sont pas disponibles.")
        else:
            print("❌ Erreur : Le modèle n'a pas pu être chargé.")

    print("✅ Pipeline terminé avec succès !")

if __name__ == "__main__":
    main()
