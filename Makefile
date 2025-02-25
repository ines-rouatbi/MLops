# Définition des variables
PYTHON = python3
VENV = venv
MAIN_SCRIPT = main.py
REQ_FILE = requirements.txt
MODEL_FILE = trained_model.pkl

# Installation des dépendances
install:
	@echo "📦 Installation des dépendances..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r $(REQ_FILE)

# Vérification du code (qualité et formatage)
lint:
	@echo "🔍 Vérification de la qualité du code..."
	$(VENV)/bin/python -m flake8 --max-line-length=100 --ignore=E203,W503 model_pipeline.py main.py

# Préparation des données
prepare:
	@echo "📊 Préparation des données..."
	$(VENV)/bin/python $(MAIN_SCRIPT) --prepare_data

# Entraînement du modèle
train:
	@echo "🚀 Entraînement du modèle..."
	$(VENV)/bin/python $(MAIN_SCRIPT) --train

# Validation du modèle
validate:
	@echo "📈 Validation du modèle..."
	$(VENV)/bin/python $(MAIN_SCRIPT) --validate

# Sauvegarde du modèle entraîné
save_model: train
	@echo "💾 Sauvegarde du modèle..."
	$(VENV)/bin/python -c "from model_pipeline import save_model, load_trained_model; model = load_trained_model(); save_model(model, '$(MODEL_FILE)')"
	@echo "💾 Modèle sauvegardé sous $(MODEL_FILE)"

# Nettoyage des fichiers temporaires
clean:
	@echo "🧹 Nettoyage des fichiers..."
	rm -f $(MODEL_FILE) prepared_data.pkl
	rm -rf __pycache__

# Démarrer l'API FastAPI
run_api:
	@echo "🚀 Démarrage de l'API FastAPI..."
	$(VENV)/bin/uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Aide
help:
	@echo "📘 Utilisez 'make' suivi de l'étape que vous voulez exécuter."
	@echo "Exemples :"
	@echo "  make prepare      # Prépare les données"
	@echo "  make train        # Entraîne le modèle"
	@echo "  make validate     # Valide le modèle"
	@echo "  make save_model   # Sauvegarde le modèle"
	@echo "  make clean        # Nettoie les fichiers temporaires"
