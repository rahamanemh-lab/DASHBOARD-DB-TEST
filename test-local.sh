#!/bin/bash
# Script de test local sans Docker

set -e

echo "🚀 Test local du Dashboard Souscriptions (sans Docker)"

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    exit 1
fi

# Créer l'environnement virtuel s'il n'existe pas
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "🔄 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances
echo "📥 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Créer les dossiers nécessaires
mkdir -p data logs

# Lancer l'application
echo "🚀 Lancement de l'application..."
echo "🌐 L'application sera disponible sur: http://localhost:8501"
echo "🛑 Appuyez sur Ctrl+C pour arrêter"
streamlit run app.py --server.port=8501
