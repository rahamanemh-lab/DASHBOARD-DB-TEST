#!/bin/bash

# Script de démarrage pour la production
# Avec configuration de taille de fichier élevée

echo "🚀 Démarrage du Dashboard Souscriptions en mode production"

# Variables d'environnement
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=2000
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Vérification de la configuration
echo "📋 Configuration:"
echo "  - Taille max upload: 2000MB"
echo "  - Port: 8501"
echo "  - Mode: Production"

# Démarrage avec paramètres explicites
streamlit run app.py \
  --server.maxUploadSize=2000 \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --browser.gatherUsageStats=false \
  --server.enableCORS=true \
  --server.enableXsrfProtection=false

echo "✅ Application démarrée"