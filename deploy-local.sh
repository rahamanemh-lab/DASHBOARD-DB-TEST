#!/bin/bash
# Script de déploiement local pour tester

set -e

echo "🚀 Déploiement local du Dashboard Souscriptions"

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    echo "📝 Installez Docker Desktop depuis: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Vérifier si Docker daemon est en cours d'exécution
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon n'est pas démarré"
    echo "🚀 Démarrez Docker Desktop et réessayez"
    
    # Essayer de démarrer Docker sur macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🔄 Tentative de démarrage de Docker Desktop..."
        open -a Docker
        echo "⏳ Attendez que Docker Desktop démarre (icône baleine dans la barre de menu)"
        echo "   Puis relancez ce script"
    fi
    exit 1
fi

echo "✅ Docker est opérationnel"

# Variables
APP_NAME="dashboard-app"
IMAGE_NAME="dashboard-souscriptions"
PORT="8501"

# Arrêter et supprimer le conteneur existant
echo "🛑 Arrêt du conteneur existant..."
docker stop $APP_NAME 2>/dev/null || true
docker rm $APP_NAME 2>/dev/null || true

# Essayer de télécharger l'image Python d'abord
echo "📥 Téléchargement de l'image Python (peut prendre du temps)..."
if ! docker pull python:3.9-slim; then
    echo "❌ Échec du téléchargement de l'image Python"
    echo "🔄 Vérifiez votre connexion internet et réessayez"
    echo "💡 Ou utilisez le script test-local.sh pour tester sans Docker"
    exit 1
fi

# Construire l'image avec timeout plus long
echo "🐳 Construction de l'image Docker..."
if ! timeout 300 docker build -t $IMAGE_NAME:latest .; then
    echo "❌ Échec de la construction Docker (timeout 5min)"
    echo "💡 Essayez à nouveau ou utilisez test-local.sh"
    exit 1
fi

# Lancer le conteneur
echo "🚀 Lancement du conteneur..."
docker run -d \
  --name $APP_NAME \
  -p $PORT:8501 \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  $IMAGE_NAME:latest

# Attendre que l'application démarre
echo "⏳ Attente du démarrage (30s)..."
sleep 30

# Vérifier que l'application fonctionne
echo "🔍 Vérification de santé..."
if curl -f http://localhost:$PORT/_stcore/health &>/dev/null || curl -f http://localhost:$PORT &>/dev/null; then
    echo "✅ Application déployée avec succès!"
    echo "🌐 Accès: http://localhost:$PORT"
    echo "📊 Logs récents:"
    docker logs $APP_NAME --tail 5
else
    echo "⚠️ L'application ne répond pas encore, mais le conteneur fonctionne"
    echo "🔍 Vérifiez les logs:"
    docker logs $APP_NAME
    echo "🌐 Essayez d'accéder à: http://localhost:$PORT"
fi