#!/bin/bash
# Script de déploiement robuste avec fallback

set -e

echo "🚀 Déploiement robuste du Dashboard Souscriptions"

# Variables
APP_NAME="dashboard-app"
IMAGE_NAME="dashboard-souscriptions"
PORT="8501"

# Fonction de retry
retry_command() {
    local cmd="$1"
    local retries=3
    local delay=5
    
    for i in $(seq 1 $retries); do
        echo "🔄 Tentative $i/$retries..."
        if eval "$cmd"; then
            return 0
        fi
        
        if [ $i -lt $retries ]; then
            echo "⏳ Attente ${delay}s avant retry..."
            sleep $delay
            delay=$((delay * 2))  # Backoff exponentiel
        fi
    done
    
    return 1
}

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker non installé. Utilisation du mode Python direct..."
    ./test-local.sh
    exit 0
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker non démarré. Utilisation du mode Python direct..."
    ./test-local.sh
    exit 0
fi

echo "✅ Docker opérationnel"

# Arrêter conteneur existant
docker stop $APP_NAME 2>/dev/null || true
docker rm $APP_NAME 2>/dev/null || true

# Essayer différentes approches
echo "🐳 Tentative de build Docker..."
build_success=false

# Approche 1: Dockerfile Ubuntu (plus fiable)
if retry_command "docker build -f Dockerfile.ubuntu -t $IMAGE_NAME:latest ."; then
    echo "✅ Build réussi avec Dockerfile Ubuntu"
    build_success=true
# Approche 2: Dockerfile rapide
elif retry_command "docker build -f Dockerfile.fast -t $IMAGE_NAME:latest ."; then
    echo "✅ Build réussi avec Dockerfile rapide"
    build_success=true
# Approche 3: Dockerfile principal
elif retry_command "docker build -t $IMAGE_NAME:latest ."; then
    echo "✅ Build réussi avec Dockerfile principal"
    build_success=true
fi

if [ "$build_success" = true ]; then
    # Lancer le conteneur Docker
    echo "🚀 Lancement du conteneur..."
    docker run -d \
      --name $APP_NAME \
      -p $PORT:8501 \
      --restart unless-stopped \
      -v $(pwd)/data:/app/data \
      $IMAGE_NAME:latest

    echo "⏳ Attente du démarrage (30s)..."
    sleep 30

    if curl -f http://localhost:$PORT &>/dev/null; then
        echo "✅ Application Docker déployée avec succès!"
        echo "🌐 Accès: http://localhost:$PORT"
        docker logs $APP_NAME --tail 5
    else
        echo "⚠️ Conteneur lancé mais application pas encore prête"
        echo "🔍 Logs du conteneur:"
        docker logs $APP_NAME
        echo "🌐 Essayez: http://localhost:$PORT"
    fi
else
    echo "❌ Échec des builds Docker. Basculement vers Python direct..."
    ./test-local.sh
fi