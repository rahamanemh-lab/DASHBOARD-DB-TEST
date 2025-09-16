#!/bin/bash
# Script de lancement Docker

set -e

IMAGE_NAME="dashboard-souscriptions"
IMAGE_TAG="${1:-latest}"
CONTAINER_NAME="dashboard-app"
PORT="${2:-8501}"

echo "🚀 Lancement du container Docker..."

# Arrêter le container existant s'il existe
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Lancer le nouveau container
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8501 \
    --restart unless-stopped \
    --health-cmd="curl -f http://localhost:8501/_stcore/health || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    $IMAGE_NAME:$IMAGE_TAG

echo "✅ Container lancé: $CONTAINER_NAME"
echo "🌐 Application disponible sur: http://localhost:$PORT"

# Afficher les logs
echo "📋 Logs du container:"
docker logs -f $CONTAINER_NAME