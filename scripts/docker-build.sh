#!/bin/bash
# Script de build Docker

set -e

# Variables
IMAGE_NAME="dashboard-souscriptions"
IMAGE_TAG="${1:-latest}"
REGISTRY="${REGISTRY:-}"

echo "🐳 Construction de l'image Docker..."

# Build de l'image
docker build -t $IMAGE_NAME:$IMAGE_TAG .

# Tagging pour le registry si défini
if [ ! -z "$REGISTRY" ]; then
    docker tag $IMAGE_NAME:$IMAGE_TAG $REGISTRY/$IMAGE_NAME:$IMAGE_TAG
    echo "✅ Image taguée pour le registry: $REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
fi

echo "✅ Image construite: $IMAGE_NAME:$IMAGE_TAG"

# Afficher la taille de l'image
docker images $IMAGE_NAME:$IMAGE_TAG