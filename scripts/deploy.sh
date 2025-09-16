#!/bin/bash
# Script de déploiement manuel

set -e

echo "🚀 Déploiement du Dashboard Souscriptions"

# Variables d'environnement
SERVER_IP=${SERVER_IP:-"votre-ip-lightsail"}
SERVER_USER=${SERVER_USER:-"ubuntu"}
APP_PATH=${APP_PATH:-"/var/www/dashboard-souscriptions"}
SERVICE_NAME="dashboard-app"

# Vérification des variables
if [ -z "$SERVER_IP" ]; then
    echo "❌ Erreur: SERVER_IP non défini"
    exit 1
fi

# Synchronisation du code (exclure les fichiers inutiles)
echo "📦 Synchronisation du code..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='streamlit_env' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    ./ $SERVER_USER@$SERVER_IP:$APP_PATH/

# Installation des dépendances
echo "📋 Installation des dépendances..."
ssh $SERVER_USER@$SERVER_IP "cd $APP_PATH && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"

# Redémarrage du service
echo "🔄 Redémarrage du service..."
ssh $SERVER_USER@$SERVER_IP "sudo systemctl restart $SERVICE_NAME"

# Vérification
echo "✅ Vérification du déploiement..."
ssh $SERVER_USER@$SERVER_IP "sudo systemctl is-active $SERVICE_NAME"

echo "🎉 Déploiement terminé avec succès !"
echo "🌐 Application disponible sur votre domaine"