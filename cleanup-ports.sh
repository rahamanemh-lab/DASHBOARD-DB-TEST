#!/bin/bash
# Script de nettoyage des ports

echo "🧹 Nettoyage des ports et processus..."

# Arrêter tous les processus Streamlit
echo "🛑 Arrêt des processus Streamlit..."
pkill -f streamlit || echo "Aucun processus Streamlit trouvé"

# Arrêter les conteneurs Docker
echo "🐳 Arrêt des conteneurs Docker..."
docker stop dashboard-app 2>/dev/null || echo "Aucun conteneur dashboard-app trouvé"
docker rm dashboard-app 2>/dev/null || echo "Aucun conteneur dashboard-app à supprimer"

# Vérifier les ports occupés
echo "🔍 Vérification des ports..."
if command -v lsof &> /dev/null; then
    echo "Processus utilisant le port 8501:"
    lsof -i :8501 || echo "Port 8501 libre"
    echo "Processus utilisant le port 8502:"
    lsof -i :8502 || echo "Port 8502 libre"
else
    echo "⚠️ lsof non disponible, utilisez 'netstat -tulpn | grep 850'"
fi

echo "✅ Nettoyage terminé"