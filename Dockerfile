# Dockerfile pour Dashboard Souscriptions
# Multi-stage build pour optimiser la taille de l'image
# ==============================================================================
# Stage 1: Builder - Installation des dépendances
# ==============================================================================
FROM python:3.12-slim AS builder

# Métadonnées
LABEL maintainer="raja.hamdi@570easi.com"
LABEL description="Dashboard Souscriptions - Application Streamlit"
LABEL version="1.0.0"

# Variables d'environnement pour Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installation des dépendances système nécessaires pour la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root pour la sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Créer et donner les permissions au répertoire home
RUN mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser

# Créer le répertoire de travail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements.txt .
USER appuser
RUN pip install --user --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Runtime - Image finale optimisée
# ==============================================================================
FROM python:3.12-slim AS runtime

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Installation des dépendances runtime uniquement
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Créer l'utilisateur non-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Créer les répertoires nécessaires
RUN mkdir -p /app /home/appuser/.streamlit \
    && chown -R appuser:appuser /app /home/appuser

# Copier depuis le bon chemin avec les bonnes permissions
COPY --from=builder --chown=appuser:appuser /home/appuser/.local /home/appuser/.local

# Changer vers l'utilisateur non-root
USER appuser

# Définir le répertoire de travail
WORKDIR /app

# Configuration Streamlit par défaut
RUN echo '[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
headless = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = "#1f77b4"\n\
backgroundColor = "#ffffff"\n\
secondaryBackgroundColor = "#f0f2f6"\n\
textColor = "#000000"' > /home/appuser/.streamlit/config.toml

# Copier le code de l'application
COPY --chown=appuser:appuser . .

# Créer les répertoires manquants si nécessaire
RUN mkdir -p logs data

# Variables d'environnement (injectées par GitLab CI/CD au runtime)
# ❌ AUCUNE VALEUR EN DUR
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Exposer le port Streamlit
EXPOSE 8501

# Health check pour vérifier que l'application fonctionne
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Point d'entrée par défaut
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]