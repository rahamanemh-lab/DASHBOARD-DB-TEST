"""
Dashboard Commercial - Application principale
Version 2.0 avec configuration centralisée et amélirations
"""

import streamlit as st
import pandas as pd
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import calendar
import plotly.graph_objects as go
import numpy as np
from contextlib import contextmanager
import json
import requests
from typing import Optional, Dict, Any

# =============================================================================
# SECTION CRYPTO - TEST CONNEXION LIGHTSAIL
# =============================================================================
import os
import pandas as pd
import pymysql
import streamlit as st

st.set_page_config(page_title="Crypto Lightsail (SSL)", layout="wide")
st.title("🔐 Connexion SSL à Lightsail MySQL")

def get_db_conf():
    # 1) Streamlit Cloud secrets
    if "database" in st.secrets:
        cfg = st.secrets["database"]
        return {
            "host": cfg["host"],
            "user": cfg["user"],
            "password": cfg["password"],
            "database": cfg["dbname"],
            "port": int(cfg.get("port", 3306)),
            "ca": cfg.get("ssl_ca", None)  # chemin optionnel du CA si fourni
        }
    # 2) Fallback: env vars (local)
    return {
        "host": os.getenv("DB_HOST"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME", "crypto_datalake"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "ca": os.getenv("DB_SSL_CA")  # ex: ./certs/rds-combined-ca-bundle.pem
    }

cfg = get_db_conf()
if not all([cfg["host"], cfg["user"], cfg["password"], cfg["database"]]):
    st.error("Config manquante: host/user/password/database")
    st.stop()

# SSL kwargs: avec CA si dispo, sinon SSL rapide (sans vérification)
ssl_kwargs = {"ssl": {"ca": cfg["ca"]}} if cfg.get("ca") else {"ssl": {"ssl": {}}}

# Connexion + ping
try:
    conn = pymysql.connect(
        host=cfg["host"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        port=cfg["port"],
        charset="utf8mb4",
        connect_timeout=10,
        **ssl_kwargs
    )
    with conn.cursor() as c:
        c.execute("SELECT NOW(), USER(), CURRENT_USER(), VERSION()")
        now, user, cur_user, ver = c.fetchone()
    st.success(f"✅ SSL OK — NOW={now} | USER()={user} | CURRENT_USER()={cur_user} | MySQL={ver}")
except Exception as e:
    st.error(f"❌ Connexion échouée : {e}")
    st.stop()


def render_crypto_test_section():
    # Lecture + graphe
    df = pd.read_sql("""
        SELECT ts_utc AS ts, asset, fiat, price
        FROM crypto_prices
        WHERE ts_utc >= NOW() - INTERVAL 1 DAY
        ORDER BY ts_utc DESC
        LIMIT 500
    """, conn)
    conn.close()
    
    if df.empty:
        st.warning("Aucune ligne dans crypto_prices (24h).")
    else:
        st.dataframe(df.head(20), use_container_width=True)
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts","price"]).sort_values(["ts","asset"])
        pivot = (
            df.groupby(["ts","asset"], as_index=False)["price"].mean()
              .pivot(index="ts", columns="asset", values="price")
              .sort_index()
        )
        st.subheader("📈 Prix par asset (dernières 24h)")
        st.line_chart(pivot)



# =============================================================================

# =============================================================================
def make_epargne_from_crypto(df_crypto: pd.DataFrame) -> pd.DataFrame:
    """
    Map crypto -> schéma ÉPARGNE attendu par le dashboard.
    Colonnes produites : 
      - Date de souscription (datetime)
      - Montant (float)
      - Montant du placement (float)  # utile pour 'Retard cumulé'
      - Conseiller (str)
      - Produit (str)
      - Statut/Étape (str)
    """
    if df_crypto is None or df_crypto.empty:
        return pd.DataFrame(columns=[
            "Date de souscription","Montant","Montant du placement","Conseiller","Produit","Statut/Étape"
        ])
    df = df_crypto.copy()
    df["Date de souscription"] = pd.to_datetime(df["ts"], errors="coerce")  # ts vient de ts_utc AS ts
    df["Montant"] = pd.to_numeric(df["price"], errors="coerce")
    df["Montant du placement"] = df["Montant"]  # pour la section 'Retard cumulé' :contentReference[oaicite:3]{index=3}
    df["Conseiller"] = "Demo (Lightsail)"
    df["Produit"] = df["asset"].astype(str).str.upper()
    df["Statut/Étape"] = "Validé"
    out = df[[
        "Date de souscription","Montant","Montant du placement","Conseiller","Produit","Statut/Étape"
    ]].dropna(subset=["Date de souscription","Montant"])
    return out


def make_immo_from_crypto(df_crypto: pd.DataFrame) -> pd.DataFrame:
    """
    Map crypto -> schéma IMMOBILIER attendu par le dashboard.
    Colonnes produites :
      - Date de création (datetime)
      - Conseiller (str)
      - Statut (str)
      - Montant (float)
      - Type de bien (str)
    """
    if df_crypto is None or df_crypto.empty:
        return pd.DataFrame(columns=[
            "Date de création","Conseiller","Statut","Montant","Type de bien"
        ])
    df = df_crypto.copy()
    df["Date de création"] = pd.to_datetime(df["ts"], errors="coerce")
    df["Montant"] = pd.to_numeric(df["price"], errors="coerce")
    df["Conseiller"] = "Demo (Lightsail)"
    df["Statut"] = "En cours"
    # on “réutilise” asset comme pseudo-type
    df["Type de bien"] = df["asset"].astype(str).str.upper()
    out = df[["Date de création","Conseiller","Statut","Montant","Type de bien"]].dropna(subset=["Date de création","Montant"])
    return out


# =============================================================================

#S3 CONFIG

# =============================================================================
import boto3
from io import BytesIO
from urllib.parse import urlparse

@st.cache_resource(ttl=300, show_spinner=False)
def _get_s3_client():
    # lit d’abord dans st.secrets, sinon variables d’env
    region = (st.secrets.get("s3", {}) or {}).get("region") or os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
    aws_id = (st.secrets.get("s3", {}) or {}).get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    aws_sk = (st.secrets.get("s3", {}) or {}).get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    params = {"region_name": region}
    if aws_id and aws_sk:
        params.update({"aws_access_key_id": aws_id, "aws_secret_access_key": aws_sk})
    return boto3.client("s3", **params)

def _s3_conf():
    # récupère bucket + mapping des préfixes
    s3sec = st.secrets.get("s3", {}) or {}
    return {
        "bucket": s3sec.get("bucket") or os.getenv("S3_BUCKET"),
        "region": s3sec.get("region") or os.getenv("AWS_DEFAULT_REGION", "eu-north-1"),
        "prefixes": {
            "epargne": s3sec.get("epargne_prefix") or os.getenv("S3_EPARGNE_PREFIX", "inputs/epargne/"),
            "immo": s3sec.get("immo_prefix") or os.getenv("S3_IMMO_PREFIX", "inputs/immo/"),
            "rdv": s3sec.get("rdv_prefix") or os.getenv("S3_RDV_PREFIX", "inputs/rdv/"),
            "clients": s3sec.get("clients_prefix") or os.getenv("S3_CLIENTS_PREFIX", "inputs/clients/"),
            "entretiens_epargne": s3sec.get("entretiens_epargne_prefix") or os.getenv("S3_ENT_EP_PREFIX", "inputs/entretiens_epargne/"),
            "entretiens_immo": s3sec.get("entretiens_immo_prefix") or os.getenv("S3_ENT_IMMO_PREFIX", "inputs/entretiens_immo/"),
            "paiements_epargne": s3sec.get("paiements_epargne_prefix") or os.getenv("S3_PAY_EP_PREFIX", "inputs/paiements_epargne/"),
            "paiements_immo": s3sec.get("paiements_immo_prefix") or os.getenv("S3_PAY_IMMO_PREFIX", "inputs/paiements_immo/"),
            "analyse_2025": s3sec.get("analyse_2025_prefix") or os.getenv("S3_2025_PREFIX", "inputs/analyse_2025/"),
        },
        "extensions": (s3sec.get("extensions") or "xlsx,csv").split(",")
    }

@st.cache_data(ttl=120, show_spinner=False, hash_funcs={"botocore.client.S3": lambda _: None})
def s3_latest_key(bucket: str, prefix: str, allowed_ext=("xlsx","xls","csv")) -> str | None:
    s3 = _get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    latest = None
    latest_ts = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if any(key.lower().endswith("."+ext.lower()) for ext in allowed_ext):
                ts = obj["LastModified"]
                if latest_ts is None or ts > latest_ts:
                    latest, latest_ts = key, ts
    return latest

@st.cache_data(ttl=120, show_spinner=False, hash_funcs={"botocore.client.S3": lambda _: None})
def s3_read_excel_or_csv(bucket: str, key: str) -> pd.DataFrame | None:
    s3 = _get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    if key.lower().endswith((".xlsx",".xls")):
        return pd.read_excel(BytesIO(data))
    elif key.lower().endswith(".csv"):
        # essai auto d'encodage ; ajuste si besoin
        try:
            return pd.read_csv(BytesIO(data))
        except Exception:
            return pd.read_csv(BytesIO(data), encoding="utf-8", sep=";")
    return None

def auto_load_from_s3(kind: str, description: str = "") -> pd.DataFrame | None:
    cfg = _s3_conf()
    bucket = cfg["bucket"]
    if not bucket:
        st.warning("S3 non configuré (bucket manquant).")
        return None
    prefix = cfg["prefixes"].get(kind)
    if not prefix:
        st.warning(f"Préfixe S3 manquant pour {kind}.")
        return None
    key = s3_latest_key(bucket, prefix, cfg["extensions"])
    if not key:
        st.info(f"Aucun fichier trouvé sur S3 sous `{prefix}` pour {kind}.")
        return None
    with st.spinner(f"📥 Chargement S3 ({description or kind})…"):
        df = s3_read_excel_or_csv(bucket, key)
    if df is not None:
        st.success(f"✅ Chargé depuis S3: s3://{bucket}/{key}")
    return df






# =============================================================================

# ⚠️ CECI DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
# st.set_page_config(
#     page_title="Dashboard Souscriptions",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Configuration du path pour les imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import du système de configuration
try:
    from utils.config_loader import config, get_config, get_theme_colors, get_chart_config
    CONFIG_LOADED = True
except ImportError:
    logging.warning("Configuration loader non disponible, utilisation des valeurs par défaut")
    CONFIG_LOADED = False
    
# Configuration des objectifs
OBJECTIF_MENSUEL_EPARGNE = 1_830_000
OBJECTIF_ANNUEL_EPARGNE = 22_000_000  # 22M€ objectif annuel

# Import des modules d'analyse
from analyses.epargne import analyser_souscriptions_epargne
from analyses.souscription_epargne import analyser_pipe_collecte_epargne
from analyses.epargne_groupe import analyser_groupes_epargne, analyser_performance_conseillers_epargne
from analyses.immo import analyser_suivi_immo
from analyses.immo_status import analyser_statuts_dossiers_immo
from analyses.immo_groupe import analyser_groupes_dossiers_immo
from analyses.entretiens import analyser_entretiens
import analyses.entretiens
from analyses.rdv import analyser_rdv
from analyses.conversion import analyser_conversion
from analyses.parcours_client import analyser_parcours_client
from analyses.paiements import analyser_paiements
from analyses.clients import analyser_clients
from analyses.clients_integration import analyser_clients_integration

# Import des fonctions utilitaires
from utils.data_processing import read_excel_robust, safe_to_datetime



# ================================================================================
# Classe LLM Assistant
# ================================================================================

class LLMAssistant:
    """Assistant LLM intelligent pour l'analyse de données"""
    
    def __init__(self):
        self.available_models = {
            "openai": {
                "name": "OpenAI GPT-4",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "requires_key": True
            },
            "anthropic": {
                "name": "Claude 3.5 Sonnet", 
                "endpoint": "https://api.anthropic.com/v1/messages",
                "requires_key": True
            },
            "local": {
                "name": "Ollama Local",
                "endpoint": "http://localhost:11434/api/chat",
                "requires_key": False
            }
        }
        
        # Configuration par défaut
        self.selected_model = None
        self.api_key = None
        
    def configure_model(self, model_type: str, api_key: Optional[str] = None):
        """Configure le modèle LLM à utiliser"""
        if model_type in self.available_models:
            self.selected_model = model_type
            self.api_key = api_key
            return True
        return False
    
    def is_configured(self) -> bool:
        """Vérifie si le modèle est configuré"""
        if not self.selected_model:
            return False
        
        model_info = self.available_models[self.selected_model]
        if model_info["requires_key"] and not self.api_key:
            return False
            
        return True
    
    def analyze_data_with_llm(self, question: str, data_summary: Dict[str, Any]) -> str:
        """Analyse les données avec le LLM"""
        if not self.is_configured():
            return self._fallback_analysis(question, data_summary)
        
        try:
            # Construire le prompt intelligent
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(question, data_summary)
            
            # Appeler le modèle approprié
            if self.selected_model == "openai":
                return self._call_openai(system_prompt, user_prompt)
            elif self.selected_model == "anthropic":
                return self._call_anthropic(system_prompt, user_prompt)
            elif self.selected_model == "local":
                return self._call_ollama(system_prompt, user_prompt)
            else:
                return self._fallback_analysis(question, data_summary)
                
        except Exception as e:
            st.error(f"❌ Erreur LLM: {str(e)}")
            return self._fallback_analysis(question, data_summary)
    
    def _build_system_prompt(self) -> str:
        """Construit le prompt système"""
        return """Tu es un assistant IA expert en analyse de données financières et commerciales.

CONTEXTE: Tu analyses des données de souscriptions, ventes, et performance commerciale pour une entreprise financière.

DONNÉES DISPONIBLES: Tu recevras un résumé des données avec des métriques clés.

INSTRUCTIONS:
1. Réponds en français de manière professionnelle
2. Utilise des emojis pour structurer tes réponses (📊 💰 👥 📈 etc.)
3. Fournis des chiffres précis quand disponibles
4. Donne des insights et recommandations pertinentes
5. Sois concis mais informatif (max 300 mots)
6. Utilise des formats structurés avec des puces
7. Ajoute des recommandations d'actions si pertinent

STYLE: Direct, analytique, avec des insights business."""
    
    def _build_user_prompt(self, question: str, data_summary: Dict[str, Any]) -> str:
        """Construit le prompt utilisateur avec les données"""
        prompt = f"""QUESTION: {question}

RÉSUMÉ DES DONNÉES:
"""
        
        for key, value in data_summary.items():
            if isinstance(value, (int, float)):
                prompt += f"• {key}: {value:,}\n" if isinstance(value, int) else f"• {key}: {value:,.2f}\n"
            else:
                prompt += f"• {key}: {value}\n"
        
        prompt += "\nAnalyse ces données et réponds à la question avec des insights pertinents."
        return prompt
    
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Appelle l'API OpenAI"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o-mini",  # Modèle plus économique
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(self.available_models["openai"]["endpoint"], 
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error: {response.status_code}")
    
    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Appelle l'API Anthropic"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}]
        }
        
        response = requests.post(self.available_models["anthropic"]["endpoint"],
                               headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            raise Exception(f"API Error: {response.status_code}")
    
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Appelle Ollama en local"""
        data = {
            "model": "llama3.2",  # ou autre modèle local
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        response = requests.post(self.available_models["local"]["endpoint"],
                               json=data, timeout=60)
        
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            raise Exception(f"Ollama Error: {response.status_code}")
    
    def _fallback_analysis(self, question: str, data_summary: Dict[str, Any]) -> str:
        """Analyse de secours sans LLM"""
        return f"""🤖 **Analyse des données** (Mode basique)

📊 **Résumé des métriques disponibles:**
{chr(10).join([f"• {k}: {v}" for k, v in data_summary.items()])}

💡 **Note:** Pour des analyses plus intelligentes et contextuelles, configurez un modèle LLM dans les paramètres du chatbot.

Votre question: "{question}"
"""
    
    def test_model_connection(self, model_type: str, api_key: str = None) -> bool:
        """Teste la connexion au modèle LLM"""
        try:
            if model_type == "openai":
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Test"}],
                    "max_tokens": 5
                }
                response = requests.post(self.available_models["openai"]["endpoint"], 
                                       headers=headers, json=data, timeout=10)
                return response.status_code == 200
                
            elif model_type == "anthropic":
                headers = {
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                data = {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "Test"}]
                }
                response = requests.post(self.available_models["anthropic"]["endpoint"],
                                       headers=headers, json=data, timeout=10)
                return response.status_code == 200
                
            elif model_type == "local":
                # Test simple de ping pour Ollama
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                return response.status_code == 200
                
        except Exception:
            return False
        
        return False

class DashboardApp:
    """Classe principale du dashboard commercial"""
    
    def __init__(self):
        """Initialisation de l'application"""
        self.setup_logging()
        self.setup_streamlit_config()
        self.setup_theme()
        
        # Variables d'état pour les données
        self.data_files = {
            'df_epargne': None,
            'df_immo': None,
            'df_entretiens_epargne': None,
            'df_entretiens_immo': None,
            'df_rdv': None,
            'df_clients': None,
            'df_clients_analyse': None,  # Nouveau fichier pour l'analyse clients
            'df_analyse_2025': None,  # Nouveau fichier pour l'analyse 2025
            'df_paiements_epargne': None,
            'df_paiements_immo': None
        }
        
        # Initialize LLM Assistant
        self.llm_assistant = LLMAssistant()
        
        # Performance settings
        self.performance_mode = st.sidebar.checkbox(
            "⚡ Mode Performance", 
            value=True, 
            help="Active l'optimisation des performances (cache, lazy loading)"
        )
        
        # Variable pour stocker les messages de traitement des données
        if 'data_processing_messages' not in st.session_state:
            st.session_state.data_processing_messages = []
    
    def setup_logging(self):
        """Configuration du système de logging"""
        if CONFIG_LOADED:
            config.configure_logging()
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Application Dashboard Commercial démarrée")
    
    def setup_streamlit_config(self):
        """Configuration de Streamlit"""
        if CONFIG_LOADED:
            #config.configure_streamlit()
            logging.info("Configuration config_loader désactivée temporairement")
        else:
            # Configuration par défaut
            #st.set_page_config(
            #    page_title="Dashboard Commercial",
            #    page_icon="📊",
            #    layout="wide",
            #    initial_sidebar_state="expanded",
            #    menu_items={
            #        'Get Help': 'mailto:support@votredomaine.com',
            #        'Report a bug': 'https://gitlab.com/datanaly-group/dashboard-souscriptions/-/issues',
            #        'About': "Dashboard Commercial v2.0"
            #    }
            #)
            # Juste un message de log à la place
            logging.info("Configuration Streamlit par défaut utilisée")
    
    def setup_theme(self):
        """Configuration du thème et des styles CSS"""
        if CONFIG_LOADED:
            theme_colors = get_theme_colors()
            chart_config = get_chart_config()
        else:
            theme_colors = {
                'primary_color': '#1f77b4',
                'background_color': '#ffffff',
                'secondary_background_color': '#13c2c2',
                'text_color': '#000000',
                'success_color': '#51cf66',
                'warning_color': '#ffd43b',
                'error_color': '#ff6b6b'
            }
            chart_config = {'default_height': 400}
        
        # CSS personnalisé avec les couleurs du thème
        st.markdown(f"""
        <style>
            /* Styles principaux */
            .main-header {{
                color: {theme_colors.get('primary_color', '#1f77b4')};
                font-size: 3rem;
                text-align: center;
                margin-bottom: 2rem;
                font-weight: 700;
            }}
            
            .sub-header {{
                color: {theme_colors.get('primary_color', '#1f77b4')};
                font-size: 2rem;
                margin-bottom: 1rem;
                border-bottom: 2px solid {theme_colors.get('primary_color', '#1f77b4')};
                padding-bottom: 0.5rem;
            }}
            
            /* Cartes métriques */
            .metric-card {{
                background: linear-gradient(135deg, {theme_colors.get('secondary_background_color', '#13c2c2')}, #13c2c2);
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 4px solid {theme_colors.get('primary_color', '#1f77b4')};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }}
            
            /* Status indicators */
            .status-success {{
                background-color: {theme_colors.get('success_color', '#51cf66')};
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: bold;
                display: inline-block;
                margin: 0.2rem;
            }}
            
            .status-warning {{
                background-color: {theme_colors.get('warning_color', '#ffd43b')};
                color: #333;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: bold;
                display: inline-block;
                margin: 0.2rem;
            }}
            
            .status-error {{
                background-color: {theme_colors.get('error_color', '#ff6b6b')};
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: bold;
                display: inline-block;
                margin: 0.2rem;
            }}
            
            /* Sidebar améliorée */
            .sidebar-section {{
                background-color: {theme_colors.get('secondary_background_color', '#13c2c2')};
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                border: 1px solid #e0e0e0;
            }}
            
            /* Onglets personnalisés - Bleu canard */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background-color: #E0F2F1;
                border-radius: 8px 8px 0 0;
                padding: 0.5rem 1rem;
                border: none;
                color: #008B8B;
                transition: all 0.3s ease;
            }}
            
            .stTabs [data-baseweb="tab"]:hover {{
                background-color: #B2DFDB;
                color: #006666;
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: #008B8B !important;
                color: white !important;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,139,139,0.3);
            }}
            
            /* Footer */
            .footer {{
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: {theme_colors.get('primary_color', '#1f77b4')};
                color: white;
                text-align: center;
                padding: 10px;
                font-size: 0.8rem;
            }}
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Affichage de l'en-tête de l'application"""
        app_name = get_config("app.name", "Dashboard Commercial") if CONFIG_LOADED else "Dashboard Commercial"
        app_version = get_config("app.version", "2.0") if CONFIG_LOADED else "2.0"
        
        st.markdown(f'<h1 class="main-header">{app_name}</h1>', unsafe_allow_html=True)
        
        # Informations système en colonnes
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.write("📊 Analyse des souscriptions, entretiens et rendez-vous")
        
        with col2:
            environment = get_config("deployment.environment", "development") if CONFIG_LOADED else "development"
            if environment == "production":
                st.success("🌐 Environnement de production")
            else:
                st.info(f"🔧 Environnement: {environment}")
        
        with col3:
            st.caption(f"Version {app_version}")
    
    def render_sidebar(self):
        """Affichage de la sidebar avec chargement des fichiers"""
        with st.sidebar:
            # Chatbot IA en haut
            self.render_sidebar_chatbot()
            
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("📁 Information")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.info("💡 **Nouveau !** Les fichiers se chargent maintenant directement dans chaque page d'analyse pour plus de clarté.")
            
            # Configuration des types de fichiers acceptés pour référence
            accepted_formats = get_config("data.formats.upload", ["xlsx", "xls", "csv"]) if CONFIG_LOADED else ["xlsx", "xls", "csv"]
            max_file_size = get_config("data.formats.max_file_size", "2000MB") if CONFIG_LOADED else "2000MB"
            
            st.success(f"📋 **Formats supportés:** {', '.join(accepted_formats)}\n📏 **Taille max:** {max_file_size}")
            
            # Section Souscriptions
            with st.expander("💰 Souscriptions", expanded=True):
                st.markdown("**📊 Données de souscriptions pour analyses financières**")
                
                # Épargne
                st.markdown("**🏦 Épargne**")
                st.caption("Colonnes attendues: Date de souscription, Montant, Conseiller, Produit, Statut/Étape")
                self.data_files['df_epargne'] = self.load_file(
                    "📈 Souscriptions épargne", 
                    "file_epargne",
                    accepted_formats,
                    description="Fichier des souscriptions d'épargne avec montants et conseillers"
                )
                
                # Immobilier
                st.markdown("**🏠 Immobilier**")
                st.caption("Colonnes attendues: Date de création, Conseiller, Statut, Montant, Type de bien")
                self.data_files['df_immo'] = self.load_file(
                    "🏘️ Souscriptions immobilières", 
                    "file_immo",
                    accepted_formats,
                    description="Fichier des dossiers immobiliers avec statuts et conseillers"
                )
            
            # Section Entretiens
            with st.expander("🗣️ Entretiens", expanded=True):
                st.markdown("**📞 Données d'entretiens clients pour analyses de performance**")
                
                # Entretiens Épargne
                st.markdown("**💼 Entretiens Épargne**")
                st.caption("Colonnes attendues: Date, Conseiller, Type d'entretien, Résultat, Client")
                self.data_files['df_entretiens_epargne'] = self.load_file(
                    "📋 Entretiens épargne", 
                    "file_entretiens_epargne",
                    accepted_formats,
                    process_entretiens=True,
                    type_entretien="Épargne",
                    description="Fichier des entretiens clients pour l'épargne"
                )
                
                # Entretiens Immobilier
                st.markdown("**🏗️ Entretiens Immobilier**")
                st.caption("Colonnes attendues: Date, Conseiller, Type d'entretien, Résultat, Client")
                self.data_files['df_entretiens_immo'] = self.load_file(
                    "🏘️ Entretiens immobiliers", 
                    "file_entretiens_immo",
                    accepted_formats,
                    process_entretiens=True,
                    type_entretien="Immobilier",
                    description="Fichier des entretiens clients pour l'immobilier"
                )
            
            # Section RDV et Clients
            with st.expander("📅 RDV et Clients", expanded=True):
                st.markdown("**👥 Données de rendez-vous et informations clients**")
                
                # RDV
                st.markdown("**📆 Rendez-vous**")
                st.caption("Colonnes attendues: Date RDV, Conseiller, Type RDV, Statut, Client")
                self.data_files['df_rdv'] = self.load_file(
                    "🕐 Fichier des RDV", 
                    "file_rdv",
                    accepted_formats,
                    description="Planning et suivi des rendez-vous clients"
                )
                
                # Clients
                st.markdown("**👤 Base clients**")
                st.caption("Colonnes attendues: Nom, Prénom, Email, Téléphone, Conseiller affecté")
                self.data_files['df_clients'] = self.load_file(
                    "📇 Base de données clients", 
                    "file_clients",
                    accepted_formats,
                    description="Informations et parcours des clients"
                )
                
                # Analyse clients
                st.markdown("**📊 Analyse clients détaillée**")
                st.caption("Colonnes: Nom & Prénom, Email, Date entretien, Nb Souscriptions, VR, Métier, Secteur, Revenus, TMI, Profil épargnant, etc.")
                self.data_files['df_clients_analyse'] = self.load_file(
                    "🎯 Fichier d'analyse clients", 
                    "file_clients_analyse",
                    accepted_formats,
                    description="Données détaillées des clients pour analyse approfondie"
                )
                
                # Analyse 2025
                st.markdown("**🚀 Analyse 2025**")
                st.caption("Colonnes: Full Name, Email, Phone, Mobile, Profession, Contact Owner, Opportunité Name, Produit, Premier versement, Stage, Date versement initial, Date validation 570, Date de passage comité, Apport net")
                self.data_files['df_analyse_2025'] = self.load_file(
                    "🚀 Fichier d'analyse 2025", 
                    "file_analyse_2025",
                    accepted_formats,
                    description="Données 2025 avec pipeline commercial et opportunités"
                )
            
            # Section Paiements
            with st.expander("💸 Paiements", expanded=True):
                st.markdown("**💳 Données de paiements et transactions financières**")
                
                # Paiements Épargne
                st.markdown("**💰 Paiements Épargne**")
                st.caption("Colonnes attendues: Date paiement, Montant, Conseiller, Statut, Référence")
                self.data_files['df_paiements_epargne'] = self.load_file(
                    "💵 Paiements épargne", 
                    "file_paiements_epargne",
                    accepted_formats,
                    description="Suivi des paiements et encaissements épargne"
                )
                
                # Paiements Immobilier
                st.markdown("**🏦 Paiements Immobilier**")
                st.caption("Colonnes attendues: Date paiement, Montant, Conseiller, Statut, Référence")
                self.data_files['df_paiements_immo'] = self.load_file(
                    "🏠 Paiements immobiliers", 
                    "file_paiements_immo",
                    accepted_formats,
                    description="Suivi des paiements et commissions immobilier"
                )
            
            # Statut global des fichiers
            st.markdown("---")
            self.render_file_status()
            
            # Informations système
            st.markdown("---")
            self.render_system_info()
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
    def _process_uploaded_file(_file_content, _file_name, process_entretiens=False, type_entretien=None):
        """Process uploaded file with caching and memory optimization - internal method"""
        import io
        df = read_excel_robust(io.BytesIO(_file_content))
        
        if df is not None:
            # Memory optimization: reduce data types where possible
            try:
                df = DashboardApp._optimize_dataframe_memory(df)
            except Exception as e:
                # If optimization fails, continue with original DataFrame
                pass
            
            if process_entretiens and type_entretien:
                # Renommer la colonne "Date de création" en "Date" si elle existe
                if "Date de création" in df.columns:
                    df = df.rename(columns={"Date de création": "Date"})
                # Ajouter une colonne Type
                df["Type"] = type_entretien
            
        return df
    
    @staticmethod
    def _optimize_dataframe_memory(df):
        """Optimize DataFrame memory usage by converting data types"""
        import pandas as pd
        
        for col in df.columns:
            try:
                # Convert object columns that are actually numeric
                if df[col].dtype == 'object':
                    # Try to convert to numeric if it looks numeric
                    numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                    if numeric_conversion.notna().sum() > len(df) * 0.5:  # If >50% are numeric
                        # Use smaller int types where possible
                        if numeric_conversion.min() >= 0 and numeric_conversion.max() <= 255:
                            df[col] = numeric_conversion.astype('Int8')
                        elif numeric_conversion.min() >= -128 and numeric_conversion.max() <= 127:
                            df[col] = numeric_conversion.astype('Int8')
                        elif numeric_conversion.min() >= 0 and numeric_conversion.max() <= 65535:
                            df[col] = numeric_conversion.astype('Int16')
                        else:
                            df[col] = numeric_conversion.astype('Int32')
                    else:
                        # For string columns, use category if many duplicates
                        if df[col].nunique() / len(df) < 0.5:  # If <50% unique values
                            df[col] = df[col].astype('category')
            except Exception:
                # Skip optimization for this column if it fails
                continue
        
        return df
    
    def load_file(self, label, key, formats, process_entretiens=False, type_entretien=None, description=None):
        """Chargement d'un fichier avec traitement spécifique et mise en cache"""
        # Afficher la description si fournie
        if description:
            st.caption(f"ℹ️ {description}")
        
        # Diagnostic de la configuration de taille
        try:
            from streamlit.runtime.config import get_option
            max_size = get_option("server.maxUploadSize")
            st.caption(f"ℹ️ Taille max autorisée: {max_size}MB")
        except:
            st.caption("ℹ️ Configuration par défaut")
            
        file = st.file_uploader(label, type=formats, key=key)
        
        if file is not None:
            try:
                with st.spinner(f"Chargement de {label.lower()}..."):
                    # Use cached processing if performance mode is enabled
                    if self.performance_mode:
                        df = DashboardApp._process_uploaded_file(
                            file.getvalue(), 
                            file.name, 
                            process_entretiens=process_entretiens, 
                            type_entretien=type_entretien
                        )
                    else:
                        # Direct processing without caching for lighter memory usage
                        df = read_excel_robust(file)
                        
                        if df is not None and process_entretiens and type_entretien:
                            # Renommer la colonne "Date de création" en "Date" si elle existe
                            if "Date de création" in df.columns:
                                df = df.rename(columns={"Date de création": "Date"})
                            # Ajouter une colonne Type
                            df["Type"] = type_entretien
                    
                    if df is not None:
                        # Affichage amélioré du succès
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.success(f"✅ **{label}** chargé avec succès")
                        with col2:
                            st.metric("Lignes", f"{len(df):,}")
                        
                        # Afficher un aperçu des colonnes avec un toggle
                        if st.checkbox("🔍 Voir les colonnes détectées", key=f"cols_{key}"):
                            cols_preview = list(df.columns)[:10]  # Première 10 colonnes
                            if len(df.columns) > 10:
                                cols_preview.append(f"... et {len(df.columns) - 10} autres")
                            st.write("📋 **Colonnes détectées:**")
                            for col in cols_preview:
                                st.write(f"• {col}")
                        
                        self.logger.info(f"Fichier chargé: {label} - {len(df)} lignes")
                        return df
                    else:
                        st.error(f"❌ Erreur lors du chargement de {label.lower()}")
                        self.logger.error(f"Erreur chargement: {label}")
                        return None
            except Exception as e:
                st.error(f"❌ **Erreur:** {str(e)}")
                self.logger.error(f"Exception lors du chargement {label}: {e}")
                return None
        
        return None
    
    def process_entretiens_file(self, df, type_entretien):
        """Traitement spécifique des fichiers d'entretiens"""
        # Renommer la colonne "Date de création" en "Date" si elle existe
        if "Date de création" in df.columns:
            df = df.rename(columns={"Date de création": "Date"})
        
        # Ajouter une colonne Type
        df["Type"] = type_entretien
        
        return df
    
    def render_file_status(self):
        """Affichage du statut global des fichiers chargés"""
        st.subheader("📊 Statut des fichiers")
        
        # Définir les catégories de fichiers avec leurs icônes
        file_categories = {
            "💰 Souscriptions": {
                "df_epargne": "🏦 Épargne",
                "df_immo": "🏠 Immobilier"
            },
            "🗣️ Entretiens": {
                "df_entretiens_epargne": "💼 Épargne",
                "df_entretiens_immo": "🏗️ Immobilier"
            },
            "📅 RDV & Clients": {
                "df_rdv": "🕐 RDV",
                "df_clients": "👤 Clients",
                "df_clients_analyse": "🎯 Analyse clients",
                "df_analyse_2025": "🚀 Analyse 2025"
            },
            "💸 Paiements": {
                "df_paiements_epargne": "💰 Épargne",
                "df_paiements_immo": "🏦 Immobilier"
            }
        }
        
        # Afficher le statut par catégorie
        for category, files in file_categories.items():
            st.markdown(f"**{category}**")
            
            for file_key, file_name in files.items():
                df = self.data_files.get(file_key)
                if df is not None:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.success(f"✅ {file_name}")
                    with col2:
                        st.caption(f"{len(df):,} lignes")
                    with col3:
                        st.caption(f"{len(df.columns)} colonnes")
                else:
                    st.error(f"❌ {file_name}")
            
            st.write("")  # Espacement
        
        # Résumé global
        loaded_files = sum(1 for df in self.data_files.values() if df is not None)
        total_files = len(self.data_files)
        total_rows = sum(len(df) for df in self.data_files.values() if df is not None)
        
        st.markdown("**📈 Résumé global**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fichiers", f"{loaded_files}/{total_files}")
        with col2:
            st.metric("Total lignes", f"{total_rows:,}" if total_rows > 0 else "0")
        with col3:
            completion_rate = (loaded_files / total_files) * 100
            st.metric("Complétude", f"{completion_rate:.0f}%")
        
        # Indicateur de progression
        progress = loaded_files / total_files
        st.progress(progress)
        
        if loaded_files == total_files:
            st.success("🎉 Tous les fichiers sont chargés !")
        elif loaded_files > 0:
            st.info(f"📋 {total_files - loaded_files} fichier(s) restant(s) à charger")
        else:
            st.warning("⚠️ Aucun fichier chargé")
    
    def render_system_info(self):
        """Affichage des informations système"""
        st.subheader("ℹ️ Informations système")
        
        # Dernière mise à jour
        now = datetime.now()
        st.write(f"🕒 Dernière mise à jour : {now.strftime('%d/%m/%Y %H:%M:%S')}")        
        # Statistiques des fichiers chargés
        loaded_files = sum(1 for df in self.data_files.values() if df is not None)
        total_files = len(self.data_files)
        st.metric("Fichiers chargés", f"{loaded_files}/{total_files}")
        
        # Total des lignes
        total_rows = sum(len(df) for df in self.data_files.values() if df is not None)
        if total_rows > 0:
            st.metric("Total lignes", f"{total_rows:,}")
        
        # Bouton pour vider le cache Streamlit
        st.markdown("---")
        st.subheader("🔄 Rafraîchissement")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🧹 Vider le cache", type="primary"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        with col2:
            st.caption("Utilisez ce bouton si vous rencontrez des problèmes d'affichage ou si les modifications de code ne sont pas prises en compte.")
        
        # Environnement
        if CONFIG_LOADED:
            env = get_config("deployment.environment", "development")
            domain = get_config("deployment.domain", "localhost")
            st.caption(f"🌐 Env: {env}")
            if env == "production":
                st.caption(f"🔗 {domain}")
    
    def render_file_status_overview(self):
        """Affichage de l'état des fichiers dans l'accueil"""
        st.subheader("📊 État des fichiers")
        
        # Créer un DataFrame pour l'état des fichiers
        file_status = []
        file_labels = {
            'df_epargne': 'Souscriptions épargne',
            'df_immo': 'Souscriptions immobilières',
            'df_entretiens_epargne': 'Entretiens épargne',
            'df_entretiens_immo': 'Entretiens immobiliers',
            'df_rdv': 'Rendez-vous',
            'df_clients': 'Clients',
            'df_clients_analyse': 'Analyse clients',
            'df_analyse_2025': 'Analyse 2025',
            'df_paiements_epargne': 'Paiements épargne',
            'df_paiements_immo': 'Paiements immobiliers'
        }
        
        for key, label in file_labels.items():
            df = self.data_files[key]
            status = "✅ Chargé" if df is not None else "❌ Non chargé"
            rows = len(df) if df is not None else 0
            file_status.append({
                'Fichier': label,
                'Statut': status,
                'Lignes': f"{rows:,}" if rows > 0 else "-"
            })
        
        # Affichage en colonnes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📁 Souscriptions**")
            for item in file_status[:2]:
                status_class = "status-success" if "✅" in item['Statut'] else "status-error"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{item['Fichier']}</strong><br>
                    <span class="{status_class}">{item['Statut']}</span><br>
                    <small>Lignes: {item['Lignes']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🗣️ Entretiens et RDV**")
            for item in file_status[2:5]:
                status_class = "status-success" if "✅" in item['Statut'] else "status-error"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{item['Fichier']}</strong><br>
                    <span class="{status_class}">{item['Statut']}</span><br>
                    <small>Lignes: {item['Lignes']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("**👥 Clients et Paiements**")
            for item in file_status[5:]:
                status_class = "status-success" if "✅" in item['Statut'] else "status-error"
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{item['Fichier']}</strong><br>
                    <span class="{status_class}">{item['Statut']}</span><br>
                    <small>Lignes: {item['Lignes']}</small>
                </div>
                """, unsafe_allow_html=True)
            
        # Ajout d'une option pour analyser les paiements combinés
        st.markdown("---")
        st.subheader("🔗 Analyse combinée des paiements")
        
        has_epargne = self.data_files['df_paiements_epargne'] is not None
        has_immo = self.data_files['df_paiements_immo'] is not None
        
        if has_epargne and has_immo:
            if st.button("📊 Analyser tous les paiements (Épargne + Immobilier)", type="primary"):
                with st.expander("💸 Analyse combinée des paiements", expanded=True):
                    df_paiements_combined = pd.concat([
                        self.data_files['df_paiements_epargne'], 
                        self.data_files['df_paiements_immo']
                    ], ignore_index=True)
                    st.info("📊 Analyse combinée des paiements épargne et immobiliers")
                    analyser_paiements(df_paiements_combined)
        elif has_epargne or has_immo:
            st.info("ℹ️ Chargez les deux fichiers de paiements pour une analyse combinée")
        else:
            st.warning("⚠️ Aucun fichier de paiement chargé")
    
    def render_welcome_tab(self):
        """Onglet d'accueil"""
        st.markdown('<h2 class="sub-header">🏠 Accueil</h2>', unsafe_allow_html=True)
        
        # Description de l'application
        st.markdown("""
        ## Bienvenue dans le Dashboard Commercial 2.0
        
        Ce dashboard vous permet d'analyser en détail les données commerciales de votre entreprise avec une interface moderne et des fonctionnalités avancées.
        
        ### 🎯 Fonctionnalités principales :
        
        - **💰 Épargne** : Analyse complète des souscriptions d'épargne
        - **🏢 Immobilier** : Suivi détaillé des souscriptions immobilières  
        - **🗣️ Entretiens** : Analyse des entretiens par type (épargne/immobilier)
        - **📅 RDV** : Gestion et analyse des rendez-vous
        - **🔄 Conversions** : Calcul des taux de conversion inter-fichiers
        - **👣 Parcours Client** : Analyse du parcours et comportement client
        - **🎯 Analyse Clients** : Analyse approfondie des données clients
        - **💸 Paiements Épargne** : Suivi spécialisé des paiements épargne
        - **💸 Paiements Immobilier** : Suivi spécialisé des paiements immobiliers
        - **📊 Retard Cumulé** : Suivi des objectifs annuels et retard cumulé
        - **👤 Clients Détaillés** : Analyse avancée et segmentation des clients
        
        ### 🚀 Nouveautés v2.0 :
        - Configuration centralisée
        - Interface améliorée avec thème personnalisable
        - Logging et monitoring intégrés
        - Gestion d'erreurs renforcée
        - Performance optimisée
        """)
        
        # État des fichiers avec style amélioré
        self.render_file_status_overview()
        
        # Statistiques rapides si des données sont chargées
        self.render_quick_stats()
    
    def render_quick_stats(self):
        """Affichage de statistiques rapides"""
        loaded_data = [df for df in self.data_files.values() if df is not None]
        
        if loaded_data:
            st.subheader("📈 Statistiques rapides")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_rows = sum(len(df) for df in loaded_data)
                st.metric("Total enregistrements", f"{total_rows:,}")
            
            with col2:
                files_loaded = len(loaded_data)
                st.metric("Fichiers chargés", files_loaded)
            
            with col3:
                # Calculer le nombre de colonnes moyen
                avg_cols = sum(len(df.columns) for df in loaded_data) // len(loaded_data)
                st.metric("Colonnes moyennes", avg_cols)
            
            with col4:
                # Taille mémoire approximative
                memory_mb = sum(df.memory_usage(deep=True).sum() for df in loaded_data) / (1024 * 1024)
                st.metric("Mémoire utilisée", f"{memory_mb:.1f} MB")
    
    def render_debug_info(self, tab_name=""):
        """Affiche les informations de débogage à la fin de chaque onglet"""
        st.markdown("---")
        
        with st.expander("🐛 Informations de Débogage", expanded=False):
            debug_col1, debug_col2, debug_col3 = st.columns(3)
            
            with debug_col1:
                st.subheader("📊 État des Données")
                
                # État des fichiers chargés
                files_status = {}
                for key, df in self.data_files.items():
                    if df is not None:
                        files_status[key] = {
                            "status": "✅ Chargé",
                            "rows": len(df),
                            "columns": len(df.columns),
                            "memory": f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.1f} MB"
                        }
                    else:
                        files_status[key] = {
                            "status": "❌ Non chargé",
                            "rows": 0,
                            "columns": 0,
                            "memory": "0 MB"
                        }
                
                st.json(files_status)
            
            with debug_col2:
                st.subheader("🔧 Configuration Système")
                
                system_info = {
                    "tab_actuel": tab_name,
                    "config_loaded": CONFIG_LOADED,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_memory_used": f"{sum(df.memory_usage(deep=True).sum() for df in self.data_files.values() if df is not None) / (1024 * 1024):.1f} MB",
                    "files_loaded": sum(1 for df in self.data_files.values() if df is not None),
                    "total_records": sum(len(df) for df in self.data_files.values() if df is not None)
                }
                
                st.json(system_info)
            
            with debug_col3:
                st.subheader("📋 Colonnes Détectées")
                
                # Affichage des colonnes pour les fichiers chargés
                columns_info = {}
                for key, df in self.data_files.items():
                    if df is not None:
                        columns_info[key] = list(df.columns)[:10]  # Premières 10 colonnes
                        if len(df.columns) > 10:
                            columns_info[key].append(f"... et {len(df.columns) - 10} autres")
                
                if columns_info:
                    st.json(columns_info)
                else:
                    st.info("Aucun fichier chargé")
            
            # Messages de traitement des données
            st.subheader("📋 Messages de Traitement des Données")
            
            # Afficher les messages stockés dans session_state
            if hasattr(st.session_state, 'data_processing_messages') and st.session_state.data_processing_messages:
                st.markdown("**Messages de conversion et traitement :**")
                
                # Grouper les messages par type
                success_messages = []
                info_messages = []
                warning_messages = []
                error_messages = []
                
                for msg in st.session_state.data_processing_messages:
                    if msg.startswith("✅"):
                        success_messages.append(msg)
                    elif msg.startswith("ℹ️"):
                        info_messages.append(msg)
                    elif msg.startswith("⚠️"):
                        warning_messages.append(msg)
                    elif msg.startswith("❌"):
                        error_messages.append(msg)
                
                # Afficher les messages groupés
                for msg in success_messages:
                    st.success(msg)
                for msg in info_messages:
                    st.info(msg)
                for msg in warning_messages:
                    st.warning(msg)
                for msg in error_messages:
                    st.error(msg)
                
                # Bouton pour vider les messages
                if st.button("🧹 Vider les Messages", key=f"clear_messages_{tab_name}"):
                    st.session_state.data_processing_messages = []
                    st.success("Messages vidés !")
                    st.rerun()
            else:
                st.info("Aucun message de traitement des données disponible")
            
            # Section avancée de débogage
            st.subheader("🔬 Débogage Avancé")
            
            debug_advanced_col1, debug_advanced_col2 = st.columns(2)
            
            with debug_advanced_col1:
                if st.button("🧹 Vider le Cache Streamlit", key=f"clear_cache_{tab_name}"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Cache vidé !")
                    st.rerun()
                
                if st.button("🔄 Recharger la Page", key=f"reload_{tab_name}"):
                    st.rerun()
            
            with debug_advanced_col2:
                if st.button("💾 Export État Debug", key=f"export_debug_{tab_name}"):
                    debug_export = {
                        "timestamp": datetime.now().isoformat(),
                        "tab": tab_name,
                        "files_status": files_status,
                        "system_info": system_info,
                        "columns_info": columns_info,
                        "processing_messages": getattr(st.session_state, 'data_processing_messages', [])
                    }
                    
                    debug_json = str(debug_export).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger Debug JSON",
                        data=debug_json,
                        file_name=f"debug_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key=f"download_debug_{tab_name}"
                    )
    
    def render_tabs(self):
        """Affichage des onglets principaux avec lazy loading"""
        # Configuration des onglets
        tab_names = [
            "🏠 Accueil",
            "💰 Épargne", 
            "🏢 Immobilier",
            "📞 Entretiens",
            "🔄 Conversions",
            "👥 Parcours Client",
            "🎯 Analyse Clients",
            "💳 Paiements",
            "📊 Retard Cumulé",
            "👤 Clients Détaillés",
            "🚀 Analyse 2025",
            "🤖 Chatbot IA"
        ]
        
        # Use selectbox instead of tabs for better performance (only active tab renders)
        selected_tab = st.selectbox(
            "📊 Sélectionner une analyse:",
            tab_names,
            index=0,
            key="main_tab_selector"
        )
        
        # Add a visual separator
        st.markdown("---")
        
        # Render only the selected tab content
        if selected_tab == "🏠 Accueil":
            self.render_welcome_tab()
            self.render_debug_info("Accueil")
        
        elif selected_tab == "💰 Épargne":
            st.markdown('<h2 class="sub-header">💰 Analyse Épargne</h2>', unsafe_allow_html=True)
            
            # Chargement du fichier spécifique à cette analyse
            st.subheader("📁 Chargement des données")
            df_epargne = self.load_file(
                "📈 Fichier souscriptions épargne", 
                "file_epargne_tab",
                ["xlsx", "csv"],
                description="Colonnes attendues: Date de souscription, Montant, Conseiller, Produit, Statut/Étape"
            )


# =============================================================================

            if df_epargne is None:
                df_epargne = auto_load_from_s3("epargne", description="souscriptions épargne")
# =============================================================================


            if df_epargne is not None:
                st.markdown("---")
                
                # Créer des sous-onglets pour l'analyse épargne
                epargne_subtab = st.selectbox(
                    "📊 Type d'analyse épargne:",
                    ["Performance Globale", "Performance par Conseiller", "Analyse par Groupe", "Pipe de Collecte"],
                    key="epargne_subtab"
                )
                
                st.markdown("---")
                
                if epargne_subtab == "Performance Globale":
                    with st.spinner("📊 Analyse de la performance globale..."):
                        analyser_souscriptions_epargne(df_epargne)
                elif epargne_subtab == "Performance par Conseiller":
                    with st.spinner("📊 Analyse par conseiller..."):
                        analyser_performance_conseillers_epargne(df_epargne)
                elif epargne_subtab == "Analyse par Groupe":
                    with st.spinner("📊 Analyse par groupe..."):
                        analyser_groupes_epargne(df_epargne)
                elif epargne_subtab == "Pipe de Collecte":
                    with st.spinner("📊 Analyse du pipe de collecte..."):
                        analyser_pipe_collecte_epargne(df_epargne)
            
                self.render_debug_info("Épargne")


# =============================================================================
#           if df_epargne is None:
#                 df_epargne = make_epargne_from_crypto(df)
#                 if not df_epargne.empty:
#                     st.info("🧪 Mode test : données crypto (Lightsail) mappées vers Épargne.")
#                     st.markdown("---")
                
#                     # Créer des sous-onglets pour l'analyse épargne
#                     epargne_subtab = st.selectbox(
#                         "📊 Type d'analyse épargne:",
#                         ["Performance Globale", "Performance par Conseiller", "Analyse par Groupe", "Pipe de Collecte"],
#                         key="epargne_subtab"
#                     )
                    
#                     st.markdown("---")
                    
#                     if epargne_subtab == "Performance Globale":
#                         with st.spinner("📊 Analyse de la performance globale..."):
#                             analyser_souscriptions_epargne(df_epargne)
#                     elif epargne_subtab == "Performance par Conseiller":
#                         with st.spinner("📊 Analyse par conseiller..."):
#                             analyser_performance_conseillers_epargne(df_epargne)
#                     elif epargne_subtab == "Analyse par Groupe":
#                         with st.spinner("📊 Analyse par groupe..."):
#                             analyser_groupes_epargne(df_epargne)
#                     elif epargne_subtab == "Pipe de Collecte":
#                         with st.spinner("📊 Analyse du pipe de collecte..."):
#                             analyser_pipe_collecte_epargne(df_epargne)
                
#                 self.render_debug_info("Épargne")
                # =============================================================================



            elif selected_tab == "🏢 Immobilier":
                st.markdown('<h2 class="sub-header">🏢 Analyse Immobilier</h2>', unsafe_allow_html=True)
                
                # Chargement du fichier spécifique à cette analyse
                st.subheader("📁 Chargement des données")
                df_immo = self.load_file(
                    "🏠 Fichier souscriptions immobilières", 
                    "file_immo_tab",
                    ["xlsx", "csv"],
                    description="Colonnes attendues: Date de création, Conseiller, Statut, Montant, Type de bien"
                )

# =============================================================================

            if df_immo is None:
                df_immo = auto_load_from_s3("immo", description="souscriptions immobilières")
# =============================================================================




            if df_immo is not None:
                st.markdown("---")
                
                # Créer des sous-onglets pour l'analyse immobilière
                immo_subtab = st.selectbox(
                    "🏢 Type d'analyse immobilier:",
                    ["Suivi Global", "Analyse par Statut", "Analyse par Groupe"],
                    key="immo_subtab"
                )
                
                st.markdown("---")
                
                if immo_subtab == "Suivi Global":
                    with st.spinner("🏢 Analyse du suivi global..."):
                        analyser_suivi_immo(df_immo)
                elif immo_subtab == "Analyse par Statut":
                    with st.spinner("🏢 Analyse par statut..."):
                        analyser_statuts_dossiers_immo(df_immo)
                elif immo_subtab == "Analyse par Groupe":
                    with st.spinner("🏢 Analyse par groupe..."):
                        analyser_groupes_dossiers_immo(df_immo)
            
            self.render_debug_info("Immobilier")
                # =============================================================================
#             if df_immo is None:
#                 df_immo = make_immo_from_crypto(df)
#                 if not df_immo.empty:
#                     st.info("🧪 Mode test : données crypto (Lightsail) mappées vers Immobilier.")
#                     st.markdown("---")
                
#                     # Créer des sous-onglets pour l'analyse immobilière
#                     immo_subtab = st.selectbox(
#                         "🏢 Type d'analyse immobilier:",
#                         ["Suivi Global", "Analyse par Statut", "Analyse par Groupe"],
#                         key="immo_subtab"
#                         )
                        
#                     st.markdown("---")
                        
#                     if immo_subtab == "Suivi Global":
#                         with st.spinner("🏢 Analyse du suivi global..."):
#                             analyser_suivi_immo(df_immo)
#                     elif immo_subtab == "Analyse par Statut":
#                         with st.spinner("🏢 Analyse par statut..."):
#                             analyser_statuts_dossiers_immo(df_immo)
#                     elif immo_subtab == "Analyse par Groupe":
#                         with st.spinner("🏢 Analyse par groupe..."):
#                             analyser_groupes_dossiers_immo(df_immo)

  
#             self.render_debug_info("Immobilier")
                

               # =============================================================================

        
        elif selected_tab == "📞 Entretiens":
            with st.spinner("📞 Chargement des analyses d'entretiens..."):
                self.render_entretiens_tab()
            self.render_debug_info("Entretiens")
        
        elif selected_tab == "🔄 Conversions":
            with st.spinner("🔄 Calcul des taux de conversion..."):
                self.render_conversions_tab()
            self.render_debug_info("Conversions")
        
        elif selected_tab == "👥 Parcours Client":
            st.markdown('<h2 class="sub-header">👥 Analyse Parcours Client</h2>', unsafe_allow_html=True)
            
            # Chargement du fichier spécifique à cette analyse
            st.subheader("📁 Chargement des données")
            df_clients = self.load_file(
                "📇 Base de données clients", 
                "file_clients_tab",
                ["xlsx", "csv"],
                description="Colonnes attendues: Nom, Prénom, Email, Téléphone, Conseiller affecté"
            )
            
            if df_clients is not None:
                st.markdown("---")
                with st.spinner("👥 Analyse du parcours client..."):
                    analyser_parcours_client(df_clients)
            
            self.render_debug_info("Parcours Client")
        
        elif selected_tab == "🎯 Analyse Clients":
            st.markdown('<h2 class="sub-header">🎯 Analyse Clients Détaillée</h2>', unsafe_allow_html=True)
            
            st.subheader("📁 Chargement des données - Analyse Clients")
            df_clients_analyse = self.load_file(
                "📊 Fichier analyse clients détaillée", 
                "file_analyse_clients_tab",
                ["xlsx", "csv"],
                description="Colonnes attendues: Nom & Prénom, Email, Date de l'entretien, Nb Souscriptions, VR, Métier, Secteur d'activité, Revenus, etc."
            )
            
            if df_clients_analyse is not None:
                st.markdown("---")
                with st.spinner("🎯 Analyse détaillée des clients..."):
                    analyser_clients(df_clients_analyse)
            else:
                st.info("💡 Chargez le fichier d'analyse clients pour voir les analyses détaillées.")
            
            self.render_debug_info("Analyse Clients")
        
        elif selected_tab == "💳 Paiements":
            paiement_type = st.selectbox(
                "💳 Type de paiement:",
                ["Paiements Épargne", "Paiements Immobilier"],
                key="paiement_subtab"
            )
            
            st.markdown("---")
            
            if paiement_type == "Paiements Épargne":
                # Chargement de fichier pour les paiements épargne
                df_paiements_epargne = self.load_file(
                    "Upload du fichier paiements épargne", 
                    "paiements_epargne",
                    "Chargez le fichier Excel des paiements épargne pour analyser les transactions et les tendances de versements."
                )
                
                if df_paiements_epargne is not None:
                    with st.spinner("💳 Analyse des paiements épargne..."):
                        analyser_paiements(df_paiements_epargne, "épargne")
                else:
                    st.warning("⚠️ Veuillez charger le fichier des paiements épargne.")
            else:
                # Chargement de fichier pour les paiements immobilier
                df_paiements_immo = self.load_file(
                    "Upload du fichier paiements immobilier", 
                    "paiements_immo",
                    "Chargez le fichier Excel des paiements immobilier pour analyser les transactions et les versements immobiliers."
                )
                
                if df_paiements_immo is not None:
                    with st.spinner("💳 Analyse des paiements immobilier..."):
                        analyser_paiements(df_paiements_immo, "immobilier")
                else:
                    st.warning("⚠️ Veuillez charger le fichier des paiements immobiliers.")
            
            self.render_debug_info("Paiements")
        
        elif selected_tab == "📊 Retard Cumulé":
            st.markdown('<h2 class="sub-header">📊 Retard Cumulé par Rapport à l\'Objectif Annuel</h2>', unsafe_allow_html=True)
            st.info("💡 Cette section analyse le retard cumulé depuis le début de l'année par rapport à l'objectif annuel de 22M€, avec un objectif mensuel de 1,83M€.")
            
            st.subheader("📁 Chargement des données - Épargne")
            df_epargne_retard = self.load_file(
                "💰 Fichier souscriptions épargne", 
                "file_epargne_retard_tab",
                ["xlsx", "csv"],
                description="Colonnes attendues: Date de souscription, Montant, Conseiller pour calculer le retard cumulé"
            )
            
            if df_epargne_retard is not None:
                st.markdown("---")
                with st.spinner("📊 Calcul du retard cumulé..."):
                    self.analyser_retard_cumule(df_epargne_retard)
            else:
                st.info("💡 Chargez le fichier des souscriptions épargne pour analyser le retard cumulé.")
            
            self.render_debug_info("Retard Cumulé")
        
        elif selected_tab == "👤 Clients Détaillés":
            st.markdown('<h2 class="sub-header">👤 Analyse Détaillée des Clients</h2>', unsafe_allow_html=True)
            st.info("💡 Cette section propose une analyse approfondie des données clients avec des fonctionnalités de segmentation et de profilage avancées basées sur vos colonnes Excel spécifiques.")
            
            # Chargement de fichier pour cette analyse
            df_clients = self.load_file(
                "Upload du fichier clients détaillés", 
                "clients_analyse",
                "Chargez le fichier Excel contenant l'analyse détaillée des clients avec leurs profils professionnels et personnels."
            )
            
            if df_clients is not None:
                # Utiliser le nouveau module d'analyse intégrée
                with st.spinner("👤 Analyse détaillée des clients en cours..."):
                    analyser_clients_integration(df_clients)
            else:
                st.warning("⚠️ Veuillez charger le fichier d'analyse clients détaillée.")
                
                # Afficher les colonnes attendues basées sur l'image fournie
                st.markdown("### 📋 Structure Attendue du Fichier Excel")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    **Informations de base :**
                    - Nom & Prénom
                    - Email
                    - Date de l'entretien
                    - Nb Souscriptions
                    - Dernière Souscription
                    - VR (Versements Réguliers - OUI/NON)
                    """)
                
                with col2:
                    st.markdown("""
                    **Profil professionnel :**
                    - Métier
                    - Secteur d'activité
                    - Revenus
                    - Type de contrat
                    - Éligibilité
                    - TMI (Tranche Marginale d'Imposition)
                    """)
                
                with col3:
                    st.markdown("""
                    **Profil personnel :**
                    - Profil épargnant
                    - Épargne disponible
                    - Situation familiale
                    - Nb d'enfants
                    - Date d'inscription
                    - Conseiller
                    """)
                
                st.markdown("---")
                st.info("🔄 **VR (Versements Réguliers) :** Cette colonne indique si le client a des versements programmés. **OUI** = versements réguliers actifs, **NON** = pas de versements programmés.")
                
                st.success("✨ **Fonctionnalités de l'analyse intégrée :**")
                
                feature_cols = st.columns(2)
                with feature_cols[0]:
                    st.markdown("""
                    🔍 **Filtres avancés :**
                    - Multi-critères (conseiller, secteur, éligibilité, revenus)
                    - Plages de revenus personnalisables
                    - Filtrage par profil épargnant
                    
                    📊 **Analyses détaillées :**
                    - Vue d'ensemble avec graphiques interactifs
                    - Performance par conseiller
                    - Répartition par secteur d'activité
                    - Matrice de corrélations financières
                    """)
                
                with feature_cols[1]:
                    st.markdown("""
                    🎯 **Segmentation avancée :**
                    - Classification automatique (Champions, Gros Potentiel, Très Actifs, Fidèles, À Développer)
                    - Analyse familiale et démographique
                    - Évolution temporelle des entretiens
                    
                    📤 **Exports & Rapports :**
                    - Export CSV avec données filtrées
                    - Résumé statistique automatique
                    - Recommandations stratégiques par segment
                    """)
            
            self.render_debug_info("Clients Détaillés")
        
        elif selected_tab == "🚀 Analyse 2025":
            st.markdown('<h2 class="sub-header">🚀 Analyse 2025</h2>', unsafe_allow_html=True)
            st.info("💡 Cette section analyse les données avec les nouvelles colonnes 2025 pour le suivi des opportunités et du pipeline commercial.")
            
            # Chargement de fichier pour l'analyse 2025
            df_analyse_2025 = self.load_file(
                "Upload du fichier analyse 2025", 
                "analyse_2025",
                ["xlsx", "xls", "csv"],
                description="Chargez le fichier Excel 2025 avec les colonnes Contact Owner, Opportunité Name, Stage, Premier versement, Apport net, etc."
            )
            
            if df_analyse_2025 is not None:
                # Sauvegarder dans session_state pour le chatbot
                st.session_state['chatbot_data_2025'] = df_analyse_2025
                
                with st.spinner("🚀 Analyse des données 2025 en cours..."):
                    self.analyser_donnees_2025(df_analyse_2025)
                
                # Analyse de saisonnalité des ventes 2025
                st.markdown("---")
                with st.spinner("📈 Analyse de saisonnalité en cours..."):
                    self.analyser_saisonnalite_ventes(df_analyse_2025)
            else:
                st.warning("⚠️ Veuillez charger le fichier d'analyse 2025.")
                
                # Afficher la structure attendue
                st.markdown("### 📋 Structure Attendue du Fichier Excel 2025")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    **Informations client :**
                    - Full Name
                    - Email
                    - Phone
                    - Mobile
                    - Profession
                    """)
                
                with col2:
                    st.markdown("""
                    **Gestion commerciale :**
                    - Contact Owner
                    - Opportunité Name
                    - Produit
                    - Stage
                    """)
                
                with col3:
                    st.markdown("""
                    **Données financières & temporelles :**
                    - Premier versement
                    - Apport net
                    - Date versement initial
                    - Date validation 570
                    - Date de passage comité
                    """)
            
            self.render_debug_info("Analyse 2025")
        
        elif selected_tab == "🤖 Chatbot IA":
            self.render_chatbot_tab()
    
    def render_chatbot_tab(self):
        """Onglet Chatbot IA pour l'assistance directeur"""
        st.markdown('<h2 class="sub-header">🤖 Assistant IA - Questions sur les Analyses</h2>', unsafe_allow_html=True)
        st.info("💡 Posez vos questions sur les analyses et obtenez des réponses avec les chiffres clés correspondants")
        
        # Instructions d'utilisation
        with st.expander("📋 Comment utiliser l'Assistant IA", expanded=False):
            st.markdown("""
            ### 🎯 **Questions que vous pouvez poser :**
            
            **📊 Analyses générales :**
            - "Quel est notre CA total pour 2024 ?"
            - "Combien de nouveaux clients avons-nous acquis cette année ?"
            - "Quel est notre top 3 des produits ?"
            
            **👥 Analyses par conseiller :**
            - "Qui est le meilleur conseiller en termes de CA ?"
            - "Combien de clients a Pierre ce mois-ci ?"
            
            **📈 Saisonnalité et tendances :**
            - "Quel trimestre est le plus performant ?"
            - "Comment évoluent nos ventes SCPI ?"
            
            **🎯 Segmentation :**
            - "Combien de clients dans le segment CPP Actif ?"
            - "Quel est le panier moyen des clients IMMO ?"
            """)
        
        # Interface de chat
        st.subheader("💬 Posez votre question")
        
        # Historique de conversation
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Interface de saisie
        user_question = st.text_input(
            "Votre question :",
            placeholder="Ex: Quel est notre CA total pour 2024 ?",
            key="chatbot_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🤖 Demander", type="primary")
        with col2:
            clear_button = st.button("🗑️ Effacer historique")
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        # Traitement de la question
        if ask_button and user_question.strip():
            with st.spinner("🧠 L'IA analyse vos données..."):
                response = self.process_ai_question(user_question)
                
                # Ajouter à l'historique
                st.session_state.chat_history.append({
                    'question': user_question,
                    'response': response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
        
        # Affichage de l'historique de conversation
        if st.session_state.chat_history:
            st.subheader("📝 Historique de conversation")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # 5 derniers échanges
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 10px 0;">
                        <strong>👤 Vous ({chat['timestamp']}) :</strong><br>
                        {chat['question']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background-color: #e8f4fd; padding: 10px; border-radius: 10px; margin: 10px 0;">
                        <strong>🤖 Assistant IA :</strong><br>
                        {chat['response']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if i < len(st.session_state.chat_history[-5:]) - 1:
                        st.markdown("---")
        
        # Suggestions de questions
        st.subheader("💡 Questions suggérées")
        
        suggestions = [
            "📊 Quel est notre CA total cette année ?",
            "👥 Combien de nouveaux clients en 2024 ?",
            "🏆 Qui est le meilleur conseiller ?",
            "📈 Quel trimestre est le plus performant ?",
            "🎯 Répartition des segments clients ?",
            "💰 Panier moyen par produit ?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    with st.spinner("🧠 L'IA analyse vos données..."):
                        response = self.process_ai_question(suggestion.replace("📊 ", "").replace("👥 ", "").replace("🏆 ", "").replace("📈 ", "").replace("🎯 ", "").replace("💰 ", ""))
                        
                        st.session_state.chat_history.append({
                            'question': suggestion,
                            'response': response,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        st.rerun()
    
    def create_data_summary(self, available_data):
        """Crée un résumé des données pour l'analyse LLM"""
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        summary = {}
        total_ca = 0
        total_clients = 0
        total_transactions = 0
        
        for key in available_data:
            df = self.data_files.get(key)
            if df is not None and len(df) > 0:
                # Informations générales
                summary[f"{key}_nombre_lignes"] = len(df)
                summary[f"{key}_nombre_colonnes"] = len(df.columns)
                
                # Analyser colonnes monétaires
                montant_cols = [col for col in df.columns if any(word in col.lower() for word in ['montant', 'premier', 'versement', 'ca', 'apport'])]
                if montant_cols:
                    try:
                        montant_col = montant_cols[0]
                        montants = pd.to_numeric(df[montant_col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
                        montants = montants.dropna()
                        if len(montants) > 0:
                            ca_total = montants.sum()
                            panier_moyen = montants.mean()
                            summary[f"{key}_ca_total"] = ca_total
                            summary[f"{key}_panier_moyen"] = panier_moyen
                            total_ca += ca_total
                            total_transactions += len(montants)
                    except:
                        pass
                
                # Analyser clients uniques
                if 'Email' in df.columns:
                    nb_clients = df['Email'].nunique()
                    summary[f"{key}_clients_uniques"] = nb_clients
                    total_clients += nb_clients
                
                # Analyser dates
                date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'versement', 'inscription'])]
                if date_cols:
                    try:
                        date_col = date_cols[0]
                        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                        if len(dates) > 0:
                            summary[f"{key}_periode_debut"] = dates.min().strftime('%Y-%m-%d')
                            summary[f"{key}_periode_fin"] = dates.max().strftime('%Y-%m-%d')
                    except:
                        pass
                
                # Top produits/conseillers
                if 'Conseiller' in df.columns:
                    top_conseiller = df['Conseiller'].value_counts().head(3)
                    summary[f"{key}_top_conseillers"] = dict(top_conseiller)
                
                if 'Produit' in df.columns:
                    top_produit = df['Produit'].value_counts().head(3)
                    summary[f"{key}_top_produits"] = dict(top_produit)
        
        # Résumé global
        if total_ca > 0:
            summary["ca_total_global"] = total_ca
        if total_clients > 0:
            summary["clients_total_global"] = total_clients
        if total_transactions > 0:
            summary["transactions_total_global"] = total_transactions
        
        return summary
    
    def process_ai_question(self, question):
        """Traite une question IA et retourne une réponse basée sur les données"""
        from datetime import datetime
        import pandas as pd
        import numpy as np
        
        question_lower = question.lower()
        
        # Réponses basées sur les données chargées
        try:
            # Vérifier quelles données sont disponibles
            available_data = []
            
            # Vérifier data_files classiques
            for key, df in self.data_files.items():
                if df is not None and len(df) > 0:
                    available_data.append(key)
            
            # Ajouter données spécifiques du session_state
            if 'chatbot_data_2025' in st.session_state and st.session_state.chatbot_data_2025 is not None:
                # Copier temporairement pour l'analyse
                self.data_files['temp_analyse_2025'] = st.session_state.chatbot_data_2025
                available_data.append('temp_analyse_2025')
            
            if 'chatbot_data_epargne' in st.session_state and st.session_state.chatbot_data_epargne is not None:
                # Copier temporairement pour l'analyse
                self.data_files['temp_epargne'] = st.session_state.chatbot_data_epargne
                available_data.append('temp_epargne')
            
            all_data = available_data
            
            if not all_data:
                return """❌ **Aucune donnée chargée**

💡 **Comment obtenir des réponses précises :**

**Option 1 - Chargement dans le chatbot :**
• Utilisez le widget 'Chargement rapide pour IA' ci-dessus
• Chargez votre fichier principal (ex: analyse 2025)

**Option 2 - Chargement par page :**
• Allez dans l'onglet d'analyse souhaité
• Chargez le fichier correspondant
• Revenez poser votre question

**Exemple de questions que je peux traiter :**
• "Quel est notre CA total 2024 ?"
• "Combien de clients CPP Actif ?"
• "Top 3 des conseillers par CA"
• "Évolution par trimestre"
"""
            
            # Analyser la question et générer une réponse
            response = "🤖 **Analyse en cours...**\n\n"
            
            # Essayer d'abord l'analyse LLM si configurée
            llm_response = None
            try:
                # Vérifier si un LLM est configuré
                configured_model = None
                configured_key = None
                
                for model in self.llm_assistant.available_models.keys():
                    if st.session_state.get(f'llm_configured_{model}', False):
                        configured_model = model
                        configured_key = st.session_state.get(f'llm_key_{model}')
                        break
                
                if configured_model:
                    # Configurer le LLM assistant
                    self.llm_assistant.configure_model(configured_model, configured_key)
                    
                    # Créer résumé des données pour le LLM
                    data_summary = self.create_data_summary(all_data)
                    
                    # Obtenir l'analyse LLM
                    llm_response = self.llm_assistant.analyze_data_with_llm(question, data_summary)
                    
                    if llm_response and llm_response.strip():
                        response = llm_response
                        response += f"\n\n📋 **Sources utilisées :** {', '.join(all_data)}"
                        response += f"\n🤖 **Analyse IA :** {self.llm_assistant.available_models[configured_model]['name']}"
                        response += f"\n🕒 **Analyse effectuée le :** {datetime.now().strftime('%d/%m/%Y à %H:%M')}"
                        
                        # Nettoyer les données temporaires
                        for key in list(self.data_files.keys()):
                            if key.startswith('temp_'):
                                del self.data_files[key]
                        
                        return response
            
            except Exception as e:
                # Si erreur avec LLM, continuer avec analyse basique
                st.warning(f"⚠️ Erreur avec l'IA configurée : {str(e)}")
                pass
            
            # Fallback vers analyse basique si pas d'LLM ou erreur
            # Questions sur le CA / Chiffre d'affaires
            if any(word in question_lower for word in ['ca', 'chiffre', 'affaires', 'revenus', 'montant']):
                response += self.analyze_ca_data()
            
            # Questions sur les clients
            elif any(word in question_lower for word in ['client', 'nouveaux', 'acquisition']):
                response += self.analyze_client_data()
            
            # Questions sur les conseillers
            elif any(word in question_lower for word in ['conseiller', 'commercial', 'vendeur', 'meilleur']):
                response += self.analyze_conseiller_data()
            
            # Questions sur les produits
            elif any(word in question_lower for word in ['produit', 'top', 'meilleur', 'scpi', 'per', 'immo']):
                response += self.analyze_produit_data()
            
            # Questions sur la saisonnalité
            elif any(word in question_lower for word in ['trimestre', 'saison', 'mois', 'tendance', 'évolution']):
                response += self.analyze_seasonality_data()
            
            # Questions sur la segmentation
            elif any(word in question_lower for word in ['segment', 'cpp', 'pgp', 'mono', 'multi']):
                response += self.analyze_segmentation_data()
            
            else:
                response += self.generate_general_summary()
            
            # Ajouter les sources de données
            response += f"\n\n📋 **Sources utilisées :** {', '.join(all_data)}"
            response += f"\n🕒 **Analyse effectuée le :** {datetime.now().strftime('%d/%m/%Y à %H:%M')}"
            
            # Nettoyer les données temporaires
            for key in list(self.data_files.keys()):
                if key.startswith('temp_'):
                    del self.data_files[key]
            
            return response
            
        except Exception as e:
            # Nettoyer les données temporaires même en cas d'erreur
            for key in list(self.data_files.keys()):
                if key.startswith('temp_'):
                    del self.data_files[key]
            return f"❌ **Erreur lors de l'analyse :** {str(e)}\n\n💡 Veuillez vérifier que vos données sont correctement chargées."
    
    def analyze_ca_data(self):
        """Analyse les données de CA"""
        response = "💰 **Analyse du Chiffre d'Affaires :**\n\n"
        
        for key, df in self.data_files.items():
            if df is not None and len(df) > 0:
                # Rechercher des colonnes de montant
                montant_cols = [col for col in df.columns if any(word in col.lower() for word in ['montant', 'premier', 'versement', 'ca', 'apport'])]
                
                if montant_cols:
                    try:
                        montant_col = montant_cols[0]
                        # Nettoyer et calculer
                        montants = pd.to_numeric(df[montant_col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
                        montants = montants.dropna()
                        
                        if len(montants) > 0:
                            ca_total = montants.sum()
                            nb_transactions = len(montants)
                            panier_moyen = montants.mean()
                            
                            response += f"📊 **{key}** :\n"
                            response += f"- CA Total : **{ca_total:,.0f}€**\n"
                            response += f"- Nombre de transactions : **{nb_transactions:,}**\n"
                            response += f"- Panier moyen : **{panier_moyen:,.0f}€**\n\n"
                    except:
                        continue
        
        if "📊" not in response:
            response += "ℹ️ Aucune donnée financière trouvée dans les fichiers chargés.\n"
        
        return response
    
    def analyze_client_data(self):
        """Analyse les données clients"""
        response = "👥 **Analyse des Clients :**\n\n"
        
        for key, df in self.data_files.items():
            if df is not None and len(df) > 0:
                if 'Email' in df.columns:
                    nb_clients_uniques = df['Email'].nunique()
                    nb_total_transactions = len(df)
                    
                    response += f"📊 **{key}** :\n"
                    response += f"- Clients uniques : **{nb_clients_uniques:,}**\n"
                    response += f"- Total transactions : **{nb_total_transactions:,}**\n"
                    
                    # Analyser l'acquisition par année si possible
                    date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'versement', 'inscription'])]
                    if date_cols:
                        try:
                            date_col = date_cols[0]
                            df_temp = df.copy()
                            df_temp['Date_Clean'] = pd.to_datetime(df_temp[date_col], errors='coerce')
                            df_temp = df_temp.dropna(subset=['Date_Clean'])
                            
                            if len(df_temp) > 0:
                                # Nouveaux clients par année
                                premiers_clients = df_temp.groupby('Email')['Date_Clean'].min()
                                acquisitions_par_annee = premiers_clients.dt.year.value_counts().sort_index()
                                
                                response += f"- Nouveaux clients 2024 : **{acquisitions_par_annee.get(2024, 0):,}**\n"
                                response += f"- Nouveaux clients 2023 : **{acquisitions_par_annee.get(2023, 0):,}**\n"
                        except:
                            pass
                    
                    response += "\n"
        
        if "📊" not in response:
            response += "ℹ️ Aucune donnée client trouvée dans les fichiers chargés.\n"
        
        return response
    
    def analyze_conseiller_data(self):
        """Analyse les données conseillers"""
        response = "👨‍💼 **Analyse des Conseillers :**\n\n"
        
        for key, df in self.data_files.items():
            if df is not None and len(df) > 0:
                conseiller_cols = [col for col in df.columns if any(word in col.lower() for word in ['conseiller', 'owner', 'commercial', 'agent'])]
                
                if conseiller_cols:
                    try:
                        conseiller_col = conseiller_cols[0]
                        
                        # Top conseillers par nombre de transactions
                        top_conseillers = df[conseiller_col].value_counts().head(5)
                        
                        response += f"🏆 **Top 5 Conseillers - {key}** :\n"
                        for i, (conseiller, nb) in enumerate(top_conseillers.items(), 1):
                            response += f"{i}. **{conseiller}** : {nb:,} transactions\n"
                        
                        # Si on a des montants, calculer le CA par conseiller
                        montant_cols = [col for col in df.columns if any(word in col.lower() for word in ['montant', 'premier', 'versement'])]
                        if montant_cols:
                            montant_col = montant_cols[0]
                            ca_par_conseiller = df.groupby(conseiller_col)[montant_col].apply(
                                lambda x: pd.to_numeric(x.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').sum()
                            ).sort_values(ascending=False).head(3)
                            
                            response += f"\n💰 **Top 3 CA - {key}** :\n"
                            for i, (conseiller, ca) in enumerate(ca_par_conseiller.items(), 1):
                                response += f"{i}. **{conseiller}** : {ca:,.0f}€\n"
                        
                        response += "\n"
                    except:
                        continue
        
        if "🏆" not in response:
            response += "ℹ️ Aucune donnée conseiller trouvée dans les fichiers chargés.\n"
        
        return response
    
    def analyze_produit_data(self):
        """Analyse les données produits"""
        response = "🏷️ **Analyse des Produits :**\n\n"
        
        for key, df in self.data_files.items():
            if df is not None and len(df) > 0 and 'Produit' in df.columns:
                try:
                    # Top produits par volume
                    top_produits = df['Produit'].value_counts().head(5)
                    
                    response += f"📊 **Top 5 Produits - {key}** :\n"
                    for i, (produit, nb) in enumerate(top_produits.items(), 1):
                        response += f"{i}. **{produit}** : {nb:,} ventes\n"
                    
                    # CA par produit si disponible
                    montant_cols = [col for col in df.columns if any(word in col.lower() for word in ['montant', 'premier', 'versement'])]
                    if montant_cols:
                        montant_col = montant_cols[0]
                        ca_par_produit = df.groupby('Produit')[montant_col].apply(
                            lambda x: pd.to_numeric(x.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').sum()
                        ).sort_values(ascending=False).head(3)
                        
                        response += f"\n💰 **Top 3 CA par produit - {key}** :\n"
                        for i, (produit, ca) in enumerate(ca_par_produit.items(), 1):
                            response += f"{i}. **{produit}** : {ca:,.0f}€\n"
                    
                    response += "\n"
                except:
                    continue
        
        if "📊" not in response:
            response += "ℹ️ Aucune donnée produit trouvée dans les fichiers chargés.\n"
        
        return response
    
    def analyze_seasonality_data(self):
        """Analyse la saisonnalité"""
        response = "📈 **Analyse de Saisonnalité :**\n\n"
        
        for key, df in self.data_files.items():
            if df is not None and len(df) > 0:
                date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'versement'])]
                
                if date_cols:
                    try:
                        date_col = date_cols[0]
                        df_temp = df.copy()
                        df_temp['Date_Clean'] = pd.to_datetime(df_temp[date_col], errors='coerce')
                        df_temp = df_temp.dropna(subset=['Date_Clean'])
                        
                        if len(df_temp) > 0:
                            # Analyse par trimestre
                            df_temp['Trimestre'] = df_temp['Date_Clean'].dt.quarter
                            ventes_par_trimestre = df_temp['Trimestre'].value_counts().sort_index()
                            
                            response += f"🗓️ **Répartition trimestrielle - {key}** :\n"
                            trimestre_labels = {1: 'T1 (Hiver)', 2: 'T2 (Printemps)', 3: 'T3 (Été)', 4: 'T4 (Automne)'}
                            for trimestre, nb in ventes_par_trimestre.items():
                                response += f"- **{trimestre_labels[trimestre]}** : {nb:,} ventes\n"
                            
                            meilleur_trimestre = ventes_par_trimestre.idxmax()
                            response += f"🏆 **Meilleur trimestre** : {trimestre_labels[meilleur_trimestre]}\n\n"
                    except:
                        continue
        
        if "🗓️" not in response:
            response += "ℹ️ Aucune donnée temporelle trouvée dans les fichiers chargés.\n"
        
        return response
    
    def analyze_segmentation_data(self):
        """Analyse les données de segmentation"""
        response = "🎯 **Analyse de Segmentation :**\n\n"
        
        # Cette fonction nécessiterait d'avoir accès aux analyses de segmentation
        # Pour l'instant, on donne des informations générales
        response += "ℹ️ L'analyse de segmentation est disponible dans l'onglet '🎯 Segmentation'.\n"
        response += "Vous y trouverez :\n"
        response += "- 🏠 **IMMO Mono-équipé** : Clients avec uniquement produits IMMO\n"
        response += "- 🟢 **CPP Actif** : Gros potentiel >30k€ (actif 2024)\n"
        response += "- 🟡 **PGP+ Actif** : Moyen potentiel 10-30k€ (actif 2024)\n"
        response += "- 🟠 **PGP Actif** : Petit potentiel <10k€ (actif 2024)\n"
        response += "- 🔴 **CPP/PGP+ Inactifs** : Segments inactifs par taille de panier\n\n"
        response += "💡 Chargez vos données dans l'onglet '🚀 Analyse 2025' pour voir la segmentation détaillée.\n"
        
        return response
    
    def generate_general_summary(self):
        """Génère un résumé général"""
        response = "📊 **Résumé Général des Données :**\n\n"
        
        total_files = len([df for df in self.data_files.values() if df is not None])
        response += f"📁 **Fichiers chargés** : {total_files}\n\n"
        
        for key, df in self.data_files.items():
            if df is not None and len(df) > 0:
                response += f"📈 **{key}** :\n"
                response += f"- {len(df):,} lignes\n"
                response += f"- {len(df.columns)} colonnes\n"
                if 'Email' in df.columns:
                    response += f"- {df['Email'].nunique():,} clients uniques\n"
                response += "\n"
        
        response += "💡 **Posez des questions plus spécifiques pour obtenir des analyses détaillées !**\n"
        
        return response
    
    def render_entretiens_tab(self):
        """Onglet des entretiens avec sous-onglets"""
        st.markdown('<h2 class="sub-header">🗣️ Analyse des Entretiens</h2>', unsafe_allow_html=True)
        
        entretiens_tabs = st.tabs(["Entretiens Épargne", "Entretiens Immobiliers"])
        
        with entretiens_tabs[0]:
            st.subheader("📁 Chargement des données - Entretiens Épargne")
            df_entretiens_epargne = self.load_file(
                "📞 Fichier entretiens épargne", 
                "file_entretiens_epargne_tab",
                ["xlsx", "csv"],
                description="Colonnes attendues: Date, Contact, Conseiller, Statut, Durée, Type d'entretien"
            )
            
            if df_entretiens_epargne is not None:
                st.markdown("---")
                analyses.entretiens.analyser_entretiens_epargne(df_entretiens_epargne, None, None)
            else:
                st.info("💡 Chargez le fichier des entretiens épargne pour voir les analyses.")
        
        with entretiens_tabs[1]:
            st.subheader("📁 Chargement des données - Entretiens Immobiliers")
            df_entretiens_immo = self.load_file(
                "🏠 Fichier entretiens immobiliers", 
                "file_entretiens_immo_tab",
                ["xlsx", "csv"],
                description="Colonnes attendues: Date, Contact, Conseiller, Statut, Type de bien, Durée"
            )
            
            if df_entretiens_immo is not None:
                st.markdown("---")
                analyser_entretiens(df_entretiens_immo, key_suffix="_immo")
            else:
                st.info("💡 Chargez le fichier des entretiens immobiliers pour voir les analyses.")
    
    def render_conversions_tab(self):
        """Onglet des conversions avec liaison de données"""
        st.markdown('<h2 class="sub-header">🔄 Analyse des Conversions</h2>', unsafe_allow_html=True)
        
        st.info("💡 Cette section lie les données de plusieurs fichiers (entretiens, RDV et souscriptions) pour calculer les taux de conversion.")
        
        conversion_tabs = st.tabs(["Conversion Épargne", "Conversion Immobilier"])
        
        with conversion_tabs[0]:
            st.subheader("📁 Chargement des données - Conversion Épargne")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                df_entretiens_epargne = self.load_file(
                    "📞 Entretiens épargne", 
                    "conv_entretiens_epargne",
                    ["xlsx", "csv"],
                    description="Fichier des entretiens épargne"
                )
            with col2:
                df_rdv_epargne = self.load_file(
                    "📅 RDV épargne", 
                    "conv_rdv_epargne",
                    ["xlsx", "csv"],
                    description="Fichier des RDV épargne"
                )
            with col3:
                df_souscriptions_epargne = self.load_file(
                    "💰 Souscriptions épargne", 
                    "conv_souscriptions_epargne",
                    ["xlsx", "csv"],
                    description="Fichier des souscriptions épargne"
                )
            
            if any([df_entretiens_epargne, df_rdv_epargne, df_souscriptions_epargne]):
                st.markdown("---")
                self.analyze_conversion(df_entretiens_epargne, df_rdv_epargne, df_souscriptions_epargne, "épargne")
            else:
                st.info("💡 Chargez au moins 2 fichiers parmi les 3 pour calculer les taux de conversion.")
        
        with conversion_tabs[1]:
            st.subheader("📁 Chargement des données - Conversion Immobilier")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                df_entretiens_immo = self.load_file(
                    "🏠 Entretiens immo", 
                    "conv_entretiens_immo",
                    ["xlsx", "csv"],
                    description="Fichier des entretiens immobiliers"
                )
            with col2:
                df_rdv_immo = self.load_file(
                    "📅 RDV immo", 
                    "conv_rdv_immo",
                    ["xlsx", "csv"],
                    description="Fichier des RDV immobiliers"
                )
            with col3:
                df_souscriptions_immo = self.load_file(
                    "🏢 Souscriptions immo", 
                    "conv_souscriptions_immo",
                    ["xlsx", "csv"],
                    description="Fichier des souscriptions immobilières"
                )
            
            if any([df_entretiens_immo, df_rdv_immo, df_souscriptions_immo]):
                st.markdown("---")
                self.analyze_conversion(df_entretiens_immo, df_rdv_immo, df_souscriptions_immo, "immobilier")
            else:
                st.info("💡 Chargez au moins 2 fichiers parmi les 3 pour calculer les taux de conversion.")
    
    def analyze_conversion(self, df_entretiens, df_rdv, df_souscriptions, type_conversion):
        """Analyse de conversion avec gestion des fichiers manquants"""
        files_available = [df for df in [df_entretiens, df_rdv, df_souscriptions] if df is not None]
        
        if len(files_available) >= 2:
            analyser_conversion(df_entretiens, df_rdv, df_souscriptions)
        elif len(files_available) == 1:
            st.warning(f"⚠️ Analyse limitée - seulement un fichier disponible pour la conversion {type_conversion}.")
            analyser_conversion(df_entretiens, df_rdv, df_souscriptions)
        else:
            st.error(f"❌ Veuillez charger au moins deux fichiers parmi : entretiens {type_conversion}, RDV et souscriptions {type_conversion}.")
    
    def analyser_retard_cumule(self, df):
        """Analyse et affiche le retard cumulé par rapport à l'objectif annuel.
        Calcule également le retard sans les produits SOSK et BMSK.
        
        Args:
            df (DataFrame): DataFrame contenant les données de souscriptions
        """
        # S'assurer que les colonnes nécessaires existent
        if 'Date de souscription' not in df.columns or 'Montant du placement' not in df.columns:
            # Essayer de trouver des colonnes alternatives
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'souscription' in col.lower()]
            montant_columns = [col for col in df.columns if 'montant' in col.lower() or 'placement' in col.lower() or 'collecte' in col.lower()]
            
            if date_columns and montant_columns:
                st.warning(f"⚠️ Colonnes standards non trouvées. Utilisation de {date_columns[0]} et {montant_columns[0]} à la place.")
                date_column = date_columns[0]
                montant_column = montant_columns[0]
            else:
                st.error("❌ Les colonnes nécessaires pour l'analyse du retard cumulé n'ont pas été trouvées.")
                st.info("Colonnes disponibles: " + ", ".join(df.columns.tolist()))
                return
        else:
            date_column = 'Date de souscription'
            montant_column = 'Montant du placement'
        
        # Identifier la colonne contenant le type de produit
        produit_column = None
        produit_candidates = ['Produit', 'Type de produit', 'Nom du produit', 'Product', 'Product Type']
        for col in produit_candidates:
            if col in df.columns:
                produit_column = col
                break
        
        # Si aucune colonne standard n'est trouvée, chercher des colonnes contenant des mots-clés
        if produit_column is None:
            for col in df.columns:
                if 'produit' in col.lower() or 'product' in col.lower() or 'type' in col.lower():
                    produit_column = col
                    st.info(f"Utilisation de la colonne '{col}' pour identifier les produits.")
                    break
        
        # Convertir la date de souscription en datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Filtrer les lignes avec des dates valides
        df = df.dropna(subset=[date_column])
        
        # Ajouter une colonne pour l'année et le mois
        df['Année'] = df[date_column].dt.year
        df['Mois'] = df[date_column].dt.month
        
        # Obtenir l'année actuelle
        annee_actuelle = datetime.now().year
        
        # Filtrer les données pour l'année actuelle
        df_annee = df[df['Année'] == annee_actuelle].copy()
        
        if df_annee.empty:
            st.warning(f"⚠️ Aucune donnée disponible pour l'année {annee_actuelle}.")
            return
        
        # Obtenir le mois actuel
        mois_actuel = datetime.now().month
        
        # Calculer l'objectif proratisé à la date actuelle
        objectif_proratise = (OBJECTIF_ANNUEL_EPARGNE / 12) * mois_actuel
        
        # Calculer le montant total collecté depuis le début de l'année (tous produits)
        montant_collecte = df_annee[montant_column].sum()
        
        # Calculer le retard (ou l'avance) par rapport à l'objectif
        retard = objectif_proratise - montant_collecte
        
        # Calcul sans les produits SOSK et BMSK
        df_sans_sosk_bmsk = df_annee.copy()
        if produit_column is not None:
            # Filtrer les produits SOSK et BMSK (recherche insensible à la casse)
            mask_sosk_bmsk = df_annee[produit_column].astype(str).str.contains('SOSK|BMSK', case=False, regex=True)
            if mask_sosk_bmsk.any():
                df_sans_sosk_bmsk = df_annee[~mask_sosk_bmsk]
                montant_collecte_sans_sosk_bmsk = df_sans_sosk_bmsk[montant_column].sum()
                retard_sans_sosk_bmsk = objectif_proratise - montant_collecte_sans_sosk_bmsk
                
                # Calculer le montant des produits SOSK et BMSK
                montant_sosk_bmsk = montant_collecte - montant_collecte_sans_sosk_bmsk
            else:
                # Pas de produits SOSK ou BMSK trouvés
                montant_collecte_sans_sosk_bmsk = montant_collecte
                retard_sans_sosk_bmsk = retard
                montant_sosk_bmsk = 0
        else:
            # Pas de colonne produit identifiée
            montant_collecte_sans_sosk_bmsk = montant_collecte
            retard_sans_sosk_bmsk = retard
            montant_sosk_bmsk = 0
        
        # Afficher les métriques principales
        st.subheader("💰 Retard cumulé - Tous produits")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Objectif annuel", f"{OBJECTIF_ANNUEL_EPARGNE/1_000_000:.1f}M€")
        with col2:
            st.metric("Objectif à date", f"{objectif_proratise/1_000_000:.2f}M€")
        with col3:
            st.metric("Montant collecté", f"{montant_collecte/1_000_000:.2f}M€")
        with col4:
            delta_label = "de retard" if retard > 0 else "d'avance"
            st.metric("Écart", f"{abs(retard)/1_000_000:.2f}M€ {delta_label}", delta=-retard if retard > 0 else retard)
        
        # Afficher les métriques sans SOSK et BMSK
        st.subheader("💳 Retard cumulé - Sans produits SOSK/BMSK")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Montant collecté", f"{montant_collecte_sans_sosk_bmsk/1_000_000:.2f}M€")
        with col2:
            delta_label_sans_sosk = "de retard" if retard_sans_sosk_bmsk > 0 else "d'avance"
            st.metric("Écart", f"{abs(retard_sans_sosk_bmsk)/1_000_000:.2f}M€ {delta_label_sans_sosk}", delta=-retard_sans_sosk_bmsk if retard_sans_sosk_bmsk > 0 else retard_sans_sosk_bmsk)
        with col3:
            st.metric("Montant SOSK/BMSK", f"{montant_sosk_bmsk/1_000_000:.2f}M€", delta=montant_sosk_bmsk)
        
        # Calculer le pourcentage de réalisation de l'objectif
        pourcentage_realisation = (montant_collecte / objectif_proratise) * 100
        
        # Créer une jauge pour visualiser le pourcentage de réalisation
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pourcentage_realisation,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Pourcentage de réalisation de l'objectif à date"},
            delta={'reference': 100, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 150], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "yellow"},
                    {'range': [100, 150], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculer l'évolution mensuelle
        evolution_mensuelle = df_annee.groupby('Mois').agg(
            Montant=pd.NamedAgg(column=montant_column, aggfunc='sum')
        ).reset_index()
        
        # Ajouter le nom du mois
        evolution_mensuelle['Nom_Mois'] = evolution_mensuelle['Mois'].apply(lambda x: calendar.month_name[x])
        
        # Calculer le montant cumulé
        evolution_mensuelle['Montant_Cumule'] = evolution_mensuelle['Montant'].cumsum()
        
        # Calculer l'objectif mensuel et l'objectif cumulé
        evolution_mensuelle['Objectif_Mensuel'] = OBJECTIF_MENSUEL_EPARGNE
        evolution_mensuelle['Objectif_Cumule'] = evolution_mensuelle['Mois'].apply(lambda x: (OBJECTIF_ANNUEL_EPARGNE / 12) * x)
        
        # Calculer le retard cumulé
        evolution_mensuelle['Retard_Cumule'] = evolution_mensuelle['Objectif_Cumule'] - evolution_mensuelle['Montant_Cumule']
        
        # Créer un graphique pour visualiser l'évolution du retard cumulé
        fig_evolution = go.Figure()
        
        # Ajouter la ligne de montant cumulé
        fig_evolution.add_trace(go.Scatter(
            x=evolution_mensuelle['Nom_Mois'],
            y=evolution_mensuelle['Montant_Cumule'],
            mode='lines+markers',
            name='Montant Cumulé',
            line=dict(color='blue', width=3)
        ))
        
        # Ajouter la ligne d'objectif cumulé
        fig_evolution.add_trace(go.Scatter(
            x=evolution_mensuelle['Nom_Mois'],
            y=evolution_mensuelle['Objectif_Cumule'],
            mode='lines+markers',
            name='Objectif Cumulé',
            line=dict(color='green', width=3, dash='dash')
        ))
        
        # Ajouter des barres pour le retard cumulé
        fig_evolution.add_trace(go.Bar(
            x=evolution_mensuelle['Nom_Mois'],
            y=evolution_mensuelle['Retard_Cumule'],
            name='Retard Cumulé',
            marker_color=['red' if x > 0 else 'green' for x in evolution_mensuelle['Retard_Cumule']]
        ))
        
        fig_evolution.update_layout(
            title="📈 Évolution du Retard Cumulé par Rapport à l'Objectif Annuel",
            xaxis_title="Mois",
            yaxis_title="Montant (€)",
            height=600,
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Afficher le tableau détaillé
        st.subheader("📋 Détail Mensuel")
        
        # Préparer le tableau pour l'affichage
        tableau = evolution_mensuelle.copy()
        tableau['Nom_Mois'] = tableau['Mois'].apply(lambda x: calendar.month_name[x])
        tableau['Montant'] = tableau['Montant'].apply(lambda x: f"{x:,.0f}€")
        tableau['Montant_Cumule'] = tableau['Montant_Cumule'].apply(lambda x: f"{x:,.0f}€")
        tableau['Objectif_Mensuel'] = tableau['Objectif_Mensuel'].apply(lambda x: f"{x:,.0f}€")
        tableau['Objectif_Cumule'] = tableau['Objectif_Cumule'].apply(lambda x: f"{x:,.0f}€")
        tableau['Retard_Cumule'] = tableau['Retard_Cumule'].apply(lambda x: f"{x:,.0f}€")
        
        # Renommer les colonnes pour l'affichage
        tableau = tableau[['Nom_Mois', 'Montant', 'Montant_Cumule', 'Objectif_Mensuel', 'Objectif_Cumule', 'Retard_Cumule']]
        tableau.columns = ['Mois', 'Montant Mensuel', 'Montant Cumulé', 'Objectif Mensuel', 'Objectif Cumulé', 'Retard Cumulé']
        
        st.dataframe(tableau, use_container_width=True)
        
        # Téléchargement des données
        csv = tableau.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name=f"retard_cumule_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def analyser_donnees_2025(self, df):
        """Analyse spécialisée pour les données 2025 avec pipeline commercial"""
        st.subheader("🎯 Vue d'Ensemble Pipeline 2025")
        
        if df is None or df.empty:
            st.error("❌ Aucune donnée disponible pour l'analyse 2025")
            return
        
        # Préparation des données - regroupement par email
        df_clean = self.preparer_donnees_2025(df)
        
        # Première ligne de métriques - Informations générales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Total Lignes", f"{len(df):,}")
        
        with col2:
            emails_uniques = df_clean['Email'].nunique() if 'Email' in df_clean.columns else len(df_clean)
            st.metric("👥 Clients Uniques", f"{emails_uniques:,}", 
                     help="Basé sur les emails uniques")
        
        with col3:
            if 'Contact Owner' in df_clean.columns:
                owners_uniques = df_clean['Contact Owner'].nunique()
                st.metric("👨‍💼 Contact Owners", f"{owners_uniques}")
            else:
                st.metric("👨‍💼 Contact Owners", "N/A")
        
        with col4:
            if 'Produit' in df_clean.columns:
                produits_uniques = df_clean['Produit'].nunique()
                st.metric("📦 Produits", f"{produits_uniques}")
            else:
                st.metric("📦 Produits", "N/A")
        
        with col5:
            if 'Stage' in df_clean.columns:
                stages_uniques = df_clean['Stage'].nunique()
                st.metric("📊 Stages", f"{stages_uniques}")
            else:
                st.metric("📊 Stages", "N/A")
        
        # Deuxième ligne de métriques - Données financières
        if any(col in df_clean.columns for col in ['Premier versement', 'Apport net']):
            st.subheader("💰 Métriques Financières")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'Premier versement' in df_clean.columns:
                    # Convertir en numérique et calculer
                    from utils.data_processing_debug import safe_to_numeric_debug
                    premiers_versements = safe_to_numeric_debug(df_clean['Premier versement'])
                    if not premiers_versements.isna().all():
                        total_premiers = premiers_versements.sum()
                        st.metric("💸 Total Premiers Versements", f"{total_premiers:,.0f}€")
                    else:
                        st.metric("💸 Total Premiers Versements", "N/A")
                else:
                    st.metric("💸 Total Premiers Versements", "N/A")
            
            with col2:
                if 'Apport net' in df_clean.columns:
                    apports_nets = safe_to_numeric_debug(df_clean['Apport net'])
                    if not apports_nets.isna().all():
                        total_apports = apports_nets.sum()
                        st.metric("🏦 Total Apports Nets", f"{total_apports:,.0f}€")
                    else:
                        st.metric("🏦 Total Apports Nets", "N/A")
                else:
                    st.metric("🏦 Total Apports Nets", "N/A")
            
            with col3:
                if 'Premier versement' in df_clean.columns:
                    clients_avec_versement = (premiers_versements > 0).sum()
                    pourcentage_versement = (clients_avec_versement / len(df_clean)) * 100
                    st.metric("📈 % avec Versement", f"{pourcentage_versement:.1f}%")
                else:
                    st.metric("📈 % avec Versement", "N/A")
            
            with col4:
                if 'Profession' in df_clean.columns:
                    professions_uniques = df_clean['Profession'].nunique()
                    st.metric("💼 Professions", f"{professions_uniques}")
                else:
                    st.metric("💼 Professions", "N/A")
        
        # Afficher les statistiques de regroupement
        if 'Email' in df.columns and len(df) != len(df_clean):
            doublons = len(df) - len(df_clean)
            st.info(f"ℹ️ {doublons:,} doublons détectés et regroupés par email. Analyse basée sur {len(df_clean):,} clients uniques.")
        
        # Analyses organisées par catégories - 2 niveaux de navigation
        st.markdown("### 🎯 Choisissez votre analyse :")
        
        # Première rangée d'onglets - Analyses principales
        main_tabs = st.tabs([
            "📊 Analyses Pipeline", 
            "👥 Analyses Clients",
            "🎯 Segmentation",
            "📤 Export"
        ])
        
        with main_tabs[0]:  # Analyses Pipeline
            pipeline_subtabs = st.tabs([
                "📊 Pipeline par Stage",
                "👨‍💼 Contact Owners", 
                "📦 Produits",
                "💰 Financière"
            ])
            
            with pipeline_subtabs[0]:
                self.analyser_pipeline_stages(df_clean)
            
            with pipeline_subtabs[1]:
                self.analyser_par_contact_owner(df_clean)
            
            with pipeline_subtabs[2]:
                self.analyser_par_produit(df_clean)
            
            with pipeline_subtabs[3]:
                self.analyser_financier_2025(df_clean)
        
        with main_tabs[1]:  # Analyses Clients
            client_subtabs = st.tabs([
                "💼 Professions",
                "📅 Temporel",
                "💰 Hors Immo"
            ])
            
            with client_subtabs[0]:
                self.analyser_professions(df_clean)
            
            with client_subtabs[1]:
                self.analyser_suivi_temporel_2025(df_clean)
            
            with client_subtabs[2]:
                self.analyser_opportunites(df_clean)
        
        with main_tabs[2]:  # Segmentation
            segmentation_subtabs = st.tabs([
                "🎯 Multi-équipement",
                "🏠 Segmentation 7 Segments"
            ])
            
            with segmentation_subtabs[0]:
                self.analyser_multi_equipement(df_clean)
            
            with segmentation_subtabs[1]:
                self.analyser_segmentation_immo(df_clean)
        
        with main_tabs[3]:  # Export
            self.generer_export_2025(df_clean)
    
    def preparer_donnees_2025(self, df):
        """Prépare et nettoie les données 2025 en regroupant par email"""
        import pandas as pd
        import numpy as np
        
        df_clean = df.copy()
        
        # Nettoyer et normaliser les emails
        if 'Email' in df_clean.columns:
            df_clean['Email'] = df_clean['Email'].astype(str).str.strip().str.lower()
            df_clean['Email'] = df_clean['Email'].replace(['nan', 'none', ''], np.nan)
            
            # Vérifier s'il y a des doublons
            emails_avant = len(df_clean)
            emails_uniques = df_clean['Email'].nunique()
            doublons = emails_avant - emails_uniques
            
            if doublons > 0:
                # Stratégies de regroupement pour les données 2025
                agg_dict = {}
                
                # Fonctions d'agrégation
                def first_non_null(x):
                    return x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
                
                def concat_unique(x):
                    unique_vals = x.dropna().astype(str).unique()
                    unique_vals = unique_vals[unique_vals != 'nan']
                    return ' | '.join(unique_vals) if len(unique_vals) > 0 else np.nan
                
                def latest_date(x):
                    return x.dropna().max() if len(x.dropna()) > 0 else np.nan
                
                # Colonnes texte simples : prendre la première valeur
                colonnes_simples = ['Full Name', 'Phone', 'Mobile', 'Profession', 'Contact Owner']
                for col in colonnes_simples:
                    if col in df_clean.columns:
                        agg_dict[col] = first_non_null
                
                # Traitement spécial pour Produit avec normalisation
                if 'Produit' in df_clean.columns:
                    def first_non_null_normalized_produit(x):
                        x_clean = x.dropna()
                        if len(x_clean) == 0:
                            return np.nan
                        # Normaliser avant de prendre la première valeur
                        x_str = x_clean.astype(str).str.strip()
                        x_str = x_str.str.replace(r'\bimmo\b', 'Immo', regex=True, case=False)
                        x_str = x_str.str.replace(r'\bSCPIFI\b', 'SCPI', regex=True, case=False)
                        x_str = x_str.str.replace(r'\bperinsa\b', 'PERINSA', regex=True, case=False)
                        x_str = x_str.str.replace(r'\b(AVIESA|AVSA|AVPERENYS)\b', 'AVIE', regex=True, case=False)
                        return x_str.iloc[0]
                    agg_dict['Produit'] = first_non_null_normalized_produit
                
                # Colonnes à concaténer : garder toutes les valeurs uniques avec normalisation
                if 'Opportunité Name' in df_clean.columns:
                    def concat_unique_normalized(x):
                        unique_vals = x.dropna().astype(str)
                        # Normaliser avant de concaténer
                        unique_vals = unique_vals.str.replace(r'\bimmo\b', 'Immo', regex=True, case=False)
                        unique_vals = unique_vals.str.replace(r'\bSCPIFI\b', 'SCPI', regex=True, case=False)
                        unique_vals = unique_vals.str.replace(r'\bperinsa\b', 'PERINSA', regex=True, case=False)
                        unique_vals = unique_vals.str.replace(r'\b(AVIESA|AVSA|AVPERENYS)\b', 'AVIE', regex=True, case=False)
                        unique_vals = unique_vals.unique()
                        unique_vals = unique_vals[unique_vals != 'nan']
                        return ' | '.join(unique_vals) if len(unique_vals) > 0 else np.nan
                    agg_dict['Opportunité Name'] = concat_unique_normalized
                
                if 'Stage' in df_clean.columns:
                    # Pour le stage, prendre le plus avancé (dernière valeur non nulle)
                    agg_dict['Stage'] = lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan
                
                # Colonnes de dates : prendre la plus récente
                colonnes_dates = ['Date versement initial', 'Date validation 570', 'Date de passage comité']
                for col in colonnes_dates:
                    if col in df_clean.columns:
                        agg_dict[col] = latest_date
                
                # Colonnes financières : sommer
                def safe_sum_financier(x):
                    from utils.data_processing_debug import safe_to_numeric_debug
                    x_numeric = safe_to_numeric_debug(x)
                    return x_numeric.sum() if not x_numeric.isna().all() else 0
                
                colonnes_financieres = ['Premier versement', 'Apport net']
                for col in colonnes_financieres:
                    if col in df_clean.columns:
                        agg_dict[col] = safe_sum_financier
                
                # Effectuer le regroupement
                try:
                    df_clean = df_clean.groupby('Email').agg(agg_dict).reset_index()
                except Exception as e:
                    # En cas d'erreur, retourner les données sans regroupement
                    pass
        
        # Nettoyage final des colonnes texte
        colonnes_texte = ['Full Name', 'Phone', 'Mobile', 'Profession', 'Contact Owner', 'Opportunité Name', 'Stage', 'Produit']
        for col in colonnes_texte:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
                df_clean[col] = df_clean[col].replace(['nan', 'None', ''], np.nan)
        
        # Normalisation spécialisée pour les produits et termes métier
        if 'Produit' in df_clean.columns:
            # Normaliser les variantes de produits
            df_clean['Produit'] = df_clean['Produit'].str.replace(
                r'\bimmo\b', 'Immo', regex=True, case=False
            )
            df_clean['Produit'] = df_clean['Produit'].str.replace(
                r'\bSCPIFI\b', 'SCPI', regex=True, case=False
            )
            df_clean['Produit'] = df_clean['Produit'].str.replace(
                r'\bperinsa\b', 'PERINSA', regex=True, case=False
            )
            # Normaliser les variantes AVIE
            df_clean['Produit'] = df_clean['Produit'].str.replace(
                r'\b(AVIESA|AVSA|AVPERENYS)\b', 'AVIE', regex=True, case=False
            )
        
        # Normaliser aussi dans Opportunité Name si elle existe
        if 'Opportunité Name' in df_clean.columns:
            df_clean['Opportunité Name'] = df_clean['Opportunité Name'].str.replace(
                r'\bimmo\b', 'Immo', regex=True, case=False
            )
            df_clean['Opportunité Name'] = df_clean['Opportunité Name'].str.replace(
                r'\bSCPIFI\b', 'SCPI', regex=True, case=False
            )
            df_clean['Opportunité Name'] = df_clean['Opportunité Name'].str.replace(
                r'\bperinsa\b', 'PERINSA', regex=True, case=False
            )
            # Normaliser les variantes AVIE
            df_clean['Opportunité Name'] = df_clean['Opportunité Name'].str.replace(
                r'\b(AVIESA|AVSA|AVPERENYS)\b', 'AVIE', regex=True, case=False
            )
        
        return df_clean
    
    def analyser_pipeline_stages(self, df):
        """Analyse de la répartition par stage du pipeline"""
        st.subheader("📊 Répartition par Stage")
        
        if 'Stage' not in df.columns:
            st.warning("⚠️ Colonne 'Stage' non trouvée")
            return
        
        # Compter les stages
        stage_counts = df['Stage'].value_counts()
        
        if not stage_counts.empty:
            # Graphique en barres
            import plotly.express as px
            fig = px.bar(
                x=stage_counts.index,
                y=stage_counts.values,
                title="Distribution des Contacts par Stage",
                labels={'x': 'Stage', 'y': 'Nombre de Contacts'},
                text=stage_counts.values
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            stage_df = pd.DataFrame({
                'Stage': stage_counts.index,
                'Nombre': stage_counts.values,
                'Pourcentage': (stage_counts.values / len(df) * 100).round(1)
            })
            st.dataframe(stage_df, use_container_width=True)
    
    def analyser_par_contact_owner(self, df):
        """Analyse par propriétaire de contact"""
        st.subheader("👨‍💼 Performance par Contact Owner")
        
        if 'Contact Owner' not in df.columns:
            st.warning("⚠️ Colonne 'Contact Owner' non trouvée")
            return
        
        # Analyse par contact owner
        owner_stats = df['Contact Owner'].value_counts()
        
        if not owner_stats.empty:
            # Top 10 contact owners
            top_owners = owner_stats.head(10)
            
            import plotly.express as px
            fig = px.bar(
                x=top_owners.values,
                y=top_owners.index,
                orientation='h',
                title="Top 10 Contact Owners",
                labels={'x': 'Nombre de Contacts', 'y': 'Contact Owner'},
                text=top_owners.values
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau complet
            owner_df = pd.DataFrame({
                'Contact Owner': owner_stats.index,
                'Nombre de Contacts': owner_stats.values,
                'Part (%)': (owner_stats.values / len(df) * 100).round(1)
            })
            st.dataframe(owner_df, use_container_width=True)
    
    def analyser_par_produit(self, df):
        """Analyse par produit"""
        st.subheader("📦 Répartition par Produit")
        
        if 'Produit' not in df.columns:
            st.warning("⚠️ Colonne 'Produit' non trouvée")
            return
        
        # Compter les produits
        produit_counts = df['Produit'].value_counts()
        
        if not produit_counts.empty:
            # Graphique en secteurs
            import plotly.express as px
            fig = px.pie(
                values=produit_counts.values,
                names=produit_counts.index,
                title="Répartition des Clients par Produit"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique en barres
            fig_bar = px.bar(
                x=produit_counts.index,
                y=produit_counts.values,
                title="Distribution des Produits",
                labels={'x': 'Produit', 'y': 'Nombre de Clients'},
                text=produit_counts.values
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Tableau détaillé avec croisement Stage si disponible
            produit_df = pd.DataFrame({
                'Produit': produit_counts.index,
                'Nombre de Clients': produit_counts.values,
                'Pourcentage': (produit_counts.values / len(df) * 100).round(1)
            })
            st.dataframe(produit_df, use_container_width=True)
            
            # Croisement Produit x Stage si disponible
            if 'Stage' in df.columns:
                st.subheader("📊 Croisement Produit × Stage")
                crosstab_produit_stage = pd.crosstab(df['Produit'], df['Stage'], margins=True)
                st.dataframe(crosstab_produit_stage, use_container_width=True)
    
    def analyser_financier_2025(self, df):
        """Analyse des données financières"""
        st.subheader("💰 Analyse Financière")
        
        # Analyser Premier versement
        if 'Premier versement' in df.columns:
            st.subheader("💸 Premiers Versements")
            
            from utils.data_processing_debug import safe_to_numeric_debug
            premiers_versements = safe_to_numeric_debug(df['Premier versement'])
            
            if not premiers_versements.isna().all():
                # Métriques
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total = premiers_versements.sum()
                    st.metric("💰 Total", f"{total:,.0f}€")
                with col2:
                    moyenne = premiers_versements.mean()
                    st.metric("📊 Moyenne", f"{moyenne:,.0f}€")
                with col3:
                    mediane = premiers_versements.median()
                    st.metric("📈 Médiane", f"{mediane:,.0f}€")
                with col4:
                    avec_versement = (premiers_versements > 0).sum()
                    st.metric("👥 Avec Versement", f"{avec_versement:,}")
                
                # Distribution
                import plotly.express as px
                versements_non_null = premiers_versements[premiers_versements > 0]
                if not versements_non_null.empty:
                    fig = px.histogram(
                        versements_non_null,
                        title="Distribution des Premiers Versements",
                        labels={'value': 'Montant (€)', 'count': 'Nombre de Clients'},
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Analyser Apport net
        if 'Apport net' in df.columns:
            st.subheader("🏦 Apports Nets")
            
            apports_nets = safe_to_numeric_debug(df['Apport net'])
            
            if not apports_nets.isna().all():
                # Métriques
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_apport = apports_nets.sum()
                    st.metric("💰 Total Apports", f"{total_apport:,.0f}€")
                with col2:
                    moyenne_apport = apports_nets.mean()
                    st.metric("📊 Moyenne", f"{moyenne_apport:,.0f}€")
                with col3:
                    mediane_apport = apports_nets.median()
                    st.metric("📈 Médiane", f"{mediane_apport:,.0f}€")
                with col4:
                    avec_apport = (apports_nets > 0).sum()
                    st.metric("👥 Avec Apport", f"{avec_apport:,}")
                
                # Distribution
                apports_non_null = apports_nets[apports_nets > 0]
                if not apports_non_null.empty:
                    fig = px.histogram(
                        apports_non_null,
                        title="Distribution des Apports Nets",
                        labels={'value': 'Montant (€)', 'count': 'Nombre de Clients'},
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Analyse croisée si les deux colonnes existent
        if 'Premier versement' in df.columns and 'Apport net' in df.columns:
            st.subheader("🔄 Relation Premiers Versements vs Apports Nets")
            
            versements_clean = safe_to_numeric_debug(df['Premier versement'])
            apports_clean = safe_to_numeric_debug(df['Apport net'])
            
            # Créer un DataFrame pour l'analyse croisée
            df_financier = pd.DataFrame({
                'Premier versement': versements_clean,
                'Apport net': apports_clean
            }).dropna()
            
            if not df_financier.empty:
                fig = px.scatter(
                    df_financier,
                    x='Premier versement',
                    y='Apport net',
                    title="Relation entre Premiers Versements et Apports Nets",
                    labels={'Premier versement': 'Premier Versement (€)', 'Apport net': 'Apport Net (€)'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Corrélation
                if len(df_financier) > 1:
                    correlation = df_financier.corr().iloc[0, 1]
                    st.metric("📊 Corrélation", f"{correlation:.3f}")
    
    def analyser_professions(self, df):
        """Analyse des professions"""
        st.subheader("💼 Répartition par Profession")
        
        if 'Profession' not in df.columns:
            st.warning("⚠️ Colonne 'Profession' non trouvée")
            return
        
        # Compter les professions
        prof_counts = df['Profession'].value_counts().head(15)
        
        if not prof_counts.empty:
            import plotly.express as px
            fig = px.pie(
                values=prof_counts.values,
                names=prof_counts.index,
                title="Top 15 Professions"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau
            prof_df = pd.DataFrame({
                'Profession': prof_counts.index,
                'Nombre': prof_counts.values,
                'Pourcentage': (prof_counts.values / len(df) * 100).round(1)
            })
            st.dataframe(prof_df, use_container_width=True)
    
    def analyser_suivi_temporel_2025(self, df):
        """Analyse temporelle des dates"""
        st.subheader("📅 Suivi Temporel")
        
        # Analyser les colonnes de date disponibles
        date_columns = ['Date versement initial', 'Date validation 570', 'Date de passage comité']
        
        for col in date_columns:
            if col in df.columns:
                st.subheader(f"📊 {col}")
                
                # Convertir en datetime
                try:
                    from utils.data_processing_debug import safe_to_datetime_debug
                    dates = safe_to_datetime_debug(df[col])
                    dates_valides = dates.dropna()
                    
                    if not dates_valides.empty:
                        # Évolution mensuelle
                        monthly_counts = dates_valides.dt.to_period('M').value_counts().sort_index()
                        
                        if not monthly_counts.empty:
                            import plotly.express as px
                            fig = px.line(
                                x=monthly_counts.index.astype(str),
                                y=monthly_counts.values,
                                title=f"Évolution mensuelle - {col}",
                                labels={'x': 'Mois', 'y': 'Nombre'},
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.metric(f"📊 {col} - Données valides", f"{len(dates_valides):,}")
                    else:
                        st.warning(f"⚠️ Aucune date valide trouvée pour {col}")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse de {col}: {str(e)}")
    
    def analyser_opportunites(self, df):
        """Analyse des montants moyens de souscription hors Immo"""
        st.subheader("💰 Montants Moyens de Souscription (hors Immo)")
        
        # Vérifier les colonnes nécessaires
        colonnes_requises = ['Produit', 'Premier versement']
        colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
        
        if colonnes_manquantes:
            st.warning(f"⚠️ Colonnes manquantes : {', '.join(colonnes_manquantes)}")
            return
        
        # Filtrer pour exclure les produits Immo
        df_hors_immo = df[~df['Produit'].str.contains('Immo', case=False, na=False)].copy()
        
        if df_hors_immo.empty:
            st.warning("⚠️ Aucune donnée trouvée pour les produits hors Immo")
            return
        
        st.info(f"📊 Analyse basée sur {len(df_hors_immo):,} clients avec produits hors Immo")
        
        # Convertir les montants en numérique
        from utils.data_processing_debug import safe_to_numeric_debug
        df_hors_immo['premier_versement_num'] = safe_to_numeric_debug(df_hors_immo['Premier versement'])
        
        # Filtrer les montants valides (> 0)
        df_valide = df_hors_immo[
            (df_hors_immo['premier_versement_num'] > 0) & 
            (df_hors_immo['premier_versement_num'].notna())
        ].copy()
        
        if df_valide.empty:
            st.warning("⚠️ Aucun montant de premier versement valide trouvé")
            return
        
        # Calculer les statistiques par produit
        stats_par_produit = df_valide.groupby('Produit')['premier_versement_num'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(0)
        
        stats_par_produit.columns = ['Nb Clients', 'Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max']
        stats_par_produit = stats_par_produit.sort_values('Moyenne', ascending=False)
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_clients = len(df_valide)
            st.metric("👥 Clients avec montant", f"{total_clients:,}")
        with col2:
            moyenne_globale = df_valide['premier_versement_num'].mean()
            st.metric("💰 Moyenne globale", f"{moyenne_globale:,.0f}€")
        with col3:
            mediane_globale = df_valide['premier_versement_num'].median()
            st.metric("📊 Médiane globale", f"{mediane_globale:,.0f}€")
        with col4:
            nb_produits = stats_par_produit.shape[0]
            st.metric("📦 Produits analysés", f"{nb_produits}")
        
        # Graphique des montants moyens par produit
        import plotly.express as px
        fig = px.bar(
            stats_par_produit.reset_index(),
            x='Produit',
            y='Moyenne',
            title="Montants Moyens de Premier Versement par Produit (hors Immo)",
            labels={'Moyenne': 'Montant Moyen (€)', 'Produit': 'Produit'},
            text='Moyenne'
        )
        fig.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
        fig.update_layout(xaxis_tickangle=45, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique en boîtes pour voir les distributions
        fig_box = px.box(
            df_valide,
            x='Produit',
            y='premier_versement_num',
            title="Distribution des Montants par Produit (hors Immo)",
            labels={'premier_versement_num': 'Premier Versement (€)', 'Produit': 'Produit'}
        )
        fig_box.update_layout(xaxis_tickangle=45, height=600)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Tableau détaillé
        st.subheader("📊 Statistiques Détaillées par Produit")
        
        # Formater le tableau pour l'affichage
        stats_display = stats_par_produit.copy()
        for col in ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max']:
            stats_display[col] = stats_display[col].apply(lambda x: f"{x:,.0f}€" if pd.notna(x) else "N/A")
        
        st.dataframe(stats_display, use_container_width=True)
        
        # Analyse complémentaire avec Apport net si disponible
        if 'Apport net' in df_valide.columns:
            st.subheader("🏦 Analyse Complémentaire - Apports Nets")
            
            df_valide['apport_net_num'] = safe_to_numeric_debug(df_valide['Apport net'])
            df_apports = df_valide[
                (df_valide['apport_net_num'] > 0) & 
                (df_valide['apport_net_num'].notna())
            ].copy()
            
            if not df_apports.empty:
                # Statistiques apports nets par produit
                stats_apports = df_apports.groupby('Produit')['apport_net_num'].agg([
                    'count', 'mean', 'median'
                ]).round(0)
                stats_apports.columns = ['Nb Clients', 'Apport Moyen', 'Apport Médiane']
                
                # Graphique apports nets
                fig_apport = px.bar(
                    stats_apports.reset_index(),
                    x='Produit',
                    y='Apport Moyen',
                    title="Apports Nets Moyens par Produit (hors Immo)",
                    text='Apport Moyen'
                )
                fig_apport.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
                fig_apport.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_apport, use_container_width=True)
        
        # Export des statistiques
        stats_csv = stats_par_produit.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Télécharger statistiques par produit (CSV)",
            data=stats_csv,
            file_name=f"montants_moyens_hors_immo_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def safe_get_numeric(self, row, column_name):
        """Récupère une valeur numérique d'une ligne en gérant les erreurs"""
        value = row.get(column_name, 0)
        if pd.isna(value) or value == '' or value == 'nan':
            return 0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0
    
    def extraire_type_produit(self, produit_complet):
        """
        Extrait le type de produit de base à partir d'une chaîne complète.
        Exemples:
        - '152128-Part-PERINSA-n°1' -> 'PERINSA'
        - 'SCPI-n°2' -> 'SCPI'  
        - 'AVIE' -> 'AVIE'
        - 'Immo' -> 'IMMO'
        """
        if not produit_complet or pd.isna(produit_complet):
            return None
            
        produit = str(produit_complet).strip().upper()
        
        # Dictionnaire de normalisation des produits (comme déjà défini ailleurs)
        normalisation_produits = {
            # Immobilier
            'IMMO': 'IMMO',
            'IMMOBILIER': 'IMMO',
            
            # SCPI/SCPIFI
            'SCPI': 'SCPI',
            'SCPIFI': 'SCPI',
            
            # PERINSA/PERINSA
            'PERINSA': 'PERINSA',
            'PERINSAV': 'PERINSA',
            
            # AVIE et variantes
            'AVIE': 'AVIE',
            'AVIESA': 'AVIE',
            'AVSA': 'AVIE', 
            'AVPERENYS': 'AVIE',
            
            # Autres produits courants
            'PER': 'PER',
            'ASSURANCE VIE': 'AVIE',
            'AV': 'AVIE'
        }
        
        # Chercher des motifs connus dans la chaîne
        for motif, type_normalise in normalisation_produits.items():
            if motif in produit:
                return type_normalise
        
        # Si aucun motif connu, essayer d'extraire le mot principal
        # Supprimer les préfixes numériques, les tirets, les 'n°', etc.
        import re
        # Pattern pour extraire le type de produit principal
        patterns = [
            r'.*?-([A-Z]+)-.*',  # 152128-Part-PERINSA-n°1 -> PERINSA
            r'([A-Z]+)-N°\d+',    # SCPI-n°2 -> SCPI
            r'([A-Z]+)\s*N°\d+',  # SCPI n°2 -> SCPI
            r'^([A-Z]+)',        # Premier mot en majuscules
        ]
        
        for pattern in patterns:
            match = re.match(pattern, produit)
            if match:
                type_extrait = match.group(1).strip()
                # Vérifier si c'est un type connu
                if type_extrait in normalisation_produits:
                    return normalisation_produits[type_extrait]
                else:
                    return type_extrait
        
        # Si rien ne fonctionne, retourner le produit nettoyé
        # Supprimer les chiffres, tirets et caractères spéciaux du début/fin
        produit_nettoye = re.sub(r'^[\d\-\s]+', '', produit)  # Supprimer chiffres/tirets du début
        produit_nettoye = re.sub(r'[\-\s]+N°.*$', '', produit_nettoye)  # Supprimer -n°X de la fin
        produit_nettoye = produit_nettoye.strip()
        
        return produit_nettoye if produit_nettoye else None
    
    def analyser_saisonnalite_ventes(self, df):
        """Analyse de la saisonnalité des ventes par produit par trimestre et nouveaux clients par an"""
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        from datetime import datetime
        
        st.subheader("📈 Analyse de Saisonnalité des Ventes")
        st.info("📊 Analyse de l'activité saisonnière par produit par trimestre et acquisition de nouveaux clients par année")
        
        # Vérifier les colonnes nécessaires
        colonnes_requises = ['Date versement initial', 'Produit', 'Premier versement']
        colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
        
        if colonnes_manquantes:
            st.error(f"❌ Colonnes manquantes : {', '.join(colonnes_manquantes)}")
            st.info("💡 Cette analyse nécessite : Date versement initial, Produit, Premier versement")
            return
        
        # Préparation des données
        df_work = df.copy()
        
        # Nettoyer et convertir la date
        df_work['Date_Clean'] = pd.to_datetime(df_work['Date versement initial'], errors='coerce')
        df_work = df_work.dropna(subset=['Date_Clean'])
        
        if len(df_work) == 0:
            st.warning("⚠️ Aucune date valide trouvée")
            return
        
        # Nettoyer les montants
        df_work['Montant_Clean'] = pd.to_numeric(
            df_work['Premier versement'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        ).fillna(0)
        
        # Nettoyer les produits
        df_work['Produit_Clean'] = df_work['Produit'].str.strip().str.upper()
        
        # Ajouter les dimensions temporelles
        df_work['Annee'] = df_work['Date_Clean'].dt.year
        df_work['Trimestre'] = df_work['Date_Clean'].dt.quarter
        df_work['Trimestre_Label'] = 'T' + df_work['Trimestre'].astype(str) + ' ' + df_work['Annee'].astype(str)
        df_work['Annee_Trimestre'] = df_work['Annee'].astype(str) + '-Q' + df_work['Trimestre'].astype(str)
        
        # Métriques générales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total ventes", f"{len(df_work):,}")
        with col2:
            st.metric("📅 Période", f"{df_work['Annee'].min()}-{df_work['Annee'].max()}")
        with col3:
            st.metric("🏷️ Produits uniques", f"{df_work['Produit_Clean'].nunique()}")
        with col4:
            ca_total = df_work['Montant_Clean'].sum()
            st.metric("💰 CA Total", f"{ca_total:,.0f}€")
        
        # Section 1: Saisonnalité par produit par trimestre
        st.subheader("🔄 Saisonnalité par Produit et Trimestre")
        
        # Top produits pour l'analyse
        top_produits = df_work['Produit_Clean'].value_counts().head(10).index.tolist()
        
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            produits_selectionnes = st.multiselect(
                "Sélectionner les produits à analyser",
                options=top_produits,
                default=top_produits[:5] if len(top_produits) >= 5 else top_produits,
                key="produits_saisonnalite"
            )
        
        with col2:
            annees_disponibles = sorted(df_work['Annee'].unique())
            annees_selectionnees = st.multiselect(
                "Sélectionner les années",
                options=annees_disponibles,
                default=annees_disponibles,
                key="annees_saisonnalite"
            )
        
        if produits_selectionnes and annees_selectionnees:
            # Filtrer les données
            df_filtre = df_work[
                (df_work['Produit_Clean'].isin(produits_selectionnes)) &
                (df_work['Annee'].isin(annees_selectionnees))
            ]
            
            # Agrégation par produit et trimestre
            saisonnalite = df_filtre.groupby(['Produit_Clean', 'Trimestre', 'Annee']).agg({
                'Montant_Clean': ['sum', 'count', 'mean'],
                'Date_Clean': 'count'
            }).round(0)
            
            saisonnalite.columns = ['CA', 'Nb_Ventes', 'Panier_Moyen', 'Volume']
            saisonnalite = saisonnalite.reset_index()
            
            # Graphiques de saisonnalité
            tab1, tab2, tab3 = st.tabs(["📊 Volume par Trimestre", "💰 CA par Trimestre", "📈 Évolution Temporelle"])
            
            with tab1:
                # Heatmap volume par produit/trimestre
                pivot_volume = df_filtre.groupby(['Produit_Clean', 'Trimestre']).size().reset_index(name='Volume')
                pivot_volume = pivot_volume.pivot(index='Produit_Clean', columns='Trimestre', values='Volume').fillna(0)
                
                fig = px.imshow(
                    pivot_volume.values,
                    labels=dict(x="Trimestre", y="Produit", color="Volume"),
                    x=['T1', 'T2', 'T3', 'T4'],
                    y=pivot_volume.index.tolist(),
                    title="🔥 Heatmap Volume des Ventes par Produit et Trimestre",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # CA par trimestre et produit
                ca_trimestre = df_filtre.groupby(['Trimestre', 'Produit_Clean'])['Montant_Clean'].sum().reset_index()
                
                fig = px.bar(
                    ca_trimestre,
                    x='Trimestre',
                    y='Montant_Clean',
                    color='Produit_Clean',
                    title="💰 Chiffre d'Affaires par Trimestre et Produit",
                    labels={'Montant_Clean': 'CA (€)', 'Trimestre': 'Trimestre'}
                )
                fig.update_layout(xaxis=dict(tickmode='array', tickvals=[1,2,3,4], ticktext=['T1', 'T2', 'T3', 'T4']))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Évolution temporelle par produit
                evolution = df_filtre.groupby(['Annee_Trimestre', 'Produit_Clean']).agg({
                    'Montant_Clean': 'sum',
                    'Date_Clean': 'count'
                }).reset_index()
                
                # Graphique en lignes pour chaque produit
                for produit in produits_selectionnes[:3]:  # Limiter à 3 produits pour la lisibilité
                    data_produit = evolution[evolution['Produit_Clean'] == produit]
                    if not data_produit.empty:
                        fig = px.line(
                            data_produit,
                            x='Annee_Trimestre',
                            y='Montant_Clean',
                            title=f"📈 Évolution {produit}",
                            markers=True
                        )
                        fig.update_layout(xaxis_title="Période", yaxis_title="CA (€)")
                        st.plotly_chart(fig, use_container_width=True)
        
        # Section 2: Acquisition de nouveaux clients par année
        st.subheader("👥 Acquisition de Nouveaux Clients par Année")
        
        if 'Email' in df_work.columns:
            # Identifier les nouveaux clients par année (première souscription)
            premiers_clients = df_work.groupby('Email')['Date_Clean'].min().reset_index()
            premiers_clients['Annee_Acquisition'] = premiers_clients['Date_Clean'].dt.year
            
            # Compter les nouveaux clients par année
            nouveaux_clients_par_annee = premiers_clients['Annee_Acquisition'].value_counts().sort_index()
            
            # Métriques nouveaux clients
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("👥 Total nouveaux clients", f"{len(premiers_clients):,}")
            with col2:
                if len(nouveaux_clients_par_annee) > 1:
                    croissance = ((nouveaux_clients_par_annee.iloc[-1] / nouveaux_clients_par_annee.iloc[-2]) - 1) * 100
                    st.metric("📈 Croissance annuelle", f"{croissance:+.1f}%")
                else:
                    st.metric("📈 Croissance annuelle", "N/A")
            with col3:
                moyenne_annuelle = nouveaux_clients_par_annee.mean()
                st.metric("📊 Moyenne annuelle", f"{moyenne_annuelle:.0f}")
            with col4:
                annee_peak = nouveaux_clients_par_annee.idxmax()
                st.metric("🎯 Meilleure année", f"{annee_peak}")
            
            # Graphiques nouveaux clients
            col1, col2 = st.columns(2)
            
            with col1:
                # Évolution des nouveaux clients par année
                fig = px.bar(
                    x=nouveaux_clients_par_annee.index,
                    y=nouveaux_clients_par_annee.values,
                    title="👥 Nouveaux Clients par Année",
                    labels={'x': 'Année', 'y': 'Nombre de nouveaux clients'},
                    text=nouveaux_clients_par_annee.values
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Répartition trimestrielle des acquisitions
                premiers_clients['Trimestre'] = premiers_clients['Date_Clean'].dt.quarter
                acquisition_trimestre = premiers_clients['Trimestre'].value_counts().sort_index()
                
                fig = px.pie(
                    values=acquisition_trimestre.values,
                    names=[f'T{i}' for i in acquisition_trimestre.index],
                    title="🔄 Saisonnalité Acquisition Clients",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Analyse détaillée par année et produit
            st.subheader("🔍 Analyse Croisée: Nouveaux Clients et Produits")
            
            # Croiser données clients avec produits préférés
            clients_produits = df_work.merge(
                premiers_clients[['Email', 'Annee_Acquisition']],
                on='Email'
            )
            
            # Produits préférés des nouveaux clients par année
            produits_nouveaux = clients_produits.groupby(['Annee_Acquisition', 'Produit_Clean']).size().reset_index(name='Volume')
            
            # Top produits par année pour nouveaux clients
            for annee in sorted(clients_produits['Annee_Acquisition'].unique())[-3:]:  # 3 dernières années
                data_annee = produits_nouveaux[produits_nouveaux['Annee_Acquisition'] == annee]
                top_produits_annee = data_annee.nlargest(5, 'Volume')
                
                if not top_produits_annee.empty:
                    with st.expander(f"📊 Top Produits Nouveaux Clients {annee}"):
                        fig = px.bar(
                            top_produits_annee,
                            x='Volume',
                            y='Produit_Clean',
                            orientation='h',
                            title=f"Top 5 Produits - Nouveaux Clients {annee}",
                            text='Volume'
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("⚠️ Colonne 'Email' manquante - Impossible d'analyser l'acquisition de nouveaux clients")
        
        # Export des données d'analyse
        st.subheader("📥 Export des Analyses")
        
        # Préparer les données d'export
        export_data = df_work.copy()
        
        # Ajouter des colonnes calculées
        export_data['Saison'] = export_data['Trimestre'].map({
            1: 'Hiver', 2: 'Printemps', 3: 'Été', 4: 'Automne'
        })
        
        csv_export = export_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger analyse saisonnalité (CSV)",
            data=csv_export,
            file_name=f"analyse_saisonnalite_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    def analyser_multi_equipement(self, df):
        """Analyse des clients mono-équipés vs multi-équipés basée sur les types de produits distincts"""
        st.subheader("🎯 Analyse du Multi-équipement Client")
        st.info("📊 Cette analyse compte les **types de produits distincts** par client (ex: PERINSA-n°1 et PERINSA-n°2 = 1 seul type)")
        
        if 'Produit' not in df.columns:
            st.warning("⚠️ Colonne 'Produit' non trouvée")
            return
        
        # Analyser les données uniquement après regroupement par email
        if 'Email' not in df.columns:
            st.warning("⚠️ Colonne 'Email' non trouvée pour analyser les clients uniques")
            return
        
        st.info("📊 Cette analyse est basée sur les clients uniques (regroupés par email)")
        
        # Pour analyser le multi-équipement, nous devons regarder les données originales
        # car après regroupement, nous n'avons plus le détail des produits multiples
        # Nous allons créer une analyse basée sur les colonnes 'Produit' et 'Opportunité Name'
        
        # Analyser les produits par client
        clients_produits = {}
        
        # Si les produits sont dans 'Produit', les analyser directement
        for idx, row in df.iterrows():
            email = row.get('Email', '')
            produit = row.get('Produit', '')
            opportunite = row.get('Opportunité Name', '')
            
            if pd.notna(email) and email != 'nan' and email != '':
                if email not in clients_produits:
                    clients_produits[email] = {
                        'produits': set(),
                        'opportunites': set(),
                        'full_name': row.get('Full Name', ''),
                        'profession': row.get('Profession', ''),
                        'stage': row.get('Stage', ''),
                        'phone': row.get('Phone', ''),
                        'mobile': row.get('Mobile', ''),
                        'contact_owner': row.get('Contact Owner', ''),
                        'premier_versement': self.safe_get_numeric(row, 'Premier versement'),
                        'apport_net': self.safe_get_numeric(row, 'Apport net')
                    }
                
                if pd.notna(produit) and produit != 'nan' and produit != '':
                    # Séparer les produits s'ils sont concaténés avec |
                    produits_split = str(produit).split('|') if '|' in str(produit) else [str(produit)]
                    for p in produits_split:
                        p_clean = p.strip()
                        if p_clean and p_clean != 'nan':
                            # Extraire le type de produit de base (sans les numéros)
                            type_produit = self.extraire_type_produit(p_clean)
                            if type_produit:
                                clients_produits[email]['produits'].add(type_produit)
                
                if pd.notna(opportunite) and opportunite != 'nan' and opportunite != '':
                    # Séparer les opportunités s'elles sont concaténées avec |
                    opp_split = str(opportunite).split('|') if '|' in str(opportunite) else [str(opportunite)]
                    for o in opp_split:
                        o_clean = o.strip()
                        if o_clean and o_clean != 'nan':
                            # Extraire le type de produit de base de l'opportunité
                            type_produit = self.extraire_type_produit(o_clean)
                            if type_produit:
                                clients_produits[email]['opportunites'].add(type_produit)
        
        # Créer un DataFrame d'analyse
        analyse_data = []
        for email, data in clients_produits.items():
            nb_produits = len(data['produits'])
            nb_opportunites = len(data['opportunites'])
            # Combiner produits et opportunités pour avoir une vue complète
            tous_produits = data['produits'].union(data['opportunites'])
            nb_total_produits = len(tous_produits)
            
            if nb_total_produits > 0:  # Ignorer les clients sans produits
                # Créer un numéro de téléphone consolidé (Phone ou Mobile)
                telephone = ''
                if data['phone'] and str(data['phone']) not in ['', 'nan', 'None']:
                    telephone = str(data['phone'])
                elif data['mobile'] and str(data['mobile']) not in ['', 'nan', 'None']:
                    telephone = str(data['mobile'])
                
                # Calculer le panier moyen hors Immo
                panier_moyen_hors_immo = 0
                if data['premier_versement'] and pd.notna(data['premier_versement']) and data['premier_versement'] > 0:
                    # Vérifier si le client a des produits immobiliers
                    produits_immo = {'IMMO', 'Immobilier'}
                    produits_client = tous_produits
                    has_immo = any(p in produits_immo for p in produits_client)
                    
                    # Si le client n'a que de l'immo, panier = 0, sinon utiliser le premier versement
                    if not (has_immo and len(produits_client) == 1):
                        panier_moyen_hors_immo = data['premier_versement']
                
                # Calculer la segmentation basée sur le panier moyen hors Immo
                segmentation = "PGP"  # Par défaut
                if panier_moyen_hors_immo >= 30000:
                    segmentation = "CPP"
                elif panier_moyen_hors_immo >= 10000:
                    segmentation = "PGP+"
                # Sinon reste "PGP" pour < 10k€

                analyse_data.append({
                    'Email': email,
                    'Full Name': data['full_name'],
                    'Profession': data['profession'],
                    'Stage': data['stage'],
                    'Phone': data['phone'],
                    'Mobile': data['mobile'],
                    'Telephone': telephone,
                    'Contact Owner': data['contact_owner'],
                    'Nb_Produits_Directs': nb_produits,
                    'Nb_Opportunites': nb_opportunites,
                    'Nb_Total_Produits': nb_total_produits,
                    'Produits': ' | '.join(sorted(data['produits'])),
                    'Opportunites': ' | '.join(sorted(data['opportunites'])),
                    'Tous_Produits': ' | '.join(sorted(tous_produits)),
                    'Type_Client': 'Multi-équipé' if nb_total_produits > 1 else 'Mono-équipé',
                    'Premier_Versement': data['premier_versement'],
                    'Apport_Net': data['apport_net'],
                    'Panier_Moyen_Hors_Immo': panier_moyen_hors_immo,
                    'Segmentation': segmentation
                })
        
        if not analyse_data:
            st.warning("⚠️ Aucune donnée de produit trouvée pour l'analyse")
            return
        
        df_analyse = pd.DataFrame(analyse_data)
        
        # Métriques principales
        total_clients = len(df_analyse)
        clients_mono = len(df_analyse[df_analyse['Type_Client'] == 'Mono-équipé'])
        clients_multi = len(df_analyse[df_analyse['Type_Client'] == 'Multi-équipé'])
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total Clients", f"{total_clients:,}")
        with col2:
            st.metric("🎯 Mono-équipés", f"{clients_mono:,}", 
                     help="Clients avec 1 seul produit")
        with col3:
            st.metric("🏆 Multi-équipés", f"{clients_multi:,}", 
                     help="Clients avec plusieurs produits")
        with col4:
            taux_multi = (clients_multi / total_clients) * 100 if total_clients > 0 else 0
            st.metric("📈 Taux Multi-équip.", f"{taux_multi:.1f}%")
        
        # Explication de la logique de calcul
        with st.expander("💡 Comment fonctionne le calcul multi-équipement ?", expanded=False):
            st.markdown("""
            **Méthode de calcul :**
            - ✅ **Mono-équipé** : Client avec un seul **type** de produit
                - Exemple : `PERINSA-n°1` et `PERINSA-n°2` = **1 seul type** (PERINSA) 
            - ✅ **Multi-équipé** : Client avec plusieurs **types** de produits
                - Exemple : `PERINSA-n°1` + `SCPI-n°1` = **2 types** (PERINSA + SCPI)
            
            **Normalisation des produits :**
            - `SCPIFI` → `SCPI`
            - `AVIESA`, `AVSA`, `AVPERENYS` → `AVIE`  
            - `IMMO`, `Immobilier` → `IMMO`
            
            **Sources de données :**
            - Colonne `Produit` : produits directs
            - Colonne `Opportunité Name` : opportunités commerciales
            """)
            
            # Montrer quelques exemples concrets s'il y en a
            if len(df_analyse) > 0:
                st.markdown("**Exemples concrets de votre dataset :**")
                
                # Exemples de clients mono-équipés
                mono_exemples = df_analyse[df_analyse['Type_Client'] == 'Mono-équipé'].head(3)
                if not mono_exemples.empty:
                    st.markdown("**Clients mono-équipés :**")
                    for _, row in mono_exemples.iterrows():
                        st.write(f"• {row['Full Name']} : {row['Tous_Produits']}")
                
                # Exemples de clients multi-équipés
                multi_exemples = df_analyse[df_analyse['Type_Client'] == 'Multi-équipé'].head(3)
                if not multi_exemples.empty:
                    st.markdown("**Clients multi-équipés :**")
                    for _, row in multi_exemples.iterrows():
                        st.write(f"• {row['Full Name']} : {row['Tous_Produits']} ({row['Nb_Total_Produits']} types)")
        
        # Graphiques
        import plotly.express as px
        
        # Répartition mono vs multi
        repartition = df_analyse['Type_Client'].value_counts()
        fig_pie = px.pie(
            values=repartition.values,
            names=repartition.index,
            title="Répartition Mono-équipés vs Multi-équipés",
            color_discrete_map={'Mono-équipé': '#ff7f7f', 'Multi-équipé': '#7fbf7f'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Distribution du nombre de produits
        distribution_produits = df_analyse['Nb_Total_Produits'].value_counts().sort_index()
        fig_bar = px.bar(
            x=distribution_produits.index,
            y=distribution_produits.values,
            title="Distribution du Nombre de Produits par Client",
            labels={'x': 'Nombre de Produits', 'y': 'Nombre de Clients'},
            text=distribution_produits.values
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Analyse par profession si disponible
        if 'Profession' in df_analyse.columns and df_analyse['Profession'].notna().any():
            st.subheader("💼 Multi-équipement par Profession")
            
            prof_analysis = df_analyse.groupby(['Profession', 'Type_Client']).size().unstack(fill_value=0)
            if not prof_analysis.empty:
                # Calculer le taux de multi-équipement par profession
                prof_analysis['Total'] = prof_analysis.sum(axis=1)
                prof_analysis['Taux_Multi'] = (prof_analysis.get('Multi-équipé', 0) / prof_analysis['Total'] * 100).round(1)
                prof_analysis_sorted = prof_analysis.sort_values('Taux_Multi', ascending=False).head(10)
                
                fig_prof = px.bar(
                    prof_analysis_sorted.reset_index(),
                    x='Profession',
                    y='Taux_Multi',
                    title="Top 10 Professions - Taux de Multi-équipement",
                    labels={'Taux_Multi': 'Taux Multi-équipement (%)', 'Profession': 'Profession'},
                    text='Taux_Multi'
                )
                fig_prof.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_prof.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_prof, use_container_width=True)
        
        # Analyse financière si disponible
        from utils.data_processing_debug import safe_to_numeric_debug
        df_analyse['Premier_Versement_Num'] = safe_to_numeric_debug(df_analyse['Premier_Versement'])
        df_analyse['Apport_Net_Num'] = safe_to_numeric_debug(df_analyse['Apport_Net'])
        
        df_financier = df_analyse[
            (df_analyse['Premier_Versement_Num'] > 0) | 
            (df_analyse['Apport_Net_Num'] > 0)
        ].copy()
        
        if not df_financier.empty:
            st.subheader("💰 Impact Financier du Multi-équipement")
            
            # Comparaison montants moyens
            financial_comparison = df_financier.groupby('Type_Client').agg({
                'Premier_Versement_Num': ['count', 'mean', 'median'],
                'Apport_Net_Num': ['mean', 'median']
            }).round(0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Premier_Versement_Num' in df_financier.columns:
                    mono_avg = df_financier[df_financier['Type_Client'] == 'Mono-équipé']['Premier_Versement_Num'].mean()
                    multi_avg = df_financier[df_financier['Type_Client'] == 'Multi-équipé']['Premier_Versement_Num'].mean()
                    
                    if pd.notna(mono_avg) and pd.notna(multi_avg):
                        st.metric("💸 Premier Versement Moyen - Mono", f"{mono_avg:,.0f}€")
                        st.metric("💸 Premier Versement Moyen - Multi", f"{multi_avg:,.0f}€")
                        
                        ecart = ((multi_avg - mono_avg) / mono_avg * 100) if mono_avg > 0 else 0
                        st.metric("📊 Écart Multi vs Mono", f"{ecart:+.1f}%")
            
            with col2:
                if 'Apport_Net_Num' in df_financier.columns:
                    mono_apport = df_financier[df_financier['Type_Client'] == 'Mono-équipé']['Apport_Net_Num'].mean()
                    multi_apport = df_financier[df_financier['Type_Client'] == 'Multi-équipé']['Apport_Net_Num'].mean()
                    
                    if pd.notna(mono_apport) and pd.notna(multi_apport):
                        st.metric("🏦 Apport Net Moyen - Mono", f"{mono_apport:,.0f}€")
                        st.metric("🏦 Apport Net Moyen - Multi", f"{multi_apport:,.0f}€")
        
        # Tableaux détaillés
        st.subheader("📊 Détail des Clients Multi-équipés")
        
        df_multi = df_analyse[df_analyse['Type_Client'] == 'Multi-équipé'].copy()
        if not df_multi.empty:
            df_multi_display = df_multi[['Full Name', 'Profession', 'Nb_Total_Produits', 'Tous_Produits', 'Stage']].copy()
            df_multi_display = df_multi_display.sort_values('Nb_Total_Produits', ascending=False)
            st.dataframe(df_multi_display, use_container_width=True)
        
        st.subheader("📋 Résumé par Nombre de Produits")
        summary = df_analyse.groupby('Nb_Total_Produits').agg({
            'Email': 'count',
            'Premier_Versement_Num': 'mean',
            'Apport_Net_Num': 'mean'
        }).round(0)
        summary.columns = ['Nombre de Clients', 'Premier Versement Moyen', 'Apport Net Moyen']
        summary = summary.reset_index()
        summary.rename(columns={'Nb_Total_Produits': 'Nombre de Produits'}, inplace=True)
        
        # Formater les montants
        for col in ['Premier Versement Moyen', 'Apport Net Moyen']:
            summary[col] = summary[col].apply(lambda x: f"{x:,.0f}€" if pd.notna(x) and x > 0 else "N/A")
        
        st.dataframe(summary, use_container_width=True)
        
        # Export
        # Colonnes à exporter avec les nouveaux champs demandés
        colonnes_export = [
            'Email', 'Full Name', 'Telephone', 'Contact Owner', 
            'Profession', 'Stage', 'Type_Client', 'Nb_Total_Produits', 
            'Tous_Produits', 'Premier_Versement', 'Apport_Net', 
            'Panier_Moyen_Hors_Immo', 'Segmentation'
        ]
        
        # Vérifier que les colonnes existent dans le DataFrame
        colonnes_disponibles = [col for col in colonnes_export if col in df_analyse.columns]
        
        export_data = df_analyse[colonnes_disponibles].copy()
        
        # Formatter les montants pour l'export avec gestion d'erreur
        def format_montant(x):
            """Formate un montant en gérant les valeurs vides ou non numériques"""
            if pd.isna(x) or x == '' or x == 0:
                return ""
            try:
                # Essayer de convertir en float
                montant = float(x)
                if montant == 0:
                    return ""
                return f"{montant:,.0f}€"
            except (ValueError, TypeError):
                # Si la conversion échoue, retourner vide
                return ""
        
        if 'Premier_Versement' in export_data.columns:
            export_data['Premier_Versement'] = export_data['Premier_Versement'].apply(format_montant)
        if 'Apport_Net' in export_data.columns:
            export_data['Apport_Net'] = export_data['Apport_Net'].apply(format_montant)
        if 'Panier_Moyen_Hors_Immo' in export_data.columns:
            export_data['Panier_Moyen_Hors_Immo'] = export_data['Panier_Moyen_Hors_Immo'].apply(format_montant)
        csv_multi = export_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger analyse multi-équipement (CSV)",
            data=csv_multi,
            file_name=f"analyse_multi_equipement_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Aperçu de l'export multi-équipement
        with st.expander("📄 Aperçu de l'export multi-équipement", expanded=False):
            # Créer une version avec colonnes renommées pour l'aperçu
            export_preview = export_data.copy()
            nouvelles_colonnes = []
            if 'Telephone' in export_preview.columns and 'Contact Owner' in export_preview.columns:
                nouvelles_colonnes.append("Téléphone, Contact Owner")
            if 'Panier_Moyen_Hors_Immo' in export_preview.columns and 'Segmentation' in export_preview.columns:
                nouvelles_colonnes.append("Panier Moyen Hors Immo, Segmentation (PGP/PGP+/CPP)")
            
            if nouvelles_colonnes:
                st.write(f"**✅ Nouvelles colonnes ajoutées :** {' | '.join(nouvelles_colonnes)}")
            
            colonnes_actuelles = list(export_preview.columns)
            st.write(f"**Colonnes exportées :** {', '.join(colonnes_actuelles)}")
            st.write(f"**Nombre de clients :** {len(export_preview):,}")
            
            # Afficher les critères de segmentation
            st.markdown("""
            **📊 Critères de Segmentation :**
            - **PGP** : Panier moyen < 10 000€
            - **PGP+** : Panier moyen entre 10 000€ et 30 000€  
            - **CPP** : Panier moyen ≥ 30 000€
            - *Note : Panier calculé hors produits immobiliers*
            """)
            
            if not export_preview.empty:
                st.dataframe(export_preview.head(5), use_container_width=True)
    
    def analyser_segmentation_immo(self, df):
        """Analyse de segmentation IMMO avec mono-équipés et 6 segments basés sur activité et panier moyen"""
        import plotly.express as px
        import pandas as pd
        from datetime import datetime
        
        st.subheader("🏠 Segmentation Complète")
        st.info("📊 Segmentation avec 7 segments : IMMO mono-équipé + 6 segments CPP/PGP actifs/inactifs")
        st.markdown("💡 **Les 7 segments :** IMMO mono-équipé, CPP/PGP+ Actifs (>30k€, 10-30k€, <10k€), CPP/PGP+ Inactifs (>30k€, 10-30k€, <10k€)")
        
        # Identifier les clients mono-équipés IMMO uniquement
        if 'Produit' not in df.columns:
            st.warning("⚠️ Colonne 'Produit' non trouvée. Impossible d'analyser les mono-équipés IMMO.")
            return
        
        # Préparer tous les clients pour la segmentation en 7 segments
        df_work = df.copy()
        
        # Vérifier l'Email pour identifier les clients uniques
        if 'Email' not in df_work.columns:
            st.warning("⚠️ Colonne 'Email' non trouvée. Impossible d'identifier les clients uniques.")
            return
        
        # Analyser les produits de chaque client
        client_produits = df_work.groupby('Email')['Produit'].agg(['unique', 'count']).reset_index()
        client_produits.columns = ['Email', 'Liste_Produits_Uniques', 'Nb_Total_Souscriptions']
        
        # Identifier les clients qui ont UNIQUEMENT le produit "IMMO" (segment spécial)
        def est_segment_immo_mono(produits_array):
            # Convertir numpy array en liste si nécessaire
            if hasattr(produits_array, 'tolist'):
                produits_list = produits_array.tolist()
            elif not isinstance(produits_array, list):
                produits_list = [produits_array]
            else:
                produits_list = produits_array
            
            # Nettoyer et normaliser les noms de produits
            produits_clean = [str(p).strip().upper() for p in produits_list if pd.notna(p) and str(p) != 'nan']
            
            if len(produits_clean) == 0:
                return False
            
            # Vérifier si tous les produits contiennent "IMMO" mais pas d'autres types
            for produit in produits_clean:
                if 'IMMO' not in produit:
                    return False
                # Exclure les produits qui contiennent d'autres mots-clés
                if any(autre in produit for autre in ['PER', 'SCPI', 'SCI', 'FONCIERE', 'EPARGNE']):
                    return False
            
            return True
        
        client_produits['Est_IMMO_Mono'] = client_produits['Liste_Produits_Uniques'].apply(est_segment_immo_mono)
        
        # Préparer les données pour la segmentation complète (tous les clients)
        df_segmentation = df_work.copy()
        
        # Nettoyer et convertir les montants
        montant_col = None
        for col in ['Premier versement', 'Premier Versement', 'Montant', 'Premier_versement']:
            if col in df_segmentation.columns:
                montant_col = col
                break
        
        if montant_col is None:
            st.warning("⚠️ Aucune colonne de montant trouvée. Utilisation d'un montant par défaut de 15 000€")
            df_segmentation['Montant_Clean'] = 15000
        else:
            # Nettoyer les montants
            df_segmentation['Montant_Clean'] = pd.to_numeric(
                df_segmentation[montant_col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            ).fillna(15000)
        
        # Vérifier l'activité 2024 (par défaut considérer tous comme actifs si pas de date)
        date_col = None
        for col in ['Date versement initial', 'Date_versement', 'Date', 'Date de souscription']:
            if col in df_segmentation.columns:
                date_col = col
                break
        
        if date_col:
            df_segmentation['Date_Clean'] = pd.to_datetime(df_segmentation[date_col], errors='coerce')
            df_segmentation['Actif_2024'] = df_segmentation['Date_Clean'].dt.year >= 2024
        else:
            st.info("ℹ️ Aucune colonne de date trouvée. Tous les clients sont considérés comme actifs.")
            df_segmentation['Actif_2024'] = True
        
        # Calculer le panier moyen par client
        client_stats = df_segmentation.groupby('Email').agg({
            'Montant_Clean': ['mean', 'sum', 'count'],
            'Actif_2024': 'any'
        }).round(0)
        
        client_stats.columns = ['Panier_Moyen', 'Montant_Total', 'Nb_Souscriptions', 'Actif_2024']
        client_stats = client_stats.reset_index()
        
        # Ajouter l'information IMMO mono aux statistiques clients
        client_stats = client_stats.merge(
            client_produits[['Email', 'Est_IMMO_Mono']], 
            on='Email', 
            how='left'
        ).fillna(False)
        
        # Définir les 7 segments selon vos critères
        def attribuer_segment(row):
            # Segment spécial IMMO mono-équipé (priorité)
            if row['Est_IMMO_Mono']:
                return "🏠 IMMO Mono-équipé"
            
            # Les 6 autres segments basés sur activité et panier moyen
            panier_moyen = row['Panier_Moyen']
            actif = row['Actif_2024']
            
            if actif:
                if panier_moyen > 30000:
                    return "🟢 CPP Actif (Gros potentiel >30k€)"
                elif panier_moyen >= 10000:
                    return "🟡 PGP+ Actif (Moyen potentiel 10-30k€)"
                else:
                    return "🟠 PGP Actif (Petit potentiel <10k€)"
            else:
                if panier_moyen > 30000:
                    return "🔴 CPP Inactif (Gros potentiel >30k€)"
                elif panier_moyen >= 10000:
                    return "🔴 PGP+ Inactif (Moyen potentiel 10-30k€)"
                else:
                    return "🔴 PGP Inactif (Petit potentiel <10k€)"
        
        client_stats['Segment'] = client_stats.apply(attribuer_segment, axis=1)
        
        # Affichage des métriques générales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total clients", f"{len(client_stats):,}")
        with col2:
            immo_mono = len(client_stats[client_stats['Est_IMMO_Mono']])
            st.metric("🏠 IMMO Mono-équipés", f"{immo_mono:,}")
        with col3:
            actifs = len(client_stats[client_stats['Actif_2024']])
            st.metric("✅ Clients actifs 2024", f"{actifs:,}")
        with col4:
            panier_moyen_global = client_stats['Panier_Moyen'].mean()
            st.metric("💰 Panier moyen global", f"{panier_moyen_global:,.0f}€")
        
        # Visualisations de la segmentation
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition par segments
            segment_counts = client_stats['Segment'].value_counts()
            fig_segments = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="🎯 Répartition par Segments",
                hole=0.4
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col2:
            # Panier moyen par segment
            segment_stats = client_stats.groupby('Segment').agg({
                'Panier_Moyen': 'mean',
                'Email': 'count'
            }).reset_index()
            segment_stats.columns = ['Segment', 'Panier_Moyen', 'Nb_Clients']
            
            fig_panier = px.bar(
                segment_stats,
                x='Nb_Clients',
                y='Segment',
                title="📊 Nombre de clients par segment",
                orientation='h',
                text='Nb_Clients'
            )
            fig_panier.update_traces(textposition='outside')
            st.plotly_chart(fig_panier, use_container_width=True)
        
        # Tableau détaillé des segments
        st.subheader("📋 Détail des Segments")
        segment_detail = client_stats.groupby('Segment').agg({
            'Email': 'count',
            'Panier_Moyen': ['mean', 'median', 'min', 'max'],
            'Montant_Total': 'sum',
            'Nb_Souscriptions': 'sum'
        }).round(0)
        
        # Aplatir les colonnes multi-niveaux
        segment_detail.columns = ['Nb_Clients', 'Panier_Moyen', 'Panier_Median', 'Panier_Min', 'Panier_Max', 'CA_Total', 'Souscriptions_Total']
        segment_detail = segment_detail.reset_index()
        
        # Ajouter le pourcentage
        segment_detail['Pourcentage'] = (segment_detail['Nb_Clients'] / len(client_stats) * 100).round(1)
        
        # Formater les montants
        for col in ['Panier_Moyen', 'Panier_Median', 'Panier_Min', 'Panier_Max']:
            segment_detail[col] = segment_detail[col].apply(lambda x: f"{x:,.0f}€")
        segment_detail['CA_Total'] = segment_detail['CA_Total'].apply(lambda x: f"{x:,.0f}€")
        segment_detail['Pourcentage'] = segment_detail['Pourcentage'].apply(lambda x: f"{x}%")
        
        st.dataframe(segment_detail, use_container_width=True)
        
        # Analyse détaillée pour tous les segments - Affichage direct
        st.subheader("🔍 Analyse Détaillée des 7 Segments")
        
        # Afficher tous les segments directement, organisés par catégorie
        all_segments = [
            "🏠 IMMO Mono-équipé",
            "🟢 CPP Actif (Gros potentiel >30k€)", 
            "🟡 PGP+ Actif (Moyen potentiel 10-30k€)", 
            "🟠 PGP Actif (Petit potentiel <10k€)",
            "🔴 CPP Inactif (Gros potentiel >30k€)", 
            "🔴 PGP+ Inactif (Moyen potentiel 10-30k€)", 
            "🔴 PGP Inactif (Petit potentiel <10k€)"
        ]
        
        # Organisé en colonnes pour un affichage compact
        col1, col2 = st.columns(2)
        
        segments_gauche = all_segments[:4]  # IMMO + 3 premiers
        segments_droite = all_segments[4:]  # 3 derniers
        
        with col1:
            st.markdown("### 🏠 IMMO & Segments Actifs")
            for segment in segments_gauche:
                if segment in client_stats['Segment'].values:
                    segment_data = client_stats[client_stats['Segment'] == segment]
                    with st.expander(f"{segment} ({len(segment_data)} clients)", expanded=False):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Panier moyen", f"{segment_data['Panier_Moyen'].mean():,.0f}€")
                        with col_b:
                            st.metric("CA total", f"{segment_data['Montant_Total'].sum():,.0f}€")
                        with col_c:
                            st.metric("Souscriptions", f"{segment_data['Nb_Souscriptions'].sum():.0f}")
                        
                        if "IMMO" in segment:
                            st.write("**Caractéristiques :** Clients ayant uniquement le produit IMMO")
                        elif "Actif" in segment:
                            st.write("**Caractéristiques :** Clients avec souscriptions en 2024")
                else:
                    st.info(f"Aucun client dans le segment {segment}")
        
        with col2:
            st.markdown("### 🔴 Segments Inactifs")
            for segment in segments_droite:
                if segment in client_stats['Segment'].values:
                    segment_data = client_stats[client_stats['Segment'] == segment]
                    with st.expander(f"{segment} ({len(segment_data)} clients)", expanded=False):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Panier moyen", f"{segment_data['Panier_Moyen'].mean():,.0f}€")
                        with col_b:
                            st.metric("CA total", f"{segment_data['Montant_Total'].sum():,.0f}€")
                        with col_c:
                            st.metric("Souscriptions", f"{segment_data['Nb_Souscriptions'].sum():.0f}")
                        
                        st.write("**Caractéristiques :** Clients sans souscription en 2024")
                else:
                    st.info(f"Aucun client dans le segment {segment}")
        
        # Vue d'ensemble rapide des 7 segments
        st.subheader("📊 Vue d'Ensemble Rapide")
        segment_summary = client_stats['Segment'].value_counts().reset_index()
        segment_summary.columns = ['Segment', 'Nombre_Clients']
        segment_summary['Pourcentage'] = (segment_summary['Nombre_Clients'] / len(client_stats) * 100).round(1)
        
        # Affichage en colonnes
        cols = st.columns(4)
        for i, (_, row) in enumerate(segment_summary.iterrows()):
            with cols[i % 4]:
                st.metric(
                    row['Segment'][:20] + "..." if len(row['Segment']) > 20 else row['Segment'],
                    f"{row['Nombre_Clients']} ({row['Pourcentage']}%)"
                )
        
        # Export des données segmentées
        st.subheader("📥 Export Segmentation Complète")
        
        # Enrichir les données client avec les segments
        df_export = df_segmentation.merge(
            client_stats[['Email', 'Segment', 'Panier_Moyen', 'Est_IMMO_Mono']],
            on='Email',
            how='left'
        )
        
        csv_export = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger segmentation complète - 7 segments (CSV)",
            data=csv_export,
            file_name=f"segmentation_complete_7segments_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Résumé de la segmentation
        st.info(f"""
        **📊 Résumé de la segmentation :**
        - **🏠 IMMO Mono-équipé** : {len(client_stats[client_stats['Segment'] == '🏠 IMMO Mono-équipé'])} clients
        - **🟢 Segments Actifs** : {len(client_stats[client_stats['Segment'].str.contains('Actif', na=False)])} clients  
        - **🔴 Segments Inactifs** : {len(client_stats[client_stats['Segment'].str.contains('Inactif', na=False)])} clients
        - **📊 Total** : {len(client_stats)} clients segmentés
        """)
    
    def generer_export_2025(self, df):
        """Génération des exports pour l'analyse 2025"""
        st.subheader("📤 Export des Données 2025")
        
        # Export du dataset complet
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger toutes les données (CSV)",
            data=csv_data,
            file_name=f"analyse_2025_complet_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Export par stage si disponible
        if 'Stage' in df.columns:
            stage_summary = df['Stage'].value_counts().reset_index()
            stage_summary.columns = ['Stage', 'Nombre']
            stage_csv = stage_summary.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📊 Télécharger résumé par Stage (CSV)",
                data=stage_csv,
                file_name=f"stages_2025_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def render_footer(self):
        """Affichage du footer"""
        if get_config("ui.footer.enabled", False) if CONFIG_LOADED else False:
            app_name = get_config("app.name", "Dashboard Commercial") if CONFIG_LOADED else "Dashboard Commercial"
            app_version = get_config("app.version", "2.0") if CONFIG_LOADED else "2.0"
            
            st.markdown(f"""
            <div class="footer">
                {app_name} v{app_version} | © 2025 | 
                <a href="mailto:support@votredomaine.com" style="color: white;">Support</a>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar_chatbot(self):
        """Widget chatbot dans la sidebar, toujours accessible"""
        # Initialiser l'historique si nécessaire
        if 'sidebar_chat_history' not in st.session_state:
            st.session_state.sidebar_chat_history = []
        
        # CSS pour styler le chatbot dans la sidebar
        st.markdown("""
        <style>
        .sidebar-chatbot {
            background: linear-gradient(135deg, #13c2c3, #0a9b9c);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            color: white;
        }
        .sidebar-chatbot h4 {
            color: white !important;
            margin-bottom: 10px;
        }
        .chat-history {
            max-height: 200px;
            overflow-y: auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding: 8px;
            margin: 10px 0;
            font-size: 12px;
        }
        .quick-buttons {
            display: flex;
            flex-direction: column;
            gap: 5px;
            margin-top: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Chatbot dans la sidebar (on est déjà dans le contexte sidebar)
        st.markdown('<div class="sidebar-chatbot">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Assistant IA")
        st.markdown("*Questions rapides*")
        
        # Vérifier les données disponibles de manière simple
        chatbot_data_available = False
        data_sources = []
        
        # Vérifier les données principales
        if 'chatbot_data_2025' in st.session_state and st.session_state.chatbot_data_2025 is not None:
            chatbot_data_available = True
            data_sources.append("Analyse 2025")
            
        if 'chatbot_data_epargne' in st.session_state and st.session_state.chatbot_data_epargne is not None:
            chatbot_data_available = True
            data_sources.append("Épargne")
        
        # Vérifier data_files
        for key, df in self.data_files.items():
            if df is not None and len(df) > 0:
                chatbot_data_available = True
                data_sources.append(key.replace('df_', '').replace('_', ' ').title())
        
        if not chatbot_data_available:
            st.warning("⚠️ Aucune donnée chargée. Le chatbot peut répondre de manière limitée.")
            
            # Widget de chargement rapide pour le chatbot
            with st.expander("📁 Chargement rapide pour IA", expanded=False):
                uploaded_file = st.file_uploader(
                    "Charger un fichier pour améliorer les réponses",
                    type=['xlsx', 'xls', 'csv'],
                    key="chatbot_file_upload",
                    help="Chargez votre fichier principal pour que l'IA puisse répondre précisément"
                )
                if uploaded_file:
                    try:
                        # Sauvegarder temporairement sans rerun immédiat
                        st.info("📊 Fichier détecté. Traitement en cours...")
                        
                        from utils.data_processing import read_excel_robust
                        import io
                        df_chatbot = read_excel_robust(io.BytesIO(uploaded_file.getvalue()))
                        if df_chatbot is not None and not df_chatbot.empty:
                            self.data_files['df_chatbot_temp'] = df_chatbot
                            st.success(f"✅ Fichier chargé ! {len(df_chatbot)} lignes, {len(df_chatbot.columns)} colonnes")
                            st.info("💡 Vous pouvez maintenant poser vos questions à l'IA")
                        else:
                            st.error("❌ Erreur lors du chargement du fichier")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
                        st.error("💡 Essayez de charger le fichier dans la page d'analyse correspondante")
        else:
            st.success(f"✅ Données disponibles : {', '.join(data_sources)}")
        
        # Configuration LLM
        with st.expander("⚙️ Configuration LLM", expanded=False):
            st.markdown("**🧠 Modèles disponibles :**")
            
            # Sélection du modèle
            selected_model = st.selectbox(
                "Modèle IA :",
                options=list(self.llm_assistant.available_models.keys()),
                format_func=lambda x: self.llm_assistant.available_models[x]["name"],
                key="llm_model_selection",
                help="Choisissez le modèle d'IA pour des analyses plus poussées"
            )
            
            # Configuration des clés API si nécessaire
            if selected_model and self.llm_assistant.available_models[selected_model]["requires_key"]:
                api_key = st.text_input(
                    "Clé API :",
                    type="password",
                    key=f"api_key_{selected_model}",
                    help=f"Entrez votre clé API pour {self.llm_assistant.available_models[selected_model]['name']}"
                )
                
                # Test de connexion
                if api_key:
                    if st.button(f"🔗 Tester {selected_model}", key=f"test_{selected_model}"):
                        try:
                            test_result = self.llm_assistant.test_model_connection(selected_model, api_key)
                            if test_result:
                                st.success("✅ Connexion réussie !")
                                st.session_state[f'llm_configured_{selected_model}'] = True
                                st.session_state[f'llm_key_{selected_model}'] = api_key
                            else:
                                st.error("❌ Échec de la connexion")
                        except Exception as e:
                            st.error(f"❌ Erreur : {str(e)}")
                            
            # Configuration pour modèle local
            elif selected_model == "local":
                st.info("💡 Modèle local Ollama - Assurez-vous qu'Ollama est lancé sur localhost:11434")
                if st.button("🔗 Tester connexion locale", key="test_local"):
                    try:
                        test_result = self.llm_assistant.test_model_connection("local")
                        if test_result:
                            st.success("✅ Ollama détecté !")
                            st.session_state['llm_configured_local'] = True
                        else:
                            st.error("❌ Ollama non disponible")
                    except Exception as e:
                        st.error(f"❌ Erreur : {str(e)}")
            
            # Statut de la configuration
            config_status = False
            for model in self.llm_assistant.available_models.keys():
                if st.session_state.get(f'llm_configured_{model}', False):
                    config_status = True
                    st.success(f"🚀 {self.llm_assistant.available_models[model]['name']} configuré")
                    break
            
            if not config_status:
                st.warning("⚠️ Aucun modèle LLM configuré - analyses basiques uniquement")
        
        # Interface de saisie compacte
        with st.form(key="sidebar_chatbot_form", clear_on_submit=True):
            quick_question = st.text_input(
                "Votre question :",
                placeholder="Ex: CA total 2024 ?",
                key="sidebar_chat_input",
                label_visibility="collapsed"
            )
            ask_btn = st.form_submit_button("🔍 Analyser", use_container_width=True)
            
            if ask_btn and quick_question.strip():
                with st.spinner("🤔 Analyse..."):
                    response = self.process_ai_question(quick_question)
                    st.session_state.sidebar_chat_history.append({
                        'question': quick_question,
                        'response': response[:100] + "..." if len(response) > 100 else response,  # Version courte
                        'full_response': response,
                        'timestamp': datetime.now().strftime("%H:%M")
                    })
        
        # Dernière réponse
        if st.session_state.sidebar_chat_history:
            last_chat = st.session_state.sidebar_chat_history[-1]
            st.success(f"**Q:** {last_chat['question']}")
            st.info(f"**R:** {last_chat['response']}")
            
            # Bouton pour voir la réponse complète
            if len(last_chat['full_response']) > 100:
                if st.button("📄 Réponse complète", key="full_response"):
                    st.text_area("Réponse complète:", last_chat['full_response'], height=150)
        
        # Historique compact
        if len(st.session_state.sidebar_chat_history) > 1:
            with st.expander("📜 Historique (5 derniers)"):
                for chat in reversed(st.session_state.sidebar_chat_history[-5:-1]):
                    st.markdown(f"**[{chat['timestamp']}]** {chat['question']}")
                    st.markdown(f"*{chat['response']}*")
                    st.markdown("---")
        
        # Boutons de questions rapides
        st.markdown("**💡 Questions suggérées:**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💰 CA", key="sidebar_ca", use_container_width=True):
                response = self.process_ai_question("Quel est notre CA total pour 2024 ?")
                st.session_state.sidebar_chat_history.append({
                    'question': "CA total 2024",
                    'response': response[:100] + "..." if len(response) > 100 else response,
                    'full_response': response,
                    'timestamp': datetime.now().strftime("%H:%M")
                })
            
            if st.button("👥 Clients", key="sidebar_clients", use_container_width=True):
                response = self.process_ai_question("Combien de nouveaux clients avons-nous acquis cette année ?")
                st.session_state.sidebar_chat_history.append({
                    'question': "Nouveaux clients 2024",
                    'response': response[:100] + "..." if len(response) > 100 else response,
                    'full_response': response,
                    'timestamp': datetime.now().strftime("%H:%M")
                })
        
        with col2:
            if st.button("🏆 Top", key="sidebar_top", use_container_width=True):
                response = self.process_ai_question("Qui est le meilleur conseiller en termes de CA ?")
                st.session_state.sidebar_chat_history.append({
                    'question': "Top conseiller CA",
                    'response': response[:100] + "..." if len(response) > 100 else response,
                    'full_response': response,
                    'timestamp': datetime.now().strftime("%H:%M")
                })
            
            if st.button("📊 Produits", key="sidebar_products", use_container_width=True):
                response = self.process_ai_question("Quel est notre top 3 des produits ?")
                st.session_state.sidebar_chat_history.append({
                    'question': "Top 3 produits",
                    'response': response[:100] + "..." if len(response) > 100 else response,
                    'full_response': response,
                    'timestamp': datetime.now().strftime("%H:%M")
                })
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Lien vers le chatbot complet
        st.markdown("---")
        st.markdown("💬 *Pour des conversations approfondies, utilisez l'onglet **🤖 Chatbot IA***")
    
    def run(self):
        """Point d'entrée principal de l'application"""
        try:
            self.render_header()
            self.render_sidebar()
            self.render_tabs()
            # self.render_footer()  # Optionnel
            
            self.logger.info("Application rendue avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors du rendu de l'application: {e}")
            st.error("❌ **Une erreur est survenue - Application redémarrée**")
            
            # Information pour l'utilisateur
            st.info("💡 **Solutions :**")
            st.markdown("• Actualisez la page (F5)")
            st.markdown("• Rechargez vos fichiers si nécessaire")
            st.markdown("• Contactez le support si le problème persiste")
            
            # Bouton pour rafraîchir
            if st.button("🔄 Rafraîchir l'application"):
                st.rerun()
            
            if get_config("server.debug", False) if CONFIG_LOADED else False:
                with st.expander("🔧 Détails techniques (debug)", expanded=False):
                    st.exception(e)


def health_check():
    """Endpoint de santé pour Docker health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": os.getenv("VERSION", "1.0.0"),
        "service": "dashboard-souscriptions"
    }

# Vous pouvez aussi ajouter une page de santé visible
if st.sidebar.button("🔍 Health Check"):
    st.json(health_check())
    

def main():
    """Fonction principale"""
    try:
        app = DashboardApp()
        app.run()
    except Exception as e:
        logging.error(f"Erreur critique lors du démarrage de l'application: {e}")
        st.error("❌ Erreur critique lors du démarrage de l'application")
        if st.button("🔄 Recharger l'application"):
            st.rerun()


if __name__ == "__main__":
    render_crypto_test_section()
    main()
