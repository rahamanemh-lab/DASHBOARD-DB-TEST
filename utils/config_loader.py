"""
Configuration loader pour Dashboard Souscriptions
Charge et gère la configuration depuis config.yaml
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import streamlit as st

class ConfigLoader:
    """Gestionnaire de configuration pour l'application"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise le loader de configuration
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        if config_path is None:
            # Chercher le fichier de config depuis la racine du projet
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier YAML"""
        try:
            if not self.config_path.exists():
                logging.warning(f"Fichier de configuration non trouvé: {self.config_path}")
                self._config = self._get_default_config()
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file) or {}
            
            # Remplacer les variables d'environnement
            self._resolve_env_variables(self._config)
            
            logging.info(f"Configuration chargée depuis: {self.config_path}")
            
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la configuration: {e}")
            self._config = self._get_default_config()
    
    def _resolve_env_variables(self, config: Dict[str, Any]) -> None:
        """Résout les variables d'environnement dans la configuration"""
        def resolve_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value
        
        for key, value in config.items():
            config[key] = resolve_value(value)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut si le fichier n'existe pas"""
        return {
            "app": {
                "name": "Dashboard Souscriptions",
                "version": "1.0.0",
                "streamlit": {
                    "title": "📊 Dashboard Souscriptions",
                    "page_icon": "📈",
                    "layout": "wide"
                }
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8501,
                "debug": False
            },
            "ui": {
                "theme": {
                    "primary_color": "#1f77b4",
                    "background_color": "#ffffff"
                }
            },
            "logging": {
                "level": "INFO"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration avec notation pointée
        
        Args:
            key: Clé de configuration (ex: "app.name" ou "server.port")
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Valeur de configuration ou valeur par défaut
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Récupère une section complète de la configuration
        
        Args:
            section: Nom de la section
            
        Returns:
            Dictionnaire de la section ou dictionnaire vide
        """
        return self._config.get(section, {})
    
    def set(self, key: str, value: Any) -> None:
        """
        Définit une valeur de configuration
        
        Args:
            key: Clé de configuration
            value: Nouvelle valeur
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def reload(self) -> None:
        """Recharge la configuration depuis le fichier"""
        self._load_config()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Retourne la configuration complète"""
        return self._config.copy()
    
    def configure_streamlit(self) -> None:
        """Configure Streamlit selon la configuration"""
        streamlit_config = self.get_section("app").get("streamlit", {})
        
        # Configuration de la page
        st.set_page_config(
            page_title=streamlit_config.get("title", "Dashboard"),
            page_icon=streamlit_config.get("page_icon", "📊"),
            layout=streamlit_config.get("layout", "wide"),
            initial_sidebar_state=streamlit_config.get("initial_sidebar_state", "expanded"),
            menu_items=streamlit_config.get("menu_items", {})
        )
    
    def configure_logging(self) -> None:
        """Configure le système de logging"""
        log_config = self.get_section("logging")
        
        # Créer le répertoire de logs s'il n'existe pas
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_config.get("files", {}).get("app", "logs/app.log"))
            ]
        )
    
    def get_database_url(self) -> Optional[str]:
        """Construit l'URL de connexion à la base de données"""
        db_config = self.get_section("database").get("primary", {})
        
        if not db_config:
            return None
        
        db_type = db_config.get("type")
        if db_type == "postgresql":
            return f"postgresql://{db_config.get('user')}:{db_config.get('password')}@{db_config.get('host')}:{db_config.get('port')}/{db_config.get('name')}"
        elif db_type == "mysql":
            return f"mysql://{db_config.get('user')}:{db_config.get('password')}@{db_config.get('host')}:{db_config.get('port')}/{db_config.get('name')}"
        elif db_type == "sqlite":
            return f"sqlite:///{db_config.get('name', 'database.db')}"
        
        return None
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Vérifie si une fonctionnalité est activée
        
        Args:
            feature: Nom de la fonctionnalité
            
        Returns:
            True si la fonctionnalité est activée
        """
        return self.get(f"features.beta_features.{feature}", False) or \
               self.get(f"features.experimental.{feature}", False)
    
    def get_environment(self) -> str:
        """Retourne l'environnement actuel"""
        return self.get("deployment.environment", "development")
    
    def is_production(self) -> bool:
        """Vérifie si on est en environnement de production"""
        return self.get_environment() == "production"
    
    def is_debug(self) -> bool:
        """Vérifie si le mode debug est activé"""
        env = self.get_environment()
        return self.get(f"environments.{env}.debug", self.get("server.debug", False))


# Instance globale du loader de configuration
config = ConfigLoader()

# Fonctions utilitaires
def get_config(key: str, default: Any = None) -> Any:
    """Fonction utilitaire pour récupérer une configuration"""
    return config.get(key, default)

def get_theme_colors() -> Dict[str, str]:
    """Retourne les couleurs du thème"""
    return config.get_section("ui").get("theme", {})

def get_chart_config() -> Dict[str, Any]:
    """Retourne la configuration des graphiques"""
    return config.get_section("ui").get("charts", {})

def is_feature_enabled(feature: str) -> bool:
    """Vérifie si une fonctionnalité est activée"""
    return config.is_feature_enabled(feature)