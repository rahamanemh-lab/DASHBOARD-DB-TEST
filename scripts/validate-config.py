#!/usr/bin/env python3
"""
Script de validation de la configuration
Vérifie que tous les paramètres sont corrects
"""

import sys
from pathlib import Path

# Ajouter le projet au path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config_loader import config
import logging

def validate_config():
    """Valide la configuration"""
    errors = []
    warnings = []
    
    # Vérifier les sections obligatoires
    required_sections = ["app", "server", "ui", "logging"]
    for section in required_sections:
        if not config.get_section(section):
            errors.append(f"Section manquante: {section}")
    
    # Vérifier le port serveur
    port = config.get("server.port")
    if not isinstance(port, int) or port < 1 or port > 65535:
        errors.append(f"Port serveur invalide: {port}")
    
    # Vérifier les couleurs du thème
    theme = config.get_section("ui").get("theme", {})
    color_keys = ["primary_color", "background_color", "text_color"]
    for color_key in color_keys:
        color = theme.get(color_key)
        if color and not color.startswith("#"):
            warnings.append(f"Couleur {color_key} ne commence pas par #: {color}")
    
    # Vérifier les répertoires de données
    data_dirs = config.get("data.directories", {})
    for dir_name, dir_path in data_dirs.items():
        if dir_path and not Path(dir_path).exists():
            warnings.append(f"Répertoire {dir_name} n'existe pas: {dir_path}")
    
    return errors, warnings

if __name__ == "__main__":
    print("🔍 Validation de la configuration...")
    
    errors, warnings = validate_config()
    
    if errors:
        print("❌ Erreurs détectées:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("⚠️ Avertissements:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("✅ Configuration valide !")
    
    sys.exit(1 if errors else 0)