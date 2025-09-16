"""
Version modifiée des fonctions de traitement de données qui capture les messages
pour les afficher dans la section debug au lieu de les afficher directement
"""

import pandas as pd
import numpy as np
import streamlit as st
from .data_processing import *  # Importer toutes les autres fonctions

def capture_message(message):
    """Capture un message dans la session state pour l'affichage debug"""
    if 'data_processing_messages' not in st.session_state:
        st.session_state.data_processing_messages = []
    st.session_state.data_processing_messages.append(message)

def capture_success(message):
    """Capture un message de succès"""
    capture_message(f"✅ {message}")

def capture_info(message):
    """Capture un message d'information"""
    capture_message(f"ℹ️ {message}")

def capture_warning(message):
    """Capture un message d'avertissement"""
    capture_message(f"⚠️ {message}")

def capture_error(message):
    """Capture un message d'erreur"""
    capture_message(f"❌ {message}")

def safe_to_datetime_debug(series):
    """Version debug de safe_to_datetime qui capture les messages"""
    if series.empty:
        return series
    
    # Créer une copie pour éviter de modifier l'original
    result = pd.Series([pd.NaT] * len(series), index=series.index)
    
    # Ignorer les valeurs nulles
    mask_notna = series.notna()
    if not mask_notna.any():
        return result
    
    # Convertir les valeurs en chaînes de caractères pour uniformiser le traitement
    series_str = series[mask_notna].astype(str)
    
    # Nettoyer les chaînes (supprimer les espaces supplémentaires, etc.)
    series_str = series_str.str.strip()
    
    # Créer un dictionnaire pour stocker les raisons des échecs de conversion
    conversion_issues = {}
    
    # Pré-traitement: nettoyer les chaînes pour les cas spéciaux
    # 1. Insérer un espace entre le texte et les chiffres pour les dates collées au texte
    series_str = series_str.str.replace(r'([A-Za-z])([0-3]?\d[/.-])', r'\1 \2', regex=True)
    
    # Identifier et ignorer les chaînes qui ne sont clairement pas des dates
    # (par exemple, les chaînes contenant des informations de souscription)
    mask_potential_dates = ~series_str.str.contains(r'\d{5,}|\$|\%|\#')
    non_date_format = mask_notna & ~mask_potential_dates
    if non_date_format.any():
        conversion_issues['format_non_date'] = series_str[non_date_format].tolist()
    
    # Identifier les valeurs qui semblent être des nombres (dates Excel)
    mask_numeric = series_str.str.match(r'^\d+(\.\d+)?$')
    if mask_numeric.any():
        try:
            # Convertir les dates au format Excel
            excel_epoch = pd.Timestamp('1899-12-30')
            numeric_values = pd.to_numeric(series_str[mask_numeric], errors='coerce')
            temp_dates = excel_epoch + pd.to_timedelta(numeric_values, unit='D')
            
            # Ne pas filtrer par année pour accepter toutes les dates valides
            valid_year_mask = temp_dates.notna()
            if valid_year_mask.any():
                result.loc[mask_numeric[valid_year_mask].index] = temp_dates[valid_year_mask]
            
            # Enregistrer les dates Excel qui ne sont pas valides
            invalid_excel_dates = mask_numeric & ~valid_year_mask
            if invalid_excel_dates.any():
                conversion_issues['invalid_excel_dates'] = series_str[invalid_excel_dates].tolist()
        except Exception as e:
            conversion_issues['excel_date_error'] = str(e)
    
    # Traiter les formats de date courants
    formats_to_try = [
        # Format spécifique de l'utilisateur (jj/mm/aaaa) - PRIORITAIRE ABSOLU
        '%d/%m/%Y',
        
        # Autres formats européens (jour/mois/année) - HAUTE PRIORITÉ
        '%d-%m-%Y', '%d.%m.%Y', 
        
        # Formats courts européens
        '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',
        
        # Formats avec heure (format européen)
        '%d-%m-%Y %H:%M:%S', '%d-%m-%Y %H:%M',
        '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M', 
        
        # Formats textuels (format européen)
        '%d %B %Y', '%d %b %Y',
        
        # Formats avec jour de la semaine (format européen)
        '%a %d/%m/%Y', '%A %d/%m/%Y', '%a %d %b %Y', '%A %d %B %Y',
        
        # Format ISO - PRIORITÉ MOYENNE
        '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
        '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
        
        # Format américain (mois/jour/année) - DERNIER RECOURS
        '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y', '%m/%d/%y', '%m-%d-%y', '%m.%d.%y',
        '%B %d, %Y', '%b %d, %Y'
    ]
    
    # Essayer les formats un par un
    remaining_mask = mask_notna & result.isna() & mask_potential_dates
    format_success_count = {}
    
    # Essayer d'abord pandas auto-detection (très robuste pour les dates)
    if remaining_mask.any():
        try:
            auto_parsed = pd.to_datetime(series_str[remaining_mask], errors='coerce')
            valid_auto = auto_parsed.notna()
            if valid_auto.any():
                result.loc[remaining_mask[valid_auto].index] = auto_parsed[valid_auto]
                format_success_count['auto_detection'] = valid_auto.sum()
        except Exception as e:
            conversion_issues['auto_detection_error'] = str(e)
    
    # Essayer les formats spécifiques pour les valeurs restantes
    remaining_mask = mask_notna & result.isna() & mask_potential_dates
    for date_format in formats_to_try:
        if not remaining_mask.any():
            break
            
        try:
            parsed_dates = pd.to_datetime(series_str[remaining_mask], format=date_format, errors='coerce')
            valid_mask = parsed_dates.notna()
            
            if valid_mask.any():
                result.loc[remaining_mask[valid_mask].index] = parsed_dates[valid_mask]
                format_success_count[date_format] = valid_mask.sum()
                
                # Mettre à jour le masque pour les valeurs restantes
                remaining_mask = mask_notna & result.isna() & mask_potential_dates
        except Exception as e:
            conversion_issues[f'format_error_{date_format}'] = str(e)
    
    # Gérer les valeurs qui n'ont pas pu être converties
    if remaining_mask.any():
        conversion_issues['unconverted_values'] = series_str[remaining_mask].tolist()
    
    # Capturer les statistiques sur les conversions réussies
    if format_success_count:
        capture_message(f"✅ Conversion de dates réussie pour {result.notna().sum()} valeurs sur {mask_notna.sum()}")
        
        # Capturer les formats qui ont fonctionné
        formats_info = ", ".join([f"{format}: {count}" for format, count in format_success_count.items()])
        capture_message(f"ℹ️ Formats détectés: {formats_info}")
    
    # Capturer des avertissements pour les valeurs non converties
    if remaining_mask.any():
        capture_message(f"⚠️ {remaining_mask.sum()} valeurs de date n'ont pas pu être converties")
        
        # Capturer quelques exemples de valeurs non converties
        examples = series_str[remaining_mask].sample(min(3, remaining_mask.sum())).tolist()
        capture_message(f"⚠️ Exemples de valeurs non converties: {', '.join(examples)}")
    
    return result

def safe_to_numeric_debug(series):
    """Version debug de safe_to_numeric qui capture les messages"""
    if series.empty:
        return series
        
    # Créer une copie pour éviter de modifier l'original
    result = pd.Series([np.nan] * len(series), index=series.index)
    
    # Ignorer les valeurs nulles
    mask_notna = series.notna()
    if not mask_notna.any():
        return result
    
    # Convertir en chaîne et nettoyer
    series_str = series[mask_notna].astype(str)
    
    # Nettoyage avancé des valeurs numériques
    # Détection et traitement des formats numériques internationaux
    
    # Afficher les valeurs avant conversion pour diagnostic
    sample_values = series_str.sample(min(5, len(series_str))).tolist()
    
    # Remplacer les virgules par des points (format européen)
    series_str = series_str.str.replace(',', '.', regex=False)
    
    # Supprimer les symboles monétaires et les espaces
    series_str = series_str.str.replace(r'[€$£\s]', '', regex=True)
    
    # Supprimer les points des milliers (ex: 1.000.000 -> 1000000)
    # On cherche les motifs comme 1.000 ou 1.000.000
    series_str = series_str.str.replace(r'(\d)\.(\d{3})', r'\1\2', regex=True)
    
    # Essayer de convertir en float
    try:
        numeric_values = pd.to_numeric(series_str, errors='coerce')
        result[mask_notna] = numeric_values
        
        # Capturer des statistiques sur la conversion
        success_count = result.notna().sum()
        total_count = mask_notna.sum()
        
        if success_count < total_count:
            capture_message(f"⚠️ {total_count - success_count} valeurs numériques n'ont pas pu être converties")
            
            # Capturer quelques exemples de valeurs non converties
            failed_mask = mask_notna & result.isna()
            if failed_mask.any():
                examples = series[failed_mask].astype(str).sample(min(3, failed_mask.sum())).tolist()
                capture_message(f"⚠️ Exemples de valeurs non converties: {', '.join(examples)}")
        else:
            capture_message(f"✅ Conversion numérique réussie pour toutes les {total_count} valeurs")
    except Exception as e:
        capture_message(f"❌ Erreur lors de la conversion numérique: {str(e)}")
    
    return result