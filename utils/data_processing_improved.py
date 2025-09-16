import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import re

@st.cache_data
def safe_to_datetime_improved(series):
    """
    Version améliorée de safe_to_datetime avec meilleure gestion des formats de date.
    
    Args:
        series: Série pandas à convertir en dates
        
    Returns:
        Série pandas avec les dates converties
    """
    if series.empty:
        return series
    
    # Créer une copie pour éviter de modifier l'original
    result = pd.Series([pd.NaT] * len(series), index=series.index)
    
    # Ignorer les valeurs nulles
    mask_notna = series.notna()
    if not mask_notna.any():
        return result
    
    # Convertir la série en chaînes de caractères pour le traitement
    series_str = series.astype(str)
    
    # Essayer d'abord une conversion directe avec pd.to_datetime
    direct_result = pd.to_datetime(series, errors='coerce')
    direct_valid_mask = direct_result.notna()
    
    if direct_valid_mask.any():
        # Copier les dates valides dans le résultat
        result[direct_valid_mask] = direct_result[direct_valid_mask]
        
        # Afficher des statistiques
        st.write(f"Conversion directe réussie pour {direct_valid_mask.sum()} dates sur {len(series)}")
    
    # Pour les valeurs restantes, essayer différentes approches
    remaining_mask = mask_notna & result.isna()
    
    if remaining_mask.any():
        # 1. Essayer de traiter les valeurs comme des dates Excel
        try:
            numeric_values = pd.to_numeric(series[remaining_mask], errors='coerce')
            valid_numeric_mask = numeric_values.notna()
            
            if valid_numeric_mask.any():
                # Convertir les dates au format Excel (base 1899-12-30)
                excel_epoch = pd.Timestamp('1899-12-30')
                excel_dates = excel_epoch + pd.to_timedelta(numeric_values[valid_numeric_mask], unit='D')
                
                # Mapper les indices
                for idx, date_val in zip(numeric_values[valid_numeric_mask].index, excel_dates):
                    result.loc[idx] = date_val
                
                # Mettre à jour le masque
                remaining_mask = mask_notna & result.isna()
        except Exception as e:
            st.write(f"Erreur lors de la conversion des dates Excel: {str(e)}")
    
    # 2. Essayer des formats de date spécifiques pour les valeurs restantes
    if remaining_mask.any():
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',  # Formats européens
            '%Y/%m/%d', '%Y-%m-%d', '%Y.%m.%d',  # Formats ISO
            '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y',  # Formats américains
            '%d/%m/%y', '%Y/%m/%d %H:%M:%S',     # Autres formats courants
            '%d/%m/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S'
        ]
        
        for date_format in date_formats:
            try:
                # Essayer de convertir avec le format spécifique
                temp_result = pd.to_datetime(series_str[remaining_mask], format=date_format, errors='coerce')
                valid_mask = temp_result.notna()
                
                if valid_mask.any():
                    # Mapper les indices
                    for idx, date_val in zip(temp_result[valid_mask].index, temp_result[valid_mask]):
                        result.loc[idx] = date_val
                    
                    # Mettre à jour le masque
                    remaining_mask = mask_notna & result.isna()
                    
                    if not remaining_mask.any():
                        break
            except Exception:
                continue
    
    # 3. Essayer d'extraire des dates de chaînes complexes
    if remaining_mask.any():
        date_patterns = [
            r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',  # 31/12/2023, 31-12-2023
            r'(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})'     # 2023/12/31, 2023-12-31
        ]
        
        for pattern in date_patterns:
            try:
                extracted_dates = series_str[remaining_mask].str.extract(pattern, expand=False)
                valid_dates = extracted_dates.notna()
                
                if valid_dates.any():
                    temp_result = pd.to_datetime(extracted_dates[valid_dates], errors='coerce')
                    valid_temp = temp_result.notna()
                    
                    # Mapper les indices
                    for idx, date_val in zip(temp_result[valid_temp].index, temp_result[valid_temp]):
                        result.loc[idx] = date_val
                    
                    # Mettre à jour le masque
                    remaining_mask = mask_notna & result.isna()
                    
                    if not remaining_mask.any():
                        break
            except Exception:
                continue
    
    # Afficher des statistiques finales
    na_count = result[mask_notna].isna().sum()
    if na_count > 0:
        st.write(f"Date de souscription (dates manquantes): {na_count}")
        
        # Afficher quelques exemples de valeurs problématiques
        problematic_values = series[mask_notna & result.isna()].unique()
        if len(problematic_values) > 0:
            st.write(f"Exemples de valeurs non converties: {problematic_values[:5]}")
    
    return result

@st.cache_data
def safe_to_numeric_improved(series):
    """
    Version améliorée de safe_to_numeric avec meilleure gestion des formats de montants.
    
    Args:
        series: Série pandas à convertir en valeurs numériques
        
    Returns:
        Série pandas avec les valeurs numériques converties
    """
    if series.empty:
        return series
    
    # Si la série est déjà numérique, la retourner directement
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # Créer une copie pour éviter de modifier l'original
    series_clean = series.copy()
    
    # Convertir en string pour le traitement
    series_clean = series_clean.astype(str)
    
    # Nettoyer les chaînes
    # 1. Remplacer les symboles monétaires et autres caractères spéciaux
    series_clean = series_clean.str.replace(r'[\s\u20ac$%\u00a0£¥¢\u20a0-\u20bf]', '', regex=True)
    
    # 2. Remplacer les virgules par des points (format français -> anglais)
    series_clean = series_clean.str.replace(',', '.', regex=False)
    
    # 3. Supprimer tout caractère non numérique sauf le point décimal
    series_clean = series_clean.str.replace(r'[^0-9.-]', '', regex=True)
    
    # 4. Gérer les cas où il y a plusieurs points décimaux (garder seulement le premier)
    series_clean = series_clean.apply(lambda x: x.split('.')[0] + '.' + ''.join(x.split('.')[1:]) if '.' in x else x)
    
    # Convertir en numérique
    result = pd.to_numeric(series_clean, errors='coerce')
    
    # Afficher des statistiques
    na_count = result.isna().sum()
    zero_count = (result == 0).sum()
    
    if na_count > 0 or zero_count > 0:
        total_invalid = na_count + zero_count
        if total_invalid > 0:
            st.write(f"⚠️ {total_invalid} valeurs non valides ou à 0 dans la colonne. Vérifiez les données.")
            
            if na_count > 0:
                st.write(f"Montant du placement (montants manquants): {na_count}")
            
            if zero_count > 0:
                st.write(f"Montant du placement (montants à 0): {zero_count}")
    
    return result

def extract_conseiller_improved(df, column=None):
    """
    Version améliorée de extract_conseiller avec meilleure détection des conseillers.
    
    Args:
        df: DataFrame contenant les données
        column: Nom de la colonne contenant l'information du conseiller (optionnel)
        
    Returns:
        DataFrame avec une colonne 'Conseiller' standardisée
    """
    df = df.copy()
    
    # Si la colonne Conseiller existe déjà et contient des valeurs non nulles, on la garde
    if 'Conseiller' in df.columns and df['Conseiller'].notna().any() and (df['Conseiller'] != 'Inconnu').any():
        return df
    
    # Si une colonne spécifique est fournie, on l'utilise
    if column and column in df.columns:
        colonne_trouvee = column
    else:
        # Sinon, on cherche parmi les colonnes possibles
        colonne_trouvee = None
        
        # 1. Recherche intelligente parmi toutes les colonnes
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['conseiller', 'Conseiller', 'agent', 'staff', 'vendeur', 'commercial']):
                colonne_trouvee = col
                break
        
        # 2. Si rien n'est trouvé, chercher dans une liste étendue de colonnes possibles
        if colonne_trouvee is None:
            colonnes_possibles = [
                'Conseiller', 'conseiller', 'Conseiller', 'Conseiller', 'Staff', 'staff',
                'Vendeur', 'vendeur', 'Commercial', 'commercial', 'Agent', 'agent',
                'Responsable', 'responsable', 'Consultant', 'consultant', 'Représentant', 'représentant',
                'Souscription', 'souscription', 'User', 'user', 'Utilisateur', 'utilisateur'
            ]
            for col in colonnes_possibles:
                if col in df.columns and df[col].notna().any():
                    colonne_trouvee = col
                    break
    
    if colonne_trouvee:
        # Extraire le nom du conseiller avec plusieurs patterns possibles
        patterns = [
            r"Conseiller\s*['\"]([^'\"]+)['\"]?",  # Conseiller 'Nom' ou Conseiller "Nom"
            r"Conseiller\s*['\"]([^'\"]+)['\"]?",     # Conseiller 'Nom' ou Conseiller "Nom"
            r"Staff\s*['\"]([^'\"]+)['\"]?",       # Staff 'Nom' ou Staff "Nom"
            r"Agent:\s*([^,;]+)",                  # Agent: Nom
            r"Par:\s*([^,;]+)"                     # Par: Nom
        ]
        
        # Appliquer chaque pattern et garder la première correspondance
        df['Conseiller'] = df[colonne_trouvee]
        
        # Compteur pour le débogage
        matches_count = 0
        
        for pattern in patterns:
            mask = df['Conseiller'].astype(str).str.contains(pattern, na=False, regex=True)
            if mask.any():
                matches_count += mask.sum()
                df.loc[mask, 'Conseiller'] = df.loc[mask, 'Conseiller'].astype(str).str.extract(pattern, expand=False)
        
        # Si aucun pattern n'a fonctionné, utiliser la valeur brute
        if matches_count == 0:
            # Pas besoin de faire quoi que ce soit, on garde les valeurs brutes
            pass
        
        # Nettoyer les valeurs (supprimer espaces en début/fin)
        df['Conseiller'] = df['Conseiller'].astype(str).str.strip()
        
        # Remplacer les valeurs vides ou NaN par 'Inconnu'
        df['Conseiller'] = df['Conseiller'].fillna('Inconnu')
        df.loc[df['Conseiller'] == '', 'Conseiller'] = 'Inconnu'
        
        return df
    else:
        # Si aucune colonne de conseiller n'est trouvée, créer une colonne par défaut
        df['Conseiller'] = 'Inconnu'
        return df

def ensure_arrow_compatibility_improved(df):
    """
    Assure que toutes les colonnes du DataFrame sont compatibles avec Arrow pour Streamlit.
    Version améliorée avec meilleure gestion des types.
    
    Args:
        df: DataFrame à convertir
        
    Returns:
        DataFrame avec des types compatibles Arrow
    """
    # Créer une copie pour éviter de modifier l'original
    result = df.copy()
    
    # Convertir chaque colonne selon son type
    for col in result.columns:
        # Convertir selon le type
        if result[col].dtype == 'object':
            # Convertir les objets en chaînes de caractères
            result[col] = result[col].astype(str)
        elif pd.api.types.is_datetime64_any_dtype(result[col]):
            # Convertir les dates en chaînes de caractères formatées
            result[col] = result[col].dt.strftime('%Y-%m-%d')
        elif pd.api.types.is_integer_dtype(result[col]):
            # Convertir les entiers en float pour éviter les problèmes avec NaN
            result[col] = result[col].astype(float)
    
    return result
