import pandas as pd
import numpy as np
import re
import streamlit as st
from datetime import datetime
import calendar
from dateutil.relativedelta import relativedelta

def clean_dataframe(df):
    """
    Nettoie un DataFrame en supprimant les lignes et colonnes vides,
    en remplaçant les valeurs NaN par des chaînes vides,
    et en convertissant toutes les colonnes en chaînes de caractères.
    
    Args:
        df (pandas.DataFrame): DataFrame à nettoyer
        
    Returns:
        pandas.DataFrame: DataFrame nettoyé
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Supprimer les lignes où toutes les valeurs sont NaN
    df = df.dropna(how='all')
    
    # Supprimer les colonnes où toutes les valeurs sont NaN
    df = df.dropna(axis=1, how='all')
    
    # Remplacer les valeurs NaN par des chaînes vides
    df = df.fillna('')
    
    # Convertir toutes les colonnes en chaînes de caractères
    for col in df.columns:
        if df[col].dtype != 'datetime64[ns]':
            df[col] = df[col].astype(str)
    
    return df

def read_excel_robust(file, sheet_name=0):
    """
    Lit un fichier Excel de manière robuste en essayant différentes méthodes
    et moteurs en cas d'échec.
    
    Args:
        file: Chemin du fichier ou objet de type fichier
        sheet_name: Nom ou index de la feuille à lire (par défaut: 0)
        
    Returns:
        pandas.DataFrame: DataFrame contenant les données du fichier Excel
    """
    try:
        # Première tentative avec pandas
        df = pd.read_excel(file, sheet_name=sheet_name)
        return clean_dataframe(df)
    except Exception as e1:
        st.warning(f"Erreur lors de la lecture avec pandas: {e1}")
        
        try:
            # Deuxième tentative avec engine='openpyxl'
            df = pd.read_excel(file, sheet_name=sheet_name, engine='openpyxl')
            return clean_dataframe(df)
        except Exception as e2:
            st.warning(f"Erreur lors de la lecture avec openpyxl: {e2}")
            
            try:
                # Troisième tentative avec engine='xlrd'
                df = pd.read_excel(file, sheet_name=sheet_name, engine='xlrd')
                return clean_dataframe(df)
            except Exception as e3:
                st.warning(f"Erreur lors de la lecture avec xlrd: {e3}")
                
                try:
                    # Quatrième tentative en utilisant BytesIO
                    import io
                    if hasattr(file, 'read'):
                        # Si c'est un objet de type fichier, lire le contenu
                        content = file.read()
                        file_obj = io.BytesIO(content)
                        df = pd.read_excel(file_obj, sheet_name=sheet_name)
                        return clean_dataframe(df)
                except Exception as e4:
                    st.warning(f"Erreur lors de la lecture avec BytesIO: {e4}")
                    
                    try:
                        # Cinquième tentative en sauvegardant dans un fichier temporaire
                        import tempfile
                        import os
                        
                        if hasattr(file, 'read'):
                            # Si c'est un objet de type fichier, lire le contenu
                            file.seek(0)
                            content = file.read()
                            
                            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                                tmp.write(content)
                                tmp_path = tmp.name
                            
                            try:
                                df = pd.read_excel(tmp_path, sheet_name=sheet_name)
                                os.unlink(tmp_path)
                                return clean_dataframe(df)
                            except Exception as e5:
                                os.unlink(tmp_path)
                                st.error(f"Impossible de lire le fichier Excel: {e5}")
                                return pd.DataFrame()
                    except Exception as e5:
                        st.error(f"Erreur lors de la lecture du fichier: {e5}")
                        return pd.DataFrame()
    
    st.error("Toutes les tentatives de lecture du fichier Excel ont échoué.")
    return pd.DataFrame()

def extract_conseiller(df):
    """
    Extrait et standardise la colonne 'Conseiller' à partir de multiples colonnes possibles.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        
    Returns:
        pandas.DataFrame: DataFrame avec la colonne 'Conseiller' standardisée
    """
    # Liste des noms de colonnes possibles pour le conseiller
    colonnes_possibles = [
        'Conseiller', 'conseiller', 'CONSEILLER',
        'Advisor', 'advisor', 'ADVISOR',
        'Agent', 'agent', 'AGENT',
        'Staff', 'staff', 'STAFF',
        'Vendeur', 'vendeur', 'VENDEUR',
        'Commercial', 'commercial', 'COMMERCIAL',
        'User', 'user', 'USER',
        'Utilisateur', 'utilisateur', 'UTILISATEUR'
    ]
    
    # Vérifier si une colonne de conseiller existe déjà
    found = False
    for col in colonnes_possibles:
        if col in df.columns and not df[col].isna().all() and not (df[col] == '').all():
            # Renommer la colonne en 'Conseiller'
            if col != 'Conseiller':
                df['Conseiller'] = df[col]
            found = True
            break
    
    # Si aucune colonne n'est trouvée, créer une colonne par défaut
    if not found:
        st.warning("⚠️ Aucune colonne de conseiller trouvée. Utilisation de 'Inconnu' par défaut.")
        df['Conseiller'] = 'Inconnu'
    
    # Nettoyer les valeurs
    if 'Conseiller' in df.columns:
        # Extraire le nom du conseiller si au format "Conseiller 'Nom'" ou "Conseiller 'Nom'"
        # Pattern simple pour la détection (sans groupes de capture)
        detect_pattern = r"Conseiller|Agent|Staff|Par|Commercial|Vendeur"
        # Pattern avec groupe de capture pour l'extraction
        extract_pattern = r"(?:Conseiller|Conseiller|Agent|Staff|Par|Commercial|Vendeur)['\s:]*([^']+)['\s]*"
        mask = df['Conseiller'].astype(str).str.contains(detect_pattern, case=False, regex=True)
        if mask.any():
            df.loc[mask, 'Conseiller'] = df.loc[mask, 'Conseiller'].astype(str).str.extract(extract_pattern, expand=False)
        
        # Nettoyer les espaces et remplacer les valeurs vides
        df['Conseiller'] = df['Conseiller'].astype(str).str.strip().replace('', 'Inconnu').fillna('Inconnu')
    
    return df

def extract_date(df):
    """
    Extrait et standardise la colonne 'Date' à partir de multiples colonnes possibles.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        
    Returns:
        pandas.DataFrame: DataFrame avec la colonne 'Date' standardisée
    """
    # Liste des noms de colonnes possibles pour la date
    colonnes_date = [
        'Date', 'date', 'DATE',
        'Date de souscription', 'date de souscription', 'DATE DE SOUSCRIPTION',
        'Date souscription', 'date souscription', 'DATE SOUSCRIPTION',
        'Date de création', 'date de création', 'DATE DE CREATION',
        'Création', 'création', 'CREATION',
        'Date d\'ouverture', 'date d\'ouverture', 'DATE D\'OUVERTURE'
    ]
    
    # Vérifier si une colonne de date existe déjà
    found = False
    for col in colonnes_date:
        if col in df.columns and not df[col].isna().all() and not (df[col] == '').all():
            # Renommer la colonne en 'Date'
            if col != 'Date':
                df['Date'] = df[col]
            found = True
            break
    
    # Si aucune colonne n'est trouvée, créer une colonne par défaut
    if not found:
        st.warning("⚠️ Aucune colonne de date trouvée. Utilisation de la date du jour par défaut.")
        df['Date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Convertir la colonne 'Date' en datetime
    try:
        # Essayer de convertir directement
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    except:
        # Si échec, essayer de nettoyer la colonne avant conversion
        series_str = df['Date'].astype(str)
        # Supprimer les caractères non numériques sauf / et -
        series_str = series_str.str.replace(r'[^\d/-]', '', regex=True)
        # Essayer de convertir à nouveau
        df['Date'] = pd.to_datetime(series_str, errors='coerce')
    
    # Remplir les valeurs NaT avec la date du jour
    df['Date'] = df['Date'].fillna(pd.Timestamp(datetime.now().date()))
    
    # Ajouter les colonnes 'Année', 'Mois' et 'Jour'
    df['Année'] = df['Date'].dt.year
    df['Mois'] = df['Date'].dt.strftime('%Y-%m')
    df['Jour'] = df['Date'].dt.day
    
    return df

def extract_montant(df):
    """
    Extrait et standardise la colonne 'Montant' à partir de multiples colonnes possibles.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        
    Returns:
        pandas.DataFrame: DataFrame avec la colonne 'Montant' standardisée
    """
    # Liste des noms de colonnes possibles pour le montant
    colonnes_montant = [
        'Montant', 'montant', 'MONTANT',
        'Montant du prêt', 'montant du prêt', 'MONTANT DU PRET',
        'Montant prêt', 'montant prêt', 'MONTANT PRET',
        'Montant crédit', 'montant crédit', 'MONTANT CREDIT',
        'Montant du crédit', 'montant du crédit', 'MONTANT DU CREDIT',
        'Somme', 'somme', 'SOMME',
        'Valeur', 'valeur', 'VALEUR',
        'Financement', 'financement', 'FINANCEMENT'
    ]
    
    # Vérifier si une colonne de montant existe déjà
    found = False
    for col in colonnes_montant:
        if col in df.columns and not df[col].isna().all() and not (df[col] == '').all():
            # Renommer la colonne en 'Montant'
            if col != 'Montant':
                df['Montant'] = df[col]
            found = True
            break
    
    # Si aucune colonne n'est trouvée, créer une colonne par défaut
    if not found:
        st.warning("⚠️ Aucune colonne de montant trouvée. Utilisation de 0 par défaut.")
        df['Montant'] = 0
    
    # Nettoyer et convertir la colonne 'Montant' en float
    if 'Montant' in df.columns:
        # Convertir en string d'abord
        series_str = df['Montant'].astype(str)
        
        # Remplacer les virgules par des points
        series_str = series_str.str.replace(',', '.', regex=False)
        
        # Supprimer les symboles monétaires et les espaces
        series_str = series_str.str.replace(r'[€$£\s]', '', regex=True)
        
        # Supprimer les points des milliers (ex: 1.000.000 -> 1000000)
        # On cherche les motifs comme 1.000 ou 1.000.000
        series_str = series_str.str.replace(r'(\d)\.(\d{3})', r'\1\2', regex=True)
        
        # Essayer de convertir en float
        try:
            df['Montant'] = pd.to_numeric(series_str, errors='coerce')
        except:
            st.warning("⚠️ Erreur lors de la conversion des montants. Certaines valeurs peuvent être incorrectes.")
            df['Montant'] = 0
        
        # Remplacer les valeurs NaN par 0
        df['Montant'] = df['Montant'].fillna(0)
    
    return df

def extract_statut(df):
    """
    Extrait et standardise la colonne 'Statut' à partir de multiples colonnes possibles.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        
    Returns:
        pandas.DataFrame: DataFrame avec la colonne 'Statut' standardisée
    """
    # Liste des noms de colonnes possibles pour le statut
    colonnes_statut = [
        'Statut', 'statut', 'STATUT',
        'État', 'état', 'ETAT',
        'Status', 'status', 'STATUS',
        'Phase', 'phase', 'PHASE',
        'Étape', 'étape', 'ETAPE'
    ]
    
    # Vérifier si une colonne de statut existe déjà
    found = False
    for col in colonnes_statut:
        if col in df.columns and not df[col].isna().all() and not (df[col] == '').all():
            # Renommer la colonne en 'Statut'
            if col != 'Statut':
                df['Statut'] = df[col]
            found = True
            break
    
    # Si aucune colonne n'est trouvée, créer une colonne par défaut
    if not found:
        st.warning("⚠️ Aucune colonne de statut trouvée. Utilisation de 'En cours' par défaut.")
        df['Statut'] = 'En cours'
    
    # Standardiser les valeurs de statut
    if 'Statut' in df.columns:
        # Convertir en string et mettre en minuscules
        df['Statut'] = df['Statut'].astype(str).str.lower().str.strip()
        
        # Mapper les statuts similaires
        statut_mapping = {
            'en cours': 'En cours',
            'encours': 'En cours',
            'in progress': 'En cours',
            'progress': 'En cours',
            'pending': 'En cours',
            'en attente': 'En attente',
            'attente': 'En attente',
            'waiting': 'En attente',
            'wait': 'En attente',
            'validé': 'Validé',
            'valide': 'Validé',
            'validated': 'Validé',
            'valid': 'Validé',
            'approved': 'Validé',
            'refusé': 'Refusé',
            'refuse': 'Refusé',
            'rejected': 'Refusé',
            'reject': 'Refusé',
            'denied': 'Refusé',
            'annulé': 'Annulé',
            'annule': 'Annulé',
            'canceled': 'Annulé',
            'cancel': 'Annulé',
            'abandonné': 'Abandonné',
            'abandonne': 'Abandonné',
            'abandoned': 'Abandonné',
            'abandon': 'Abandonné',
            'clôturé': 'Clôturé',
            'cloture': 'Clôturé',
            'closed': 'Clôturé',
            'close': 'Clôturé',
            'terminé': 'Terminé',
            'termine': 'Terminé',
            'completed': 'Terminé',
            'complete': 'Terminé',
            'done': 'Terminé',
            'finished': 'Terminé',
            'finish': 'Terminé'
        }
        
        # Appliquer le mapping
        df['Statut'] = df['Statut'].map(lambda x: next((v for k, v in statut_mapping.items() if k in x), 'Autre'))
        
        # Remplacer les valeurs vides
        df['Statut'] = df['Statut'].replace('', 'Autre').fillna('Autre')
    
    return df
def safe_to_datetime(series):
    """Convertit une série en datetime avec gestion des erreurs et formats multiples."""
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
    
    # Afficher des statistiques sur les conversions réussies
    if format_success_count:
        st.success(f"✅ Conversion de dates réussie pour {result.notna().sum()} valeurs sur {mask_notna.sum()}")
        
        # Afficher les formats qui ont fonctionné
        formats_info = ", ".join([f"{format}: {count}" for format, count in format_success_count.items()])
        st.info(f"ℹ️ Formats détectés: {formats_info}")
    
    # Afficher des avertissements pour les valeurs non converties
    if remaining_mask.any():
        st.warning(f"⚠️ {remaining_mask.sum()} valeurs de date n'ont pas pu être converties")
        
        # Afficher quelques exemples de valeurs non converties
        examples = series_str[remaining_mask].sample(min(3, remaining_mask.sum())).tolist()
        st.warning(f"⚠️ Exemples de valeurs non converties: {', '.join(examples)}")
    
    return result

def safe_to_numeric(series):
    """Convertit une série en numérique avec gestion des erreurs et diagnostic."""
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
        
        # Afficher des statistiques sur la conversion
        success_count = result.notna().sum()
        total_count = mask_notna.sum()
        
        if success_count < total_count:
            st.warning(f"⚠️ {total_count - success_count} valeurs numériques n'ont pas pu être converties")
            
            # Afficher quelques exemples de valeurs non converties
            failed_mask = mask_notna & result.isna()
            if failed_mask.any():
                examples = series[failed_mask].astype(str).sample(min(3, failed_mask.sum())).tolist()
                st.warning(f"⚠️ Exemples de valeurs non converties: {', '.join(examples)}")
        else:
            st.success(f"✅ Conversion numérique réussie pour toutes les {total_count} valeurs")
    except Exception as e:
        st.error(f"❌ Erreur lors de la conversion numérique: {str(e)}")
    
    return result

def adjust_dates_to_month_range(df, date_col, start_date=None, end_date=None):
    """
    Ajuste les dates d'un DataFrame pour qu'elles soient dans une plage de mois spécifiée.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données
        date_col (str): Nom de la colonne de date à ajuster
        start_date (str, optional): Date de début au format 'YYYY-MM-DD'
        end_date (str, optional): Date de fin au format 'YYYY-MM-DD'
        
    Returns:
        pandas.DataFrame: DataFrame filtré avec les dates dans la plage spécifiée
    """
    if df.empty or date_col not in df.columns:
        return df
    
    # S'assurer que la colonne de date est au format datetime
    if not pd.api.types.is_datetime64_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Filtrer les valeurs NaT
    df = df.dropna(subset=[date_col])
    
    # Si aucune date de début n'est spécifiée, utiliser le premier jour du mois le plus ancien
    if start_date is None:
        min_date = df[date_col].min()
        if pd.notna(min_date):
            start_date = min_date.replace(day=1)
    else:
        start_date = pd.to_datetime(start_date)
    
    # Si aucune date de fin n'est spécifiée, utiliser le dernier jour du mois le plus récent
    if end_date is None:
        max_date = df[date_col].max()
        if pd.notna(max_date):
            # Calculer le dernier jour du mois
            next_month = max_date + pd.DateOffset(months=1)
            end_date = next_month.replace(day=1) - pd.DateOffset(days=1)
    else:
        end_date = pd.to_datetime(end_date)
    
    # Filtrer le DataFrame pour ne garder que les dates dans la plage spécifiée
    if start_date is not None and end_date is not None:
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    
    return df
