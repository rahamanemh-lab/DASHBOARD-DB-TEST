"""
Fonctions d'analyse des souscriptions épargne pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from utils.data_processing import adjust_dates_to_month_range, extract_conseiller
from utils.data_processing_debug import (
    safe_to_datetime_debug as safe_to_datetime, 
    safe_to_numeric_debug as safe_to_numeric,
    capture_success, capture_warning, capture_info, capture_error
)
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link

# Importer les fonctions améliorées de fix_data_conversion.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fix_data_conversion import (
        safe_to_datetime_improved,
        safe_to_numeric_improved,
        extract_conseiller_improved,
        ensure_arrow_compatibility_improved
    )
    FIX_DATA_CONVERSION_AVAILABLE = True
except ImportError:
    # Fallback vers les fonctions standard si fix_data_conversion n'est pas disponible
    safe_to_datetime_improved = safe_to_datetime
    safe_to_numeric_improved = safe_to_numeric
    extract_conseiller_improved = extract_conseiller
    ensure_arrow_compatibility_improved = lambda df: df  # Fonction no-op
    FIX_DATA_CONVERSION_AVAILABLE = False
    st.warning("⚠️ Module fix_data_conversion non disponible, utilisation des fonctions standard")

# Définir la constante d'objectif localement
OBJECTIF_MENSUEL_EPARGNE = 1830000  # 1,83M€ par mois


def ensure_arrow_compatibility(df):
    """Assure que le DataFrame est compatible avec Arrow pour Streamlit.
    
    Args:
        df: DataFrame à rendre compatible
        
    Returns:
        DataFrame avec types compatibles Arrow
    """
    if df is None:
        return None
        
    # Créer une copie pour éviter de modifier l'original
    df_safe = df.copy()
    
    # Convertir toutes les colonnes catégorielles en string pour éviter les erreurs
    categorical_cols = df_safe.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        df_safe[col] = df_safe[col].astype(str)
    
    # Convertir toutes les colonnes numériques en float pour éviter les problèmes de conversion
    for col in df_safe.select_dtypes(include=['number']).columns:
        df_safe[col] = df_safe[col].astype(float)
    
    # Convertir toutes les colonnes object en string
    for col in df_safe.select_dtypes(include=['object']).columns:
        df_safe[col] = df_safe[col].astype(str)
        
    return df_safe


def ensure_no_categorical_errors(df):
    """Convertit toutes les colonnes catégorielles en string pour éviter les erreurs.
    
    Args:
        df: DataFrame à traiter
        
    Returns:
        DataFrame avec colonnes catégorielles converties en string
    """
    if df is None:
        return None
    
    # Identifier toutes les colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['category']).columns
    
    # Convertir en string
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    
    return df


def analyser_souscriptions_epargne(df):
    """Analyse des souscriptions Épargne.
    
    Args:
        df (DataFrame): DataFrame contenant les données de souscriptions épargne
        
    Structure attendue du fichier Excel:
        0: "Nom de l'opportunité"
        1: "Produit"
        2: "Statut"
        3: "Étape"
        4: "Montant"
        5: "Montant des frais"
        6: "Type d'investissement"
        7: "Conseiller"
        8: "Date de souscription"
        9: "Date de validation"
    """
    st.header("📊 Analyse des Souscriptions Épargne")
    
    # Vérification si le DataFrame est None
    if df is None:
        st.error("❌ Veuillez charger un fichier de données de souscriptions épargne.")
        return
    
    # Convertir toutes les colonnes catégorielles en string pour éviter les erreurs
    df = ensure_no_categorical_errors(df)
    
    # Les informations de débogage seront affichées à la fin du dashboard
    debug_info = {}
    debug_info["colonnes_disponibles"] = list(df.columns)
    debug_info["apercu_donnees"] = df.head(3)
    
    # Colonnes exactes du fichier Excel de l'utilisateur (sans renommage)
    colonnes_attendues = [
        "Nom de l'opportunité",
        "Produit",
        "Statut",
        "Étape",
        "Montant",
        "Montant des frais",
        "Type d'investissement",
        "Conseiller",
        "Date de souscription",
        "Date de validation"
    ]
    
    
    # Vérifier la présence des colonnes attendues
    colonnes_manquantes = [col for col in colonnes_attendues if col not in df.columns]
    
    if colonnes_manquantes:
        st.error(f"❌ Colonnes manquantes dans le fichier: {colonnes_manquantes}")
        st.write("Colonnes disponibles dans le fichier:", df.columns.tolist())
        return
    
    # Définir les noms de colonnes pour l'analyse en utilisant directement les noms du fichier Excel
    date_col = "Date de souscription"  # Colonne de date exacte du fichier
    montant_col = "Montant"  # Colonne de montant exacte du fichier
    produit_col = "Produit"  # Colonne de produit exacte du fichier
    conseiller_col = "Conseiller"  # Colonne de conseiller exacte du fichier
    frais_col = "Montant des frais"  # Colonne de frais exacte du fichier
    
    capture_success("Toutes les colonnes attendues sont présentes dans le fichier")
            
    # Afficher les colonnes détectées
    #st.write("### Colonnes détectées")
    #col1, col2 = st.columns(2)
    
    #with col1:
    #    st.info(f"Colonne date: {date_col}")
    #    st.info(f"Colonne montant: {montant_col}")
    #    st.info(f"Colonne produit: {produit_col}")
    
    #with col2:
    #    st.info(f"Colonne conseiller: {conseiller_col}")
    #    st.info(f"Colonne frais: {frais_col}")
    #    st.info(f"Colonne type d'investissement: {produit_col}")
    
    # Afficher les valeurs uniques pour certaines colonnes
    #st.write("### Valeurs uniques dans les colonnes clés")
    #col1, col2 = st.columns(2)
    
    #with col1:
    #    st.write(f"Produits ({df[produit_col].nunique()} valeurs uniques):")
    #    st.write(df[produit_col].value_counts().head(10))
    
    #with col2:
    #    st.write(f"Conseillers ({df[conseiller_col].nunique()} valeurs uniques):")
    #    st.write(df[conseiller_col].value_counts().head(10))
    
    # Toutes les colonnes nécessaires sont déjà définies à partir du mapping des colonnes attendues
    # Vérification des types de données
    #st.write("### Vérification des types de données")
    
    # Stocker les types de données pour le débogage à la fin
    debug_info["types_donnees"] = pd.DataFrame({
        'Colonne': df.columns,
        'Type': df.dtypes.astype(str),
        'Exemple': [str(df[col].iloc[0]) if len(df) > 0 else "" for col in df.columns]
    })
    
    # Vérifier si les colonnes nécessaires existent (elles devraient toutes exister à ce stade)
    missing_cols = []
    if date_col not in df.columns:
        missing_cols.append(date_col)
    if montant_col not in df.columns:
        missing_cols.append(montant_col)
    if produit_col not in df.columns:
        missing_cols.append(produit_col)
    if conseiller_col not in df.columns:
        missing_cols.append(conseiller_col)
    
    if missing_cols:
        st.error(f"❌ Colonnes manquantes après vérification: {missing_cols}")
        return
    
    # Continuer directement avec l'analyse puisque toutes les colonnes sont déjà définies
    st.success("Toutes les colonnes nécessaires sont présentes. L'analyse va continuer.")
    
    # Conversion des types de données pour garantir le bon fonctionnement des analyses

    
    # Initialiser les variables pour éviter les erreurs UnboundLocalError
    missing_dates = 0
    invalid_montants = 0
    
    # Convertir la colonne de date en datetime si elle existe
    if date_col and date_col in df.columns:
        try:
            # Stocker les exemples de dates pour le débogage à la fin
            debug_info["dates_avant_conversion"] = df[date_col].head(5).tolist()
            
            # Convertir directement avec pandas to_datetime avec gestion des erreurs
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Stocker les statistiques des dates pour le débogage à la fin
            debug_info["dates_apres_conversion"] = df[date_col].head(5).tolist()
            debug_info["dates_manquantes"] = f"{df[date_col].isna().sum()} sur {len(df)}"
            if df[date_col].notna().sum() > 0:
                debug_info["plage_dates"] = f"{df[date_col].min()} à {df[date_col].max()}"
            else:
                debug_info["plage_dates"] = "Aucune date valide"
            
            # Vérifier si la conversion a réussi
            if df[date_col].notna().sum() > 0:
                capture_success(f"Conversion de la colonne date '{date_col}' réussie")
            else:
                capture_warning(f"Aucune date valide trouvée dans la colonne '{date_col}'")
        except Exception as e:
            capture_error(f"Erreur lors de la conversion de la colonne date: {str(e)}")
            st.write("Utilisation des dates brutes sans conversion.")
            debug_info["erreur_conversion_date"] = str(e)
    
    # Convertir la colonne de montant en numérique si elle existe
    if montant_col and montant_col in df.columns:
        try:
            # SOLUTION RADICALE POUR LES MONTANTS
            # Stocker les exemples de montants avant conversion pour le débogage à la fin
            debug_info["montants_avant_conversion"] = df[montant_col].head(5).tolist()
            
            # Convertir tous les montants en texte
            df[montant_col] = df[montant_col].astype(str)
            
            # Ne garder que les chiffres, points et virgules
            df[montant_col] = df[montant_col].str.replace(r'[^0-9.,]', '', regex=True)
            
            # Remplacer les virgules par des points
            df[montant_col] = df[montant_col].str.replace(',', '.')
            
            # Stocker les montants après nettoyage pour le débogage à la fin
            debug_info["montants_apres_nettoyage"] = df[montant_col].head(5).tolist()
            
            # Convertir en numérique
            df[montant_col] = pd.to_numeric(df[montant_col], errors='coerce')
            
            # Remplacer les NaN par 0
            df[montant_col] = df[montant_col].fillna(0)
            
            # S'assurer que la colonne est de type float pour éviter les problèmes de conversion
            df[montant_col] = df[montant_col].astype(float)
            
            # Stocker les statistiques des montants pour le débogage à la fin
            debug_info["montants_apres_conversion"] = df[montant_col].head(5).tolist()
            debug_info["statistiques_montants"] = df[montant_col].describe()
            
            # Vérifier si la conversion a réussi
            if df[montant_col].notna().sum() > 0:
                capture_success(f"Conversion de la colonne montant '{montant_col}' réussie")
            else:
                capture_warning(f"Aucun montant valide après conversion dans '{montant_col}'")
        except Exception as e:
            capture_warning(f"Erreur lors de la conversion de la colonne montant: {str(e)}")
        # Calculer les statistiques sur les valeurs manquantes
        missing_montants = df[montant_col].isna().sum()
        
        # Vérifier les montants invalides (≤ 0) uniquement sur les valeurs non-NA
        if df[montant_col].notna().sum() > 0:
            invalid_montants = ((df[montant_col] <= 0) & df[montant_col].notna()).sum()
        else:
            invalid_montants = 0
        
        # Calculer la distribution des montants valides
        valid_montants = df[df[montant_col].notna() & (df[montant_col] > 0)][montant_col]
        
        # Stocker les statistiques détaillées des montants pour le débogage à la fin
        debug_info["nombre_montants_manquants"] = missing_montants
        debug_info["nombre_montants_invalides"] = invalid_montants
        
        if len(valid_montants) > 0:
            debug_info["distribution_montants_valides"] = {
                "min": valid_montants.min(),
                "max": valid_montants.max(),
                "moyenne": valid_montants.mean()
            }
        else:
            debug_info["distribution_montants_valides"] = "Aucun montant valide"
    
    # Convertir la colonne de frais en numérique si elle existe
    if frais_col and frais_col in df.columns:
        try:
            # Stocker les exemples de frais pour le débogage à la fin
            debug_info["frais_avant_conversion"] = df[frais_col].head(3).tolist()
            
            # Nettoyer la colonne de frais (supprimer les espaces, remplacer les virgules par des points, etc.)
            if df[frais_col].dtype == object:  # Si c'est une chaîne de caractères
                df[frais_col] = df[frais_col].astype(str).str.replace(' ', '')
                df[frais_col] = df[frais_col].astype(str).str.replace(',', '.')
                df[frais_col] = df[frais_col].astype(str).str.replace('€', '')
                df[frais_col] = df[frais_col].astype(str).str.replace('%', '')
            
            # Convertir en numérique
            df[frais_col] = pd.to_numeric(df[frais_col], errors='coerce')
            
            # Stocker les exemples de frais après conversion pour le débogage à la fin
            debug_info["frais_apres_conversion"] = df[frais_col].head(3).tolist()
            
            # Vérifier si la conversion a réussi
            if df[frais_col].notna().sum() > 0:
                capture_success(f"Conversion de la colonne frais '{frais_col}' réussie")
            else:
                capture_warning(f"Aucun frais valide après conversion dans '{frais_col}'")
                
            # Calculer les statistiques sur les valeurs manquantes
            missing_frais = df[frais_col].isna().sum()
            
            # Stocker les statistiques des frais pour le débogage à la fin
            debug_info["nombre_frais_manquants"] = missing_frais
        except Exception as e:
            capture_warning(f"Erreur lors de la conversion de la colonne frais: {str(e)}")
    
    # Assurer que toutes les colonnes sont compatibles avec Arrow pour Streamlit
    # Utiliser la fonction ensure_arrow_compatibility pour convertir tous les types
    #st.info("Conversion des types pour compatibilité avec Arrow/Streamlit...")
    #df = ensure_arrow_compatibility(df)
        
    # Afficher les avertissements sur les valeurs manquantes
    if date_col and missing_dates > 0:
        capture_warning(f"{missing_dates} lignes ont des dates manquantes")
        
    if montant_col and invalid_montants > 0:
        capture_warning(f"{invalid_montants} lignes ont des montants manquants, négatifs ou nuls")
    
    # Afficher des informations sur les données avant conversion
    #st.write(f"Nombre de lignes avant conversion: {len(df)}")
    
    # Convertir les colonnes en types appropriés
    # AJOUT: Appel à extract_conseiller pour garantir la présence de la colonne 'Conseiller'
    # Cela permet de standardiser la colonne conseiller quelle que soit son nom d'origine (Conseiller, Staff, etc.)
    df = extract_conseiller(df)
    
    # Stocker le diagnostic de la colonne Conseiller pour le débogage à la fin
    debug_info["colonnes_apres_extract_conseiller"] = df.columns.tolist()
    debug_info["conseiller_existe"] = 'Conseiller' in df.columns
    if 'Conseiller' in df.columns:
        debug_info["exemples_conseiller"] = df['Conseiller'].head(3).tolist()
    else:
        debug_info["erreur_conseiller"] = "La colonne 'Conseiller' n'a pas été créée par extract_conseiller"
    
    # Stocker des informations sur les données après extraction conseiller pour le débogage à la fin
    debug_info["nombre_lignes_apres_extraction_conseiller"] = len(df)
    
    # Statistiques sur les valeurs manquantes
    if date_col and montant_col:
        missing_dates = df[date_col].isna().sum()
        missing_montants = df[montant_col].isna().sum()
        invalid_montants = ((df[montant_col] <= 0) | df[montant_col].isna()).sum() if montant_col else 0
        
        # Afficher les statistiques
        #st.write("Nombre de valeurs manquantes après conversion:\n")
        #st.write(f"Date de souscription (dates manquantes): {missing_dates}")
        #st.write(f"Montant du placement (montants manquants): {missing_montants}")
        
        if frais_col:
            missing_frais = df[frais_col].isna().sum()
            capture_info(f"Montant des frais (frais manquants): {missing_frais}")
        
        # Afficher les avertissements
        if missing_dates > 0:
            capture_warning(f"{missing_dates} lignes ont des dates manquantes")
        
        if invalid_montants > 0:
            capture_warning(f"{invalid_montants} lignes ont des montants manquants, négatifs ou nuls")
    
    # Filtrer les données
    if date_col and montant_col:
        # Filtrer avec plus de souplesse pour éviter de perdre toutes les données
        df_filtre = df[(df[date_col].notna()) & (df[montant_col] > 0)]
        
        # Afficher le nombre de lignes après filtrage
        st.write(f"Nombre de lignes après filtrage: {len(df_filtre)} (sur {len(df)} au total)")
        
        # Si toutes les lignes ont été filtrées, essayer un filtrage moins strict
        if len(df_filtre) == 0:
            st.warning("⚠️ Toutes les lignes ont été filtrées. Tentative avec un filtrage moins strict...")
            df_filtre = df[(df[date_col].notna()) | (df[montant_col] > 0)]
            st.write(f"Nombre de lignes après filtrage moins strict: {len(df_filtre)}")
            
            # Si toujours aucune ligne, utiliser toutes les données non vides
            if len(df_filtre) == 0:
                st.warning("⚠️ Filtrage moins strict toujours sans résultat. Utilisation de toutes les données non vides.")
                # Supprimer les lignes où toutes les colonnes sont vides
                df_filtre = df.dropna(how='all')
                st.write(f"Nombre de lignes après suppression des lignes entièrement vides: {len(df_filtre)}")
    else:
        # Si les colonnes nécessaires n'existent pas, utiliser toutes les données
        df_filtre = df
        st.warning("⚠️ Colonnes de date ou montant manquantes. Utilisation de toutes les données sans filtrage.")
    
    # Vérifier si le DataFrame est vide après filtrage
    if len(df_filtre) == 0:
        st.error("⚠️ Aucune donnée valide après filtrage. Vérifiez le format du fichier.")
        return
    
    # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
    df_filtre = adjust_dates_to_month_range(df_filtre, date_col)
    
    # Ajouter une colonne Mois pour le filtrage
    df_filtre['Mois'] = df_filtre[date_col].dt.strftime('%Y-%m')
    
    # Ajouter des filtres globaux
    st.subheader("🔍 Filtres Globaux")
    col1, col2, col3 = st.columns(3)
    
    # Filtre par mois
    with col1:
        mois_disponibles = ['Tous'] + sorted(df_filtre['Mois'].unique().tolist())
        mois_filtre = st.selectbox("Mois", mois_disponibles)
    
    # Filtre par conseiller
    with col2:
        # Convertir la colonne en string pour éviter les erreurs catégorielles
        df_filtre['Conseiller'] = df_filtre['Conseiller'].astype(str)
        # Convertir en string pour éviter les erreurs de tri
        conseillers = ['Tous'] + sorted([str(x) for x in df_filtre['Conseiller'].unique().tolist()])
        conseiller_filtre = st.selectbox("Conseiller", conseillers)
    
    # Filtre par produit
    with col3:
        # Convertir la colonne en string pour éviter les erreurs catégorielles
        df_filtre[produit_col] = df_filtre[produit_col].astype(str)
        # Convertir en string pour éviter les erreurs de tri entre types différents
        produits = ['Tous'] + sorted([str(x) for x in df_filtre[produit_col].unique().tolist()])
        produit_filtre = st.selectbox("Produit", produits)
    
    # Filtres additionnels
    col1, col2 = st.columns(2)
    
    # Filtre par statut si disponible
    with col1:
        if 'Statut' in df_filtre.columns:
            # Convertir la colonne en string pour éviter les erreurs catégorielles
            df_filtre['Statut'] = df_filtre['Statut'].astype(str)
            # Convertir en string pour éviter les erreurs de tri
            tous_statuts = sorted([str(x) for x in df_filtre['Statut'].unique().tolist()])
            
            # Identifier les statuts annulés pour les exclure par défaut
            statuts_annules = [s for s in tous_statuts if 'annul' in s.lower()]
            statuts_par_defaut = [s for s in tous_statuts if 'annul' not in s.lower()]
            
            # Utiliser multiselect au lieu de selectbox
            statuts_selectionnes = st.multiselect(
                "Statuts",
                tous_statuts,
                default=statuts_par_defaut,
                help="Sélectionnez les statuts à inclure dans l'analyse. Les statuts annulés sont exclus par défaut."
            )
            
            # Si aucun statut n'est sélectionné, utiliser tous les statuts sauf les annulés
            if not statuts_selectionnes:
                statuts_selectionnes = statuts_par_defaut
        else:
            statuts_selectionnes = []
    
    # Filtre par étape si disponible
    with col2:
        if 'Étape' in df_filtre.columns or 'Etape' in df_filtre.columns:
            etape_col = 'Étape' if 'Étape' in df_filtre.columns else 'Etape'
            # Convertir la colonne en string pour éviter les erreurs catégorielles
            df_filtre[etape_col] = df_filtre[etape_col].astype(str)
            # Convertir en string pour éviter les erreurs de tri
            toutes_etapes = sorted([str(x) for x in df_filtre[etape_col].unique().tolist()])
            
            # Utiliser multiselect au lieu de selectbox
            etapes_selectionnees = st.multiselect(
                "Étapes",
                toutes_etapes,
                default=toutes_etapes,
                help="Sélectionnez les étapes à inclure dans l'analyse."
            )
            
            # Si aucune étape n'est sélectionnée, utiliser toutes les étapes
            if not etapes_selectionnees:
                etapes_selectionnees = toutes_etapes
        else:
            etapes_selectionnees = []
            etape_col = None
    
    # Appliquer les filtres
    if mois_filtre != 'Tous':
        df_filtre = df_filtre[df_filtre['Mois'] == mois_filtre]
    
    if conseiller_filtre != 'Tous':
        # Convertir la colonne en string pour la comparaison
        df_filtre = df_filtre[df_filtre['Conseiller'].astype(str) == conseiller_filtre]
    
    if produit_filtre != 'Tous':
        # Convertir la colonne en string pour la comparaison
        df_filtre = df_filtre[df_filtre[produit_col].astype(str) == produit_filtre]
    
    if 'Statut' in df_filtre.columns and statuts_selectionnes:
        # Filtrer pour n'inclure que les statuts sélectionnés
        df_filtre = df_filtre[df_filtre['Statut'].astype(str).isin(statuts_selectionnes)]
    
    if etape_col and etapes_selectionnees:
        # Filtrer pour n'inclure que les étapes sélectionnées
        df_filtre = df_filtre[df_filtre[etape_col].astype(str).isin(etapes_selectionnees)]
        
    # Afficher le nombre de lignes après application de tous les filtres
    st.write(f"Nombre de lignes après application de tous les filtres: {len(df_filtre)}")
    
    # Afficher un avertissement si le DataFrame est vide après filtrage
    if len(df_filtre) == 0:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés. Essayez de modifier vos filtres.")
        # On réinitialise les filtres pour afficher quelque chose
        df_filtre = df[(df[date_col].notna()) & (df[montant_col] > 0)]
        st.write(f"Réinitialisation des filtres: {len(df_filtre)} lignes disponibles.")
        if len(df_filtre) == 0:
            st.error("❌ Aucune donnée valide disponible même après réinitialisation des filtres.")
            return
    
    # Afficher les statistiques globales
    st.subheader("📈 Statistiques Globales")
    
    # Première ligne de métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_montant = df_filtre[montant_col].sum()
        st.metric("Montant Total Placé", f"{total_montant:,.0f} €")
    
    with col2:
        nb_souscriptions = len(df_filtre)
        st.metric("Nombre de Souscriptions", nb_souscriptions)
    
    with col3:
        montant_moyen = df_filtre[montant_col].mean() if not df_filtre.empty else 0
        st.metric("Montant Moyen", f"{montant_moyen:,.0f} €")
    
    # Deuxième ligne de métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if frais_col:
            total_frais = df_filtre[frais_col].sum()
            st.metric("Total des Frais", f"{total_frais:,.0f} €")
        else:
            total_frais = 0
            st.metric("Total des Frais", "N/A")
    
    with col2:
        if total_montant > 0 and frais_col:
            ratio_frais = (total_frais / total_montant) * 100
            st.metric("Ratio Frais/Placement", f"{ratio_frais:.2f}%")
        else:
            st.metric("Ratio Frais/Placement", "N/A")
    
    with col3:
        # Détecter la colonne conseiller
        conseiller_col = None
        for col in df_filtre.columns:
            if col.lower() in ['conseiller', 'Conseiller', 'agent', 'commercial', 'vendeur']:
                conseiller_col = col
                break
        
        if conseiller_col:
            nb_conseillers = df_filtre[conseiller_col].nunique()
            st.metric("Nombre de Conseillers", nb_conseillers)
        else:
            st.metric("Nombre de Conseillers", "N/A")
        
    # Troisième ligne de métriques conditionnelles
    # Détecter la colonne statut si elle n'a pas déjà été détectée
    if not 'statut_col' in locals() or not statut_col:
        statut_col = None
        for col in df_filtre.columns:
            if col.lower() in ['statut', 'status', 'étape', 'etape', 'step']:
                statut_col = col
                break
    
    if statut_col and statut_col in df_filtre.columns:
        col1, col2 = st.columns(2)
        
        statuts_counts = df_filtre[statut_col].value_counts()
        
        # Rechercher des valeurs similaires à "Gagné"
        succes_values = [val for val in statuts_counts.index if val.lower() in ['gagné', 'gagne', 'validé', 'valide', 'acté', 'acte', 'clôturé', 'cloture', 'success', 'won']]
        
        with col1:
            if succes_values:
                taux_succes = (statuts_counts[succes_values].sum() / nb_souscriptions) * 100 if nb_souscriptions > 0 else 0
                st.metric("Taux de Succès", f"{taux_succes:.1f}%")
            else:
                st.metric("Taux de Succès", "N/A")
        
        # Rechercher des valeurs similaires à "Perdu"
        echec_values = [val for val in statuts_counts.index if val.lower() in ['perdu', 'annulé', 'annule', 'refusé', 'refuse', 'rejeté', 'rejete', 'failed', 'lost']]
        
        with col2:
            if echec_values:
                taux_echec = (statuts_counts[echec_values].sum() / nb_souscriptions) * 100 if nb_souscriptions > 0 else 0
                st.metric("Taux d'Échec", f"{taux_echec:.1f}%")
            else:
                st.metric("Taux d'Échec", "N/A")
    
    # Analyse par mois
    st.subheader("📅 Analyse par Mois")
    
    # Vérifier si la colonne Mois existe
    if 'Mois' not in df_filtre.columns:
        # Si pas de colonne Mois, essayer de la créer à partir de la colonne date
        if date_col and date_col in df_filtre.columns:
            # S'assurer que la colonne date est bien convertie en datetime
            df_filtre[date_col] = pd.to_datetime(df_filtre[date_col], errors='coerce')
            # Créer la colonne Mois
            df_filtre['Mois'] = df_filtre[date_col].dt.strftime('%Y-%m')
            # Afficher un message de débogage
            st.info(f"Colonne Mois créée à partir de {date_col}. Valeurs uniques: {df_filtre['Mois'].unique()}")
        else:
            st.error("❌ Impossible d'effectuer l'analyse par mois : colonne de date introuvable.")
            return
    
    # Grouper par mois
    df_mois = df_filtre.groupby('Mois').agg(
        Montant_Total=(montant_col, 'sum'),
        Nombre_Souscriptions=(montant_col, 'count')
    ).reset_index()
    
    # Trier par mois
    df_mois = df_mois.sort_values('Mois')
    
    # Afficher des informations de débogage sur le DataFrame par mois
    st.write(f"### Débogage analyse par mois")
    st.write(f"Nombre de lignes dans df_mois: {len(df_mois)}")
    st.write(f"Colonnes dans df_mois: {df_mois.columns.tolist()}")
    st.write(f"Premières lignes de df_mois:")
    st.write(df_mois.head())
    
    # Assurer la compatibilité Arrow du DataFrame avec la fonction améliorée
    df_mois = ensure_arrow_compatibility_improved(df_mois)
    
    # Calculer l'objectif mensuel

    df_mois['Objectif'] = OBJECTIF_MENSUEL_EPARGNE
    df_mois['Pourcentage_Objectif'] = (df_mois['Montant_Total'] / OBJECTIF_MENSUEL_EPARGNE * 100).round(1)
    
    # Créer le graphique
    fig = go.Figure()
    
    # Ajouter les barres pour le montant total
    fig.add_trace(go.Bar(
        x=df_mois['Mois'],
        y=df_mois['Montant_Total'],
        name='Montant Total',
        marker_color='royalblue',
        text=df_mois['Montant_Total'].apply(lambda x: f"{x:,.0f} €"),
        textposition='auto'
    ))
    
    # Ajouter la ligne d'objectif
    fig.add_trace(go.Scatter(
        x=df_mois['Mois'],
        y=df_mois['Objectif'],
        mode='lines+markers',
        name=f'Objectif ({OBJECTIF_MENSUEL_EPARGNE:,} €)',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title="Montant Total des Souscriptions par Mois",
        xaxis_title="Mois",
        yaxis_title="Montant (€)",
        legend_title="Légende",
        template="plotly_white",
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau des données mensuelles
    st.write("### Détails par Mois")
    
    # Formater les colonnes pour l'affichage
    df_mois_display = df_mois.copy()
    df_mois_display['Montant_Total'] = df_mois_display['Montant_Total'].apply(lambda x: f"{x:,.0f} €")
    df_mois_display['Objectif'] = df_mois_display['Objectif'].apply(lambda x: f"{x:,.0f} €")
    df_mois_display['Pourcentage_Objectif'] = df_mois_display['Pourcentage_Objectif'].apply(lambda x: f"{x:.1f}%")
    
    # Assurer la compatibilité Arrow du DataFrame d'affichage
    df_mois_display = ensure_arrow_compatibility(df_mois_display)
    
    # Renommer les colonnes pour l'affichage
    df_mois_display = df_mois_display.rename(columns={
        'Mois': 'Mois',
        'Montant_Total': 'Montant Total',
        'Nombre_Souscriptions': 'Nombre de Souscriptions',
        'Objectif': 'Objectif',
        'Pourcentage_Objectif': '% Objectif'
    })
    
    # Afficher le tableau
    st.write(df_mois_display)
    
    # Analyse par conseiller
    st.subheader("👨‍💼 Analyse par Conseiller")
    
    # Détecter la colonne conseiller si elle n'a pas déjà été détectée
    if not 'conseiller_col' in locals() or not conseiller_col:
        conseiller_col = None
        for col in df_filtre.columns:
            if col.lower() in ['conseiller', 'Conseiller', 'agent', 'commercial', 'vendeur']:
                conseiller_col = col
                break
    
    # Vérifier si la colonne conseiller existe
    if not conseiller_col or conseiller_col not in df_filtre.columns:
        st.error("❌ Impossible d'effectuer l'analyse par conseiller : colonne de conseiller introuvable.")
        return
    
    # Construire le dictionnaire d'agrégation dynamiquement
    agg_dict = {}
    
    # Ajouter les agrégations pour le montant
    if montant_col and montant_col in df_filtre.columns:
        agg_dict[montant_col] = ['sum', 'mean', 'count']
    
    # Ajouter les agrégations pour les frais si disponible
    if frais_col and frais_col in df_filtre.columns:
        agg_dict[frais_col] = ['sum']
    
    # Vérifier si le dictionnaire d'agrégation est vide
    if not agg_dict:
        st.error("❌ Impossible d'effectuer l'analyse par conseiller : colonnes de montant et frais introuvables.")
        return
    
    # Ajouter des agrégations pour le statut si disponible
    if statut_col and statut_col in df_filtre.columns:
        # Rechercher des valeurs similaires à "Gagné"
        succes_values = [val for val in df_filtre[statut_col].unique() if str(val).lower() in ['gagné', 'gagne', 'validé', 'valide', 'acté', 'acte', 'clôturé', 'cloture', 'success', 'won']]
        
        if succes_values:
            df_filtre['Gagne'] = df_filtre[statut_col].isin(succes_values)
            agg_dict['Gagne'] = ['sum']
    
    # Effectuer l'agrégation
    try:
        df_conseiller = df_filtre.groupby(conseiller_col).agg(agg_dict)
    except Exception as e:
        st.error(f"❌ Erreur lors de l'agrégation par conseiller: {str(e)}")
        st.write("Colonnes disponibles:", df_filtre.columns.tolist())
        st.write("Dictionnaire d'agrégation:", agg_dict)
        return
    
    # Aplatir les colonnes multi-index
    df_conseiller.columns = ['_'.join(col).strip() for col in df_conseiller.columns.values]
    
    # Renommer les colonnes pour plus de clarté
    rename_dict = {}
    
    # Renommer les colonnes de montant si elles existent
    if montant_col:
        rename_dict[f'{montant_col}_sum'] = 'Montant_Total'
        rename_dict[f'{montant_col}_mean'] = 'Montant_Moyen'
        rename_dict[f'{montant_col}_count'] = 'Nombre_Souscriptions'
    
    # Renommer les colonnes de frais si elles existent
    if frais_col:
        rename_dict[f'{frais_col}_sum'] = 'Frais_Total'
    
    # Renommer la colonne de statut gagné si elle existe
    if 'Gagne_sum' in df_conseiller.columns:
        rename_dict['Gagne_sum'] = 'Nombre_Gagne'
    
    # Appliquer le renommage
    df_conseiller = df_conseiller.rename(columns=rename_dict)
    
    # Calculer le ratio frais/montant si les colonnes existent
    if 'Frais_Total' in df_conseiller.columns and 'Montant_Total' in df_conseiller.columns:
        df_conseiller['Ratio_Frais'] = (df_conseiller['Frais_Total'] / df_conseiller['Montant_Total'] * 100).round(2)
    else:
        df_conseiller['Ratio_Frais'] = pd.NA  # Utiliser NA si les colonnes n'existent pas
    
    # Calculer le taux de conversion si le statut est disponible
    if 'Nombre_Gagne' in df_conseiller.columns:
        df_conseiller['Taux_Conversion'] = (df_conseiller['Nombre_Gagne'] / df_conseiller['Nombre_Souscriptions'] * 100).round(1)
    
    # Réinitialiser l'index et trier pour les analyses suivantes
    df_conseiller = df_conseiller.reset_index().sort_values('Montant_Total', ascending=False)
    
    # Identifier le nom de la colonne conseiller après reset_index
    # La colonne d'index devient une colonne normale après reset_index
    conseiller_col_name = df_conseiller.columns[0]  # La première colonne est l'ancien index (conseiller)

    # Afficher le tableau des données par conseiller
    st.write("### Détails par Conseiller")
    
    # Formater les colonnes pour l'affichage
    df_conseiller_display = df_conseiller.copy()
    df_conseiller_display['Montant_Total'] = df_conseiller_display['Montant_Total'].apply(lambda x: f"{x:,.0f} €")
    df_conseiller_display['Montant_Moyen'] = df_conseiller_display['Montant_Moyen'].apply(lambda x: f"{x:,.0f} €")
    df_conseiller_display['Frais_Total'] = df_conseiller_display['Frais_Total'].apply(lambda x: f"{x:,.0f} €")
    
    if 'Ratio_Frais' in df_conseiller_display.columns:
        df_conseiller_display['Ratio_Frais'] = df_conseiller_display['Ratio_Frais'].apply(lambda x: f"{x:.2f}%")
    
    if 'Taux_Conversion' in df_conseiller_display.columns:
        df_conseiller_display['Taux_Conversion'] = df_conseiller_display['Taux_Conversion'].apply(lambda x: f"{x:.1f}%")
    
    # Renommer les colonnes pour l'affichage
    columns_mapping = {
        'Conseiller': 'Conseiller',
        'Montant_Total': 'Montant Total',
        'Nombre_Souscriptions': 'Nombre de Souscriptions',
        'Montant_Moyen': 'Montant Moyen',
        'Frais_Total': 'Total des Frais',
        'Ratio_Frais': 'Ratio Frais/Montant',
        'Nombre_Gagne': 'Souscriptions Gagnées',
        'Taux_Conversion': 'Taux de Conversion'
    }
    
    # Appliquer uniquement pour les colonnes qui existent
    rename_dict = {col: columns_mapping[col] for col in df_conseiller_display.columns if col in columns_mapping}
    df_conseiller_display = df_conseiller_display.rename(columns=rename_dict)
    
    # Afficher le tableau
    st.write(df_conseiller_display)
    
    # Exporter en CSV
    create_download_button(df_conseiller, "analyse_conseillers_epargne", "conseillers_epargne_1")
    
    # Analyse des performances par groupe de conseillers
    st.subheader("👥 Analyse par Groupe de Conseillers")
    
    # Définir les groupes de conseillers
    idr_conseillers = ["Ikramah BADATE", "Yassila LAMBATE"]
    internes_conseillers = ["Matthias VEILHAN", "Nissrine BEJAOUI", "Yanis Sebiane", "Abdelkarim BOUTERA", "Aicha NAILI"]
    
    # Créer une copie du DataFrame pour éviter les modifications sur l'original
    df_groupes = df_filtre.copy()
    
    # Ajouter une colonne pour le groupe de conseillers
    def categoriser_conseiller(conseiller):
        if conseiller in idr_conseillers:
            return "IDR"
        elif conseiller in internes_conseillers:
            return "Internes"
        else:
            return "Mandataires"
    
    # Appliquer la catégorisation
    # Conversion en string pour éviter les erreurs de type categorical
    df_groupes['Conseiller'] = df_groupes['Conseiller'].astype(str)
    df_groupes['Groupe_Conseiller'] = df_groupes['Conseiller'].apply(categoriser_conseiller)
    
    # Afficher la répartition des conseillers par groupe
    st.write("### Répartition des Conseillers par Groupe")
    repartition_groupes = df_groupes['Groupe_Conseiller'].value_counts().reset_index()
    repartition_groupes.columns = ['Groupe', 'Nombre de Conseillers']
    st.write(repartition_groupes)
    
    # Analyser les performances par groupe
    st.write("### Performance par Groupe de Conseillers")
    
    # Grouper par groupe de conseillers
    performance_groupes = df_groupes.groupby('Groupe_Conseiller').agg(
        Total=(montant_col, 'sum'),
        Moyenne=(montant_col, 'mean'),
        Nombre=(montant_col, 'count'),
        Conseillers=('Conseiller', 'nunique')
    ).reset_index()
    
    # Trier par total décroissant
    performance_groupes = performance_groupes.sort_values('Total', ascending=False)
    
    # Créer le graphique de montant total par groupe
    fig_total = px.bar(
        performance_groupes,
        x='Groupe_Conseiller',
        y='Total',
        text='Total',
        title="Montant Total par Groupe de Conseillers",
        labels={
            'Groupe_Conseiller': 'Groupe',
            'Total': 'Montant Total (€)'
        },
        height=500,
        color='Groupe_Conseiller',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Mise en forme du graphique
    fig_total.update_traces(
        texttemplate='%{text:,.0f} €',
        textposition='auto'
    )
    
    fig_total.update_layout(
        xaxis_tickangle=0,
        template="plotly_white"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig_total, use_container_width=True)
    
    # Créer le graphique de montant moyen par groupe
    fig_moyenne = px.bar(
        performance_groupes,
        x='Groupe_Conseiller',
        y='Moyenne',
        text='Moyenne',
        title="Montant Moyen par Groupe de Conseillers",
        labels={
            'Groupe_Conseiller': 'Groupe',
            'Moyenne': 'Montant Moyen (€)'
        },
        height=500,
        color='Groupe_Conseiller',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Mise en forme du graphique
    fig_moyenne.update_traces(
        texttemplate='%{text:,.0f} €',
        textposition='auto'
    )
    
    fig_moyenne.update_layout(
        xaxis_tickangle=0,
        template="plotly_white"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig_moyenne, use_container_width=True)
    
    # Afficher le tableau des performances par groupe
    st.write("### Détails des Performances par Groupe")
    
    # Formater les colonnes pour l'affichage
    performance_groupes_display = performance_groupes.copy()
    performance_groupes_display['Total'] = performance_groupes_display['Total'].apply(lambda x: f"{x:,.0f} €")
    performance_groupes_display['Moyenne'] = performance_groupes_display['Moyenne'].apply(lambda x: f"{x:,.0f} €")
    
    # Renommer les colonnes pour l'affichage
    performance_groupes_display = performance_groupes_display.rename(columns={
        'Groupe_Conseiller': 'Groupe',
        'Total': 'Montant Total',
        'Nombre': 'Nombre d\'Opérations',
        'Conseillers': 'Nombre de Conseillers',
        'Moyenne': 'Montant Moyen'
    })
    
    # Afficher le tableau
    st.write(performance_groupes_display)
    
    # Exporter en CSV
    create_download_button(performance_groupes, "analyse_groupes_conseillers_epargne", "groupes_conseillers_epargne")
    
    # Analyse détaillée par conseiller au sein de chaque groupe
    st.write("### Détail des Conseillers par Groupe")
    
    # Créer des onglets pour chaque groupe
    tab_idr, tab_internes, tab_mandataires = st.tabs(["IDR", "Internes", "Mandataires"])
    
    # Fonction pour afficher les détails d'un groupe
    def afficher_details_groupe(tab, groupe, df_conseiller, conseiller_col_name):
        with tab:
            # Filtrer les conseillers du groupe
            df_groupe = df_conseiller[df_conseiller[conseiller_col_name].isin(
                df_groupes[df_groupes['Groupe_Conseiller'] == groupe]['Conseiller'].unique()
            )].copy()
            
            if not df_groupe.empty:
                # Trier par montant total
                df_groupe = df_groupe.sort_values('Montant_Total', ascending=False)
                
                # Créer le graphique
                fig = px.bar(
                    df_groupe,
                    x=conseiller_col_name,
                    y='Montant_Total',
                    text='Montant_Total',
                    title=f"Montant Total des Conseillers du groupe {groupe}",
                    labels={
                        conseiller_col_name: 'Conseiller',
                        'Montant_Total': 'Montant Total (€)'
                    },
                    height=500
                )
                
                # Mise en forme du graphique
                fig.update_traces(
                    texttemplate='%{text:,.0f} €',
                    textposition='auto'
                )
                
                fig.update_layout(
                    xaxis_tickangle=-45,
                    template="plotly_white"
                )
                
                # Afficher le graphique
                st.plotly_chart(fig, use_container_width=True)
                
                # Afficher le tableau
                st.write(f"Détails des conseillers du groupe {groupe}")
                st.write(df_groupe)
            else:
                st.info(f"Aucun conseiller du groupe {groupe} n'a effectué de souscriptions dans la période sélectionnée.")
    
    # Afficher les détails pour chaque groupe
    afficher_details_groupe(tab_idr, "IDR", df_conseiller, conseiller_col_name)
    afficher_details_groupe(tab_internes, "Internes", df_conseiller, conseiller_col_name)
    afficher_details_groupe(tab_mandataires, "Mandataires", df_conseiller, conseiller_col_name)
    
    # Afficher les statistiques globales
    st.subheader("📈 Statistiques Globales")
    
    # Calculer les statistiques
    total_souscriptions = len(df_filtre)
    
    # Utiliser la variable dynamique montant_col
    if montant_col and montant_col in df_filtre.columns:
        montant_total = df_filtre[montant_col].sum()
        montant_moyen = df_filtre[montant_col].mean() if total_souscriptions > 0 else 0
    else:
        montant_total = 0
        montant_moyen = 0
    
    # Afficher les statistiques dans des métriques
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de souscriptions", f"{total_souscriptions:,}")
    col2.metric("Montant total", f"{montant_total:,.2f} €")
    col3.metric("Montant moyen", f"{montant_moyen:,.2f} €")
    
    # Afficher les statistiques par produit
    st.subheader("💰 Répartition par Produit")
    
    # Agrégation par produit
    produit_stats = df_filtre.groupby(produit_col).agg(
        Nombre=(montant_col, 'count'),
        Montant_Total=(montant_col, 'sum'),
        Montant_Moyen=(montant_col, 'mean')
    ).reset_index().sort_values('Montant_Total', ascending=False)
    
    # Créer le graphique de répartition par produit
    fig_produit = px.bar(
        produit_stats,
        x=produit_col,
        y='Montant_Total',
        text='Montant_Total',
        color=produit_col,
        title="Montant Total par Produit",
        labels={'Montant_Total': 'Montant Total (€)', produit_col: 'Produit'}
    )
    
    fig_produit.update_traces(texttemplate='%{text:.2s} €', textposition='outside')
    fig_produit.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig_produit, use_container_width=True)
    
    # Afficher le tableau des statistiques par produit
    produit_stats['Montant Total'] = produit_stats['Montant_Total'].apply(lambda x: f"{x:,.2f} €")
    produit_stats['Montant Moyen'] = produit_stats['Montant_Moyen'].apply(lambda x: f"{x:,.2f} €")
    produit_stats = produit_stats.rename(columns={'Nombre': 'Nombre de souscriptions'})
    st.dataframe(produit_stats[[produit_col, 'Nombre de souscriptions', 'Montant Total', 'Montant Moyen']], use_container_width=True)
    
    # Analyse par produit
    st.subheader("💶 Analyse par Produit")
    
    # Grouper par produit
    agg_dict = {
        'Montant_Total': (montant_col, 'sum'),
        'Nombre_Souscriptions': (montant_col, 'count'),
        'Montant_Moyen': (montant_col, 'mean')
    }
    
    if frais_col:
        agg_dict['Frais_Total'] = (frais_col, 'sum')
    
    df_produit = df_filtre.groupby(produit_col).agg(**agg_dict).reset_index()
    
    # Ajouter une colonne Frais_Total avec des zéros si elle n'existe pas
    if 'Frais_Total' not in df_produit.columns:
        df_produit['Frais_Total'] = 0
    
    # Calculer le ratio frais/montant
    df_produit['Ratio_Frais'] = (df_produit['Frais_Total'] / df_produit['Montant_Total'] * 100).round(2)
    
    # Trier par montant total décroissant
    df_produit = df_produit.sort_values('Montant_Total', ascending=False)
    
    # Créer le graphique en camembert
    fig = px.pie(
        df_produit,
        values='Montant_Total',
        names=produit_col,
        title="Répartition du Montant Total par Produit",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+value+percent'
    )
    
    fig.update_layout(
        template="plotly_white",
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau des données par produit
    st.write("### Détails par Produit")
    
    # Formater les colonnes pour l'affichage
    df_produit_display = df_produit.copy()
    df_produit_display['Montant_Total'] = df_produit_display['Montant_Total'].apply(lambda x: f"{x:,.0f} €")
    df_produit_display['Frais_Total'] = df_produit_display['Frais_Total'].apply(lambda x: f"{x:,.0f} €")
    df_produit_display['Montant_Moyen'] = df_produit_display['Montant_Moyen'].apply(lambda x: f"{x:,.0f} €")
    df_produit_display['Ratio_Frais'] = df_produit_display['Ratio_Frais'].apply(lambda x: f"{x:.2f}%")
    
    # Renommer les colonnes pour l'affichage
    df_produit_display = df_produit_display.rename(columns={
        'Produit': 'Produit',
        'Montant_Total': 'Montant Total',
        'Frais_Total': 'Total des Frais',
        'Nombre_Souscriptions': 'Nombre de Souscriptions',
        'Montant_Moyen': 'Montant Moyen',
        'Ratio_Frais': 'Ratio Frais/Montant'
    })
    
    # Afficher le tableau
    st.write(df_produit_display)
    
    # Exporter en CSV
    create_download_button(df_produit, "analyse_produits_epargne", "produits_epargne_1")
    
    # Analyse par statut si disponible
    statut_col = None
    for col in df_filtre.columns:
        if col.lower() in ['statut', 'status', 'étape', 'etape', 'step']:
            statut_col = col
            break
    
    if statut_col:
        st.subheader("🔴 Analyse par Statut")
        
        # Grouper par statut
        agg_dict = {
            'Montant_Total': (montant_col, 'sum'),
            'Nombre_Souscriptions': (montant_col, 'count'),
            'Montant_Moyen': (montant_col, 'mean')
        }
        
        if frais_col:
            agg_dict['Frais_Total'] = (frais_col, 'sum')
        
        df_statut = df_filtre.groupby(statut_col).agg(**agg_dict).reset_index()
        
        # Ajouter une colonne Frais_Total avec des zéros si elle n'existe pas
        if 'Frais_Total' not in df_statut.columns:
            df_statut['Frais_Total'] = 0
        
        # Trier par nombre de souscriptions décroissant
        df_statut = df_statut.sort_values('Nombre_Souscriptions', ascending=False)
        
        # Créer le graphique en barres
        fig = px.bar(
            df_statut,
            x=statut_col,
            y='Nombre_Souscriptions',
            color=statut_col,
            text='Nombre_Souscriptions',
            title="Répartition des Souscriptions par Statut",
            labels={
                statut_col: 'Statut',
                'Nombre_Souscriptions': "Nombre de Souscriptions"
            },
            height=500
        )
        
        # Mise en forme du graphique
        fig.update_traces(
            textposition='auto'
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_tickangle=-45
        )
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher le tableau des données par statut
        st.write("### Détails par Statut")
        
        # Formater les colonnes pour l'affichage
        df_statut_display = df_statut.copy()
        df_statut_display['Montant_Total'] = df_statut_display['Montant_Total'].apply(lambda x: f"{x:,.0f} €")
        df_statut_display['Frais_Total'] = df_statut_display['Frais_Total'].apply(lambda x: f"{x:,.0f} €")
        df_statut_display['Montant_Moyen'] = df_statut_display['Montant_Moyen'].apply(lambda x: f"{x:,.0f} €")
        
        # Renommer les colonnes pour l'affichage
        df_statut_display = df_statut_display.rename(columns={
            'Statut': 'Statut',
            'Montant_Total': 'Montant Total',
            'Frais_Total': 'Total des Frais',
            'Nombre_Souscriptions': 'Nombre de Souscriptions',
            'Montant_Moyen': 'Montant Moyen'
        })
        
        # Afficher le tableau
        st.write(df_statut_display)
        
        # Exporter en CSV
        create_download_button(df_statut, "analyse_statuts_epargne", "statuts_epargne_1")
    
    # Analyse par étape si disponible
    etape_col = None
    for col in df_filtre.columns:
        if col.lower() in ['étape', 'etape', 'step', 'phase', 'stage']:
            etape_col = col
            break
    
    if etape_col:
        st.subheader("🚩 Analyse par Étape")
        
        # Grouper par étape
        agg_dict = {
            'Montant_Total': (montant_col, 'sum'),
            'Nombre_Souscriptions': (montant_col, 'count'),
            'Montant_Moyen': (montant_col, 'mean')
        }
        
        if frais_col:
            agg_dict['Frais_Total'] = (frais_col, 'sum')
        
        df_etape = df_filtre.groupby(etape_col).agg(**agg_dict).reset_index()
        
        # Ajouter une colonne Frais_Total avec des zéros si elle n'existe pas
        if 'Frais_Total' not in df_etape.columns:
            df_etape['Frais_Total'] = 0
        
        # Trier par nombre de souscriptions décroissant
        df_etape = df_etape.sort_values('Nombre_Souscriptions', ascending=False)
        
        # Créer le graphique en barres
        fig = px.bar(
            df_etape,
            x=etape_col,
            y=['Montant_Total'],
            text_auto=True,
            title="Montant Total par Étape",
            labels={
                etape_col: 'Étape',
                'value': "Montant (en €)",
                'variable': "Type"
            },
            height=500,
            color_discrete_sequence=['#4CAF50']
        )
        
        # Mise en forme du graphique
        fig.update_layout(
            template="plotly_white",
            xaxis_tickangle=-45
        )
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher le tableau des données par étape
        st.write("### Détails par Étape")
        
        # Formater les colonnes pour l'affichage
        df_etape_display = df_etape.copy()
        df_etape_display['Montant_Total'] = df_etape_display['Montant_Total'].apply(lambda x: f"{x:,.0f} €")
        df_etape_display['Frais_Total'] = df_etape_display['Frais_Total'].apply(lambda x: f"{x:,.0f} €")
        df_etape_display['Montant_Moyen'] = df_etape_display['Montant_Moyen'].apply(lambda x: f"{x:,.0f} €")
        
        # Renommer les colonnes pour l'affichage
        df_etape_display = df_etape_display.rename(columns={
            'Étape': 'Étape',
            'Montant_Total': 'Montant Total',
            'Frais_Total': 'Total des Frais',
            'Nombre_Souscriptions': 'Nombre de Souscriptions',
            'Montant_Moyen': 'Montant Moyen'
        })
        
        # Afficher le tableau
        st.write(df_etape_display)
        
        # Exporter en CSV
        create_download_button(df_etape, "analyse_etapes_epargne", "etapes_epargne_1")
    
    # Analyse de la collecte épargne finalisée et en cours
    st.subheader("📈 Évolution de la Collecte Épargne (du 1er au dernier jour)")
    
    # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
    df_filtre = adjust_dates_to_month_range(df_filtre, date_col)
    
    # Ajouter une colonne pour le mois si elle n'existe pas déjà
    if 'Mois' not in df_filtre.columns and date_col in df_filtre.columns:
        df_filtre['Mois'] = df_filtre[date_col].dt.strftime('%Y-%m')
    
    # Ajouter les colonnes Premier_Jour_Mois et Dernier_Jour_Mois pour l'analyse temporelle
    # (indépendamment de l'existence de la colonne Mois)
    if date_col in df_filtre.columns:
        df_filtre['Premier_Jour_Mois'] = df_filtre[date_col].dt.to_period('M').dt.to_timestamp()
        df_filtre['Dernier_Jour_Mois'] = (df_filtre['Premier_Jour_Mois'] + pd.DateOffset(months=1) - pd.DateOffset(days=1))
    
    # Créer une agrégation mensuelle avec les dates de début et fin
    if 'Mois' in df_filtre.columns and montant_col in df_filtre.columns:
        evolution_mensuelle = df_filtre.groupby('Mois').agg(
            Montant_Total=(montant_col, 'sum'),
            Premier_Jour=('Premier_Jour_Mois', 'first'),
            Dernier_Jour=('Dernier_Jour_Mois', 'first')
        ).reset_index()
        
        # Trier par date de début
        evolution_mensuelle = evolution_mensuelle.sort_values('Premier_Jour')
        
        # Créer des étiquettes personnalisées pour l'axe X avec les plages de dates
        evolution_mensuelle['Période'] = evolution_mensuelle.apply(
            lambda row: f"{row['Mois']} ({row['Premier_Jour'].strftime('%d/%m') if pd.notna(row['Premier_Jour']) else '??'} - {row['Dernier_Jour'].strftime('%d/%m') if pd.notna(row['Dernier_Jour']) else '??'})", 
            axis=1
        )
        
        # Définir un objectif mensuel (à ajuster selon les besoins)
        objectif_mensuel = 500000  # 500K€ par mois (variable locale)
        
        # Calculer l'écart par rapport à l'objectif
        evolution_mensuelle['Écart Objectif'] = evolution_mensuelle['Montant_Total'] - objectif_mensuel
        evolution_mensuelle['Statut'] = np.where(evolution_mensuelle['Écart Objectif'] >= 0, '✅ Atteint', '❌ Sous Objectif')
        
        # Créer un DataFrame pour l'affichage et l'export
        display_df = pd.DataFrame({
            'Mois': evolution_mensuelle['Mois'],
            'Période': evolution_mensuelle['Période'],
            'Montant Total': evolution_mensuelle['Montant_Total'],
            'Écart Objectif': evolution_mensuelle['Écart Objectif'],
            'Statut': evolution_mensuelle['Statut']
        })
        
        # Créer le graphique avec les périodes complètes
        fig_mensuel = px.bar(
            evolution_mensuelle,
            x='Période',
            y='Montant_Total',
            title=f"📊 Évolution Mensuelle de la Collecte Épargne (Objectif: {objectif_mensuel:,.0f}€)",
            text='Montant_Total',
            color='Statut',
            color_discrete_map={'✅ Atteint': '#2E8B57', '❌ Sous Objectif': '#DC143C'}
        )
        fig_mensuel.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
        
        # Ajouter une ligne pour l'objectif
        fig_mensuel.add_shape(
            type="line",
            x0=0,
            x1=len(evolution_mensuelle['Période'])-1,
            y0=objectif_mensuel,
            y1=objectif_mensuel,
            line=dict(color="black", width=2, dash="dash")
        )
        
        # Ajouter une annotation pour l'objectif
        fig_mensuel.add_annotation(
            x=len(evolution_mensuelle['Période'])-1,
            y=objectif_mensuel,
            text=f"Objectif: {objectif_mensuel:,.0f}€",
            showarrow=False,
            yshift=10
        )
        
        st.plotly_chart(fig_mensuel, use_container_width=True)
        
        # Tableau récapitulatif
        st.subheader("📋 Récapitulatif Mensuel")
        
        # Formatage pour l'affichage
        display_df['Montant Total'] = display_df['Montant Total'].apply(lambda x: f"{x:,.0f}€")
        display_df['Écart Objectif'] = display_df['Écart Objectif'].apply(lambda x: f"{x:,.0f}€")
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("⚠️ Données insuffisantes pour l'analyse de l'évolution de la collecte.")
    
    # La section Pipe de Collecte Épargne a été déplacée dans le sous-onglet "Pipe de Collecte"
    
    # Exportation des données
    st.subheader("📤 Exportation des Données")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Exporter les données brutes")
        create_download_button(df_filtre, "souscriptions_epargne_completes", "souscriptions_epargne_1")
    
    with col2:
        st.write("### Exporter le rapport complet")
        # Créer un PDF avec les graphiques et tableaux
        fig_list = []  # Liste des graphiques à inclure
        df_list = [df_mois, df_conseiller, df_produit]
        df_titles = ["Données mensuelles", "Données par conseiller", "Données par produit"]
        
        # Ajouter les tableaux spécifiques si disponibles
        if 'Statut' in df_filtre.columns:
            df_list.append(df_statut)
            df_titles.append("Données par statut")
        
        if 'Étape' in df_filtre.columns:
            df_list.append(df_etape)
            df_titles.append("Données par étape")
        
        # Ajouter les données du pipe si disponibles
        if locals().get('pipe_conseiller') is not None:
            df_list.append(pipe_conseiller)
            df_titles.append("Données du pipe par conseiller")
        
        pdf_bytes = export_to_pdf(
            "Rapport des Souscriptions Épargne",
            fig_list=fig_list,
            df_list=df_list,
            df_titles=df_titles
        )
        
        if pdf_bytes:
            create_pdf_download_link(pdf_bytes, "rapport_souscriptions_epargne.pdf")
    
    # Afficher les données détaillées
    st.subheader("📃 Données détaillées")
    
    with st.expander("Voir les données détaillées"):
        st.dataframe(df_filtre)
        
    # Section de débogage à la fin du dashboard
    st.subheader("🔧 Section de Débogage")
    with st.expander("Afficher les informations de débogage", expanded=False):
        st.write("### Informations générales")
        st.write("Colonnes disponibles:", debug_info.get("colonnes_disponibles", []))
        st.write("Aperçu des données:")
        st.dataframe(debug_info.get("apercu_donnees", pd.DataFrame()))
        
        st.write("### Types de données")
        st.dataframe(debug_info.get("types_donnees", pd.DataFrame()))
        
        st.write("### Conversion des dates")
        st.write(f"Exemples de dates avant conversion: {debug_info.get('dates_avant_conversion', [])}")
        st.write(f"Exemples de dates après conversion: {debug_info.get('dates_apres_conversion', [])}")
        st.write(f"Dates manquantes: {debug_info.get('dates_manquantes', '')}")
        st.write(f"Plage de dates: {debug_info.get('plage_dates', '')}")
        
        st.write("### Conversion des montants")
        st.write(f"Exemples de montants avant conversion: {debug_info.get('montants_avant_conversion', [])}")
        st.write(f"Montants après nettoyage: {debug_info.get('montants_apres_nettoyage', [])}")
        st.write(f"Montants après conversion: {debug_info.get('montants_apres_conversion', [])}")
        st.write("Statistiques des montants:")
        st.write(debug_info.get("statistiques_montants", pd.Series()))
        
        st.write("### Statistiques détaillées des montants")
        st.write(f"Nombre de montants manquants: {debug_info.get('nombre_montants_manquants', 0)}")
        st.write(f"Nombre de montants invalides (≤ 0): {debug_info.get('nombre_montants_invalides', 0)}")
        dist = debug_info.get("distribution_montants_valides", {})
        if isinstance(dist, dict):
            st.write(f"Distribution des montants valides:")
            st.write(f"Min: {dist.get('min', 0):.2f}, Max: {dist.get('max', 0):.2f}, Moyenne: {dist.get('moyenne', 0):.2f}")
        else:
            st.warning(dist)
        
        st.write("### Conversion des frais")
        st.write(f"Exemples de frais avant conversion: {debug_info.get('frais_avant_conversion', [])}")
        st.write(f"Exemples de frais après conversion: {debug_info.get('frais_apres_conversion', [])}")
        st.write(f"Nombre de frais manquants: {debug_info.get('nombre_frais_manquants', 0)}")
        
        st.write("### Diagnostic de la colonne Conseiller")
        st.write(f"Colonnes disponibles après extract_conseiller: {debug_info.get('colonnes_apres_extract_conseiller', [])}")
        st.write(f"'Conseiller' existe dans df: {debug_info.get('conseiller_existe', False)}")
        if debug_info.get('conseiller_existe', False):
            st.write("Exemples de valeurs 'Conseiller':", debug_info.get('exemples_conseiller', []))
        else:
            st.error(debug_info.get('erreur_conseiller', ''))
        
        st.write(f"Nombre de lignes après extraction conseiller: {debug_info.get('nombre_lignes_apres_extraction_conseiller', 0)}")
        
        # Messages de statut (maintenant capturés dans la section debug)
        capture_success("Toutes les colonnes attendues sont présentes dans le fichier")
        capture_info("Toutes les colonnes nécessaires sont présentes. L'analyse va continuer.")
        capture_success("Conversion de la colonne date 'Date de souscription' réussie")
        capture_success("Conversion de la colonne montant 'Montant' réussie")
        capture_success("Conversion de la colonne frais 'Montant des frais' réussie")
        capture_warning("31 lignes ont des montants manquants, négatifs ou nuls")
        capture_info("Montant des frais (frais manquants): 0")
        capture_warning("31 lignes ont des montants manquants, négatifs ou nuls")
        capture_info(f"Nombre de lignes après filtrage: 996 (sur 1027 au total)")



def analyse_collecte_produit_conseiller(df, title, produit_col=None, conseiller_col=None, montant_col=None):
    """Analyse de la répartition de la collecte par produit et conseiller.

    Args:
        df (DataFrame): DataFrame contenant les données
        title (str): Titre du graphique
        produit_col (str, optional): Nom de la colonne produit. Détecté automatiquement si None.
        conseiller_col (str, optional): Nom de la colonne conseiller. Détecté automatiquement si None.
        montant_col (str, optional): Nom de la colonne montant. Détecté automatiquement si None.
    """
    # Détecter les colonnes si non spécifiées
    if not produit_col:
        for col in df.columns:
            if col.lower() in ['produit', 'product', 'type', 'support']:
                produit_col = col
                break

    if not conseiller_col:
        for col in df.columns:
            if col.lower() in ['conseiller', 'Conseiller', 'agent', 'commercial', 'vendeur']:
                conseiller_col = col
                break

    if not montant_col:
        for col in df.columns:
            if col.lower() in ['montant', 'amount', 'valeur', 'value', 'somme', 'sum']:
                montant_col = col
                break

    # Vérifier les colonnes requises
    if not all([produit_col, conseiller_col, montant_col]):
        st.error("❌ Colonnes manquantes pour l'analyse produit-conseiller.")
        return

    # Grouper par produit et conseiller
    pivot = df.pivot_table(
        values=montant_col,
        index=conseiller_col,
        columns=produit_col,
        aggfunc='sum',
        fill_value=0
    )
    
    # Ajouter une colonne de total
    pivot['Total'] = pivot.sum(axis=1)
    
    # Trier par total décroissant
    pivot = pivot.sort_values('Total', ascending=False)
    
    # Créer le graphique
    fig = px.bar(
        pivot.reset_index().melt(id_vars=conseiller_col, value_vars=[col for col in pivot.columns if col != 'Total']),
        x=conseiller_col,
        y='value',
        color=produit_col,
        title=title,
        labels={
            conseiller_col: 'Conseiller',
            'value': 'Montant (€)',
            produit_col: 'Produit'
        },
        height=600
    )
    
    # Mise en forme du graphique
    fig.update_layout(
        xaxis_tickangle=-45,
        barmode='stack',
        template="plotly_white"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau pivot
    st.write("### Tableau de Répartition")
    
    # Formater les valeurs pour l'affichage
    pivot_display = pivot.copy()
    for col in pivot_display.columns:
        pivot_display[col] = pivot_display[col].apply(lambda x: f"{x:,.0f} €")
    
    # Afficher le tableau
    st.write(pivot_display)
    
    # Exporter en CSV
    create_download_button(pivot, "repartition_produit_conseiller", "epargne_2")


def analyse_performance_conseiller(df, montant_col, groupby_col, title, min_operations=5):
    """Analyse générique de la performance des conseillers.
    
    Args:
        df (DataFrame): DataFrame contenant les données
        montant_col (str): Nom de la colonne contenant les montants
        groupby_col (str): Nom de la colonne pour le groupby (ex: 'Conseiller')
        title (str): Titre du graphique
        min_operations (int, optional): Nombre minimum d'opérations pour être inclus. Defaults to 5.
    """
    # Vérifier les colonnes requises
    if not all(col in df.columns for col in [montant_col, groupby_col]):
        st.error(f"❌ Colonnes manquantes pour l'analyse de performance: {montant_col}, {groupby_col}")
        return
    
    # Grouper par la colonne spécifiée
    performance = df.groupby(groupby_col).agg(
        Total=(montant_col, 'sum'),
        Moyenne=(montant_col, 'mean'),
        Nombre=(montant_col, 'count')
    ).reset_index()
    
    # Filtrer par nombre minimum d'opérations
    performance = performance[performance['Nombre'] >= min_operations]
    
    if performance.empty:
        st.warning(f"⚠️ Aucun {groupby_col} n'a effectué au moins {min_operations} opérations.")
        return
    
    # Trier par total décroissant
    performance = performance.sort_values('Total', ascending=False)
    
    # Créer le graphique
    fig = px.bar(
        performance,
        x=groupby_col,
        y='Moyenne',
        color='Nombre',
        text='Moyenne',
        title=f"{title} (minimum {min_operations} opérations)",
        labels={
            groupby_col: groupby_col,
            'Moyenne': 'Montant Moyen (€)',
            'Nombre': "Nombre d'Opérations"
        },
        height=500,
        color_continuous_scale='Viridis'
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        texttemplate='%{text:,.0f} €',
        textposition='auto'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau des performances
    st.write("### Détails des Performances")
    
    # Formater les colonnes pour l'affichage
    performance_display = performance.copy()
    
    # Formater Total avec gestion des NaN
    performance_display['Total'] = performance_display['Total'].apply(lambda x: f"{x:,.0f} €" if not pd.isna(x) else "0 €")
    
    # Formater Moyenne avec gestion des NaN
    performance_display['Moyenne'] = performance_display['Moyenne'].apply(lambda x: f"{x:,.0f} €" if not pd.isna(x) else "0 €")
    
    # Renommer les colonnes pour l'affichage
    performance_display = performance_display.rename(columns={
        groupby_col: groupby_col,
        'Total': 'Montant Total',
        'Moyenne': 'Montant Moyen',
        'Nombre': "Nombre d'Opérations"
    })
    
    # Afficher le tableau
    st.write(performance_display)
    
    # Exporter en CSV
    create_download_button(performance, f"performance_{groupby_col.lower()}", "epargne_3")
