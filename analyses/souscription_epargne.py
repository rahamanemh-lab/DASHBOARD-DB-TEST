import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import re
import sys
import os

# Ajouter le répertoire parent au chemin pour pouvoir importer les fonctions utilitaires
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_processing import safe_to_datetime, safe_to_numeric
from utils.export import create_download_button

# Définition des colonnes strictement autorisées
# Cette liste est la SEULE référence pour toutes les opérations
COLONNES_AUTORISEES = {
    # Colonnes strictes du fichier source (seules ces colonnes seront importées)
    "source": [
        "Nom de l'opportunité",
        "Produit",
        "Statut",
        "Étape",
        "Montant",
        "Montant des frais",
        "Type d'investissement",
        "Conseiller",
        "Date de souscription",
        "Date de validation",
   ],
    # Colonnes dérivées nécessaires au fonctionnement
    "derivees": [
        "Montant du placement",
        "Date de souscription",
        "Mois",
        "Premier_Jour_Mois",
        "Dernier_Jour_Mois"
    ]
}

# Mapping des colonnes alternatives vers les noms standard
COLONNE_MAPPING = {
    "Conseiller": "Conseiller",
    "Date de souscription": "Date de souscription",
    "Date de validation": "Date de validation"
}

# Importer les fonctions adaptées pour l'analyse sans colonne Produit
from analyses.utils_analyse import analyse_collecte_produit_conseiller_fallback, analyse_performance_conseiller_fallback, extract_conseiller

# Constantes
OBJECTIF_MENSUEL_EPARGNE = 2000000  # Objectif mensuel pour l'épargne (2M€)


def adjust_dates_to_month_range(df, date_column):
    """Ajoute les colonnes de premier et dernier jour du mois pour chaque entrée."""
    df = df.copy()
    
    # Extraire le premier jour du mois
    df['Premier_Jour_Mois'] = df[date_column].dt.to_period('M').dt.start_time
    
    # Extraire le dernier jour du mois
    df['Dernier_Jour_Mois'] = df[date_column].dt.to_period('M').dt.end_time
    
    return df

def analyser_souscriptions_epargne(df_original):
    """Analyse des souscriptions Épargne avec un contrôle strict des colonnes.
    
    Cette fonction applique un contrôle strict des colonnes utilisées dans l'analyse:
    1. Seules les colonnes explicitement listées dans COLONNES_AUTORISEES["source"] sont importées
    2. Aucune colonne parasite n'est autorisée
    3. Les colonnes dérivées sont créées uniquement à partir des colonnes sources autorisées
    
    Args:
        df_original: DataFrame contenant les données brutes
        
    Returns:
        DataFrame traité ou None en cas d'erreur
    """
    
    # DIAGNOSTIC: Vérifier les colonnes dans le DataFrame original
    st.write("### DIAGNOSTIC INITIAL: Colonnes dans le DataFrame original")
    st.write(f"Colonnes disponibles dans df_original: {df_original.columns.tolist()}")
    st.write(f"'Conseiller' existe dans df_original: {'Conseiller' in df_original.columns}")
    st.write(f"'Conseiller' existe dans df_original: {'Conseiller' in df_original.columns}")
    if 'Conseiller' in df_original.columns:
        st.write("Exemples de valeurs 'Conseiller':", df_original['Conseiller'].head(3).tolist())
    
    st.header("📊 Analyse des Souscriptions Épargne")
    
    # Afficher un message d'information sur les colonnes attendues
    st.info("""
    📝 **Colonnes strictement requises dans le fichier source**:
    - "Nom de l'opportunité" (texte)
    - "Produit" (texte)
    - "Statut" (texte)
    - "Étape" (texte)
    - "Montant" (nombre)
    - "Montant des frais" (nombre)
    - "Type d'investissement" (texte)
    - "Conseiller" (texte)
    - "Date de souscription" (date au format jj/mm/aaaa)
    - "Date de validation" (date au format jj/mm/aaaa)
    
    ℹ️ **Important**: Les noms des colonnes doivent correspondre **exactement** à ceux listés ci-dessus.
    Toute colonne non listée sera ignorée. Toute colonne manquante sera signalée.
    """)
    
    # 1. DIAGNOSTIC DES COLONNES DU FICHIER SOURCE
    st.subheader("🔍 Diagnostic des colonnes du fichier source")
    
    # Afficher les colonnes du fichier original pour diagnostic
    colonnes_originales = df_original.columns.tolist()
    st.write("Colonnes présentes dans le fichier source:")
    st.write(colonnes_originales)
    
    # Créer une liste des colonnes requises après avoir pris en compte les alternatives
    colonnes_requises_base = ["Nom de l'opportunité", "Produit", "Statut", "Étape", "Montant", "Montant des frais", "Type d'investissement"]
    colonnes_requises_alternatives = {
        "Conseiller": ["Conseiller", "Conseiller"],
        "Date de souscription": ["Date de souscription", "Date de souscription"],
        "Date de validation": ["Date de validation", "Date de validation"]
    }
    
    # Vérifier les colonnes manquantes en tenant compte des alternatives
    colonnes_manquantes = []
    for col in colonnes_requises_base:
        if col not in colonnes_originales:
            colonnes_manquantes.append(col)
    
    # Vérifier les colonnes avec alternatives
    for col_standard, alternatives in colonnes_requises_alternatives.items():
        if not any(alt in colonnes_originales for alt in alternatives):
            colonnes_manquantes.append(f"{col_standard} (ou alternatives: {', '.join([alt for alt in alternatives if alt != col_standard])})")
    
    if colonnes_manquantes:
        st.warning(f"⚠️ Colonnes requises manquantes: {colonnes_manquantes}")
        st.error(f"Colonnes manquantes dans le fichier: {colonnes_manquantes}")
        st.write("Colonnes disponibles dans le fichier:", colonnes_originales)
    else:
        st.success("✅ Toutes les colonnes requises sont présentes (ou leurs alternatives)")
        
    # Créer un mapping des colonnes pour ce fichier spécifique
    mapping_fichier = {}
    for col_standard, alternatives in colonnes_requises_alternatives.items():
        for alt in alternatives:
            if alt in colonnes_originales and alt != col_standard:
                mapping_fichier[alt] = col_standard
                st.info(f"Mapping: '{alt}' sera utilisé comme '{col_standard}'")
                break
    
    # Identifier les colonnes parasites dans le fichier original
    colonnes_parasites = [col for col in colonnes_originales if col not in COLONNES_AUTORISEES["source"]]
    if colonnes_parasites:
        st.warning(f"⚠️ Colonnes parasites détectées (seront ignorées): {colonnes_parasites}")
    
    # 1.5 STANDARDISATION DES COLONNES
    st.subheader("🔍 Standardisation des colonnes")
    
    # DIAGNOSTIC: Avant extract_conseiller
    st.write("### DIAGNOSTIC: Avant extract_conseiller")
    st.write(f"Colonnes disponibles avant extract_conseiller: {df_original.columns.tolist()}")
    
    # Appliquer extract_conseiller sur le DataFrame original pour identifier la colonne Conseiller
    df_with_conseiller = extract_conseiller(df_original)
    
    # DIAGNOSTIC: Après extract_conseiller
    st.write("### DIAGNOSTIC: Après extract_conseiller")
    st.write(f"Colonnes disponibles après extract_conseiller: {df_with_conseiller.columns.tolist()}")
    st.write(f"'Conseiller' existe dans df_with_conseiller: {'Conseiller' in df_with_conseiller.columns}")
    
    # Vérifier si la colonne Conseiller a été correctement extraite
    if 'Conseiller' in df_with_conseiller.columns:
        st.success(f"✅ Colonne 'Conseiller' identifiée ou créée avec succès")
        # Afficher quelques exemples de conseillers identifiés
        st.write("Exemples de conseillers identifiés:")
        st.write(df_with_conseiller['Conseiller'].head(5).tolist())
    else:
        st.error(f"❌ Impossible d'extraire ou de créer la colonne 'Conseiller'")
    
    # 2. IMPORTATION STRICTE DES COLONNES AUTORISÉES
    st.subheader("🔒 Importation stricte des colonnes")
    
    df_strict = pd.DataFrame()
    
    # Copier UNIQUEMENT les colonnes autorisées qui existent dans le fichier source
    # en tenant compte du mapping des colonnes alternatives
    for col_standard in COLONNES_AUTORISEES["source"]:
        # Cas spécial pour la colonne Conseiller qui est déjà traitée par extract_conseiller
        if col_standard == 'Conseiller' and 'Conseiller' in df_with_conseiller.columns:
            # Utiliser la colonne Conseiller standardisée
            df_strict[col_standard] = df_with_conseiller['Conseiller'].copy()
            st.success(f"✅ Colonne '{col_standard}' importée depuis la version standardisée")
            continue
            
        # Vérifier si la colonne standard existe directement
        if col_standard in df_original.columns:
            df_strict[col_standard] = df_original[col_standard].copy()
            st.success(f"✅ Colonne '{col_standard}' importée avec succès")
        else:
            # Chercher une alternative dans le mapping
            alternative_trouvee = False
            for alt_col, std_col in mapping_fichier.items():
                if std_col == col_standard and alt_col in df_original.columns:
                    df_strict[col_standard] = df_original[alt_col].copy()
                    st.success(f"✅ Colonne '{col_standard}' importée depuis '{alt_col}'")
                    alternative_trouvee = True
                    break
            
            if not alternative_trouvee:
                st.warning(f"⚠️ Colonne '{col_standard}' non trouvée dans le fichier source (ni ses alternatives)")
                
    # Afficher un résumé des colonnes importées
    st.write("Colonnes importées dans le DataFrame strict:")
    st.write(df_strict.columns.tolist())
    
    # DIAGNOSTIC: Après filtrage strict des colonnes
    st.write("### DIAGNOSTIC: Après filtrage strict des colonnes")
    st.write(f"Colonnes disponibles dans df_strict: {df_strict.columns.tolist()}")
    st.write(f"'Conseiller' existe dans df_strict: {'Conseiller' in df_strict.columns}")
    if 'Conseiller' in df_strict.columns:
        st.write("Exemples de valeurs 'Conseiller' dans df_strict:", df_strict['Conseiller'].head(3).tolist())
            

    # DIAGNOSTIC: Après purge_parasitic_columns
    st.write(f"Colonnes disponibles après purge: {df_strict.columns.tolist()}")
    st.write(f"'Conseiller' existe après purge: {'Conseiller' in df_strict.columns}")
    if 'Conseiller' in df_strict.columns:
        st.write("Exemples de valeurs 'Conseiller' après purge:", df_strict['Conseiller'].head(3).tolist())
    
    # Afficher un aperçu du DataFrame strictement filtré
    st.write("Aperçu du DataFrame après filtrage strict des colonnes:")
    st.dataframe(df_strict.head())
    
    # 3. PRÉTRAITEMENT DES DONNÉES
    st.subheader("🔧 Prétraitement des données")
    
    # Vérifier si le DataFrame contient des données
    if df_strict.empty:
        st.error("⛔ Le DataFrame est vide après filtrage des colonnes. Vérifiez que votre fichier source contient les colonnes requises.")
        return None
    
    # Vérifier si la colonne 'Conseiller' existe dans df_strict, sinon la créer à partir de 'Conseiller'
    if 'Conseiller' not in df_strict.columns:
        if 'Conseiller' in df_original.columns:
            st.warning("⚠️ Colonne 'Conseiller' manquante. Création à partir de la colonne 'Conseiller'.")
            df_strict['Conseiller'] = df_original['Conseiller'].copy()
        else:
            st.warning("⚠️ Colonnes 'Conseiller' et 'Conseiller' manquantes. Création d'une colonne par défaut.")
            df_strict['Conseiller'] = 'Inconnu'
    
    # Prétraiter les données
    df_clean, colonnes_potentielles = pretraiter_donnees(df_strict)
    
    # Conversion des dates (format jj/mm/aaaa)
    # Vérifier si nous avons une colonne de date (soit 'Date de souscription' soit 'Date de souscription')
    date_col = None
    if "Date de souscription" in df_strict.columns:
        date_col = "Date de souscription"
    elif "Date de souscription" in df_strict.columns:
        date_col = "Date de souscription"
    
    if date_col:
        st.write(f"Conversion de la colonne '{date_col}' en date...")
        # Afficher quelques exemples de dates avant conversion
        st.write(f"Exemples de dates avant conversion: {df_strict[date_col].head(3).tolist()}")
        
        # Convertir en datetime et créer/remplacer la colonne standard 'Date de souscription'
        df_strict["Date de souscription"] = safe_to_datetime(df_strict[date_col])
        
        # Afficher quelques exemples de dates après conversion
        st.write(f"Exemples de dates après conversion: {df_strict['Date de souscription'].head(3).tolist()}")
        
        # Vérifier si la conversion a réussi
        missing_dates = df_strict["Date de souscription"].isna().sum()
        st.write(f"Nombre de dates manquantes après conversion: {missing_dates}")
        
        if missing_dates > 0:
            st.warning(f"⚠️ {missing_dates} dates n'ont pas pu être converties. Vérifiez le format des dates.")
    else:
        st.error("❌ Aucune colonne de date ('Date de souscription' ou 'Date de souscription') n'a été trouvée. Impossible de continuer l'analyse temporelle.")
        return None
    
    # Conversion des montants
    montant_col = None
    if "Montant" in df_strict.columns:
        montant_col = "Montant"
    elif "Montant du placement" in df_strict.columns:
        montant_col = "Montant du placement"
    
    if montant_col:
        st.write(f"Conversion de la colonne '{montant_col}' en nombre...")
        # Nettoyer et convertir la colonne Montant
        df_strict["Montant du placement"] = safe_to_numeric(df_strict[montant_col])
        # Afficher quelques exemples pour vérification
        st.write(f"Exemples de montants convertis: {df_strict['Montant du placement'].head(3).tolist()}")
        st.write("Exemples de montants convertis:")
        montant_examples = pd.DataFrame({
            "Original": df_strict[montant_col].head(3),
            "Converti": df_strict["Montant du placement"].head(3)
        })
        st.dataframe(montant_examples)
    else:
        st.warning("⚠️ Colonne 'Montant' manquante. Impossible de créer 'Montant du placement'.")
    
    # Extraction du mois à partir de la date de souscription
    if "Date de souscription" in df.columns:
        st.write("Extraction du mois à partir de la date de souscription...")
        df["Mois"] = df["Date de souscription"].dt.strftime("%Y-%m")
        # Afficher les mois disponibles
        mois_disponibles = df["Mois"].unique().tolist()
        st.write(f"Mois disponibles dans les données: {mois_disponibles}")
    else:
        st.warning("⚠️ Colonne 'Date de souscription' manquante. Impossible de créer 'Mois'.")
    
    # Ajout des colonnes de premier et dernier jour du mois
    if "Date de souscription" in df.columns:
        st.write("Ajout des colonnes de premier et dernier jour du mois...")
        df = adjust_dates_to_month_range(df, "Date de souscription")
    
    
    # Vérification finale des colonnes
    st.write("Colonnes finales après prétraitement et purge:")
    st.write(df.columns.tolist())
    
    # Vérification des colonnes non autorisées (ne devrait jamais arriver avec cette approche)
    toutes_colonnes_autorisees = COLONNES_AUTORISEES["source"] + COLONNES_AUTORISEES["derivees"]
    colonnes_non_autorisees = [col for col in df.columns if col not in toutes_colonnes_autorisees]
    if colonnes_non_autorisees:
        st.error(f"⛔ ERREUR CRITIQUE: Des colonnes non autorisées sont présentes après prétraitement: {colonnes_non_autorisees}")
        # Suppression d'urgence des colonnes non autorisées
        df = df.drop(columns=colonnes_non_autorisees)
        st.success("Colonnes non autorisées supprimées avec succès.")
    
    # Afficher un aperçu du DataFrame après prétraitement
    st.write("Aperçu du DataFrame après prétraitement:")
    st.dataframe(df.head())
    
    # Afficher un message d'information sur les colonnes attendues
    st.info("""
    📝 **Colonnes attendues dans le fichier**:
    - "Nom de l'opportunité" (texte)
    - "Produit" (texte)
    - "Statut" (texte)
    - "Étape" (texte)
    - "Montant" (nombre)
    - "Montant des frais" (nombre)
    - "Type d'investissement" (texte)
    - "Conseiller" (texte)
    - "Date de souscription" (date au format jj/mm/aaaa)
    - "Date de validation" (date au format jj/mm/aaaa)
    
    ℹ️ **Important**: Les noms des colonnes doivent correspondre **exactement** à ceux listés ci-dessus.
    """)
    
    # FILTRAGE STRICT DÈS L'IMPORTATION: Créer un nouveau DataFrame vide qui ne contiendra
    # que les colonnes explicitement spécifiées par l'utilisateur
    st.subheader("⚠️ FILTRAGE STRICT DES COLONNES")
    st.write("Seules les colonnes explicitement spécifiées sont conservées, toutes les autres sont ignorées.")
    
    # Afficher les colonnes du fichier original pour diagnostic
    st.write("Colonnes présentes dans le fichier original:")
    st.write(df_original.columns.tolist())
    
    # Identifier les colonnes parasites dans le fichier original
    colonnes_parasites_original = [col for col in df_original.columns if col not in colonnes_strictes]
    if colonnes_parasites_original:
        st.warning(f"⚠️ ATTENTION: Colonnes parasites détectées dans le fichier original: {colonnes_parasites_original}")
        st.write("Ces colonnes seront ignorées dans l'analyse.")
    
    # Créer un DataFrame vide avec le même index que l'original
    df = pd.DataFrame(index=df_original.index)
    
    # Liste STRICTE des colonnes attendues - EXACTEMENT comme fournies par l'utilisateur
    colonnes_strictes = [
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
    
    # IMPORTANT: Cette liste est la SEULE référence, sans aucune variation ni mapping
    # Toute colonne non présente dans cette liste sera ignorée
    # Toute colonne présente dans cette liste mais absente du fichier sera signalée
    
    # Ne copier QUE les colonnes qui existent dans le fichier original
    colonnes_trouvees = []
    colonnes_manquantes = []
    for col in colonnes_strictes:
        if col in df_original.columns:
            df[col] = df_original[col]
            colonnes_trouvees.append(col)
            st.success(f"✅ Colonne '{col}' trouvée et conservée")
        else:
            colonnes_manquantes.append(col)
            st.warning(f"⚠️ Colonne '{col}' non trouvée dans le fichier original")
    
    # Afficher les colonnes conservées après filtrage strict
    st.write("Colonnes conservées après filtrage strict:")
    st.write(df.columns.tolist())
    
    # CORRECTION: Définir explicitement les colonnes attendues avec leur type
    colonnes_attendues_types = {
        "Nom de l'opportunité": "string",
        "Produit": "string",
        "Statut": "string",
        "Étape": "string",
        "Montant": "float",
        "Montant des frais": "float",
        "Type d'investissement": "string",
        "Conseiller": "string",
        "Date de souscription": "date",
        "Date de validation": "date"
    }
    
    # Afficher les colonnes attendues
    st.subheader("Colonnes attendues")
    for col, type_col in colonnes_attendues_types.items():
        st.write(f"- {col} ({type_col})")
    
    # Afficher les colonnes disponibles dans le fichier
    st.subheader("Colonnes disponibles dans le fichier")
    st.write(df.columns.tolist())
    
    # Vérifier si les colonnes attendues sont présentes
    colonnes_manquantes = [col for col in colonnes_attendues_types.keys() if col not in df.columns]
    if colonnes_manquantes:
        st.warning(f"Colonnes attendues manquantes: {', '.join(colonnes_manquantes)}")
    
    # Afficher un aperçu des données pour vérification
    with st.expander("Aperçu des données brutes"):
        st.dataframe(df.head(10))
    
    # Fonction pour assurer la compatibilité des types avec PyArrow/Streamlit
    def ensure_pyarrow_compatibility(df):
        """Assure que toutes les colonnes du DataFrame sont compatibles avec PyArrow."""
        df_clean = df.copy()
        
        # Convertir explicitement les colonnes problématiques en types compatibles
        for col in df_clean.columns:
            # Convertir les colonnes de type 'object' en string pour éviter les erreurs PyArrow
            if df_clean[col].dtype == 'object':
                # Pour les colonnes numériques, essayer d'abord la conversion en float
                if col.lower().find('montant') >= 0 or col.lower().find('frais') >= 0:
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype('float64')
                    except:
                        df_clean[col] = df_clean[col].astype(str)
                # Pour les autres colonnes de type object, convertir en string
                else:
                    df_clean[col] = df_clean[col].fillna('').astype(str)
        
        return df_clean
        
    # Fonction de prétraitement robuste pour analyser la structure du fichier
    def pretraiter_donnees(df):
        """Prétraite les données pour l'analyse des souscriptions épargne."""
        # Créer une copie du DataFrame pour éviter de modifier l'original
        df_clean = df.copy()
        
        # Afficher les types de données pour chaque colonne
        with st.expander("Types de données par colonne"):
            st.write(df_clean.dtypes)
        
        # Extraire et standardiser la colonne Conseiller
        st.write("### Extraction et standardisation de la colonne Conseiller")
        df_clean = extract_conseiller(df_clean)
        
        # Rechercher les colonnes potentielles pour chaque catégorie
        colonnes_potentielles = {
            'date': [],
            'montant': [],
            'produit': [],
            'etape': [],
            'frais': [],
        }
        
        # Afficher les colonnes potentielles détectées
        with st.expander("Colonnes potentielles détectées par catégorie"):
            st.write(colonnes_potentielles)
        
        return df_clean, colonnes_potentielles
    
    # Utiliser EXACTEMENT les colonnes fournies par l'utilisateur
    # Structure cible: ["Nom de l'opportunité" (str), "Produit" (str), "Statut" (str), "Étape" (str), 
    #                  "Montant" (float), "Montant des frais" (float), "Type d'investissement" (str), 
    #                  "Conseiller" (str), "Date de souscription" (date), "Date de validation" (date)]
    
    # Liste exacte des colonnes à prendre en compte, dans l'ordre spécifié par l'utilisateur
    colonnes_exactes = [
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
    
    # Liste STRICTE des colonnes attendues (exactement comme spécifié par l'utilisateur)
    colonnes_strictes = [
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
    
    # Afficher la liste exacte des colonnes attendues
    st.subheader("Colonnes attendues (format exact)")
    for i, col in enumerate(colonnes_strictes):
        st.write(f"{i}: \"{col}\"")
    
    # Afficher les colonnes disponibles dans le fichier pour debug
    st.write("Colonnes disponibles dans le fichier:")
    st.write(df.columns.tolist())
    
    # Vérifier si toutes les colonnes attendues sont présentes
    colonnes_manquantes_fichier = [col for col in colonnes_strictes if col not in df.columns]
    
    if colonnes_manquantes_fichier:
        st.warning(f"⚠️ Attention: Les colonnes suivantes sont attendues mais absentes du fichier: {', '.join(colonnes_manquantes_fichier)}")
    
    # Créer une copie du DataFrame pour éviter de modifier l'original
    df_mapped = df.copy()
    
    # Prétraiter les données pour analyser la structure du fichier
    df_clean, colonnes_potentielles = pretraiter_donnees(df)
    
    # SUPPRESSION DU MAPPING AUTOMATIQUE - Utilisation stricte des colonnes spécifiées
    st.subheader("Utilisation stricte des colonnes spécifiées")
    st.write("Le dashboard utilise UNIQUEMENT les colonnes exactes spécifiées, sans mapping automatique ni alternatives.")
    
    # Pas de mapping manuel - utilisation stricte des colonnes spécifiées
    mapping_manuel = {}
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Colonnes essentielles**")
        # Date de souscription
        options_date = ["Automatique"] + ["Date de souscription", "Date de souscription", "Date de validation"] + list(df.columns)
        date_col = st.selectbox("Colonne Date de souscription", options=options_date, key="date_col")
        if date_col != "Automatique" and date_col != "Aucune":
            mapping_manuel['Date de souscription'] = date_col
        
        # Montant du placement
        options_montant = ["Automatique"] + ["Montant", "Montant du placement", "Montant des frais"] + list(df.columns)
        montant_col = st.selectbox("Colonne Montant du placement", options=options_montant, key="montant_col")
        if montant_col != "Automatique" and montant_col != "Aucune":
            mapping_manuel['Montant du placement'] = montant_col
            
        # Conseiller
        options_conseiller = ["Automatique"] + ["Conseiller"] + list(df.columns)
        conseiller_col = st.selectbox("Colonne Conseiller", options=options_conseiller, key="conseiller_col")
        if conseiller_col != "Automatique" and conseiller_col != "Aucune":
            mapping_manuel['Conseiller'] = conseiller_col
    
    with col2:
        st.write("**Colonnes produit/type**")
        # Type d'investissement
        options_type = ["Automatique"] + ["Type d'investissement"] + list(df.columns)
        type_col = st.selectbox("Colonne Type d'investissement", options=options_type, key="type_col")
        if type_col != "Automatique" and type_col != "Aucune":
            mapping_manuel['Type d\'investissement'] = type_col
        
        # Produit
        options_produit = ["Automatique"] + ["Produit"] + list(df.columns)
        produit_col = st.selectbox("Colonne Produit", options=options_produit, key="produit_col")
        if produit_col != "Automatique" and produit_col != "Aucune":
            mapping_manuel['Produit'] = produit_col
            
        # Nom opportunité
        options_opportunite = ["Automatique"] + ["Nom de l'opportunité"] + list(df.columns)
        opportunite_col = st.selectbox("Colonne Nom opportunité", options=options_opportunite, key="opportunite_col")
        if opportunite_col != "Automatique" and opportunite_col != "Aucune":
            mapping_manuel['Nom opportunité'] = opportunite_col
    
    with col3:
        st.write("**Colonnes statut/étape**")
        # Étape
        options_etape = ["Automatique"] + ["Étape"] + list(df.columns)
        etape_col = st.selectbox("Colonne Étape", options=options_etape, key="etape_col")
        if etape_col != "Automatique" and etape_col != "Aucune":
            mapping_manuel['Étape'] = etape_col
            
        # Statut
        options_statut = ["Automatique"] + ["Statut"] + list(df.columns)
        statut_col = st.selectbox("Colonne Statut", options=options_statut, key="statut_col")
        if statut_col != "Automatique" and statut_col != "Aucune":
            mapping_manuel['Statut'] = statut_col
            
        # Frais
        options_frais = ["Automatique"] + ["Montant des frais", "Frais"] + list(df.columns)
        frais_col = st.selectbox("Colonne Frais", options=options_frais, key="frais_col")
        if frais_col != "Automatique" and frais_col != "Aucune":
            mapping_manuel['Frais'] = frais_col
    
    # Appliquer le mapping manuel ou automatique
    df_mapped = df.copy()
    colonnes_trouvees = {}
    colonnes_manquantes = []
    
    # UTILISATION STRICTE DES COLONNES - Sans mapping automatique ni alternatives
    # Seules les colonnes exactement nommées comme spécifié sont utilisées
    for col in colonnes_strictes:
        if col in df_mapped.columns:
            # La colonne existe exactement comme spécifiée, on la conserve telle quelle
            colonnes_trouvees[col] = col
            st.success(f"Colonne '{col}' trouvée exactement comme spécifiée")
        else:
            # La colonne n'existe pas avec ce nom exact, elle est considérée comme manquante
            colonnes_manquantes.append(col)
            st.warning(f"\u26a0\ufe0f La colonne '{col}' n'existe pas dans le fichier avec ce nom exact")
    
    # Création des colonnes dérivées UNIQUEMENT à partir des colonnes réellement présentes
    # Ne pas créer de colonne dérivée si la colonne source n'existe pas
    
    # Colonne "Date de souscription" dérivée de "Date de souscription"
    if "Date de souscription" in df.columns:
        # Créer la colonne dérivée uniquement si la colonne source existe réellement
        df_mapped["Date de souscription"] = df["Date de souscription"]
        colonnes_trouvees["Date de souscription"] = "Date de souscription"
        st.success(f"Colonne dérivée 'Date de souscription' créée à partir de 'Date de souscription'")
    else:
        st.warning(f"Impossible de créer la colonne dérivée 'Date de souscription' car 'Date de souscription' n'existe pas")
    
    # Colonne "Montant du placement" dérivée de "Montant"
    if "Montant" in df.columns:
        # Créer la colonne dérivée uniquement si la colonne source existe réellement
        df_mapped["Montant du placement"] = df["Montant"]
        colonnes_trouvees["Montant du placement"] = "Montant"
        st.success(f"Colonne dérivée 'Montant du placement' créée à partir de 'Montant'")
    else:
        st.warning(f"Impossible de créer la colonne dérivée 'Montant du placement' car 'Montant' n'existe pas")
    
    # Afficher les colonnes manquantes
    if colonnes_manquantes:
        st.warning(f"Colonnes non trouvées: {', '.join(colonnes_manquantes)}")
    
    # Afficher un résumé du mapping final
    with st.expander("Résumé du mapping final des colonnes"):
        st.write(colonnes_trouvees)
    
    # IMPORTANT: Ne créer que les colonnes qui existent réellement dans le fichier source
    # Créer un nouveau DataFrame avec uniquement les colonnes existantes
    df_clean = pd.DataFrame(index=df.index)
    
    # Ne copier que les colonnes qui existent réellement dans le fichier source
    for col in colonnes_strictes:
        if col in df.columns:
            # Si la colonne existe, la copier directement
            df_clean[col] = df[col]
            st.success(f"Colonne '{col}' trouvée directement dans le fichier source")
    
    # Avertir pour les colonnes manquantes mais ne pas les créer
    for col in colonnes_strictes:
        if col not in df.columns:
            st.warning(f"Colonne '{col}' non trouvée dans le fichier source - cette colonne ne sera pas utilisée dans l'analyse")
    
    # Remplacer le DataFrame original par le DataFrame avec uniquement les colonnes exactes
    df = df_clean.copy()
    
    # Vérifier que l'ordre des colonnes correspond exactement à celui spécifié
    if df.columns.tolist() != colonnes_exactes:
        st.warning("Réorganisation des colonnes selon l'ordre spécifié...")
        df = df[colonnes_exactes]
    
    # Utiliser directement le DataFrame nettoyé
    df_epargne = df.copy()
    
    # Vérification finale des colonnes
    # Créer une copie du DataFrame pour l'analyse
    df_epargne = df.copy()
    
    # IMPORTANT: Ne conserver que les colonnes strictement spécifiées
    colonnes_a_conserver = [col for col in colonnes_strictes if col in df_epargne.columns]
    df_epargne = df_epargne[colonnes_a_conserver].copy()
    
    # Afficher les colonnes conservées après filtrage strict
    st.write("Colonnes conservées après filtrage strict:")
    st.write(df_epargne.columns.tolist())
    
    # Conversion des colonnes avec gestion des erreurs - Utilisation STRICTE des colonnes spécifiées
    
    # 1. Conversion des dates (UNIQUEMENT "Date de souscription" et "Date de validation" comme spécifié)
    st.subheader("Conversion des types de données")
    st.write("Conversion des dates - Utilisation stricte des colonnes spécifiées")
    
    # IMPORTANT: Traitement STRICT des colonnes - Ne traiter que les colonnes réellement présentes
    # 1. Conversion des dates - UNIQUEMENT si les colonnes existent réellement
    
    # Traitement de "Date de souscription" - UNIQUEMENT si la colonne existe réellement
    if "Date de souscription" in df_epargne.columns:
        st.write("Conversion de 'Date de souscription' en date...")
        # Convertir en datetime
        df_epargne["Date de souscription"] = safe_to_datetime(df_epargne["Date de souscription"])
        
        # Créer une colonne pour l'affichage formaté en jj/mm/aaaa
        df_epargne["Date de souscription_affichage"] = df_epargne["Date de souscription"].dt.strftime("%d/%m/%Y")
        st.success("✅ Format jj/mm/aaaa appliqué à 'Date de souscription'")
        
        # Créer la colonne Mois uniquement si Date de souscription existe et contient des dates valides
        if "Date de souscription" in df_epargne.columns and not df_epargne["Date de souscription"].isna().all():
            df_epargne["Mois"] = df_epargne["Date de souscription"].dt.to_period("M").astype(str)
            st.success("Colonne 'Mois' créée avec succès")
        else:
            st.warning("Impossible de créer la colonne 'Mois' car 'Date de souscription' contient des valeurs invalides")
    else:
        st.warning("Colonne 'Date de souscription' absente - Impossible de créer les dates de souscription")
    
    # Traitement de "Date de validation" - UNIQUEMENT si la colonne existe réellement
    if "Date de validation" in df_epargne.columns:
        st.write("Conversion de 'Date de validation' en date...")
        # Convertir en datetime
        df_epargne["Date de validation"] = safe_to_datetime(df_epargne["Date de validation"])
        
        # Créer une colonne pour l'affichage formaté en jj/mm/aaaa
        df_epargne["Date de validation_affichage"] = df_epargne["Date de validation"].dt.strftime("%d/%m/%Y")
        st.success("✅ Format jj/mm/aaaa appliqué à 'Date de validation'")
    else:
        st.warning("Colonne 'Date de validation' absente - Cette information ne sera pas disponible pour l'analyse")
    
    # 2. Conversion des montants - UNIQUEMENT si les colonnes existent réellement
    st.write("Conversion des montants - Utilisation stricte des colonnes spécifiées")
    
    # Traitement de "Montant" - UNIQUEMENT si la colonne existe réellement
    if "Montant" in df_epargne.columns:
        st.write("Conversion de 'Montant' en nombre...")
        df_epargne["Montant"] = safe_to_numeric(df_epargne["Montant"])
        # Créer la colonne dérivée nécessaire au fonctionnement
        df_epargne["Montant du placement"] = df_epargne["Montant"]
        st.success("Colonne 'Montant du placement' créée avec succès")
    else:
        st.warning("Colonne 'Montant' absente - Impossible de créer les montants de placement")
    
    # Traitement de "Montant des frais" - UNIQUEMENT si la colonne existe réellement
    if "Montant des frais" in df_epargne.columns:
        st.write("Conversion de 'Montant des frais' en nombre...")
        df_epargne["Montant des frais"] = safe_to_numeric(df_epargne["Montant des frais"])
        st.success("Colonne 'Montant des frais' convertie avec succès")
    else:
        st.warning("Colonne 'Montant des frais' absente - Cette information ne sera pas disponible pour l'analyse")
    
    # IMPORTANT: FILTRAGE FINAL STRICT - Garantir qu'aucune colonne parasite n'est présente
    # Créer un nouveau DataFrame FINAL qui ne contiendra QUE les colonnes strictes + dérivées nécessaires
    st.subheader("🔍 FILTRAGE FINAL STRICT DES COLONNES")
    st.write("Création d'un DataFrame final ne contenant strictement que les colonnes spécifiées et leurs dérivées nécessaires.")
    
    # Créer un DataFrame vide pour l'analyse finale
    df_final = pd.DataFrame(index=df_epargne.index)
    
    # 1. Copier UNIQUEMENT les colonnes strictes qui existent réellement dans le fichier source
    colonnes_copiees = []
    for col in colonnes_strictes:
        if col in df_epargne.columns:
            df_final[col] = df_epargne[col]
            colonnes_copiees.append(col)
            st.success(f"✅ Colonne stricte '{col}' copiée dans le DataFrame final")
    
    # 2. Ajouter UNIQUEMENT les colonnes dérivées nécessaires au fonctionnement
    colonnes_derivees = []
    
    # 2.1 Colonne "Date de souscription" dérivée de "Date de souscription"
    if 'Date de souscription' in df_final.columns:
        df_final['Date de souscription'] = df_final['Date de souscription']
        colonnes_derivees.append('Date de souscription')
        st.success("✅ Colonne dérivée 'Date de souscription' ajoutée au DataFrame final")
    
    # 2.2 Colonne "Montant du placement" dérivée de "Montant"
    if 'Montant' in df_final.columns:
        df_final['Montant du placement'] = df_final['Montant']
        colonnes_derivees.append('Montant du placement')
        st.success("✅ Colonne dérivée 'Montant du placement' ajoutée au DataFrame final")
    
    # 2.3 Colonne "Mois" dérivée de "Date de souscription"
    if 'Date de souscription' in df_final.columns and not df_final['Date de souscription'].isna().all():
        df_final['Mois'] = df_final['Date de souscription'].dt.to_period('M').astype(str)
        colonnes_derivees.append('Mois')
        st.success("✅ Colonne dérivée 'Mois' ajoutée au DataFrame final")
    
    # Remplacer df_epargne par df_final pour la suite de l'analyse
    df_epargne = df_final.copy()
    
    # Afficher les colonnes finales pour vérification
    st.subheader("Colonnes finales utilisées pour l'analyse")
    st.write(f"Colonnes strictes ({len(colonnes_copiees)}): {colonnes_copiees}")
    st.write(f"Colonnes dérivées ({len(colonnes_derivees)}): {colonnes_derivees}")
    st.write("Toutes les colonnes du DataFrame final:")
    st.write(df_epargne.columns.tolist())
    
    # Vérification finale qu'aucune colonne parasite n'est présente
    colonnes_autorisees = colonnes_strictes + ['Date de souscription', 'Montant du placement', 'Mois']
    colonnes_parasites = [col for col in df_epargne.columns if col not in colonnes_autorisees]
    
    if colonnes_parasites:
        st.error(f"⛔ ATTENTION: Des colonnes parasites sont encore présentes: {colonnes_parasites}")
        # Supprimer les colonnes parasites
        df_epargne = df_epargne.drop(columns=colonnes_parasites)
        st.success(f"Les colonnes parasites ont été supprimées du DataFrame final")
        st.write("Colonnes finales après suppression des parasites:")
        st.write(df_epargne.columns.tolist())
    else:
        st.success("✅ Aucune colonne parasite détectée - Le DataFrame final est conforme aux spécifications")
        
    # Afficher un aperçu du DataFrame final
    with st.expander("Aperçu du DataFrame final"):
        st.dataframe(df_epargne.head(10))
        
    # Afficher un aperçu des données converties
    with st.expander("Aperçu des données après conversion"):
        st.write("Exemples de valeurs dans les colonnes principales:")
        for col in ['Montant', 'Montant des frais', 'Date de souscription', 'Date de validation']:
            if col in df_epargne.columns:
                st.write(f"Colonne '{col}':")
                st.write(df_epargne[col].head())
        df_epargne['Montant du placement'] = 0  # Valeur par défaut en cas d'erreur
    
    # FILTRAGE FINAL ULTRA-STRICT: Garantir qu'aucune colonne parasite ne subsiste
    st.subheader("🔒 FILTRAGE FINAL ULTRA-STRICT")
    st.write("Vérification finale et suppression de toute colonne parasite potentielle.")
    
    # Créer un DataFrame FINAL complètement nouveau
    df_final_strict = pd.DataFrame(index=df_epargne.index)
    
    # Vérifier si la colonne Conseiller existe exactement comme spécifiée
    if 'Conseiller' not in df_epargne.columns:
        st.warning("La colonne 'Conseiller' n'est pas présente dans le fichier source")
        # Ajouter la colonne uniquement si elle est absente
        df_epargne['Conseiller'] = 'Inconnu'
    
    # Identifier les colonnes parasites pour information
    colonnes_parasites = [col for col in df_epargne.columns if col not in colonnes_autorisees]
    if colonnes_parasites:
        st.error(f"⛔ ATTENTION: Colonnes parasites détectées avant filtrage final: {', '.join(colonnes_parasites)}")
        st.write("Ces colonnes seront supprimées définitivement.")
    
    # Copier UNIQUEMENT les colonnes strictes qui existent dans le DataFrame source
    colonnes_strictes_copiees = []
    for col in colonnes_strictes:
        if col in df_epargne.columns:
            df_final_strict[col] = df_epargne[col]
            colonnes_strictes_copiees.append(col)
            st.success(f"✅ Colonne stricte '{col}' copiée dans le DataFrame final")
    
    # Copier UNIQUEMENT les colonnes dérivées qui existent dans le DataFrame source
    colonnes_derivees_copiees = []
    for col in colonnes_derivees:
        if col in df_epargne.columns:
            df_final_strict[col] = df_epargne[col]
            colonnes_derivees_copiees.append(col)
            st.success(f"✅ Colonne dérivée '{col}' copiée dans le DataFrame final")
    
    # Remplacer df_epargne par le DataFrame final ultra-strict
    df_epargne = df_final_strict.copy()
    
    # Vérification finale qu'aucune colonne parasite n'est présente
    colonnes_finales = df_epargne.columns.tolist()
    colonnes_parasites_finales = [col for col in colonnes_finales if col not in colonnes_autorisees]
    
    if colonnes_parasites_finales:
        st.error(f"⛔ ERREUR CRITIQUE: Des colonnes parasites subsistent après filtrage ultra-strict: {colonnes_parasites_finales}")
        # Suppression d'urgence des colonnes parasites
        df_epargne = df_epargne.drop(columns=colonnes_parasites_finales)
        st.warning("Suppression d'urgence des colonnes parasites effectuée.")
    else:
        st.success("✅ VALIDATION FINALE: Aucune colonne parasite détectée. Le DataFrame est strictement conforme aux spécifications.")
    
    # Afficher les colonnes finales pour vérification
    st.write("Colonnes finales utilisées pour l'analyse:")
    st.write(df_epargne.columns.tolist())
    
    # Vérifier si la colonne Type d'investissement existe
    if 'Type d\'investissement' in df_epargne.columns:
        st.success("Colonne 'Type d\'investissement' détectée avec succès")
        # Afficher les types d'investissement uniques pour vérification
        types_investissement = df_epargne['Type d\'investissement'].dropna().unique()
        st.write("Types d'investissement détectés:")
        st.write(types_investissement)
    
    # Vérifier si la colonne Montant du placement existe
    if 'Montant du placement' in df_epargne.columns:
        # Vérifier les montants à 0
        zero_count = (df_epargne['Montant du placement'] == 0).sum()
        if zero_count > 0:
            st.warning(f"⚠️ {zero_count} souscriptions Épargne avec un montant de 0€ détectées.")
        
        # Filtrer les montants valides
        df_epargne_valid = df_epargne[df_epargne['Montant du placement'] > 0].copy()
        if df_epargne_valid.empty:
            st.warning("⚠️ Aucune souscription Épargne avec un montant supérieur à 0.")
            # Utiliser toutes les données même avec montants à 0 plutôt que de retourner
            df_epargne_valid = df_epargne.copy()
            st.info("Utilisation de toutes les données disponibles malgré l'absence de montants valides.")
    else:
        # Si la colonne n'existe pas, utiliser toutes les données
        df_epargne_valid = df_epargne.copy()
        st.warning("⚠️ Colonne 'Montant du placement' non trouvée. Analyse limitée.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Filtrer pour ne prendre en compte que les souscriptions avec étape "Acté" ou "Validé"
        # Vérifier si la colonne Étape existe (avec différentes casses possibles)
        etape_col = next((col for col in df_epargne_valid.columns if col.lower() in ['étape', 'etape']), None)
        
        if etape_col:
            df_finalise = df_epargne_valid[df_epargne_valid[etape_col].str.lower().isin(['acté', 'validé', 'acte', 'valide'])]
            total_collecte = df_finalise['Montant du placement'].sum()
            st.metric("💰 Collecte Totale Épargne (Acté/Validé)", f"{total_collecte:,.0f}€")
        else:
            total_collecte = df_epargne_valid['Montant du placement'].sum()
            st.metric("💰 Collecte Totale Épargne", f"{total_collecte:,.0f}€")
    with col2:
        nb_souscriptions = len(df_epargne_valid)
        st.metric("📝 Nombre de Souscriptions", f"{nb_souscriptions:,}")
    with col3:
        ticket_moyen = df_epargne_valid['Montant du placement'].mean()
        st.metric("🎯 Ticket Moyen", f"{ticket_moyen:,.0f}€")
    with col4:
        nb_conseillers = df_epargne_valid['Conseiller'].nunique()
        st.metric("👥 Nombre de Conseillers", f"{nb_conseillers}")
    
    st.subheader("🔍 Filtres")
    col1, col2 = st.columns(2)
    df_filtre = df_epargne_valid.copy()
    with col1:
        mois_disponibles = sorted(df_epargne_valid['Mois'].dropna().unique())
        mois_selectionne = st.selectbox("📅 Mois", options=["Tous"] + mois_disponibles, key="mois_epargne")
        if mois_selectionne != "Tous":
            df_filtre = df_filtre[df_filtre['Mois'] == mois_selectionne]
    with col2:
        conseillers_disponibles = sorted(df_epargne_valid['Conseiller'].dropna().unique())
        conseiller_selectionne = st.selectbox("👤 Conseiller", options=["Tous"] + conseillers_disponibles, key="conseiller_epargne")
        if conseiller_selectionne != "Tous":
            df_filtre = df_filtre[df_filtre['Conseiller'] == conseiller_selectionne]
    
    if not df_filtre.empty:
        st.subheader("📈 Évolution de la Collecte Épargne (du 1er au dernier jour)")
        
        # Filtrer par étape "Acté" et "Validé" si la colonne Étape existe
        if 'Étape' in df_filtre.columns:
            try:
                # Filtrer pour ne garder que les étapes "Acté" et "Validé" (insensible à la casse)
                etapes_finalisees = ['acté', 'validé', 'acte', 'valide']
                mask_etapes = df_filtre['Étape'].astype(str).str.lower().isin(etapes_finalisees)
                df_filtre_finalise = df_filtre[mask_etapes].copy()
                
                # Informer l'utilisateur du filtrage effectué
                nb_total = len(df_filtre)
                nb_finalise = len(df_filtre_finalise)
                st.info(f"ℹ️ {nb_finalise} souscriptions sur {nb_total} ont une étape 'Acté' ou 'Validé' et sont considérées comme finalisées.")
                
                # Si aucune ligne ne correspond aux étapes finalisées, utiliser toutes les données
                if df_filtre_finalise.empty and not df_filtre.empty:
                    st.warning("⚠️ Aucune souscription avec étape 'Acté' ou 'Validé' trouvée. Utilisation de toutes les données.")
                    df_filtre_finalise = df_filtre.copy()
            except Exception as e:
                st.error(f"Erreur lors du filtrage par étape: {e}")
                df_filtre_finalise = df_filtre.copy()
        else:
            # Si la colonne Étape n'existe pas, utiliser toutes les données
            df_filtre_finalise = df_filtre.copy()
            st.info("ℹ️ Colonne 'Étape' non trouvée. Toutes les souscriptions sont considérées pour l'analyse.")
        
        # Assurer la compatibilité des types avant d'afficher le DataFrame
        df_filtre_compatible = ensure_pyarrow_compatibility(df_filtre)
        
        # Afficher le DataFrame filtré
        st.subheader("📋 Données filtrées")
        st.dataframe(df_filtre_compatible)
        
        # Utiliser le DataFrame filtré pour la suite
        df_filtre = df_filtre_finalise
        
        # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
        df_filtre = adjust_dates_to_month_range(df_filtre, 'Date de souscription')
        
        # Créer une agrégation mensuelle avec les dates de début et fin
        evolution_mensuelle = df_filtre.groupby('Mois').agg(
            Montant_Total=('Montant du placement', 'sum'),
            Premier_Jour=('Premier_Jour_Mois', 'first'),
            Dernier_Jour=('Dernier_Jour_Mois', 'first')
        ).reset_index()
        
        # Trier par date de début
        evolution_mensuelle = evolution_mensuelle.sort_values('Premier_Jour')
        
        # S'assurer que les dates sont valides (pas de NaT)
        evolution_mensuelle['Premier_Jour'] = pd.to_datetime(evolution_mensuelle['Premier_Jour'], errors='coerce')
        evolution_mensuelle['Dernier_Jour'] = pd.to_datetime(evolution_mensuelle['Dernier_Jour'], errors='coerce')
        
        # Remplacer les dates NaT par les premiers et derniers jours du mois basés sur la colonne 'Mois'
        for idx, row in evolution_mensuelle.iterrows():
            if pd.isna(row['Premier_Jour']) or pd.isna(row['Dernier_Jour']):
                try:
                    mois_annee = row['Mois']
                    annee, mois = mois_annee.split('-')
                    premier_jour = pd.Timestamp(year=int(annee), month=int(mois), day=1)
                    dernier_jour = pd.Timestamp(year=int(annee), month=int(mois), day=pd.Timestamp(premier_jour).days_in_month)
                    
                    if pd.isna(row['Premier_Jour']):
                        evolution_mensuelle.at[idx, 'Premier_Jour'] = premier_jour
                    if pd.isna(row['Dernier_Jour']):
                        evolution_mensuelle.at[idx, 'Dernier_Jour'] = dernier_jour
                except Exception:
                    # En cas d'erreur, laisser les valeurs NaT
                    pass
        
        # Créer des étiquettes personnalisées pour l'axe X avec les plages de dates
        evolution_mensuelle['Période'] = evolution_mensuelle.apply(
            lambda row: f"{row['Mois']} ({row['Premier_Jour'].strftime('%d/%m') if pd.notna(row['Premier_Jour']) else '01/01'} - {row['Dernier_Jour'].strftime('%d/%m') if pd.notna(row['Dernier_Jour']) else '31/12'})", 
            axis=1
        )
        
        # Calculer l'écart par rapport à l'objectif
        evolution_mensuelle['Écart Objectif'] = evolution_mensuelle['Montant_Total'] - OBJECTIF_MENSUEL_EPARGNE
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
            title=f"📊 Évolution Mensuelle de la Collecte Épargne (Objectif: {OBJECTIF_MENSUEL_EPARGNE:,.0f}€)",
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
            y0=OBJECTIF_MENSUEL_EPARGNE,
            y1=OBJECTIF_MENSUEL_EPARGNE,
            line=dict(color="black", width=2, dash="dash")
        )
        
        # Ajouter une annotation pour l'objectif
        fig_mensuel.add_annotation(
            x=len(evolution_mensuelle['Période'])-1,
            y=OBJECTIF_MENSUEL_EPARGNE,
            text=f"Objectif: {OBJECTIF_MENSUEL_EPARGNE:,.0f}€",
            showarrow=False,
            yshift=10
        )
        
        st.plotly_chart(fig_mensuel, use_container_width=True)
        
        # Analyse des types de versements
        st.subheader("💸 Analyse par Type de Versement")
        
        # Vérifier si la colonne Type de versement existe
        if 'Type de versement' in df_filtre.columns:
            # Agrégation par type de versement
            repartition_versement = df_filtre.groupby('Type de versement').agg(
                Montant_Total=('Montant du placement', 'sum'),
                Nombre_Souscriptions=('Date de souscription', 'count')
            ).reset_index()
            
            # Trier par montant total décroissant
            repartition_versement = repartition_versement.sort_values('Montant_Total', ascending=False)
            
            # Calculer les pourcentages
            total_montant = repartition_versement['Montant_Total'].sum()
            repartition_versement['Pourcentage'] = (repartition_versement['Montant_Total'] / total_montant * 100).round(1)
            
            # Formatage pour l'affichage
            repartition_versement_display = pd.DataFrame({
                'Type de versement': repartition_versement['Type de versement'],
                'Montant Total': repartition_versement['Montant_Total'].apply(lambda x: f"{x:,.0f}€"),
                'Nombre de Souscriptions': repartition_versement['Nombre_Souscriptions'],
                'Pourcentage': repartition_versement['Pourcentage'].apply(lambda x: f"{x:.1f}%")
            })
            
            # Afficher le tableau récapitulatif
            st.dataframe(repartition_versement_display, use_container_width=True)
            
            # Créer un graphique en camembert pour la répartition par type de versement
            fig_versement = px.pie(
                repartition_versement,
                values='Montant_Total',
                names='Type de versement',
                title="Répartition par Type de Versement",
                hover_data=['Pourcentage'],
                labels={'Pourcentage': 'Pourcentage'},
                hole=0.4
            )
            
            # Ajouter les pourcentages sur le graphique
            fig_versement.update_traces(textinfo='percent+label')
            
            st.plotly_chart(fig_versement, use_container_width=True)
            
            # Analyse croisée par type de versement et étape si les deux colonnes existent
            if 'Étape' in df_filtre.columns:
                st.subheader("📆 Répartition des Types de Versement par Étape")
                
                # Créer un tableau croisé dynamique
                pivot_etape_versement = pd.pivot_table(
                    df_filtre,
                    values='Montant du placement',
                    index=['Type de versement'],
                    columns=['Étape'],
                    aggfunc='sum',
                    fill_value=0
                ).reset_index()
                
                # Formater les valeurs pour l'affichage
                for col in pivot_etape_versement.columns:
                    if col != 'Type de versement':
                        pivot_etape_versement[col] = pivot_etape_versement[col].apply(lambda x: f"{x:,.0f}€")
                
                st.dataframe(pivot_etape_versement, use_container_width=True)
                
                # Créer un graphique à barres groupées pour montrer la répartition
                # Préparer les données pour le graphique
                df_graph = df_filtre.groupby(['Type de versement', 'Étape'])['Montant du placement'].sum().reset_index()
                
                fig_etape_versement = px.bar(
                    df_graph,
                    x='Type de versement',
                    y='Montant du placement',
                    color='Étape',
                    title="Montant par Type de Versement et Étape",
                    barmode='group',
                    text_auto='.0f'
                )
                
                fig_etape_versement.update_traces(texttemplate='%{text}€', textposition='outside')
                fig_etape_versement.update_layout(height=500, yaxis_title="Montant (€)")
                
                st.plotly_chart(fig_etape_versement, use_container_width=True)
        else:
            st.warning("⚠️ La colonne 'Type de versement' n'a pas été trouvée dans les données.")
        
        # Analyse des types d'investissement
        st.subheader("💰 Analyse par Type d'Investissement")
        
        # Vérifier si la colonne Type d'investissement existe
        if 'Type d\'investissement' in df_filtre.columns:
            # Agrégation par type d'investissement
            repartition_invest = df_filtre.groupby('Type d\'investissement').agg(
                Montant_Total=('Montant du placement', 'sum'),
                Nombre_Souscriptions=('Date de souscription', 'count')
            ).reset_index()
            
            # Trier par montant total décroissant
            repartition_invest = repartition_invest.sort_values('Montant_Total', ascending=False)
            
            # Calculer les pourcentages
            total_montant = repartition_invest['Montant_Total'].sum()
            repartition_invest['Pourcentage'] = (repartition_invest['Montant_Total'] / total_montant * 100).round(1)
            
            # Formatage pour l'affichage
            repartition_invest_display = pd.DataFrame({
                'Type d\'investissement': repartition_invest['Type d\'investissement'],
                'Montant Total': repartition_invest['Montant_Total'].apply(lambda x: f"{x:,.0f}€"),
                'Nombre de Souscriptions': repartition_invest['Nombre_Souscriptions'],
                'Pourcentage': repartition_invest['Pourcentage'].apply(lambda x: f"{x:.1f}%")
            })
            
            # Afficher le tableau récapitulatif
            st.dataframe(repartition_invest_display, use_container_width=True)
            
            # Créer un graphique en camembert pour la répartition par type d'investissement
            fig_invest = px.pie(
                repartition_invest,
                values='Montant_Total',
                names='Type d\'investissement',
                title="Répartition par Type d'Investissement",
                hover_data=['Pourcentage'],
                labels={'Pourcentage': 'Pourcentage'},
                hole=0.4
            )
            
            # Ajouter les pourcentages sur le graphique
            fig_invest.update_traces(textinfo='percent+label')
            
            st.plotly_chart(fig_invest, use_container_width=True)
            
            # Analyse croisée par type d'investissement et étape si les deux colonnes existent
            if 'Étape' in df_filtre.columns:
                st.subheader("💳 Répartition des Types d'Investissement par Étape")
                
                # Créer un tableau croisé dynamique
                pivot_etape_invest = pd.pivot_table(
                    df_filtre,
                    values='Montant du placement',
                    index=['Type d\'investissement'],
                    columns=['Étape'],
                    aggfunc='sum',
                    fill_value=0
                ).reset_index()
                
                # Formater les valeurs pour l'affichage
                for col in pivot_etape_invest.columns:
                    if col != 'Type d\'investissement':
                        pivot_etape_invest[col] = pivot_etape_invest[col].apply(lambda x: f"{x:,.0f}€")
                
                st.dataframe(pivot_etape_invest, use_container_width=True)
                
                # Créer un graphique à barres groupées pour montrer la répartition
                # Préparer les données pour le graphique
                df_graph = df_filtre.groupby(['Type d\'investissement', 'Étape'])['Montant du placement'].sum().reset_index()
                
                fig_etape_invest = px.bar(
                    df_graph,
                    x='Type d\'investissement',
                    y='Montant du placement',
                    color='Étape',
                    title="Montant par Type d'Investissement et Étape",
                    barmode='group',
                    text_auto='.0f'
                )
                
                fig_etape_invest.update_traces(texttemplate='%{text}€', textposition='outside')
                fig_etape_invest.update_layout(height=500, yaxis_title="Montant (€)")
                
                st.plotly_chart(fig_etape_invest, use_container_width=True)
                
                # Analyse croisée par type d'investissement et type de versement si les deux colonnes existent
                if 'Type de versement' in df_filtre.columns:
                    st.subheader("💱 Types d'Investissement par Type de Versement")
                    
                    # Créer un tableau croisé dynamique
                    pivot_versement_invest = pd.pivot_table(
                        df_filtre,
                        values='Montant du placement',
                        index=['Type d\'investissement'],
                        columns=['Type de versement'],
                        aggfunc='sum',
                        fill_value=0
                    ).reset_index()
                    
                    # Formater les valeurs pour l'affichage
                    for col in pivot_versement_invest.columns:
                        if col != 'Type d\'investissement':
                            pivot_versement_invest[col] = pivot_versement_invest[col].apply(lambda x: f"{x:,.0f}€")
                    
                    st.dataframe(pivot_versement_invest, use_container_width=True)
                    
                    # Créer un graphique à barres groupées pour montrer la répartition
                    # Préparer les données pour le graphique
                    df_graph = df_filtre.groupby(['Type d\'investissement', 'Type de versement'])['Montant du placement'].sum().reset_index()
                    
                    fig_versement_invest = px.bar(
                        df_graph,
                        x='Type d\'investissement',
                        y='Montant du placement',
                        color='Type de versement',
                        title="Montant par Type d'Investissement et Type de Versement",
                        barmode='group',
                        text_auto='.0f'
                    )
                    
                    fig_versement_invest.update_traces(texttemplate='%{text}€', textposition='outside')
                    fig_versement_invest.update_layout(height=500, yaxis_title="Montant (€)")
                    
                    st.plotly_chart(fig_versement_invest, use_container_width=True)
        else:
            st.warning("⚠️ La colonne 'Type d\'investissement' n'a pas été trouvée dans les données.")
        
        # Tableau récapitulatif
        st.subheader("📋 Récapitulatif Mensuel")
        
        # La section Pipe de Collecte Épargne a été déplacée dans un sous-onglet dédié
        # Voir la fonction analyser_pipe_collecte_epargne()
        
        # Formatage pour l'affichage
        display_df['Montant Total'] = display_df['Montant Total'].apply(lambda x: f"{x:,.0f}€")
        display_df['Écart Objectif'] = display_df['Écart Objectif'].apply(lambda x: f"{x:,.0f}€")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Analyse par étape
        st.subheader("📊 Analyse par Étape")
        
        # Vérifier si la colonne Étape existe
        if 'Étape' in df_filtre.columns:
            # Afficher les étapes disponibles
            etapes_disponibles = sorted(df_filtre['Étape'].unique())
            st.info(f"ℹ️ Étapes disponibles dans les données: {', '.join(etapes_disponibles)}")
            
            # Vérifier la présence d'étapes importantes
            etapes_importantes = ['Mandaté', 'Soumis partenaires', 'Acté', 'Validé']
            etapes_presentes = [etape for etape in etapes_importantes if etape in etapes_disponibles or etape.lower() in [e.lower() for e in etapes_disponibles]]
            etapes_manquantes = [etape for etape in etapes_importantes if etape not in etapes_presentes and etape.lower() not in [e.lower() for e in etapes_disponibles]]
            
            if etapes_presentes:
                st.success(f"✅ Étapes importantes détectées: {', '.join(etapes_presentes)}")
            if etapes_manquantes:
                st.warning(f"⚠️ Certaines étapes importantes n'ont pas été trouvées dans les données: {', '.join(etapes_manquantes)}. Vérifiez l'orthographe ou les variations dans vos données.")
            
            # Ajouter un filtre par étape
            etapes_a_afficher = st.multiselect(
                "Sélectionner les étapes à afficher",
                options=etapes_disponibles,
                default=etapes_disponibles,
                key="etapes_filter_epargne"
            )
            
            if not etapes_a_afficher:
                st.warning("⚠️ Veuillez sélectionner au moins une étape pour l'analyse.")
                return
                
            # Filtrer les données selon les étapes sélectionnées
            df_filtre_etape = df_filtre[df_filtre['Étape'].isin(etapes_a_afficher)].copy()
            
            # Analyse des souscriptions par étape
            analyse_etape = df_filtre_etape.groupby(['Mois', 'Étape']).agg(
                Nb_Souscriptions=('Montant du placement', 'count'),
                Montant_Total=('Montant du placement', 'sum')
            ).reset_index()
            
            # Trier par mois et montant total
            analyse_etape = analyse_etape.sort_values(['Mois', 'Montant_Total'], ascending=[True, False])
            
            # Créer un graphique pour le nombre de souscriptions par étape et par mois
            fig_nb_etape = px.bar(
                analyse_etape,
                x='Mois',
                y='Nb_Souscriptions',
                color='Étape',
                title="Nombre de Souscriptions par Étape et par Mois",
                labels={
                    'Mois': 'Mois',
                    'Nb_Souscriptions': 'Nombre de Souscriptions',
                    'Étape': 'Étape'
                },
                barmode='stack'
            )
            
            # Améliorer l'apparence du graphique
            fig_nb_etape.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_nb_etape, use_container_width=True)
            
            # Créer un graphique pour le montant total par étape et par mois
            fig_montant_etape = px.bar(
                analyse_etape,
                x='Mois',
                y='Montant_Total',
                color='Étape',
                title="Montant Total par Étape et par Mois",
                labels={
                    'Mois': 'Mois',
                    'Montant_Total': 'Montant Total (€)',
                    'Étape': 'Étape'
                },
                barmode='stack',
                text_auto='.2s'
            )
            
            # Formater les étiquettes de texte pour afficher les montants en euros
            fig_montant_etape.update_traces(texttemplate='%{text:,.0f}€', textposition='inside')
            
            # Améliorer l'apparence du graphique
            fig_montant_etape.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_montant_etape, use_container_width=True)
            
            # Tableau récapitulatif par étape
            st.subheader("📋 Tableau Récapitulatif par Étape")
            
            # Créer un tableau récapitulatif par étape
            recap_etape = df_filtre_etape.groupby('Étape').agg(
                Nb_Souscriptions=('Montant du placement', 'count'),
                Montant_Total=('Montant du placement', 'sum'),
                Ticket_Moyen=('Montant du placement', 'mean')
            ).reset_index()
            
            # Trier par montant total
            recap_etape = recap_etape.sort_values('Montant_Total', ascending=False)
            
            # Formater les colonnes numériques
            recap_etape['Montant_Total'] = recap_etape['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
            recap_etape['Ticket_Moyen'] = recap_etape['Ticket_Moyen'].apply(lambda x: f"{x:,.0f}€")
            
            # Renommer les colonnes pour l'affichage
            recap_etape.columns = ['Étape', 'Nombre de Souscriptions', 'Montant Total', 'Ticket Moyen']
            
            st.dataframe(recap_etape, use_container_width=True)
        else:
            st.warning("⚠️ La colonne 'Étape' n'a pas été trouvée dans les données. Impossible de réaliser l'analyse par étape.")
        
        # Analyse par conseiller
        st.subheader("👥 Analyse par Conseiller")
        
        # Calculer les statistiques par conseiller
        analyse_conseiller = df_filtre.groupby('Conseiller').agg(
            Nb_Souscriptions=('Montant du placement', 'count'),
            Montant_Total=('Montant du placement', 'sum'),
            Ticket_Moyen=('Montant du placement', 'mean')
        ).reset_index()
        
        # Trier par montant total
        analyse_conseiller = analyse_conseiller.sort_values('Montant_Total', ascending=False)
        
        # Graphiques par conseiller
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 conseillers par montant
            fig_top_montant = px.bar(
                analyse_conseiller.head(10),
                x='Montant_Total',
                y='Conseiller',
                orientation='h',
                title="🏆 Top 10 Conseillers - Collecte Épargne",
                text='Montant_Total',
                color='Montant_Total',
                color_continuous_scale='Blues'
            )
            fig_top_montant.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
            st.plotly_chart(fig_top_montant, use_container_width=True)
        
        with col2:
            # Top 10 conseillers par ticket moyen
            top_ticket = analyse_conseiller[analyse_conseiller['Nb_Souscriptions'] >= 3].sort_values('Ticket_Moyen', ascending=False).head(10)
            fig_top_ticket = px.bar(
                top_ticket,
                x='Ticket_Moyen',
                y='Conseiller',
                orientation='h',
                title="🎯 Top 10 Conseillers - Ticket Moyen (min 3 souscriptions)",
                text='Ticket_Moyen',
                color='Ticket_Moyen',
                color_continuous_scale='Greens'
            )
            fig_top_ticket.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
            st.plotly_chart(fig_top_ticket, use_container_width=True)
        
        # Tableau détaillé par conseiller
        st.subheader("📊 Tableau Détaillé par Conseiller")
        
        # Formatage pour l'affichage
        analyse_display = analyse_conseiller.copy()
        analyse_display['Montant_Total'] = analyse_display['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
        analyse_display['Ticket_Moyen'] = analyse_display['Ticket_Moyen'].apply(lambda x: f"{x:,.0f}€")
        
        # Renommer les colonnes pour l'affichage
        analyse_display.columns = ['Conseiller', 'Nombre de Souscriptions', 'Montant Total', 'Ticket Moyen']
        
        st.dataframe(analyse_display, use_container_width=True)
        
        # Téléchargement des données
        csv = analyse_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name=f"analyse_souscriptions_epargne_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Créer un DataFrame pour le téléchargement
        df_download = df_filtre.copy()
        
        # Assurer la compatibilité des types avant le téléchargement
        df_download = ensure_pyarrow_compatibility(df_download)
        
        # Ajouter un bouton de téléchargement
        st.download_button(
            label="📥 Télécharger les données filtrées",
            data=df_download.to_csv(index=False).encode('utf-8'),
            file_name='souscriptions_epargne_filtrees.csv',
            mime='text/csv',
        )
        
        # Analyse détaillée par produit d'épargne
        st.subheader("💰 Analyse Détaillée par Produit d'Épargne")
        
        # Créer des onglets pour l'analyse par produit
        tabs_produit = st.tabs(["Vue Globale", "Analyse par Produit", "Performance par Conseiller", "Analyse par Groupe"])
        
        with tabs_produit[0]:
            # Vérifier si la colonne Produit existe
            if 'Produit' in df_filtre.columns:
                # Analyse de la répartition par produit
                repartition_produit = df_filtre.groupby('Produit').agg(
                    Collecte=('Montant du placement', 'sum'),
                    Nombre=('Montant du placement', 'count'),
                    Ticket_Moyen=('Montant du placement', 'mean')
                ).reset_index()
                
                # Ajouter le pourcentage de la collecte totale
                total_collecte = repartition_produit['Collecte'].sum()
                repartition_produit['Pourcentage'] = (repartition_produit['Collecte'] / total_collecte * 100).round(1)
                
                # Trier par collecte décroissante
                repartition_produit = repartition_produit.sort_values('Collecte', ascending=False)
                
                # Afficher les métriques par produit
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique en camembert pour la répartition de la collecte
                    fig_pie = px.pie(
                        repartition_produit,
                        values='Collecte',
                        names='Produit',
                        title="Répartition de la Collecte par Produit",
                        hover_data=['Nombre', 'Ticket_Moyen', 'Pourcentage'],
                        labels={'Collecte': 'Montant collecté (€)', 'Nombre': 'Nombre de souscriptions', 'Ticket_Moyen': 'Ticket moyen (€)', 'Pourcentage': 'Part du total (%)'},
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_pie.update_traces(textinfo='percent+label+value', texttemplate='%{label}<br>%{value:,.0f}€<br>%{percent}')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Graphique en barres pour le nombre de souscriptions par produit
                    fig_bar = px.bar(
                        repartition_produit,
                        x='Produit',
                        y='Nombre',
                        title="Nombre de Souscriptions par Produit",
                        text='Nombre',
                        color='Produit',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_bar.update_traces(textposition='outside')
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Tableau récapitulatif
                repartition_display = repartition_produit.copy()
                repartition_display['Collecte'] = repartition_display['Collecte'].apply(lambda x: f"{x:,.0f}€")
                repartition_display['Ticket_Moyen'] = repartition_display['Ticket_Moyen'].apply(lambda x: f"{x:,.0f}€")
                repartition_display['Pourcentage'] = repartition_display['Pourcentage'].apply(lambda x: f"{x:.1f}%")
                repartition_display.columns = ['Produit', 'Collecte', 'Nombre de Souscriptions', 'Ticket Moyen', 'Part du Total']
                
                st.dataframe(repartition_display, use_container_width=True)
                
                # Export des données
                csv = repartition_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les données par produit (CSV)",
                    data=csv,
                    file_name=f"repartition_produits_epargne_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_produit"
                )
            else:
                st.info("ℹ️ La colonne 'Produit' n'est pas présente dans les données. L'analyse par produit n'est pas disponible.")
                # Afficher une analyse simplifiée
                st.write("#### Résumé de la collecte épargne")
                st.metric("💰 Collecte Totale", f"{df_filtre['Montant du placement'].sum():,.0f}€")
        
        with tabs_produit[1]:
            # Analyse de l'évolution mensuelle par produit
            if 'Produit' in df_filtre.columns and 'Mois' in df_filtre.columns:
                # Créer un sélecteur de produit
                produits_disponibles = sorted(df_filtre['Produit'].unique())
                produit_selectionne = st.selectbox("Sélectionner un produit", options=produits_disponibles, key="select_produit_epargne")
                
                # Filtrer les données pour le produit sélectionné
                df_produit = df_filtre[df_filtre['Produit'] == produit_selectionne].copy()
                
                if not df_produit.empty:
                    # Afficher les métriques clés pour le produit sélectionné
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        collecte_produit = df_produit['Montant du placement'].sum()
                        st.metric(f"Collecte Totale - {produit_selectionne}", f"{collecte_produit:,.0f}€")
                    with col2:
                        nb_souscriptions = len(df_produit)
                        st.metric("Nombre de Souscriptions", f"{nb_souscriptions:,}")
                    with col3:
                        ticket_moyen = df_produit['Montant du placement'].mean()
                        st.metric("Ticket Moyen", f"{ticket_moyen:,.0f}€")
                    
                    # Évolution mensuelle du produit
                    evolution_produit = df_produit.groupby('Mois').agg(
                        Collecte=('Montant du placement', 'sum'),
                        Nombre=('Montant du placement', 'count')
                    ).reset_index()
                    
                    # Trier par mois
                    evolution_produit = evolution_produit.sort_values('Mois')
                    
                    # Graphique d'évolution mensuelle
                    fig_evolution = px.line(
                        evolution_produit,
                        x='Mois',
                        y='Collecte',
                        title=f"Évolution Mensuelle de la Collecte - {produit_selectionne}",
                        markers=True,
                        line_shape='linear'
                    )
                    fig_evolution.update_traces(line=dict(width=3), marker=dict(size=10))
                    fig_evolution.update_layout(yaxis_title="Collecte (€)")
                    st.plotly_chart(fig_evolution, use_container_width=True)
                    
                    # Graphique du nombre de souscriptions
                    fig_nombre = px.bar(
                        evolution_produit,
                        x='Mois',
                        y='Nombre',
                        title=f"Nombre de Souscriptions Mensuelles - {produit_selectionne}",
                        text='Nombre'
                    )
                    fig_nombre.update_traces(textposition='outside')
                    st.plotly_chart(fig_nombre, use_container_width=True)
                    
                    # Répartition par conseiller pour ce produit
                    if 'Conseiller' in df_produit.columns:
                        st.subheader(f"Top Conseillers - {produit_selectionne}")
                        perf_conseillers = df_produit.groupby('Conseiller').agg(
                            Collecte=('Montant du placement', 'sum'),
                            Nombre=('Montant du placement', 'count'),
                            Ticket_Moyen=('Montant du placement', 'mean')
                        ).reset_index()
                        
                        perf_conseillers = perf_conseillers.sort_values('Collecte', ascending=False)
                        
                        fig_conseillers = px.bar(
                            perf_conseillers.head(10),
                            x='Collecte',
                            y='Conseiller',
                            orientation='h',
                            title=f"Top 10 Conseillers - {produit_selectionne}",
                            text='Collecte',
                            color='Collecte',
                            color_continuous_scale='Blues'
                        )
                        fig_conseillers.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
                        st.plotly_chart(fig_conseillers, use_container_width=True)
                else:
                    st.warning(f"⚠️ Aucune donnée trouvée pour le produit {produit_selectionne}.")
            else:
                if 'Produit' not in df_filtre.columns:
                    st.info("ℹ️ La colonne 'Produit' n'est pas présente dans les données. L'analyse par produit n'est pas disponible.")
                elif 'Mois' not in df_filtre.columns:
                    st.error("❌ Colonne 'Mois' non trouvée dans les données.")
        
        with tabs_produit[2]:
            # Analyse de la performance par conseiller avec les fonctions adaptées
            analyse_collecte_produit_conseiller_fallback(df_filtre, "Épargne")
            analyse_performance_conseiller_fallback(df_filtre, 'Montant du placement', 'Conseiller', 'Performance Épargne')
        
        with tabs_produit[3]:
            # Analyse par groupe de conseiller
            st.subheader("👥 Analyse par Groupe de Conseillers")
            
            # Définir les groupes de conseillers
            idr_conseillers = ['Ikramah BADATE', 'Yassila LAMBATE']
            internes_conseillers = ['Aicha NAILI', 'Abdelkarim BOUTERA', 'Yanis Sebiane', 'Nissrine BEJAOUI']
            
            # Créer une colonne pour le groupe
            df_groupe = df_filtre.copy()
            
            # Attribuer les groupes
            def attribuer_groupe(conseiller):
                if conseiller in idr_conseillers:
                    return 'IDR'
                elif conseiller in internes_conseillers:
                    return 'Internes'
                else:
                    return 'Mandataires'
            
            df_groupe['Groupe'] = df_groupe['Conseiller'].apply(attribuer_groupe)
            
            # Calculer les statistiques par groupe
            stats_groupe = df_groupe.groupby('Groupe').agg(
                Collecte=('Montant du placement', 'sum'),
                Nombre=('Montant du placement', 'count'),
                Ticket_Moyen=('Montant du placement', 'mean'),
                Conseillers=('Conseiller', 'nunique')
            ).reset_index()
            
            # Calculer le pourcentage de la collecte totale
            total_collecte_groupe = stats_groupe['Collecte'].sum()
            stats_groupe['Pourcentage'] = (stats_groupe['Collecte'] / total_collecte_groupe * 100).round(1)
            
            # Afficher les métriques par groupe
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique en camembert pour la répartition de la collecte par groupe
                fig_pie_groupe = px.pie(
                    stats_groupe,
                    values='Collecte',
                    names='Groupe',
                    title="Répartition de la Collecte par Groupe",
                    hover_data=['Nombre', 'Ticket_Moyen', 'Pourcentage', 'Conseillers'],
                    labels={
                        'Collecte': 'Montant collecté (€)', 
                        'Nombre': 'Nombre de souscriptions', 
                        'Ticket_Moyen': 'Ticket moyen (€)', 
                        'Pourcentage': 'Part du total (%)',
                        'Conseillers': 'Nombre de conseillers'
                    },
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_pie_groupe.update_traces(textinfo='percent+label+value', texttemplate='%{label}<br>%{value:,.0f}€<br>%{percent}')
                st.plotly_chart(fig_pie_groupe, use_container_width=True)
            
            with col2:
                # Graphique en barres pour le nombre de souscriptions par groupe
                fig_bar_groupe = px.bar(
                    stats_groupe,
                    x='Groupe',
                    y='Nombre',
                    title="Nombre de Souscriptions par Groupe",
                    text='Nombre',
                    color='Groupe',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_bar_groupe.update_traces(textposition='outside')
                st.plotly_chart(fig_bar_groupe, use_container_width=True)
            
            # Tableau récapitulatif par groupe
            st.subheader("📋 Tableau Récapitulatif par Groupe")
            
            # Formatage pour l'affichage
            stats_groupe_display = stats_groupe.copy()
            stats_groupe_display['Collecte'] = stats_groupe_display['Collecte'].apply(lambda x: f"{x:,.0f}€")
            stats_groupe_display['Ticket_Moyen'] = stats_groupe_display['Ticket_Moyen'].apply(lambda x: f"{x:,.0f}€")
            stats_groupe_display['Pourcentage'] = stats_groupe_display['Pourcentage'].apply(lambda x: f"{x:.1f}%")
            stats_groupe_display.columns = ['Groupe', 'Collecte', 'Nombre de Souscriptions', 'Ticket Moyen', 'Nombre de Conseillers', 'Part du Total']
            
            st.dataframe(stats_groupe_display, use_container_width=True)
            
            # Détail par conseiller dans chaque groupe
            st.subheader("🔍 Détail par Conseiller dans chaque Groupe")
            
            # Créer un sélecteur de groupe
            groupes_disponibles = sorted(df_groupe['Groupe'].unique())
            groupe_selectionne = st.selectbox("Sélectionner un groupe", options=groupes_disponibles, key="select_groupe_epargne")
            
            # Filtrer les données pour le groupe sélectionné
            df_groupe_filtre = df_groupe[df_groupe['Groupe'] == groupe_selectionne].copy()
            
            # Calculer les statistiques par conseiller dans le groupe sélectionné
            stats_conseiller_groupe = df_groupe_filtre.groupby('Conseiller').agg(
                Collecte=('Montant du placement', 'sum'),
                Nombre=('Montant du placement', 'count'),
                Ticket_Moyen=('Montant du placement', 'mean')
            ).reset_index()
            
            # Trier par collecte décroissante
            stats_conseiller_groupe = stats_conseiller_groupe.sort_values('Collecte', ascending=False)
            
            # Graphique des conseillers du groupe
            fig_conseillers_groupe = px.bar(
                stats_conseiller_groupe,
                x='Collecte',
                y='Conseiller',
                orientation='h',
                title=f"Conseillers du groupe {groupe_selectionne}",
                text='Collecte',
                color='Collecte',
                color_continuous_scale='Viridis'
            )
            fig_conseillers_groupe.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
            st.plotly_chart(fig_conseillers_groupe, use_container_width=True)
            
            # Tableau détaillé par conseiller du groupe
            stats_conseiller_display = stats_conseiller_groupe.copy()
            stats_conseiller_display['Collecte'] = stats_conseiller_display['Collecte'].apply(lambda x: f"{x:,.0f}€")
            stats_conseiller_display['Ticket_Moyen'] = stats_conseiller_display['Ticket_Moyen'].apply(lambda x: f"{x:,.0f}€")
            stats_conseiller_display.columns = ['Conseiller', 'Collecte', 'Nombre de Souscriptions', 'Ticket Moyen']
            
            st.dataframe(stats_conseiller_display, use_container_width=True)
            
            # Téléchargement des données par groupe
            csv_groupe = stats_groupe_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📅 Télécharger les données par groupe (CSV)",
                data=csv_groupe,
                file_name=f"analyse_groupes_epargne_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_groupe"
            )


def analyser_pipe_collecte_epargne(df_original):
    """
    Analyse dédiée du Pipe de Collecte Épargne.
    
    Cette fonction analyse spécifiquement les souscriptions en cours de traitement
    (hors étapes finalisées comme "Acté" et "Validé").
    
    Args:
        df_original: DataFrame contenant les données brutes
    """
    st.header("🔄 Pipe de Collecte Épargne")
    
    # Vérification des données
    if df_original is None or df_original.empty:
        st.error("❌ Aucune donnée disponible pour l'analyse du pipe de collecte.")
        return
    
    # Appliquer extract_conseiller pour standardiser la colonne Conseiller
    df_with_conseiller = extract_conseiller(df_original)
    
    # Vérifier si la colonne Conseiller a été correctement extraite
    if 'Conseiller' not in df_with_conseiller.columns:
        st.error("❌ Impossible d'extraire la colonne 'Conseiller'.")
        return
    
    # Utiliser le DataFrame avec la colonne Conseiller standardisée
    df_epargne_valid = df_with_conseiller.copy()
    
    # Vérifier la présence des colonnes essentielles
    colonnes_essentielles = ['Date de souscription', 'Montant']
    colonnes_manquantes = [col for col in colonnes_essentielles if col not in df_epargne_valid.columns]
    
    if colonnes_manquantes:
        st.error(f"❌ Colonnes essentielles manquantes: {colonnes_manquantes}")
        return
    
    # Traitement des données de base
    # Convertir les dates
    df_epargne_valid['Date de souscription'] = safe_to_datetime(df_epargne_valid['Date de souscription'])
    
    # Créer la colonne Montant du placement
    if 'Montant des frais' in df_epargne_valid.columns:
        df_epargne_valid['Montant des frais'] = safe_to_numeric(df_epargne_valid['Montant des frais'])
        df_epargne_valid['Montant du placement'] = safe_to_numeric(df_epargne_valid['Montant']) - df_epargne_valid['Montant des frais']
    else:
        df_epargne_valid['Montant du placement'] = safe_to_numeric(df_epargne_valid['Montant'])
    
    # Filtrer les données valides
    df_epargne_valid = df_epargne_valid.dropna(subset=['Date de souscription', 'Montant du placement'])
    
    if df_epargne_valid.empty:
        st.error("❌ Aucune donnée valide après traitement.")
        return
    
    # Créer la colonne Mois pour l'agrégation
    df_epargne_valid['Mois'] = df_epargne_valid['Date de souscription'].dt.to_period('M').astype(str)
    
    # Ajouter les colonnes de premier et dernier jour du mois
    df_epargne_valid = adjust_dates_to_month_range(df_epargne_valid, 'Date de souscription')
    
    # Ajout du bloc pour l'évolution du Pipe Épargne en cours
    st.subheader("📊 Évolution du Pipe Épargne en cours")
    
    # Vérifier si la colonne Étape existe
    if 'Étape' in df_epargne_valid.columns:
        # Filtrer pour exclure les étapes "Acté" et "Validé" (insensible à la casse)
        etapes_finalisees = ['acté', 'validé', 'acte', 'valide']
        mask_etapes_en_cours = ~df_epargne_valid['Étape'].str.lower().isin(etapes_finalisees)
        df_pipe_en_cours = df_epargne_valid[mask_etapes_en_cours].copy()
        
        if df_pipe_en_cours.empty:
            st.warning("⚠️ Aucune souscription en cours dans le pipe (toutes les souscriptions sont déjà à l'étape Acté ou Validé).")
        else:
            # Informer l'utilisateur du nombre de souscriptions en cours
            nb_total = len(df_epargne_valid)
            nb_en_cours = len(df_pipe_en_cours)
            st.info(f"ℹ️ {nb_en_cours} souscriptions sur {nb_total} sont en cours de traitement (hors étapes Acté/Validé).")
            
            # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
            df_pipe_en_cours = adjust_dates_to_month_range(df_pipe_en_cours, 'Date de souscription')
            
            # Créer une agrégation mensuelle avec les dates de début et fin
            pipe_mensuel = df_pipe_en_cours.groupby('Mois').agg(
                Montant_Total=('Montant du placement', 'sum'),
                Premier_Jour=('Premier_Jour_Mois', 'first'),
                Dernier_Jour=('Dernier_Jour_Mois', 'first'),
                Nombre_Souscriptions=('Date de souscription', 'count')
            ).reset_index()
            
            # Trier par date de début
            pipe_mensuel = pipe_mensuel.sort_values('Premier_Jour')
            
            # S'assurer que les dates sont valides (pas de NaT)
            pipe_mensuel['Premier_Jour'] = pd.to_datetime(pipe_mensuel['Premier_Jour'], errors='coerce')
            pipe_mensuel['Dernier_Jour'] = pd.to_datetime(pipe_mensuel['Dernier_Jour'], errors='coerce')
            
            # Remplacer les dates NaT par les premiers et derniers jours du mois basés sur la colonne 'Mois'
            for idx, row in pipe_mensuel.iterrows():
                if pd.isna(row['Premier_Jour']) or pd.isna(row['Dernier_Jour']):
                    try:
                        mois_annee = row['Mois']
                        annee, mois = mois_annee.split('-')
                        premier_jour = pd.Timestamp(year=int(annee), month=int(mois), day=1)
                        dernier_jour = pd.Timestamp(year=int(annee), month=int(mois), day=pd.Timestamp(premier_jour).days_in_month)
                        
                        if pd.isna(row['Premier_Jour']):
                            pipe_mensuel.at[idx, 'Premier_Jour'] = premier_jour
                        if pd.isna(row['Dernier_Jour']):
                            pipe_mensuel.at[idx, 'Dernier_Jour'] = dernier_jour
                    except Exception:
                        # En cas d'erreur, laisser les valeurs NaT
                        pass
            
            # Créer des étiquettes personnalisées pour l'axe X avec les plages de dates
            pipe_mensuel['Période'] = pipe_mensuel.apply(
                lambda row: f"{row['Mois']} ({row['Premier_Jour'].strftime('%d/%m') if pd.notna(row['Premier_Jour']) else '01/01'} - {row['Dernier_Jour'].strftime('%d/%m') if pd.notna(row['Dernier_Jour']) else '31/12'})", 
                axis=1
            )
            
            # Créer un DataFrame pour l'affichage et l'export
            pipe_display_df = pd.DataFrame({
                'Mois': pipe_mensuel['Mois'],
                'Période': pipe_mensuel['Période'],
                'Montant Total': pipe_mensuel['Montant_Total'],
                'Nombre de Souscriptions': pipe_mensuel['Nombre_Souscriptions']
            })
            
            # Formatage pour l'affichage
            pipe_display_df['Montant Total'] = pipe_display_df['Montant Total'].apply(lambda x: f"{x:,.0f}€")
            
            # Afficher le tableau récapitulatif
            st.dataframe(pipe_display_df, use_container_width=True)
            
            # Espacement
            st.write("")
            
            # Créer le graphique avec les périodes complètes
            fig_pipe = px.bar(
                pipe_mensuel,
                x='Période',
                y='Montant_Total',
                title=f"📊 Évolution Mensuelle du Pipe Épargne en cours",
                text='Montant_Total',
                color_discrete_sequence=['#FFA500']  # Orange pour différencier du graphique précédent
            )
            fig_pipe.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
            
            # Ajouter une annotation pour le nombre de souscriptions
            for i, row in pipe_mensuel.iterrows():
                fig_pipe.add_annotation(
                    x=row['Période'],
                    y=row['Montant_Total'],
                    text=f"{row['Nombre_Souscriptions']} souscr.",
                    showarrow=False,
                    yshift=-30,
                    font=dict(size=10)
                )
            
            st.plotly_chart(fig_pipe, use_container_width=True)
            
            # Espacement entre les sections
            st.markdown("---")
            st.write("")
            
            # Analyse par étape pour le pipe en cours
            if 'Étape' in df_pipe_en_cours.columns:
                st.subheader("📊 Répartition du Pipe par Étape")
                
                # Agrégation par étape
                repartition_etape = df_pipe_en_cours.groupby('Étape').agg(
                    Montant_Total=('Montant du placement', 'sum'),
                    Nombre_Souscriptions=('Date de souscription', 'count')
                ).reset_index()
                
                # Trier par montant total décroissant
                repartition_etape = repartition_etape.sort_values('Montant_Total', ascending=False)
                
                # Formatage pour l'affichage
                repartition_etape['Montant Total'] = repartition_etape['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
                
                # Renommer les colonnes pour l'affichage
                repartition_etape_display = pd.DataFrame({
                    'Étape': repartition_etape['Étape'],
                    'Montant Total': repartition_etape['Montant Total'],
                    'Nombre de Souscriptions': repartition_etape['Nombre_Souscriptions']
                })
                
                # Afficher le tableau récapitulatif
                st.dataframe(repartition_etape_display, use_container_width=True)
                
                # Espacement
                st.write("")
                
                # Créer un graphique en camembert pour la répartition par étape
                fig_etape = px.pie(
                    repartition_etape,
                    values='Montant_Total',
                    names='Étape',
                    title="Répartition du Pipe Épargne par Étape",
                    hole=0.4,
                    height=500
                )
                st.plotly_chart(fig_etape, use_container_width=True)
                
                # Export des données du pipe
                st.subheader("📤 Export des Données du Pipe")
                
                # Préparer les données pour l'export
                export_data = df_pipe_en_cours.copy()
                
                # Créer le bouton de téléchargement
                create_download_button(export_data, "pipe_epargne", "pipe_epargne_export")
    else:
        st.warning("⚠️ La colonne 'Étape' n'a pas été trouvée dans les données. Impossible d'analyser le pipe en cours.")
    
    # Section Pipe de Collecte par Mois et par Conseiller
    st.subheader("📊 Pipe de Collecte par Mois et par Conseiller")
    
    if not df_original.empty:
        # Utiliser les mêmes colonnes que dans l'analyse principale
        montant_col = None
        for col in df_original.columns:
            if col.lower() in ['montant', 'amount', 'valeur', 'value', 'prix', 'price']:
                montant_col = col
                break
        
        date_col = None
        for col in df_original.columns:
            if col.lower() in ['date', 'date de souscription', 'date_souscription', 'created_at', 'timestamp']:
                date_col = col
                break
        
        # Extraire le conseiller
        df_pipe_conseiller = df_original.copy()
        df_pipe_conseiller = extract_conseiller(df_pipe_conseiller)
        conseiller_col_name = 'Conseiller'
        
        # Identifier la colonne de statut ou d'étape
        statut_col = None
        for col in df_pipe_conseiller.columns:
            if col.lower() in ['statut', 'status', 'état', 'etat', 'state']:
                statut_col = col
                break
        
        etape_col = None
        for col in df_pipe_conseiller.columns:
            if col.lower() in ['étape', 'etape', 'step', 'phase', 'stage']:
                etape_col = col
                break
        
        # Utiliser la colonne étape si disponible, sinon utiliser la colonne statut
        filtre_col = etape_col if etape_col else statut_col
        
        if filtre_col and montant_col and date_col:
            # Convertir les colonnes nécessaires
            df_pipe_conseiller[date_col] = safe_to_datetime(df_pipe_conseiller[date_col])
            df_pipe_conseiller[montant_col] = safe_to_numeric(df_pipe_conseiller[montant_col])
            
            # Créer la colonne Mois
            df_pipe_conseiller['Mois'] = df_pipe_conseiller[date_col].dt.to_period('M').astype(str)
            
            # Filtrer pour n'inclure que les souscriptions en cours (pipe)
            # Exclure les étapes/statuts finalisés (acté, validé, clôturé) et les statuts annulés
            statuts_finalises = ['acté', 'validé', 'cloturé', 'clôturé', 'acte', 'valide', 'cloture']
            
            # Identifier les statuts annulés
            statuts_annules = [s.lower() for s in df_pipe_conseiller[filtre_col].unique() if 'annul' in str(s).lower()]
            
            # Combiner les statuts à exclure
            statuts_a_exclure = [s.lower() for s in statuts_finalises] + statuts_annules
            
            # Filtrer le dataframe pour exclure les statuts finalisés et annulés
            df_pipe = df_pipe_conseiller[~df_pipe_conseiller[filtre_col].str.lower().isin(statuts_a_exclure)].copy()
            
            if not df_pipe.empty:
                # Créer des options de période pour l'analyse
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sélection de la période d'analyse
                    periode_options = ['Mois', 'Semaine', 'Trimestre']
                    periode_selectionnee = st.selectbox("Période d'analyse", periode_options, key="pipe_periode_analyse_conseiller")
                
                with col2:
                    # Sélection du conseiller pour le filtrage
                    if conseiller_col_name in df_pipe.columns:
                        conseillers = ['Tous'] + sorted([str(x) for x in df_pipe[conseiller_col_name].unique().tolist() if str(x) != 'nan'])
                        conseiller_filtre = st.selectbox("Filtrer par conseiller", conseillers, key="pipe_conseiller_analyse_conseiller")
                    else:
                        conseiller_filtre = 'Tous'
                
                # Filtrer par conseiller si nécessaire
                if conseiller_filtre != 'Tous' and conseiller_col_name in df_pipe.columns:
                    df_pipe = df_pipe[df_pipe[conseiller_col_name] == conseiller_filtre]
                
                # Déterminer la colonne de période à utiliser
                if periode_selectionnee == 'Semaine':
                    df_pipe['Semaine'] = df_pipe[date_col].dt.strftime('%Y-%U')
                    colonne_periode = 'Semaine'
                elif periode_selectionnee == 'Trimestre':
                    df_pipe['Trimestre'] = df_pipe[date_col].dt.to_period('Q').astype(str)
                    colonne_periode = 'Trimestre'
                else:
                    colonne_periode = 'Mois'
                
                # Vérifier que la colonne de période existe
                if colonne_periode in df_pipe.columns:
                    # Agréger les données par période et par conseiller
                    if conseiller_col_name in df_pipe.columns:
                        # Grouper par période et conseiller
                        pipe_periode_conseiller = df_pipe.groupby([colonne_periode, conseiller_col_name]).agg(
                            Montant_Total=(montant_col, 'sum'),
                            Nombre=(montant_col, 'count')
                        ).reset_index()
                        
                        # Trier par période
                        pipe_periode_conseiller = pipe_periode_conseiller.sort_values(colonne_periode)
                        
                        # Créer le graphique
                        fig_pipe_periode = px.bar(
                            pipe_periode_conseiller,
                            x=colonne_periode,
                            y='Montant_Total',
                            color=conseiller_col_name,
                            title=f"💰 Pipe de Collecte par {periode_selectionnee} et par Conseiller",
                            text='Montant_Total',
                            barmode='group',
                            labels={
                                colonne_periode: periode_selectionnee,
                                'Montant_Total': 'Montant Total (€)',
                                conseiller_col_name: 'Conseiller'
                            }
                        )
                        
                        # Mise en forme du graphique
                        fig_pipe_periode.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
                        fig_pipe_periode.update_layout(xaxis_tickangle=-45)
                        
                        # Afficher le graphique
                        st.plotly_chart(fig_pipe_periode, use_container_width=True)
                        
                        # Tableau récapitulatif du pipe par conseiller
                        st.write("### Détails du Pipe par Conseiller")
                        
                        # Agréger par conseiller pour le tableau récapitulatif
                        pipe_conseiller = df_pipe.groupby(conseiller_col_name).agg(
                            Montant_Total=(montant_col, 'sum'),
                            Nombre_Souscriptions=(montant_col, 'count'),
                            Montant_Moyen=(montant_col, 'mean')
                        ).reset_index().sort_values('Montant_Total', ascending=False)
                        
                        # Formater pour l'affichage
                        pipe_conseiller_display = pipe_conseiller.copy()
                        pipe_conseiller_display['Montant_Total'] = pipe_conseiller_display['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
                        pipe_conseiller_display['Montant_Moyen'] = pipe_conseiller_display['Montant_Moyen'].apply(lambda x: f"{x:,.0f}€")
                        
                        # Renommer les colonnes
                        pipe_conseiller_display = pipe_conseiller_display.rename(columns={
                            conseiller_col_name: 'Conseiller',
                            'Montant_Total': 'Montant Total',
                            'Nombre_Souscriptions': 'Nombre de Souscriptions',
                            'Montant_Moyen': 'Montant Moyen'
                        })
                        
                        st.dataframe(pipe_conseiller_display, use_container_width=True)
                        
                        # Analyse par étape/statut
                        if filtre_col:
                            st.write(f"### Répartition du Pipe par {filtre_col}")
                            
                            # Agréger par étape/statut
                            pipe_etape = df_pipe.groupby(filtre_col).agg(
                                Montant_Total=(montant_col, 'sum'),
                                Nombre=(montant_col, 'count')
                            ).reset_index().sort_values('Montant_Total', ascending=False)
                            
                            # Créer le graphique
                            fig_pipe_etape = px.pie(
                                pipe_etape,
                                values='Montant_Total',
                                names=filtre_col,
                                title=f"Répartition du Pipe par {filtre_col}",
                                hole=0.4
                            )
                            
                            # Mise en forme du graphique
                            fig_pipe_etape.update_traces(textinfo='percent+label')
                            
                            # Afficher le graphique
                            st.plotly_chart(fig_pipe_etape, use_container_width=True)
                            
                            # Tableau récapitulatif par étape/statut
                            pipe_etape_display = pipe_etape.copy()
                            pipe_etape_display['Montant_Total'] = pipe_etape_display['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
                            pipe_etape_display = pipe_etape_display.rename(columns={
                                filtre_col: filtre_col,
                                'Montant_Total': 'Montant Total',
                                'Nombre': 'Nombre de Souscriptions'
                            })
                            
                            st.dataframe(pipe_etape_display, use_container_width=True)
                    else:
                        st.warning("⚠️ Colonne conseiller non détectée pour l'analyse du pipe.")
                else:
                    st.warning(f"⚠️ Colonne {colonne_periode} non disponible pour l'analyse du pipe.")
            else:
                st.info("ℹ️ Aucune souscription en cours (pipe) trouvée.")
        else:
            st.warning("⚠️ Colonnes nécessaires manquantes pour l'analyse du pipe par conseiller.")
    else:
        st.warning("⚠️ Aucune donnée disponible pour l'analyse du pipe par conseiller.")


if __name__ == "__main__":
    st.set_page_config(page_title="Analyse Souscriptions Épargne", page_icon="👳", layout="wide")
    st.title("💳 Analyse des Souscriptions Épargne")
    
    uploaded_file = st.file_uploader("📁 Charger un fichier de souscriptions épargne", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        analyser_souscriptions_epargne(df)
