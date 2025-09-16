"""
Analyse intégrée des clients basée sur les données Excel spécifiques.
Module dédié aux colonnes : Nom & Prénom, Email, Date de l'entretien, Nb Souscriptions, 
Détails des Souscriptions, VR, Métier, Secteur d'activité, Revenus, TMI, 
Profil épargnant, Épargne disponible, Situation familiale, Nb d'enfants, 
Date d'inscription, Conseiller.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import calendar

from utils.data_processing_debug import safe_to_datetime_debug as safe_to_datetime, safe_to_numeric_debug as safe_to_numeric

@st.cache_data(ttl=600, show_spinner=False)  # Cache for 10 minutes
def _analyser_clients_integration_cached(df_dict):
    """Cached version of client analysis that works with serializable data"""
    # Convert dict back to DataFrame
    df = pd.DataFrame(df_dict)
    return _analyser_clients_integration_impl(df)

def analyser_clients_integration(df):
    """
    Analyse intégrée des clients avec toutes les fonctionnalités spécialisées.
    
    Args:
        df (DataFrame): DataFrame avec les colonnes spécifiques du fichier Excel
    """
    # Convert DataFrame to dict for caching compatibility
    return _analyser_clients_integration_cached(df.to_dict('records'))

def _analyser_clients_integration_impl(df):
    """Implementation of integrated client analysis"""
    st.header("🎯 Analyse Intégrée des Clients")
    
    if df is None or df.empty:
        st.error("❌ Veuillez charger un fichier de données clients.")
        return
    
    # Préparation des données
    df_clean = preparer_donnees_clients(df.to_dict('records'))
    
    # Séparer les clients des leads
    if 'nb_souscriptions_clean' in df_clean.columns:
        # Vrais clients (au moins 1 souscription)
        clients_mask = (df_clean['nb_souscriptions_clean'] > 0) & (df_clean['nb_souscriptions_clean'].notna())
        df_clients = df_clean[clients_mask].copy()
        
        # Leads (0 souscription ou NaN)
        leads_mask = (df_clean['nb_souscriptions_clean'] == 0) | (df_clean['nb_souscriptions_clean'].isna())
        df_leads = df_clean[leads_mask].copy()
        
        # Statistiques
        nb_clients = len(df_clients)
        nb_leads = len(df_leads)
        total_personnes = len(df_clean)
    else:
        # Si pas de colonne souscriptions, traiter tout comme clients
        df_clients = df_clean.copy()
        df_leads = pd.DataFrame()
        nb_clients = len(df_clients)
        nb_leads = 0
        total_personnes = len(df_clean)
    
    # Information sur les données
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total personnes", f"{total_personnes:,}")
    with col2:
        st.metric("✅ Vrais clients", f"{nb_clients:,}", help="Personnes avec au moins 1 souscription")
    with col3:
        st.metric("🎯 Leads", f"{nb_leads:,}", help="Personnes sans souscription")
    
    if nb_clients == 0:
        st.error("❌ Aucun vrai client trouvé (personnes avec souscriptions > 0)")
        return
    
    st.info(f"📊 Analyse focalisée sur {nb_clients:,} vrais clients avec {len(df_clients.columns)} colonnes")
    
    # Aperçu des données
    with st.expander("🔍 Aperçu des données", expanded=False):
        st.subheader("Colonnes disponibles")
        cols = st.columns(3)
        for i, col in enumerate(df_clients.columns):
            cols[i % 3].write(f"• {col}")
        
        st.subheader("Échantillon des vrais clients")
        st.dataframe(df_clients.head(10))
        
        if not df_leads.empty:
            st.subheader("Échantillon des leads (pour référence)")
            st.dataframe(df_leads.head(5))
        
        st.subheader("Statistiques des vrais clients")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total clients", len(df_clients))
        with col2:
            emails_uniques = df_clients['Email'].nunique() if 'Email' in df_clients.columns else 0
            st.metric("Emails uniques", emails_uniques)
        with col3:
            conseillers_uniques = df_clients['Conseiller'].nunique() if 'Conseiller' in df_clients.columns else 0
            st.metric("Conseillers", conseillers_uniques)
        with col4:
            secteurs_uniques = df_clients['Secteur d\'activité'].nunique() if 'Secteur d\'activité' in df_clients.columns else 0
            st.metric("Secteurs", secteurs_uniques)
    
    # Option pour inclure les leads dans l'analyse
    with st.expander("⚙️ Options d'analyse", expanded=False):
        inclure_leads = st.checkbox("Inclure les leads dans l'analyse", value=False, 
                                   help="Cocher pour analyser aussi les personnes sans souscription")
        
        if inclure_leads and not df_leads.empty:
            st.info("🎯 Mode mixte : Analyse des clients + leads")
            df_analyse = df_clean.copy()
        else:
            st.info("✅ Mode client : Analyse des vrais clients uniquement")
            df_analyse = df_clients.copy()
    
    # Filtres principaux
    df_filtre = appliquer_filtres_clients(df_analyse)
    
    # Dashboard principal
    analyser_dashboard_principal(df_filtre)
    
    # Analyses spécialisées dans des onglets
    analyses_tabs = st.tabs([
        "📊 Vue d'ensemble", 
        "👨‍💼 Par Conseiller", 
        "🏢 Par Secteur",
        "💰 Analyse Financière", 
        "👨‍👩‍👧‍👦 Profil Familial",
        "📈 Évolution Temporelle",
        "🎯 Segmentation Avancée",
        "📤 Export"
    ])
    
    with analyses_tabs[0]:
        analyser_vue_ensemble(df_filtre)
    
    with analyses_tabs[1]:
        analyser_par_conseiller(df_filtre)
    
    with analyses_tabs[2]:
        analyser_par_secteur(df_filtre)
    
    with analyses_tabs[3]:
        analyser_financier(df_filtre)
    
    with analyses_tabs[4]:
        analyser_profil_familial(df_filtre)
    
    with analyses_tabs[5]:
        analyser_evolution_temporelle(df_filtre)
    
    with analyses_tabs[6]:
        analyser_segmentation_avancee(df_filtre)
    
    with analyses_tabs[7]:
        generer_exports(df_filtre)

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def preparer_donnees_clients(df_dict):
    """Cached version of data preparation"""
    df = pd.DataFrame(df_dict)
    return _preparer_donnees_clients_impl(df)

def _preparer_donnees_clients_impl(df):
    """Prépare et nettoie les données clients en regroupant par adresse email."""
    from utils.data_processing_debug import capture_info, capture_success, capture_warning, capture_error
    
    df_clean = df.copy()
    
    # Regroupement par email si la colonne existe
    if 'Email' in df_clean.columns:
        # Nettoyer les emails
        df_clean['Email'] = df_clean['Email'].astype(str).str.strip().str.lower()
        df_clean['Email'] = df_clean['Email'].replace(['nan', 'none', ''], np.nan)
        
        # Compter les doublons avant regroupement
        emails_avant = len(df_clean)
        emails_uniques = df_clean['Email'].nunique()
        doublons = emails_avant - emails_uniques
        
        if doublons > 0:
            capture_info(f"Détection de {doublons} clients avec des emails en doublon")
            capture_info(f"Regroupement de {emails_avant} lignes en {emails_uniques} clients uniques")
            
            # Définir les stratégies d'agrégation par colonne
            agg_dict = {}
            
            # Fonctions d'agrégation améliorées
            def first_non_null(x):
                return x.dropna().iloc[0] if len(x.dropna()) > 0 else np.nan
            
            def safe_mean_non_null(x):
                # Nettoyer et convertir en numérique avant de calculer la moyenne
                x_clean = x.dropna()
                if len(x_clean) == 0:
                    return np.nan
                
                # Convertir en numérique en forçant les erreurs à NaN
                x_numeric = pd.to_numeric(x_clean, errors='coerce')
                x_numeric = x_numeric.dropna()
                
                return x_numeric.mean() if len(x_numeric) > 0 else np.nan
                
            def max_non_null(x):
                return x.dropna().max() if len(x.dropna()) > 0 else np.nan
            
            def agg_vr(x):
                x_clean = x.astype(str).str.upper().str.strip()
                has_oui = x_clean.isin(['OUI', 'YES', 'Y', 'O', '1']).any()
                return 'OUI' if has_oui else 'NON'
            
            # Fonctions spéciales pour situation familiale
            def agg_situation_familiale(x):
                # Nettoyer et normaliser avant de prendre la première valeur
                x_clean = x.dropna().astype(str).str.strip()
                if len(x_clean) == 0:
                    return np.nan
                
                # Normaliser les variations
                x_clean = x_clean.str.replace(r'PACS\(e\)|Pacsé\(e\)', 'Pacsé(e)', regex=True, case=False)
                x_clean = x_clean.str.replace(r'Marié\(e\)|Mariée?', 'Marié(e)', regex=True, case=False)
                x_clean = x_clean.str.replace(r'Divorcé\(e\)|Divorcée?', 'Divorcé(e)', regex=True, case=False)
                x_clean = x_clean.str.replace(r'Célibataire|Celibataire', 'Célibataire', regex=True, case=False)
                x_clean = x_clean.str.replace(r'Veuf/Veuve|Veuf\(ve\)|Veuve?|Veuf', 'Veuf(ve)', regex=True, case=False)
                
                return x_clean.iloc[0]
            
            # Colonnes texte : prendre la première valeur non nulle
            colonnes_texte = ['Nom & Prénom', 'Métier', 'Secteur d\'activité', 
                             'Profil épargnant', 'Conseiller']
            for col in colonnes_texte:
                if col in df_clean.columns:
                    agg_dict[col] = first_non_null
            
            # Traitement spécial pour la situation familiale
            if 'Situation familiale' in df_clean.columns:
                agg_dict['Situation familiale'] = agg_situation_familiale
            
            # Colonnes numériques : sommer les souscriptions, moyenner le reste
            if 'Nb Souscriptions' in df_clean.columns:
                agg_dict['Nb Souscriptions'] = 'sum'  # Additionner les souscriptions
            
            colonnes_moyenne = ['Revenus', 'TMI', 'Épargne disponible', 'Nb d\'enfants']
            for col in colonnes_moyenne:
                if col in df_clean.columns:
                    agg_dict[col] = safe_mean_non_null
            
            # Colonne VR : sommer les montants de versements réguliers
            if 'VR' in df_clean.columns:
                def safe_sum_vr(x):
                    # Convertir en numérique et sommer
                    x_numeric = pd.to_numeric(x, errors='coerce')
                    return x_numeric.sum() if not x_numeric.isna().all() else 0
                agg_dict['VR'] = safe_sum_vr
            
            # Colonnes dates : prendre la plus récente
            colonnes_dates = ['Date de l\'entretien', 'Date d\'inscription']
            for col in colonnes_dates:
                if col in df_clean.columns:
                    agg_dict[col] = max_non_null
            
            # Colonne Détails des Souscriptions : concaténer les informations
            if 'Détails des Souscriptions' in df_clean.columns:
                def concat_details(x):
                    details = x.dropna().astype(str)
                    unique_details = details[details != 'nan'].unique()
                    return ' | '.join(unique_details) if len(unique_details) > 0 else np.nan
                agg_dict['Détails des Souscriptions'] = concat_details
            
            # Effectuer le regroupement avec gestion d'erreur
            try:
                df_clean = df_clean.groupby('Email').agg(agg_dict).reset_index()
                capture_success(f"Regroupement terminé : {len(df_clean)} clients uniques")
            except Exception as e:
                capture_error(f"Erreur lors du regroupement par email: {str(e)}")
                # En cas d'erreur, retourner les données sans regroupement
                capture_warning("Regroupement annulé, analyse des données individuelles")
                pass
        else:
            capture_info("Aucun doublon détecté par email")
    else:
        capture_warning("Colonne 'Email' non trouvée, pas de regroupement possible")
    
    # Traitement spécial de la colonne VR (Versements Réguliers)
    if 'VR' in df_clean.columns:
        # Convertir en montant numérique
        df_clean['vr_clean'] = safe_to_numeric(df_clean['VR'])
        
        # Créer un indicateur booléen basé sur le montant (VR > 0)
        df_clean['vr_booleen'] = (df_clean['vr_clean'] > 0) & (df_clean['vr_clean'].notna())
        
        # Créer une version binaire pour les corrélations (1 si VR > 0, 0 sinon)
        df_clean['vr_binaire'] = df_clean['vr_booleen'].astype(int)
        
        # Garder aussi la version texte originale pour référence
        df_clean['vr_texte'] = df_clean['VR'].astype(str).str.strip()
    
    # Conversion des autres colonnes numériques
    colonnes_numeriques = {
        'Nb Souscriptions': 'nb_souscriptions_clean',
        'Revenus': 'revenus_clean',
        'TMI': 'tmi_clean',
        'Épargne disponible': 'epargne_disponible_clean',
        'Nb d\'enfants': 'nb_enfants_clean'
    }
    
    for col_orig, col_clean in colonnes_numeriques.items():
        if col_orig in df_clean.columns:
            df_clean[col_clean] = safe_to_numeric(df_clean[col_orig])
    
    # Analyser la répartition clients vs leads
    if 'nb_souscriptions_clean' in df_clean.columns:
        clients_count = (df_clean['nb_souscriptions_clean'] > 0).sum()
        leads_count = ((df_clean['nb_souscriptions_clean'] == 0) | (df_clean['nb_souscriptions_clean'].isna())).sum()
        capture_info(f"Répartition après regroupement: {clients_count} vrais clients, {leads_count} leads")
        
        if clients_count > 0:
            capture_success(f"Taux de conversion: {clients_count/(clients_count+leads_count)*100:.1f}% des personnes sont des clients")
            
            # Analyser les VR parmi les clients
            if 'vr_clean' in df_clean.columns:
                clients_avec_vr = (df_clean['vr_clean'] > 0).sum()
                vr_moyen = df_clean[df_clean['vr_clean'] > 0]['vr_clean'].mean()
                capture_info(f"VR: {clients_avec_vr} clients avec VR > 0€ (montant moyen: {vr_moyen:.0f}€)")
    else:
        capture_warning("Colonne 'Nb Souscriptions' non trouvée, impossible de distinguer clients vs leads")
    
    # Conversion des colonnes de date
    colonnes_dates = {
        'Date de l\'entretien': 'date_entretien_clean',
        'Date d\'inscription': 'date_inscription_clean'
    }
    
    for col_orig, col_clean in colonnes_dates.items():
        if col_orig in df_clean.columns:
            df_clean[col_clean] = safe_to_datetime(df_clean[col_orig])
    
    # Nettoyage des colonnes texte
    colonnes_texte = ['Nom & Prénom', 'Email', 'Métier', 'Secteur d\'activité', 
                      'Profil épargnant', 'Situation familiale', 'Conseiller', 
                      'Détails des Souscriptions']
    
    for col in colonnes_texte:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace(['nan', 'None', ''], np.nan)
    
    # Normalisation spéciale pour la situation familiale
    if 'Situation familiale' in df_clean.columns:
        # Normaliser les variations de PACS
        df_clean['Situation familiale'] = df_clean['Situation familiale'].str.replace(
            r'PACS\(e\)|Pacsé\(e\)', 'Pacsé(e)', regex=True, case=False
        )
        
        # Autres normalisations possibles
        df_clean['Situation familiale'] = df_clean['Situation familiale'].str.replace(
            r'Marié\(e\)|Mariée?', 'Marié(e)', regex=True, case=False
        )
        
        df_clean['Situation familiale'] = df_clean['Situation familiale'].str.replace(
            r'Divorcé\(e\)|Divorcée?', 'Divorcé(e)', regex=True, case=False
        )
        
        df_clean['Situation familiale'] = df_clean['Situation familiale'].str.replace(
            r'Célibataire|Celibataire', 'Célibataire', regex=True, case=False
        )
        
        df_clean['Situation familiale'] = df_clean['Situation familiale'].str.replace(
            r'Veuf/Veuve|Veuf\(ve\)|Veuve?|Veuf', 'Veuf(ve)', regex=True, case=False
        )
        
        # Capturer les normalisations effectuées
        from utils.data_processing_debug import capture_info
        situations_uniques = df_clean['Situation familiale'].value_counts()
        capture_info(f"Situations familiales après normalisation: {situations_uniques.to_dict()}")
    
    return df_clean

def appliquer_filtres_clients(df):
    """Interface de filtrage des données clients."""
    st.subheader("🔍 Filtres de Données")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Filtre par conseiller
    with col1:
        if 'Conseiller' in df.columns:
            conseillers = ['Tous'] + sorted(df['Conseiller'].dropna().unique().tolist())
            conseiller_selectionne = st.selectbox("👨‍💼 Conseiller", conseillers)
        else:
            conseiller_selectionne = 'Tous'
    
    # Filtre par secteur
    with col2:
        if 'Secteur d\'activité' in df.columns:
            secteurs = ['Tous'] + sorted(df['Secteur d\'activité'].dropna().unique().tolist())
            secteur_selectionne = st.selectbox("🏢 Secteur", secteurs)
        else:
            secteur_selectionne = 'Tous'
    
    # Filtre par profil épargnant
    with col3:
        if 'Profil épargnant' in df.columns:
            profils = ['Tous'] + sorted(df['Profil épargnant'].dropna().unique().tolist())
            profil_selectionne = st.selectbox("💰 Profil Épargnant", profils)
        else:
            profil_selectionne = 'Tous'
    
    # Filtre par VR (Versements Réguliers)
    with col4:
        if 'vr_clean' in df.columns:
            vr_options = ['Tous', 'Avec VR (>0€)', 'Sans VR (=0€)']
            vr_selectionne = st.selectbox("🔄 Versements Réguliers", vr_options)
        else:
            vr_selectionne = 'Tous'
    
    # Filtre par plage de revenus
    col1, col2 = st.columns(2)
    with col1:
        if 'revenus_clean' in df.columns:
            revenus_min = st.number_input("💶 Revenus minimum", 
                                        min_value=0, 
                                        value=0,
                                        step=1000)
        else:
            revenus_min = 0
    
    with col2:
        if 'revenus_clean' in df.columns:
            revenus_max = st.number_input("💶 Revenus maximum", 
                                        min_value=0, 
                                        value=int(df['revenus_clean'].max()) if 'revenus_clean' in df.columns else 500000,
                                        step=1000)
        else:
            revenus_max = 500000
    
    # Application des filtres
    df_filtre = df.copy()
    
    if conseiller_selectionne != 'Tous':
        df_filtre = df_filtre[df_filtre['Conseiller'] == conseiller_selectionne]
    
    if secteur_selectionne != 'Tous':
        df_filtre = df_filtre[df_filtre['Secteur d\'activité'] == secteur_selectionne]
    
    if profil_selectionne != 'Tous':
        df_filtre = df_filtre[df_filtre['Profil épargnant'] == profil_selectionne]
    
    if vr_selectionne == 'Avec VR (>0€)':
        df_filtre = df_filtre[df_filtre['vr_clean'] > 0]
    elif vr_selectionne == 'Sans VR (=0€)':
        df_filtre = df_filtre[(df_filtre['vr_clean'] == 0) | (df_filtre['vr_clean'].isna())]
    
    if 'revenus_clean' in df_filtre.columns:
        df_filtre = df_filtre[
            (df_filtre['revenus_clean'] >= revenus_min) & 
            (df_filtre['revenus_clean'] <= revenus_max)
        ]
    
    st.info(f"📊 {len(df_filtre):,} clients après filtrage (sur {len(df):,} total)")
    
    return df_filtre

def analyser_dashboard_principal(df):
    """Dashboard principal avec métriques clés."""
    st.subheader("📈 Métriques Principales")
    
    # Première ligne de métriques
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Total Clients", f"{len(df):,}")
    
    with col2:
        if 'nb_souscriptions_clean' in df.columns:
            total_souscriptions = df['nb_souscriptions_clean'].sum()
            st.metric("📊 Total Souscriptions", f"{total_souscriptions:,.0f}")
        else:
            st.metric("📊 Total Souscriptions", "N/A")
    
    with col3:
        if 'revenus_clean' in df.columns:
            revenus_moyen = df['revenus_clean'].mean()
            st.metric("💰 Revenus Moyen", f"{revenus_moyen:,.0f} €")
        else:
            st.metric("💰 Revenus Moyen", "N/A")
    
    with col4:
        if 'vr_clean' in df.columns:
            clients_avec_vr = (df['vr_clean'] > 0).sum()
            pourcentage_vr = (clients_avec_vr / len(df)) * 100
            st.metric("🔄 % avec VR", f"{pourcentage_vr:.1f}%", 
                     help=f"{clients_avec_vr} clients sur {len(df)} ont VR > 0€")
        else:
            st.metric("🔄 % avec VR", "N/A")
    
    with col5:
        if 'epargne_disponible_clean' in df.columns:
            epargne_moyenne = df['epargne_disponible_clean'].mean()
            st.metric("🏦 Épargne Moy.", f"{epargne_moyenne:,.0f} €" if not pd.isna(epargne_moyenne) else "N/A")
        else:
            st.metric("🏦 Épargne Moy.", "N/A")
    
    # Deuxième ligne de métriques VR détaillées
    if 'vr_clean' in df.columns:
        st.subheader("💰 Analyse des Versements Réguliers (VR)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_vr = df['vr_clean'].sum()
            st.metric("💸 VR Total", f"{total_vr:,.0f} €")
        
        with col2:
            clients_avec_vr = (df['vr_clean'] > 0).sum()
            st.metric("👥 Clients avec VR", f"{clients_avec_vr:,}")
        
        with col3:
            clients_sans_vr = ((df['vr_clean'] == 0) | (df['vr_clean'].isna())).sum()
            st.metric("👥 Clients sans VR", f"{clients_sans_vr:,}")
        
        with col4:
            if clients_avec_vr > 0:
                vr_moyen = df[df['vr_clean'] > 0]['vr_clean'].mean()
                st.metric("📊 VR Moyen", f"{vr_moyen:,.0f} €", 
                         help="Montant moyen des VR pour les clients qui en ont")
            else:
                st.metric("📊 VR Moyen", "N/A")

def analyser_vue_ensemble(df):
    """Analyse générale des clients."""
    st.subheader("📊 Vue d'Ensemble Clients")
    
    col1, col2 = st.columns(2)
    
    # Répartition par éligibilité
    with col1:
        if 'Éligibilité' in df.columns:
            eligibilite_counts = df['Éligibilité'].value_counts()
            fig = px.pie(
                values=eligibilite_counts.values,
                names=eligibilite_counts.index,
                title="🎯 Répartition par Éligibilité",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Répartition par profil épargnant
    with col2:
        if 'Profil épargnant' in df.columns:
            profil_counts = df['Profil épargnant'].value_counts()
            fig = px.pie(
                values=profil_counts.values,
                names=profil_counts.index,
                title="💰 Répartition par Profil Épargnant",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Analyse spécifique des Versements Réguliers (VR)
    if 'vr_texte' in df.columns:
        st.subheader("🔄 Analyse des Versements Réguliers (VR)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Répartition VR
            vr_counts = df['vr_texte'].value_counts()
            fig = px.pie(
                values=vr_counts.values,
                names=vr_counts.index,
                title="📈 Versements Programmés",
                hole=0.4,
                color_discrete_map={'OUI': '#28a745', 'NON': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistiques VR
            total_clients = len(df)
            clients_vr_oui = df['vr_booleen'].sum()
            clients_vr_non = total_clients - clients_vr_oui
            pourcentage_vr = (clients_vr_oui / total_clients) * 100 if total_clients > 0 else 0
            
            st.metric("👥 Clients avec VR", clients_vr_oui, f"{pourcentage_vr:.1f}%")
            st.metric("👥 Clients sans VR", clients_vr_non, f"{100-pourcentage_vr:.1f}%")
        
        with col3:
            # Analyse corrélée VR vs Souscriptions
            if 'nb_souscriptions_clean' in df.columns:
                vr_souscriptions = df.groupby('vr_texte')['nb_souscriptions_clean'].mean().reset_index()
                
                fig = px.bar(
                    vr_souscriptions,
                    x='vr_texte',
                    y='nb_souscriptions_clean',
                    title="📊 Souscriptions Moyennes par VR",
                    text='nb_souscriptions_clean',
                    color='vr_texte',
                    color_discrete_map={'OUI': '#28a745', 'NON': '#dc3545'}
                )
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
    
    # Distribution des souscriptions et revenus
    col1, col2 = st.columns(2)
    
    with col1:
        if 'nb_souscriptions_clean' in df.columns:
            fig = px.histogram(
                df,
                x='nb_souscriptions_clean',
                title="📊 Distribution du Nombre de Souscriptions",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'revenus_clean' in df.columns:
            fig = px.histogram(
                df,
                x='revenus_clean',
                title="💰 Distribution des Revenus",
                nbins=25
            )
            st.plotly_chart(fig, use_container_width=True)

def analyser_par_conseiller(df):
    """Analyse détaillée par conseiller."""
    st.subheader("👨‍💼 Analyse par Conseiller")
    
    if 'Conseiller' not in df.columns:
        st.warning("⚠️ Colonne 'Conseiller' non trouvée.")
        return
    
    # Statistiques par conseiller
    conseiller_stats = df.groupby('Conseiller').agg({
        'Conseiller': 'count',
        'nb_souscriptions_clean': 'sum' if 'nb_souscriptions_clean' in df.columns else lambda x: 0,
        'revenus_clean': 'mean' if 'revenus_clean' in df.columns else lambda x: 0,
        'vr_clean': 'mean' if 'vr_clean' in df.columns else lambda x: 0
    }).rename(columns={'Conseiller': 'Nb_Clients'}).reset_index()
    
    if 'nb_souscriptions_clean' in df.columns:
        conseiller_stats = conseiller_stats.rename(columns={'nb_souscriptions_clean': 'Total_Souscriptions'})
    if 'revenus_clean' in df.columns:
        conseiller_stats = conseiller_stats.rename(columns={'revenus_clean': 'Revenus_Moyens'})
    if 'vr_clean' in df.columns:
        conseiller_stats = conseiller_stats.rename(columns={'vr_clean': 'VR_Moyenne'})
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            conseiller_stats.sort_values('Nb_Clients', ascending=False),
            x='Conseiller',
            y='Nb_Clients',
            title="👥 Nombre de Clients par Conseiller",
            text='Nb_Clients'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Total_Souscriptions' in conseiller_stats.columns:
            fig = px.bar(
                conseiller_stats.sort_values('Total_Souscriptions', ascending=False),
                x='Conseiller',
                y='Total_Souscriptions',
                title="📊 Total Souscriptions par Conseiller",
                text='Total_Souscriptions'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    st.subheader("📋 Tableau Détaillé par Conseiller")
    
    # Formatage du tableau
    conseiller_display = conseiller_stats.copy()
    if 'Revenus_Moyens' in conseiller_display.columns:
        conseiller_display['Revenus_Moyens'] = conseiller_display['Revenus_Moyens'].apply(lambda x: f"{x:,.0f} €" if not pd.isna(x) else "N/A")
    if 'VR_Moyenne' in conseiller_display.columns:
        conseiller_display['VR_Moyenne'] = conseiller_display['VR_Moyenne'].apply(lambda x: f"{x:,.0f} €" if not pd.isna(x) else "N/A")
    
    st.dataframe(conseiller_display.sort_values('Nb_Clients', ascending=False), use_container_width=True)

def analyser_par_secteur(df):
    """Analyse par secteur d'activité."""
    st.subheader("🏢 Analyse par Secteur d'Activité")
    
    if 'Secteur d\'activité' not in df.columns:
        st.warning("⚠️ Colonne 'Secteur d'activité' non trouvée.")
        return
    
    secteur_stats = df.groupby('Secteur d\'activité').agg({
        'Secteur d\'activité': 'count',
        'nb_souscriptions_clean': 'mean' if 'nb_souscriptions_clean' in df.columns else lambda x: 0,
        'revenus_clean': 'mean' if 'revenus_clean' in df.columns else lambda x: 0,
    }).rename(columns={'Secteur d\'activité': 'Nb_Clients'}).reset_index()
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            secteur_stats.sort_values('Nb_Clients', ascending=False),
            x='Secteur d\'activité',
            y='Nb_Clients',
            title="👥 Clients par Secteur d'Activité",
            text='Nb_Clients'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top 10 secteurs pour le camembert
        top_secteurs = secteur_stats.nlargest(10, 'Nb_Clients')
        fig = px.pie(
            top_secteurs,
            values='Nb_Clients',
            names='Secteur d\'activité',
            title="🥧 Top 10 Secteurs (Répartition)",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse croisée secteur-éligibilité
    if 'Éligibilité' in df.columns:
        st.subheader("🎯 Analyse Croisée : Secteur × Éligibilité")
        
        cross_analysis = pd.crosstab(df['Secteur d\'activité'], df['Éligibilité'], normalize='index') * 100
        
        fig = px.imshow(
            cross_analysis.values,
            labels=dict(x="Éligibilité", y="Secteur d'activité", color="Pourcentage"),
            x=cross_analysis.columns,
            y=cross_analysis.index,
            title="🔥 Heatmap : Éligibilité par Secteur (%)",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)

def analyser_financier(df):
    """Analyse financière détaillée."""
    st.subheader("💰 Analyse Financière Détaillée")
    
    # Analyse spécifique VR vs Finances
    if 'vr_texte' in df.columns:
        st.subheader("🔄 Impact des Versements Réguliers sur les Finances")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenus moyens par VR
            if 'revenus_clean' in df.columns:
                vr_revenus = df.groupby('vr_texte')['revenus_clean'].agg(['mean', 'median']).reset_index()
                
                fig = px.bar(
                    vr_revenus,
                    x='vr_texte',
                    y='mean',
                    title="💰 Revenus Moyens par VR",
                    text='mean',
                    color='vr_texte',
                    color_discrete_map={'OUI': '#28a745', 'NON': '#dc3545'}
                )
                fig.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Épargne disponible par VR
            if 'epargne_disponible_clean' in df.columns:
                vr_epargne = df.groupby('vr_texte')['epargne_disponible_clean'].mean().reset_index()
                
                fig = px.bar(
                    vr_epargne,
                    x='vr_texte',
                    y='epargne_disponible_clean',
                    title="🏦 Épargne Moyenne par VR",
                    text='epargne_disponible_clean',
                    color='vr_texte',
                    color_discrete_map={'OUI': '#28a745', 'NON': '#dc3545'}
                )
                fig.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
    
    # Corrélations financières (incluant VR si disponible)
    colonnes_financieres = ['revenus_clean', 'vr_clean', 'epargne_disponible_clean', 
                           'nb_souscriptions_clean', 'tmi_clean']
    colonnes_disponibles = [col for col in colonnes_financieres if col in df.columns]
    
    if len(colonnes_disponibles) >= 2:
        st.subheader("🔗 Matrice de Corrélation")
        
        correlation_matrix = df[colonnes_disponibles].corr()
        
        # Renommer les colonnes pour l'affichage
        colonnes_affichage = {
            'revenus_clean': 'Revenus',
            'vr_clean': 'VR (1=OUI, 0=NON)',
            'epargne_disponible_clean': 'Épargne Disponible',
            'nb_souscriptions_clean': 'Nb Souscriptions',
            'tmi_clean': 'TMI'
        }
        
        correlation_renamed = correlation_matrix.rename(
            index=colonnes_affichage, 
            columns=colonnes_affichage
        )
        
        fig = px.imshow(
            correlation_renamed,
            title="💹 Corrélations entre Variables Financières (VR inclus)",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyse textuelle des corrélations avec VR
        if 'vr_clean' in df.columns:
            st.subheader("📊 Insights sur les Versements Réguliers")
            
            # Calculer les corrélations avec VR
            corr_with_vr = correlation_matrix['vr_clean'].drop('vr_clean').abs().sort_values(ascending=False)
            
            insights = []
            for var, corr_val in corr_with_vr.items():
                if corr_val > 0.3:
                    insights.append(f"**Forte corrélation** avec {colonnes_affichage.get(var, var)} : {corr_val:.3f}")
                elif corr_val > 0.1:
                    insights.append(f"**Corrélation modérée** avec {colonnes_affichage.get(var, var)} : {corr_val:.3f}")
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.info("**Aucune corrélation significative** détectée avec les versements réguliers")
    
    # Analyses par tranches
    col1, col2 = st.columns(2)
    
    with col1:
        if 'revenus_clean' in df.columns:
            st.subheader("📊 Segmentation par Revenus")
            
            # Créer des tranches de revenus
            df_revenus = df.dropna(subset=['revenus_clean'])
            bins = [0, 30000, 50000, 75000, 100000, 150000, float('inf')]
            labels = ['<30k', '30-50k', '50-75k', '75-100k', '100-150k', '>150k']
            
            df_revenus['Tranche_Revenus'] = pd.cut(df_revenus['revenus_clean'], bins=bins, labels=labels)
            tranche_stats = df_revenus.groupby('Tranche_Revenus').agg({
                'Tranche_Revenus': 'count',
                'nb_souscriptions_clean': 'mean' if 'nb_souscriptions_clean' in df.columns else lambda x: 0
            }).rename(columns={'Tranche_Revenus': 'Nb_Clients'}).reset_index()
            
            fig = px.bar(
                tranche_stats,
                x='Tranche_Revenus',
                y='Nb_Clients',
                title="👥 Répartition par Tranche de Revenus",
                text='Nb_Clients'
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'tmi_clean' in df.columns:
            st.subheader("🎯 Répartition par TMI")
            
            tmi_stats = df['tmi_clean'].value_counts().sort_index()
            fig = px.bar(
                x=tmi_stats.index,
                y=tmi_stats.values,
                title="📈 Distribution par Tranche Marginale d'Imposition",
                labels={'x': 'TMI (%)', 'y': 'Nombre de Clients'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots pour les relations
    if 'revenus_clean' in df.columns and 'nb_souscriptions_clean' in df.columns:
        st.subheader("📈 Relations Revenus-Souscriptions")
        
        fig = px.scatter(
            df,
            x='revenus_clean',
            y='nb_souscriptions_clean',
            color='Profil épargnant' if 'Profil épargnant' in df.columns else None,
            title="💰 Revenus vs Nombre de Souscriptions"
        )
        st.plotly_chart(fig, use_container_width=True)

def analyser_profil_familial(df):
    """Analyse du profil familial."""
    st.subheader("👨‍👩‍👧‍👦 Analyse du Profil Familial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Situation familiale' in df.columns:
            st.subheader("💑 Situation Familiale")
            
            situation_stats = df['Situation familiale'].value_counts()
            fig = px.pie(
                values=situation_stats.values,
                names=situation_stats.index,
                title="👪 Répartition par Situation Familiale",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'nb_enfants_clean' in df.columns:
            st.subheader("👶 Nombre d'Enfants")
            
            enfants_stats = df['nb_enfants_clean'].value_counts().sort_index()
            fig = px.bar(
                x=enfants_stats.index,
                y=enfants_stats.values,
                title="🧒 Distribution par Nombre d'Enfants",
                labels={'x': 'Nombre d\'enfants', 'y': 'Nombre de clients'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Analyse croisée famille-finance
    if all(col in df.columns or f"{col.lower().replace(' ', '_').replace('\'', '')}_clean" in df.columns 
           for col in ['Situation familiale', 'nb_enfants_clean', 'revenus_clean']):
        
        st.subheader("💰 Impact Familial sur les Finances")
        
        # Revenus moyens par situation familiale
        revenus_famille = df.groupby('Situation familiale')['revenus_clean'].mean().reset_index()
        
        fig = px.bar(
            revenus_famille,
            x='Situation familiale',
            y='revenus_clean',
            title="💶 Revenus Moyens par Situation Familiale"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def analyser_evolution_temporelle(df):
    """Analyse de l'évolution temporelle."""
    st.subheader("📅 Évolution Temporelle")
    
    # Analyses par date d'entretien
    if 'date_entretien_clean' in df.columns:
        df_entretiens = df.dropna(subset=['date_entretien_clean']).copy()
        
        if not df_entretiens.empty:
            st.subheader("📞 Évolution des Entretiens")
            
            # Grouper par mois
            df_entretiens['Mois'] = df_entretiens['date_entretien_clean'].dt.to_period('M')
            entretiens_mois = df_entretiens.groupby('Mois').size().reset_index(name='Nb_Entretiens')
            entretiens_mois['Mois_Str'] = entretiens_mois['Mois'].astype(str)
            
            fig = px.line(
                entretiens_mois,
                x='Mois_Str',
                y='Nb_Entretiens',
                title="📈 Évolution Mensuelle des Entretiens",
                markers=True
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse saisonnière
            df_entretiens['Mois_Num'] = df_entretiens['date_entretien_clean'].dt.month
            saisonnalite = df_entretiens.groupby('Mois_Num').size().reset_index(name='Nb_Entretiens')
            saisonnalite['Mois_Nom'] = saisonnalite['Mois_Num'].apply(lambda x: calendar.month_name[x])
            
            fig = px.bar(
                saisonnalite,
                x='Mois_Nom',
                y='Nb_Entretiens',
                title="🌅 Saisonnalité des Entretiens",
                text='Nb_Entretiens'
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    # Analyses par date d'inscription
    if 'date_inscription_clean' in df.columns:
        df_inscriptions = df.dropna(subset=['date_inscription_clean']).copy()
        
        if not df_inscriptions.empty:
            st.subheader("📝 Évolution des Inscriptions")
            
            df_inscriptions['Mois'] = df_inscriptions['date_inscription_clean'].dt.to_period('M')
            inscriptions_mois = df_inscriptions.groupby('Mois').size().reset_index(name='Nb_Inscriptions')
            inscriptions_mois['Mois_Str'] = inscriptions_mois['Mois'].astype(str)
            
            fig = px.line(
                inscriptions_mois,
                x='Mois_Str',
                y='Nb_Inscriptions',
                title="📈 Évolution Mensuelle des Inscriptions",
                markers=True
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

def analyser_segmentation_avancee(df):
    """Segmentation avancée des clients."""
    st.subheader("🎯 Segmentation Avancée des Clients")
    
    # Segmentation RFM adaptée
    if all(col in df.columns for col in ['nb_souscriptions_clean', 'revenus_clean']):
        st.subheader("💎 Segmentation Valeur-Volume")
        
        df_segment = df.dropna(subset=['nb_souscriptions_clean', 'revenus_clean']).copy()
        
        if not df_segment.empty:
            # Calculer les quartiles
            revenus_q75 = df_segment['revenus_clean'].quantile(0.75)
            revenus_q25 = df_segment['revenus_clean'].quantile(0.25)
            
            souscriptions_q75 = df_segment['nb_souscriptions_clean'].quantile(0.75)
            souscriptions_q25 = df_segment['nb_souscriptions_clean'].quantile(0.25)
            
            def segmenter_client(row):
                revenus = row['revenus_clean']
                souscriptions = row['nb_souscriptions_clean']
                
                if revenus >= revenus_q75 and souscriptions >= souscriptions_q75:
                    return "🌟 Champions"
                elif revenus >= revenus_q75 and souscriptions >= souscriptions_q25:
                    return "💰 Gros Potentiel"
                elif revenus >= revenus_q25 and souscriptions >= souscriptions_q75:
                    return "🔥 Très Actifs"
                elif revenus >= revenus_q25 and souscriptions >= souscriptions_q25:
                    return "✅ Fidèles"
                else:
                    return "📈 À Développer"
            
            df_segment['Segment'] = df_segment.apply(segmenter_client, axis=1)
            
            # Graphique de segmentation
            fig = px.scatter(
                df_segment,
                x='revenus_clean',
                y='nb_souscriptions_clean',
                color='Segment',
                title="🎯 Segmentation Clients : Revenus vs Souscriptions",
                hover_data=['Nom & Prénom'] if 'Nom & Prénom' in df_segment.columns else None
            )
            
            # Ajouter les lignes de démarcation
            fig.add_hline(y=souscriptions_q75, line_dash="dash", line_color="gray", opacity=0.7)
            fig.add_hline(y=souscriptions_q25, line_dash="dash", line_color="gray", opacity=0.7)
            fig.add_vline(x=revenus_q75, line_dash="dash", line_color="gray", opacity=0.7)
            fig.add_vline(x=revenus_q25, line_dash="dash", line_color="gray", opacity=0.7)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques par segment
            st.subheader("📊 Statistiques par Segment")
            
            segment_stats = df_segment.groupby('Segment').agg({
                'Segment': 'count',
                'revenus_clean': 'mean',
                'nb_souscriptions_clean': 'mean',
                'vr_clean': 'mean' if 'vr_clean' in df_segment.columns else lambda x: 0
            }).rename(columns={'Segment': 'Nb_Clients'}).reset_index()
            
            # Formatage pour l'affichage
            segment_display = segment_stats.copy()
            segment_display['revenus_clean'] = segment_display['revenus_clean'].apply(lambda x: f"{x:,.0f} €")
            segment_display['nb_souscriptions_clean'] = segment_display['nb_souscriptions_clean'].apply(lambda x: f"{x:.1f}")
            if 'vr_clean' in segment_display.columns:
                segment_display['vr_clean'] = segment_display['vr_clean'].apply(lambda x: f"{x:,.0f} €" if not pd.isna(x) else "N/A")
            
            st.dataframe(segment_display, use_container_width=True)
    
    # Recommandations par segment
    st.subheader("💡 Recommandations Stratégiques")
    
    recommendations = {
        "🌟 Champions": "Clients premium : privilégier la rétention et les produits haut de gamme",
        "💰 Gros Potentiel": "Clients à fort revenus : augmenter la fréquence d'interaction", 
        "🔥 Très Actifs": "Clients engagés : proposer des produits complémentaires",
        "✅ Fidèles": "Base solide : maintenir la relation et proposer des évolutions",
        "📈 À Développer": "Potentiel de croissance : stratégie d'activation ciblée"
    }
    
    for segment, recommendation in recommendations.items():
        st.info(f"**{segment}** : {recommendation}")

def generer_exports(df):
    """Génération des exports et rapports."""
    st.subheader("📤 Exports et Rapports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Export Données")
        
        # Export CSV
        if st.button("📥 Télécharger CSV", type="primary"):
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Télécharger le fichier CSV",
                data=csv_data,
                file_name=f"clients_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.subheader("📈 Résumé Statistique")
        
        # Génération du résumé
        if st.button("📋 Générer Résumé", type="secondary"):
            st.subheader("📊 Résumé de l'Analyse")
            
            # Statistiques générales
            st.write(f"**Nombre total de clients analysés :** {len(df):,}")
            
            if 'nb_souscriptions_clean' in df.columns:
                st.write(f"**Total des souscriptions :** {df['nb_souscriptions_clean'].sum():,.0f}")
            
            if 'revenus_clean' in df.columns:
                st.write(f"**Revenus moyen :** {df['revenus_clean'].mean():,.0f} €")
            
            # Top conseillers
            if 'Conseiller' in df.columns:
                top_conseiller = df['Conseiller'].value_counts().index[0]
                nb_clients_top = df['Conseiller'].value_counts().iloc[0]
                st.write(f"**Conseiller avec le plus de clients :** {top_conseiller} ({nb_clients_top} clients)")
            
            # Secteur dominant
            if 'Secteur d\'activité' in df.columns:
                secteur_dominant = df['Secteur d\'activité'].value_counts().index[0]
                st.write(f"**Secteur d'activité dominant :** {secteur_dominant}")
    
    # Données brutes
    with st.expander("🔍 Données Détaillées", expanded=False):
        st.dataframe(df, use_container_width=True)
        st.caption(f"Affichage de {len(df):,} lignes")