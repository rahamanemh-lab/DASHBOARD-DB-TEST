"""
🎯 Analyse Clients
Application Streamlit pour l'analyse approfondie des données clients
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import sys
import os

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration de la page
st.set_page_config(
    page_title="🎯 Analyse Clients",
    page_icon="🎯",
    layout="wide"
)

# Importer les fonctions utilitaires si disponibles
try:
    from utils.data_processing import safe_to_datetime, safe_to_numeric, extract_conseiller
    from utils.export import create_download_button
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

def safe_to_datetime_local(series):
    """Conversion sécurisée en datetime"""
    if UTILS_AVAILABLE:
        return safe_to_datetime(series)
    else:
        return pd.to_datetime(series, errors='coerce')

def safe_to_numeric_local(series):
    """Conversion sécurisée en numérique"""
    if UTILS_AVAILABLE:
        return safe_to_numeric(series)
    else:
        return pd.to_numeric(series, errors='coerce')

def analyser_clients_complet(df):
    """Analyse complète des données clients"""
    
    st.header("🎯 Analyse Clients")
    st.markdown("**Analyse approfondie des données clients**")
    st.markdown("---")
    
    # Informations générales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total clients", len(df))
    with col2:
        st.metric("📋 Colonnes", len(df.columns))
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("💾 Taille", f"{memory_mb:.1f} MB")
    with col4:
        lignes_completes = df.dropna().shape[0]
        pourcentage = (lignes_completes / len(df)) * 100
        st.metric("🔢 Données complètes", f"{pourcentage:.1f}%")
    
    # Détection intelligente des colonnes
    colonnes_mapping = {
        'nom_prenom': ['Nom & Prénom', 'Nom', 'Prénom', 'Name', 'nom_prenom', 'client_name', 'Client'],
        'email': ['Email', 'E-mail', 'Mail', 'email', 'e_mail', 'adresse_email'],
        'date_entretien': ['Date de l\'entretien', 'Date entretien', 'Date', 'date_entretien', 'entretien_date'],
        'nb_souscriptions': ['Nb Souscriptions', 'Nombre souscriptions', 'Souscriptions', 'nb_souscriptions', 'souscriptions'],
        'derniere_souscription': ['Dernière Souscription', 'Last subscription', 'derniere_souscription'],
        'vr': ['VR', 'Valeur Rachetable', 'vr', 'valeur_rachetable'],
        'metier': ['Métier', 'Job', 'Profession', 'metier', 'profession', 'job'],
        'secteur': ['Secteur d\'activité', 'Secteur', 'Sector', 'secteur', 'activite', 'secteur_activite'],
        'revenus': ['Revenus', 'Revenue', 'Income', 'revenus', 'salaire', 'income'],
        'type_contrat': ['Type de contrat', 'Contrat', 'Contract', 'type_contrat', 'contrat'],
        'eligibilite': ['Éligibilité', 'Eligibilité', 'Eligibility', 'eligibilite', 'eligible'],
        'tmi': ['TMI', 'Tranche', 'tmi', 'tranche_marginale', 'tranche_imposition'],
        'profil_epargnant': ['Profil épargnant', 'Profil', 'Profile', 'profil_epargnant', 'profil'],
        'epargne_disponible': ['Épargne disponible', 'Épargne', 'Savings', 'epargne_disponible', 'epargne'],
        'situation_familiale': ['Situation familiale', 'Situation', 'Family', 'situation_familiale', 'famille'],
        'nb_enfants': ['Nb d\'enfants', 'Nombre enfants', 'Children', 'nb_enfants', 'enfants'],
        'date_inscription': ['Date d\'inscription', 'Date inscription', 'Registration', 'date_inscription'],
        'conseiller': ['Conseiller', 'Advisor', 'Agent', 'conseiller', 'agent', 'commercial']
    }
    
    # Mapper les colonnes
    colonnes_trouvees = {}
    for cle, possibles in colonnes_mapping.items():
        for possible in possibles:
            if possible in df.columns:
                colonnes_trouvees[cle] = possible
                break
    
    # Préparation des données
    df_work = df.copy()
    
    # Conversions des données
    if 'date_entretien' in colonnes_trouvees:
        df_work['Date_Entretien_Clean'] = safe_to_datetime_local(df_work[colonnes_trouvees['date_entretien']])
    
    if 'date_inscription' in colonnes_trouvees:
        df_work['Date_Inscription_Clean'] = safe_to_datetime_local(df_work[colonnes_trouvees['date_inscription']])
    
    if 'nb_souscriptions' in colonnes_trouvees:
        df_work['Nb_Souscriptions_Clean'] = safe_to_numeric_local(df_work[colonnes_trouvees['nb_souscriptions']])
    
    if 'revenus' in colonnes_trouvees:
        df_work['Revenus_Clean'] = safe_to_numeric_local(df_work[colonnes_trouvees['revenus']])
    
    if 'nb_enfants' in colonnes_trouvees:
        df_work['Nb_Enfants_Clean'] = safe_to_numeric_local(df_work[colonnes_trouvees['nb_enfants']])
    
    # Onglets d'analyse
    tabs = st.tabs([
        "📊 Vue d'ensemble",
        "👨‍💼 Conseillers", 
        "🏢 Secteurs",
        "💰 Revenus",
        "👨‍👩‍👧‍👦 Profils",
        "📅 Temporel",
        "🎯 Segmentation"
    ])
    
    # Vue d'ensemble
    with tabs[0]:
        st.subheader("📈 Statistiques Globales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nombre total de clients", len(df_work))
        
        with col2:
            if 'Nb_Souscriptions_Clean' in df_work.columns:
                total_souscriptions = df_work['Nb_Souscriptions_Clean'].sum()
                st.metric("Total souscriptions", f"{total_souscriptions:,.0f}")
            else:
                st.metric("Total souscriptions", "N/A")
        
        with col3:
            if 'Nb_Souscriptions_Clean' in df_work.columns:
                clients_actifs = len(df_work[df_work['Nb_Souscriptions_Clean'] > 0])
                st.metric("Clients actifs", clients_actifs)
            else:
                st.metric("Clients actifs", "N/A")
        
        with col4:
            if 'Revenus_Clean' in df_work.columns:
                revenus_moyen = df_work['Revenus_Clean'].mean()
                st.metric("Revenus moyen", f"{revenus_moyen:,.0f} €")
            else:
                st.metric("Revenus moyen", "N/A")
        
        # Colonnes détectées
        st.subheader("🔗 Colonnes détectées")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**✅ Colonnes trouvées :**")
            for cle, colonne in colonnes_trouvees.items():
                st.write(f"• **{cle}**: `{colonne}`")
        
        with col2:
            st.write("**❌ Colonnes manquantes :**")
            manquantes = [cle for cle in colonnes_mapping.keys() if cle not in colonnes_trouvees]
            for manquante in manquantes:
                st.write(f"• {manquante}")
        
        # Aperçu des données
        st.subheader("👀 Aperçu des données")
        st.dataframe(df_work.head(10), use_container_width=True)
    
    # Analyse par conseiller
    with tabs[1]:
        if 'conseiller' in colonnes_trouvees:
            st.subheader("👨‍💼 Analyse par Conseiller")
            
            conseiller_stats = df_work.groupby(colonnes_trouvees['conseiller']).agg({
                colonnes_trouvees['conseiller']: 'count'
            }).rename(columns={
                colonnes_trouvees['conseiller']: 'Nombre_Clients'
            }).reset_index()
            
            if 'Nb_Souscriptions_Clean' in df_work.columns:
                souscriptions_stats = df_work.groupby(colonnes_trouvees['conseiller'])['Nb_Souscriptions_Clean'].agg(['sum', 'mean']).reset_index()
                souscriptions_stats.columns = [colonnes_trouvees['conseiller'], 'Total_Souscriptions', 'Moyenne_Souscriptions']
                conseiller_stats = conseiller_stats.merge(souscriptions_stats, on=colonnes_trouvees['conseiller'])
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    conseiller_stats.sort_values('Nombre_Clients', ascending=True),
                    y=colonnes_trouvees['conseiller'],
                    x='Nombre_Clients',
                    title="Nombre de clients par conseiller",
                    orientation='h',
                    text='Nombre_Clients'
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Total_Souscriptions' in conseiller_stats.columns:
                    fig = px.scatter(
                        conseiller_stats,
                        x='Nombre_Clients',
                        y='Total_Souscriptions',
                        size='Moyenne_Souscriptions',
                        hover_name=colonnes_trouvees['conseiller'],
                        title="Clients vs Souscriptions par conseiller",
                        size_max=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            st.write("**📊 Détail par conseiller :**")
            if 'Total_Souscriptions' in conseiller_stats.columns:
                conseiller_stats['Moyenne_Souscriptions'] = conseiller_stats['Moyenne_Souscriptions'].round(2)
            st.dataframe(conseiller_stats.sort_values('Nombre_Clients', ascending=False), use_container_width=True)
        else:
            st.info("ℹ️ Colonne 'Conseiller' non trouvée dans les données")
    
    # Analyse par secteur
    with tabs[2]:
        if 'secteur' in colonnes_trouvees:
            st.subheader("🏢 Analyse par Secteur d'Activité")
            
            secteur_stats = df_work[colonnes_trouvees['secteur']].value_counts().reset_index()
            secteur_stats.columns = ['Secteur', 'Nombre_Clients']
            secteur_stats['Pourcentage'] = (secteur_stats['Nombre_Clients'] / len(df_work) * 100).round(1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    secteur_stats,
                    values='Nombre_Clients',
                    names='Secteur',
                    title="Répartition par secteur d'activité",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    secteur_stats.sort_values('Nombre_Clients', ascending=True).tail(10),
                    y='Secteur',
                    x='Nombre_Clients',
                    title="Top 10 des secteurs",
                    orientation='h',
                    text='Pourcentage'
                )
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(secteur_stats, use_container_width=True)
        else:
            st.info("ℹ️ Colonne 'Secteur d'activité' non trouvée dans les données")
    
    # Analyse des revenus
    with tabs[3]:
        if 'Revenus_Clean' in df_work.columns:
            st.subheader("💰 Analyse des Revenus")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df_work.dropna(subset=['Revenus_Clean']),
                    x='Revenus_Clean',
                    title="Distribution des revenus",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df_work.dropna(subset=['Revenus_Clean']),
                    y='Revenus_Clean',
                    title="Boîte à moustaches des revenus"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Segmentation par tranches
            st.write("**Segmentation par tranches de revenus :**")
            revenus_clean = df_work['Revenus_Clean'].dropna()
            if len(revenus_clean) > 0:
                tranches = pd.cut(
                    revenus_clean,
                    bins=[0, 30000, 50000, 75000, 100000, 150000, float('inf')],
                    labels=['< 30K€', '30-50K€', '50-75K€', '75-100K€', '100-150K€', '> 150K€']
                )
                
                tranche_stats = tranches.value_counts().reset_index()
                tranche_stats.columns = ['Tranche_Revenus', 'Nombre_Clients']
                tranche_stats['Pourcentage'] = (tranche_stats['Nombre_Clients'] / len(revenus_clean) * 100).round(1)
                
                fig = px.bar(
                    tranche_stats,
                    x='Tranche_Revenus',
                    y='Nombre_Clients',
                    text='Pourcentage',
                    title="Répartition par tranches de revenus"
                )
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(tranche_stats, use_container_width=True)
        else:
            st.info("ℹ️ Colonne 'Revenus' non trouvée dans les données")
    
    # Profils familiaux
    with tabs[4]:
        st.subheader("👨‍👩‍👧‍👦 Profils Familiaux")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'situation_familiale' in colonnes_trouvees:
                situation_stats = df_work[colonnes_trouvees['situation_familiale']].value_counts()
                fig = px.pie(
                    values=situation_stats.values,
                    names=situation_stats.index,
                    title="Situation familiale",
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Colonne 'Situation familiale' non trouvée")
        
        with col2:
            if 'Nb_Enfants_Clean' in df_work.columns:
                enfants_stats = df_work['Nb_Enfants_Clean'].value_counts().sort_index()
                fig = px.bar(
                    x=enfants_stats.index,
                    y=enfants_stats.values,
                    title="Nombre d'enfants",
                    text=enfants_stats.values
                )
                fig.update_layout(xaxis_title="Nombre d'enfants", yaxis_title="Nombre de clients")
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Colonne 'Nombre d'enfants' non trouvée")
    
    # Analyse temporelle
    with tabs[5]:
        st.subheader("📅 Analyse Temporelle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Date_Entretien_Clean' in df_work.columns:
                entretiens_temp = df_work.dropna(subset=['Date_Entretien_Clean'])
                if len(entretiens_temp) > 0:
                    entretiens_temp['Mois'] = entretiens_temp['Date_Entretien_Clean'].dt.to_period('M')
                    entretiens_par_mois = entretiens_temp.groupby('Mois').size()
                    
                    fig = px.line(
                        x=entretiens_par_mois.index.astype(str),
                        y=entretiens_par_mois.values,
                        title="Évolution des entretiens",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Mois", yaxis_title="Nombre d'entretiens")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Pas de dates d'entretien valides")
            else:
                st.info("ℹ️ Colonne 'Date entretien' non trouvée")
        
        with col2:
            if 'Date_Inscription_Clean' in df_work.columns:
                inscriptions_temp = df_work.dropna(subset=['Date_Inscription_Clean'])
                if len(inscriptions_temp) > 0:
                    inscriptions_temp['Mois'] = inscriptions_temp['Date_Inscription_Clean'].dt.to_period('M')
                    inscriptions_par_mois = inscriptions_temp.groupby('Mois').size()
                    
                    fig = px.line(
                        x=inscriptions_par_mois.index.astype(str),
                        y=inscriptions_par_mois.values,
                        title="Évolution des inscriptions",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Mois", yaxis_title="Nombre d'inscriptions")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Pas de dates d'inscription valides")
            else:
                st.info("ℹ️ Colonne 'Date inscription' non trouvée")
    
    # Segmentation avancée
    with tabs[6]:
        st.subheader("🎯 Segmentation Clients")
        
        if all(col in df_work.columns for col in ['Nb_Souscriptions_Clean', 'Revenus_Clean']):
            df_segment = df_work.dropna(subset=['Nb_Souscriptions_Clean', 'Revenus_Clean'])
            
            if len(df_segment) > 0:
                # Calcul des seuils
                seuil_souscriptions = df_segment['Nb_Souscriptions_Clean'].median()
                seuil_revenus = df_segment['Revenus_Clean'].median()
                
                def categoriser_client(row):
                    souscriptions = row['Nb_Souscriptions_Clean']
                    revenus = row['Revenus_Clean']
                    
                    if souscriptions >= seuil_souscriptions and revenus >= seuil_revenus:
                        return "🟢 Premium (High Value)"
                    elif souscriptions >= seuil_souscriptions and revenus < seuil_revenus:
                        return "🟡 Actif (High Volume)"
                    elif souscriptions < seuil_souscriptions and revenus >= seuil_revenus:
                        return "🟠 Potentiel (High Income)"
                    else:
                        return "🔴 Basique (Low Engagement)"
                
                df_segment['Segment'] = df_segment.apply(categoriser_client, axis=1)
                
                # Graphique de segmentation
                fig = px.scatter(
                    df_segment,
                    x='Revenus_Clean',
                    y='Nb_Souscriptions_Clean',
                    color='Segment',
                    title="Matrice de segmentation clients",
                    hover_data=[colonnes_trouvees.get('nom_prenom', 'Index')]
                )
                
                # Lignes de seuil
                fig.add_hline(y=seuil_souscriptions, line_dash="dash", line_color="gray")
                fig.add_vline(x=seuil_revenus, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques par segment
                segment_stats = df_segment.groupby('Segment').agg({
                    'Segment': 'count',
                    'Revenus_Clean': 'mean',
                    'Nb_Souscriptions_Clean': 'mean'
                }).rename(columns={'Segment': 'Nombre_Clients'}).reset_index()
                
                segment_stats['Revenus_Clean'] = segment_stats['Revenus_Clean'].apply(lambda x: f"{x:,.0f} €")
                segment_stats['Nb_Souscriptions_Clean'] = segment_stats['Nb_Souscriptions_Clean'].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(segment_stats, use_container_width=True)
            else:
                st.warning("Pas assez de données pour la segmentation")
        else:
            st.info("ℹ️ Données insuffisantes pour la segmentation (nécessite revenus et nb souscriptions)")
    
    # Export des données
    st.subheader("📤 Export des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_work.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Données analysées (CSV)",
            data=csv,
            file_name=f"clients_analyses_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Résumé JSON
        resume = {
            "total_clients": len(df_work),
            "colonnes_detectees": list(colonnes_trouvees.keys()),
            "date_analyse": datetime.now().isoformat(),
            "statistiques": {
                "revenus_moyen": float(df_work['Revenus_Clean'].mean()) if 'Revenus_Clean' in df_work.columns else None,
                "souscriptions_totales": float(df_work['Nb_Souscriptions_Clean'].sum()) if 'Nb_Souscriptions_Clean' in df_work.columns else None
            }
        }
        
        st.download_button(
            label="📋 Résumé JSON",
            data=pd.Series(resume).to_json(indent=2),
            file_name=f"resume_clients_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

def main():
    """Fonction principale de l'application"""
    
    # Upload de fichier
    uploaded_file = st.file_uploader(
        "📁 Chargez votre fichier clients Excel",
        type=['xlsx', 'xls'],
        help="Fichier Excel avec toutes les données clients à analyser"
    )

    if uploaded_file is not None:
        try:
            # Lecture du fichier
            with st.spinner("📥 Chargement du fichier Excel..."):
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Fichier traité avec succès : {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Analyse
            analyser_clients_complet(df)
            
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier : {str(e)}")
            with st.expander("Détails de l'erreur"):
                st.exception(e)
    
    else:
        # Instructions d'utilisation
        st.info("👆 Veuillez charger un fichier Excel pour commencer l'analyse clients")
        
        st.markdown("""
        ### 📋 Colonnes recommandées dans votre fichier Excel :
        
        **🔹 Informations de base :**
        - Nom & Prénom, Email, Date de l'entretien, Date d'inscription, Conseiller
        
        **🔹 Données de souscription :**
        - Nb Souscriptions, Dernière Souscription, VR (Valeur Rachetable)
        
        **🔹 Profil professionnel :**
        - Métier, Secteur d'activité, Revenus, Type de contrat
        
        **🔹 Profil épargnant :**
        - Éligibilité, TMI, Profil épargnant, Épargne disponible
        
        **🔹 Situation personnelle :**
        - Situation familiale, Nb d'enfants
        
        ---
        
        ### 🚀 Fonctionnalités d'analyse disponibles :
        
        - **📊 Vue d'ensemble** : Statistiques globales et aperçu des données
        - **👨‍💼 Analyse conseillers** : Performance et répartition par conseiller
        - **🏢 Analyse secteurs** : Répartition par secteur d'activité
        - **💰 Analyse revenus** : Distribution et segmentation des revenus
        - **👨‍👩‍👧‍👦 Profils familiaux** : Situation familiale et enfants
        - **📅 Analyse temporelle** : Évolution des entretiens et inscriptions
        - **🎯 Segmentation** : Classification avancée des clients
        
        **L'analyse s'adapte automatiquement aux colonnes présentes dans votre fichier !**
        """)

if __name__ == "__main__":
    main()