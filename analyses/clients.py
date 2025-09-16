"""
Fonctions d'analyse des clients pour le dashboard.
Permet d'analyser les données clients chargées via un fichier Excel.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_processing import safe_to_datetime, safe_to_numeric, extract_conseiller
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link

def analyser_clients(df):
    """Analyse des données clients.
    
    Args:
        df (DataFrame): DataFrame contenant les données clients
        
    Structure attendue du fichier Excel:
        - Nom & Prénom
        - Email
        - Date de l'entretien
        - Nb Souscriptions
        - Dernière Souscription
        - VR (Valeur Rachetable?)
        - Métier
        - Secteur d'activité
        - Revenus
        - Type de contrat
        - Éligibilité
        - TMI (Tranche Marginale d'Imposition)
        - Profil épargnant
        - Épargne disponible
        - Situation familiale
        - Nb d'enfants
        - Date d'inscription
        - Conseiller
    """
    st.header("👥 Analyse des Clients")
    
    # Vérification si le DataFrame est None
    if df is None:
        st.error("❌ Veuillez charger un fichier de données clients.")
        return
    
    # Information sur les données chargées
    st.info(f"📊 Données chargées : {len(df)} clients avec {len(df.columns)} colonnes")
    
    # Afficher les premières colonnes pour validation
    with st.expander("🔍 Aperçu des données"):
        st.write("**Colonnes disponibles :**")
        cols_per_row = 3
        for i in range(0, len(df.columns), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(df.columns[i:i+cols_per_row]):
                if j < len(cols):
                    cols[j].write(f"• {col}")
        
        st.write("**Aperçu des données (5 premières lignes) :**")
        st.dataframe(df.head())
    
    # Préparation des données
    df_clients = df.copy()
    
    # Détecter et nettoyer les colonnes importantes
    colonnes_mapping = {
        'nom_prenom': ['Nom & Prénom', 'Nom', 'Prénom', 'Name'],
        'email': ['Email', 'E-mail', 'Mail'],
        'date_entretien': ['Date de l\'entretien', 'Date entretien', 'Date'],
        'nb_souscriptions': ['Nb Souscriptions', 'Nombre souscriptions', 'Souscriptions'],
        'derniere_souscription': ['Dernière Souscription', 'Last subscription'],
        'vr': ['VR', 'Valeur Rachetable'],
        'metier': ['Métier', 'Job', 'Profession'],
        'secteur': ['Secteur d\'activité', 'Secteur', 'Sector'],
        'revenus': ['Revenus', 'Revenue', 'Income'],
        'type_contrat': ['Type de contrat', 'Contrat', 'Contract'],
        'eligibilite': ['Éligibilité', 'Eligibilité', 'Eligibility'],
        'tmi': ['TMI', 'Tranche'],
        'profil_epargnant': ['Profil épargnant', 'Profil', 'Profile'],
        'epargne_disponible': ['Épargne disponible', 'Épargne', 'Savings'],
        'situation_familiale': ['Situation familiale', 'Situation', 'Family'],
        'nb_enfants': ['Nb d\'enfants', 'Nombre enfants', 'Children'],
        'date_inscription': ['Date d\'inscription', 'Date inscription', 'Registration'],
        'conseiller': ['Conseiller', 'Advisor', 'Agent']
    }
    
    # Mapper les colonnes
    colonnes_trouvees = {}
    for cle, possibles in colonnes_mapping.items():
        for possible in possibles:
            if possible in df_clients.columns:
                colonnes_trouvees[cle] = possible
                break
    
    # Afficher les colonnes mappées
    with st.expander("🔗 Mapping des colonnes"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Colonnes trouvées :**")
            for cle, colonne in colonnes_trouvees.items():
                st.write(f"• {cle}: `{colonne}`")
        
        with col2:
            st.write("**Colonnes manquantes :**")
            manquantes = [cle for cle in colonnes_mapping.keys() if cle not in colonnes_trouvees]
            for manquante in manquantes:
                st.write(f"• {manquante}")
    
    # Conversion des données
    if 'date_entretien' in colonnes_trouvees:
        df_clients['Date_Entretien_Clean'] = safe_to_datetime(df_clients[colonnes_trouvees['date_entretien']])
    
    if 'date_inscription' in colonnes_trouvees:
        df_clients['Date_Inscription_Clean'] = safe_to_datetime(df_clients[colonnes_trouvees['date_inscription']])
    
    if 'nb_souscriptions' in colonnes_trouvees:
        df_clients['Nb_Souscriptions_Clean'] = safe_to_numeric(df_clients[colonnes_trouvees['nb_souscriptions']])
    
    if 'revenus' in colonnes_trouvees:
        df_clients['Revenus_Clean'] = safe_to_numeric(df_clients[colonnes_trouvees['revenus']])
    
    if 'nb_enfants' in colonnes_trouvees:
        df_clients['Nb_Enfants_Clean'] = safe_to_numeric(df_clients[colonnes_trouvees['nb_enfants']])
    
    # Filtres globaux
    st.subheader("🔍 Filtres Globaux")
    col1, col2, col3 = st.columns(3)
    
    # Filtre par conseiller
    with col1:
        if 'conseiller' in colonnes_trouvees:
            conseillers = ['Tous'] + sorted(df_clients[colonnes_trouvees['conseiller']].dropna().unique().tolist())
            conseiller_filtre = st.selectbox("Conseiller", conseillers)
        else:
            conseiller_filtre = 'Tous'
    
    # Filtre par secteur d'activité
    with col2:
        if 'secteur' in colonnes_trouvees:
            secteurs = ['Tous'] + sorted(df_clients[colonnes_trouvees['secteur']].dropna().unique().tolist())
            secteur_filtre = st.selectbox("Secteur d'activité", secteurs)
        else:
            secteur_filtre = 'Tous'
    
    # Filtre par situation familiale
    with col3:
        if 'situation_familiale' in colonnes_trouvees:
            situations = ['Tous'] + sorted(df_clients[colonnes_trouvees['situation_familiale']].dropna().unique().tolist())
            situation_filtre = st.selectbox("Situation familiale", situations)
        else:
            situation_filtre = 'Tous'
    
    # Appliquer les filtres
    df_filtre = df_clients.copy()
    
    if conseiller_filtre != 'Tous' and 'conseiller' in colonnes_trouvees:
        df_filtre = df_filtre[df_filtre[colonnes_trouvees['conseiller']] == conseiller_filtre]
    
    if secteur_filtre != 'Tous' and 'secteur' in colonnes_trouvees:
        df_filtre = df_filtre[df_filtre[colonnes_trouvees['secteur']] == secteur_filtre]
    
    if situation_filtre != 'Tous' and 'situation_familiale' in colonnes_trouvees:
        df_filtre = df_filtre[df_filtre[colonnes_trouvees['situation_familiale']] == situation_filtre]
    
    # Statistiques globales
    st.subheader("📈 Statistiques Globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre total de clients", len(df_filtre))
    
    with col2:
        if 'Nb_Souscriptions_Clean' in df_filtre.columns:
            total_souscriptions = df_filtre['Nb_Souscriptions_Clean'].sum()
            st.metric("Total souscriptions", f"{total_souscriptions:,.0f}")
        else:
            st.metric("Total souscriptions", "N/A")
    
    with col3:
        if 'Nb_Souscriptions_Clean' in df_filtre.columns:
            clients_avec_souscriptions = len(df_filtre[df_filtre['Nb_Souscriptions_Clean'] > 0])
            st.metric("Clients avec souscriptions", clients_avec_souscriptions)
        else:
            st.metric("Clients avec souscriptions", "N/A")
    
    with col4:
        if 'Revenus_Clean' in df_filtre.columns:
            revenus_moyen = df_filtre['Revenus_Clean'].mean()
            st.metric("Revenus moyen", f"{revenus_moyen:,.0f} €")
        else:
            st.metric("Revenus moyen", "N/A")
    
    # Analyse par conseiller
    if 'conseiller' in colonnes_trouvees:
        st.subheader("👨‍💼 Analyse par Conseiller")
        
        conseiller_stats = df_filtre.groupby(colonnes_trouvees['conseiller']).agg({
            colonnes_trouvees['conseiller']: 'count',
            'Nb_Souscriptions_Clean': 'sum' if 'Nb_Souscriptions_Clean' in df_filtre.columns else lambda x: 0
        }).rename(columns={
            colonnes_trouvees['conseiller']: 'Nombre_Clients'
        }).reset_index()
        
        if 'Nb_Souscriptions_Clean' in df_filtre.columns:
            conseiller_stats = conseiller_stats.rename(columns={'Nb_Souscriptions_Clean': 'Total_Souscriptions'})
        else:
            conseiller_stats['Total_Souscriptions'] = 0
        
        # Graphique par conseiller
        fig = px.bar(
            conseiller_stats.sort_values('Nombre_Clients', ascending=False),
            x=colonnes_trouvees['conseiller'],
            y='Nombre_Clients',
            text='Nombre_Clients',
            title="Nombre de clients par conseiller",
            color='Total_Souscriptions',
            color_continuous_scale='Blues'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        st.write("**Détail par conseiller :**")
        st.dataframe(conseiller_stats.sort_values('Nombre_Clients', ascending=False))
    
    # Analyse par secteur d'activité
    if 'secteur' in colonnes_trouvees:
        st.subheader("🏢 Analyse par Secteur d'Activité")
        
        secteur_stats = df_filtre[colonnes_trouvees['secteur']].value_counts().reset_index()
        secteur_stats.columns = ['Secteur', 'Nombre_Clients']
        
        # Graphique en camembert
        fig = px.pie(
            secteur_stats,
            values='Nombre_Clients',
            names='Secteur',
            title="Répartition par secteur d'activité",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé
        st.write("**Répartition détaillée :**")
        secteur_stats['Pourcentage'] = (secteur_stats['Nombre_Clients'] / len(df_filtre) * 100).round(1)
        st.dataframe(secteur_stats)
    
    # Analyse des souscriptions
    if 'Nb_Souscriptions_Clean' in df_filtre.columns:
        st.subheader("💰 Analyse des Souscriptions")
        
        # Distribution des souscriptions
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramme
            fig = px.histogram(
                df_filtre,
                x='Nb_Souscriptions_Clean',
                title="Distribution du nombre de souscriptions",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistiques
            souscriptions_stats = df_filtre['Nb_Souscriptions_Clean'].describe()
            st.write("**Statistiques des souscriptions :**")
            st.write(f"• Moyenne : {souscriptions_stats['mean']:.1f}")
            st.write(f"• Médiane : {souscriptions_stats['50%']:.1f}")
            st.write(f"• Écart-type : {souscriptions_stats['std']:.1f}")
            st.write(f"• Maximum : {souscriptions_stats['max']:.0f}")
            
            # Top clients
            st.write("**Top 5 clients (souscriptions) :**")
            if 'nom_prenom' in colonnes_trouvees:
                top_clients = df_filtre.nlargest(5, 'Nb_Souscriptions_Clean')[
                    [colonnes_trouvees['nom_prenom'], 'Nb_Souscriptions_Clean']
                ]
            else:
                top_clients = df_filtre.nlargest(5, 'Nb_Souscriptions_Clean')[['Nb_Souscriptions_Clean']]
            st.dataframe(top_clients)
    
    # Analyse des revenus
    if 'Revenus_Clean' in df_filtre.columns:
        st.subheader("💵 Analyse des Revenus")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogramme des revenus
            fig = px.histogram(
                df_filtre,
                x='Revenus_Clean',
                title="Distribution des revenus",
                nbins=25
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boîte à moustaches
            fig = px.box(
                df_filtre,
                y='Revenus_Clean',
                title="Boîte à moustaches des revenus"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Segmentation par tranches de revenus
        st.write("**Segmentation par tranches de revenus :**")
        revenus_ranges = pd.cut(
            df_filtre['Revenus_Clean'].dropna(),
            bins=[0, 30000, 50000, 75000, 100000, float('inf')],
            labels=['< 30K€', '30-50K€', '50-75K€', '75-100K€', '> 100K€']
        )
        revenus_distribution = revenus_ranges.value_counts().reset_index()
        revenus_distribution.columns = ['Tranche_Revenus', 'Nombre_Clients']
        
        fig = px.bar(
            revenus_distribution,
            x='Tranche_Revenus',
            y='Nombre_Clients',
            text='Nombre_Clients',
            title="Répartition par tranches de revenus"
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(revenus_distribution)
    
    # Analyse familiale
    col1, col2 = st.columns(2)
    
    with col1:
        if 'situation_familiale' in colonnes_trouvees:
            st.subheader("👨‍👩‍👧‍👦 Situation Familiale")
            
            situation_stats = df_filtre[colonnes_trouvees['situation_familiale']].value_counts()
            fig = px.pie(
                values=situation_stats.values,
                names=situation_stats.index,
                title="Répartition par situation familiale"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Nb_Enfants_Clean' in df_filtre.columns:
            st.subheader("👶 Nombre d'Enfants")
            
            enfants_stats = df_filtre['Nb_Enfants_Clean'].value_counts().sort_index()
            fig = px.bar(
                x=enfants_stats.index,
                y=enfants_stats.values,
                title="Répartition par nombre d'enfants"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Analyse temporelle
    if 'Date_Entretien_Clean' in df_filtre.columns or 'Date_Inscription_Clean' in df_filtre.columns:
        st.subheader("📅 Analyse Temporelle")
        
        tab1, tab2 = st.tabs(["Entretiens", "Inscriptions"])
        
        with tab1:
            if 'Date_Entretien_Clean' in df_filtre.columns:
                df_entretiens = df_filtre.dropna(subset=['Date_Entretien_Clean'])
                if not df_entretiens.empty:
                    # Grouper par mois
                    df_entretiens['Mois'] = df_entretiens['Date_Entretien_Clean'].dt.to_period('M')
                    entretiens_mois = df_entretiens.groupby('Mois').size().reset_index(name='Nombre_Entretiens')
                    entretiens_mois['Mois'] = entretiens_mois['Mois'].astype(str)
                    
                    fig = px.line(
                        entretiens_mois,
                        x='Mois',
                        y='Nombre_Entretiens',
                        title="Évolution des entretiens par mois",
                        markers=True
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucune date d'entretien valide trouvée")
        
        with tab2:
            if 'Date_Inscription_Clean' in df_filtre.columns:
                df_inscriptions = df_filtre.dropna(subset=['Date_Inscription_Clean'])
                if not df_inscriptions.empty:
                    # Grouper par mois
                    df_inscriptions['Mois'] = df_inscriptions['Date_Inscription_Clean'].dt.to_period('M')
                    inscriptions_mois = df_inscriptions.groupby('Mois').size().reset_index(name='Nombre_Inscriptions')
                    inscriptions_mois['Mois'] = inscriptions_mois['Mois'].astype(str)
                    
                    fig = px.line(
                        inscriptions_mois,
                        x='Mois',
                        y='Nombre_Inscriptions',
                        title="Évolution des inscriptions par mois",
                        markers=True
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucune date d'inscription valide trouvée")
    
    # Analyse de corrélation
    if all(col in df_filtre.columns for col in ['Revenus_Clean', 'Nb_Souscriptions_Clean', 'Nb_Enfants_Clean']):
        st.subheader("🔗 Analyse de Corrélation")
        
        # Matrice de corrélation
        colonnes_numeriques = ['Revenus_Clean', 'Nb_Souscriptions_Clean', 'Nb_Enfants_Clean']
        correlation_matrix = df_filtre[colonnes_numeriques].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Matrice de corrélation",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique de dispersion
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df_filtre,
                x='Revenus_Clean',
                y='Nb_Souscriptions_Clean',
                title="Revenus vs Nombre de souscriptions"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df_filtre,
                x='Nb_Enfants_Clean',
                y='Nb_Souscriptions_Clean',
                title="Nombre d'enfants vs Nombre de souscriptions"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Profiling des clients
    st.subheader("🎯 Profiling des Clients")
    
    # Segmentation avancée
    segments = []
    
    if 'Nb_Souscriptions_Clean' in df_filtre.columns and 'Revenus_Clean' in df_filtre.columns:
        df_segment = df_filtre.dropna(subset=['Nb_Souscriptions_Clean', 'Revenus_Clean'])
        
        # Définir les seuils
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
            title="Segmentation des clients",
            size_max=20
        )
        
        # Ajouter les lignes de seuil
        fig.add_hline(y=seuil_souscriptions, line_dash="dash", line_color="gray")
        fig.add_vline(x=seuil_revenus, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques par segment
        segment_stats = df_segment.groupby('Segment').agg({
            'Segment': 'count',
            'Revenus_Clean': 'mean',
            'Nb_Souscriptions_Clean': 'mean'
        }).rename(columns={'Segment': 'Nombre_Clients'}).reset_index()
        
        st.write("**Statistiques par segment :**")
        segment_stats['Revenus_Clean'] = segment_stats['Revenus_Clean'].apply(lambda x: f"{x:,.0f} €")
        segment_stats['Nb_Souscriptions_Clean'] = segment_stats['Nb_Souscriptions_Clean'].apply(lambda x: f"{x:.1f}")
        st.dataframe(segment_stats)
    
    # Exportation des données
    st.subheader("📤 Exportation des Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Données filtrées")
        create_download_button(df_filtre, "clients_filtres", "clients_analyse")
    
    with col2:
        st.write("### Rapport complet")
        # Pour l'instant, exporter juste les données
        # TODO: Ajouter la génération PDF si nécessaire
        create_download_button(df_filtre, "rapport_clients_complet", "clients_rapport")
    
    # Données détaillées
    st.subheader("📃 Données Détaillées")
    
    with st.expander("Voir toutes les données"):
        st.dataframe(df_filtre, use_container_width=True)
    
    # Résumé final
    st.success(f"✅ Analyse terminée pour {len(df_filtre)} clients")