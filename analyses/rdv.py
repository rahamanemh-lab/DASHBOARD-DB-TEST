"""
Fonctions d'analyse des rendez-vous pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from utils.data_processing import safe_to_datetime, safe_to_numeric, adjust_dates_to_month_range, extract_conseiller
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link


def analyser_rdv(df):
    """Analyse des rendez-vous.
    
    Args:
        df (DataFrame): DataFrame contenant les données de rendez-vous
    """
    st.header("📅 Analyse des Rendez-vous")
    
    # Vérification si le DataFrame est None
    if df is None:
        st.error("❌ Veuillez charger un fichier de données de rendez-vous.")
        return
    
    # Vérification des colonnes requises
    colonnes_requises = ["Date", "Type", "Statut"]
    colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
    if colonnes_manquantes:
        st.error(f"❌ Colonnes manquantes : {', '.join(colonnes_manquantes)}")
        st.write("Colonnes disponibles:", ", ".join(df.columns))
        return
    
    # Prétraitement des données
    df = extract_conseiller(df)
    df['Date'] = safe_to_datetime(df['Date'])
    
    # Filtrer les données valides
    df_filtre = df[df['Date'].notna()]
    if df_filtre.empty:
        st.error("❌ Aucune donnée valide après filtrage.")
        return
    
    # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
    df_filtre = adjust_dates_to_month_range(df_filtre, 'Date')
    
    # Afficher les statistiques globales
    st.subheader("📈 Statistiques Globales")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nb_rdv = len(df_filtre)
        st.metric("Nombre Total de RDV", nb_rdv)
    
    with col2:
        nb_conseillers = df_filtre['Conseiller'].nunique()
        st.metric("Nombre de Conseillers", nb_conseillers)
    
    with col3:
        # Calculer le mois en cours
        mois_en_cours = datetime.now().strftime('%Y-%m')
        
        # Compter les RDV du mois en cours
        rdv_mois_en_cours = df_filtre[df_filtre['Mois'] == mois_en_cours].shape[0]
        
        # Calculer la moyenne mensuelle
        mois_uniques = df_filtre['Mois'].nunique()
        moyenne_mensuelle = nb_rdv / mois_uniques if mois_uniques > 0 else 0
        
        # Calculer la différence par rapport à la moyenne
        diff_pourcentage = ((rdv_mois_en_cours - moyenne_mensuelle) / moyenne_mensuelle * 100) if moyenne_mensuelle > 0 else 0
        
        st.metric(
            f"RDV du Mois ({mois_en_cours})",
            rdv_mois_en_cours,
            f"{diff_pourcentage:.1f}% par rapport à la moyenne"
        )
    
    # Analyse par mois
    st.subheader("📅 Analyse par Mois")
    
    # Grouper par mois
    df_mois = df_filtre.groupby('Mois').size().reset_index(name='Nombre_RDV')
    
    # Trier par mois
    df_mois = df_mois.sort_values('Mois')
    
    # Créer le graphique
    fig = px.bar(
        df_mois,
        x='Mois',
        y='Nombre_RDV',
        text='Nombre_RDV',
        title="Nombre de RDV par Mois",
        labels={
            'Mois': 'Mois',
            'Nombre_RDV': "Nombre de RDV"
        },
        height=500
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        textposition='auto',
        marker_color='royalblue'
    )
    
    fig.update_layout(
        template="plotly_white"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse par conseiller
    st.subheader("👨‍💼 Analyse par Conseiller")
    
    # Grouper par conseiller
    df_conseiller = df_filtre.groupby('Conseiller').size().reset_index(name='Nombre_RDV')
    
    # Trier par nombre de RDV décroissant
    df_conseiller = df_conseiller.sort_values('Nombre_RDV', ascending=False)
    
    # Créer le graphique
    fig = px.bar(
        df_conseiller,
        x='Conseiller',
        y='Nombre_RDV',
        text='Nombre_RDV',
        title="Nombre de RDV par Conseiller",
        labels={
            'Conseiller': 'Conseiller',
            'Nombre_RDV': "Nombre de RDV"
        },
        height=500,
        color='Nombre_RDV',
        color_continuous_scale='Viridis'
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        textposition='auto'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse par type de RDV
    st.subheader("🔄 Analyse par Type de RDV")
    
    # Grouper par type
    df_type = df_filtre.groupby('Type').size().reset_index(name='Nombre_RDV')
    
    # Trier par nombre de RDV décroissant
    df_type = df_type.sort_values('Nombre_RDV', ascending=False)
    
    # Créer le graphique en camembert
    fig = px.pie(
        df_type,
        values='Nombre_RDV',
        names='Type',
        title="Répartition des RDV par Type",
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
    
    # Analyse par statut
    st.subheader("📊 Analyse par Statut")
    
    # Grouper par statut
    df_statut = df_filtre.groupby('Statut').size().reset_index(name='Nombre_RDV')
    
    # Trier par nombre de RDV décroissant
    df_statut = df_statut.sort_values('Nombre_RDV', ascending=False)
    
    # Créer le graphique en camembert
    fig = px.pie(
        df_statut,
        values='Nombre_RDV',
        names='Statut',
        title="Répartition des RDV par Statut",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
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
    
    # Analyse par jour de la semaine
    st.subheader("📆 Analyse par Jour de la Semaine")
    
    # Ajouter le jour de la semaine
    df_filtre['Jour_Semaine'] = df_filtre['Date'].dt.day_name()
    
    # Définir l'ordre des jours de la semaine
    jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    jours_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    
    # Créer un dictionnaire de mapping
    mapping_jours = dict(zip(jours_ordre, jours_fr))
    
    # Appliquer le mapping
    df_filtre['Jour_Semaine_FR'] = df_filtre['Jour_Semaine'].map(mapping_jours)
    
    # Grouper par jour de la semaine
    df_jour = df_filtre.groupby('Jour_Semaine_FR').size().reset_index(name='Nombre_RDV')
    
    # Trier selon l'ordre des jours
    df_jour['Jour_Ordre'] = df_jour['Jour_Semaine_FR'].map(dict(zip(jours_fr, range(7))))
    df_jour = df_jour.sort_values('Jour_Ordre')
    
    # Créer le graphique
    fig = px.bar(
        df_jour,
        x='Jour_Semaine_FR',
        y='Nombre_RDV',
        text='Nombre_RDV',
        title="Nombre de RDV par Jour de la Semaine",
        labels={
            'Jour_Semaine_FR': 'Jour de la Semaine',
            'Nombre_RDV': "Nombre de RDV"
        },
        height=500,
        color='Nombre_RDV',
        color_continuous_scale='Viridis'
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        textposition='auto'
    )
    
    fig.update_layout(
        template="plotly_white"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse par heure de la journée
    st.subheader("🕒 Analyse par Heure de la Journée")
    
    # Ajouter l'heure de la journée
    df_filtre['Heure'] = df_filtre['Date'].dt.hour
    
    # Grouper par heure
    df_heure = df_filtre.groupby('Heure').size().reset_index(name='Nombre_RDV')
    
    # Trier par heure
    df_heure = df_heure.sort_values('Heure')
    
    # Créer le graphique
    fig = px.bar(
        df_heure,
        x='Heure',
        y='Nombre_RDV',
        text='Nombre_RDV',
        title="Nombre de RDV par Heure de la Journée",
        labels={
            'Heure': 'Heure',
            'Nombre_RDV': "Nombre de RDV"
        },
        height=500,
        color='Nombre_RDV',
        color_continuous_scale='Viridis'
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        textposition='auto'
    )
    
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        )
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse croisée conseiller-type
    st.subheader("🔄 Analyse Croisée Conseiller-Type")
    
    # Créer un tableau croisé dynamique
    pivot = pd.crosstab(
        index=df_filtre['Conseiller'],
        columns=df_filtre['Type'],
        margins=True,
        margins_name='Total'
    )
    
    # Trier par total décroissant
    pivot = pivot.sort_values('Total', ascending=False)
    
    # Afficher le tableau
    st.write("### Nombre de RDV par Conseiller et Type")
    st.write(pivot)
    
    # Exporter en CSV
    create_download_button(pivot, "rdv_conseiller_type", "rdv_1")
    
    # Analyse croisée conseiller-statut
    st.subheader("🔄 Analyse Croisée Conseiller-Statut")
    
    # Créer un tableau croisé dynamique
    pivot = pd.crosstab(
        index=df_filtre['Conseiller'],
        columns=df_filtre['Statut'],
        margins=True,
        margins_name='Total'
    )
    
    # Trier par total décroissant
    pivot = pivot.sort_values('Total', ascending=False)
    
    # Afficher le tableau
    st.write("### Nombre de RDV par Conseiller et Statut")
    st.write(pivot)
    
    # Exporter en CSV
    create_download_button(pivot, "rdv_conseiller_statut", "rdv_2")
    
    # Filtrage des données
    st.subheader("🔍 Filtrage des Données")
    
    # Créer des filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        types = ['Tous'] + sorted(df_filtre['Type'].unique().tolist())
        type_filtre = st.selectbox("Type de RDV", types)
    
    with col2:
        conseillers = ['Tous'] + sorted(df_filtre['Conseiller'].unique().tolist())
        conseiller_filtre = st.selectbox("Conseiller", conseillers)
    
    with col3:
        statuts = ['Tous'] + sorted(df_filtre['Statut'].unique().tolist())
        statut_filtre = st.selectbox("Statut", statuts)
    
    # Appliquer les filtres
    df_filtered = df_filtre.copy()
    
    if type_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['Type'] == type_filtre]
    
    if conseiller_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['Conseiller'] == conseiller_filtre]
    
    if statut_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['Statut'] == statut_filtre]
    
    # Afficher les résultats filtrés
    st.write(f"### Résultats ({len(df_filtered)} RDV)")
    st.write(df_filtered)
    
    # Exportation des données
    st.subheader("📤 Exportation des Données")
    create_download_button(df_filtered, "analyse_rdv", "rdv_3")
