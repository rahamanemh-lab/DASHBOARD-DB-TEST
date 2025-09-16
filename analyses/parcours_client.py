"""
Fonctions d'analyse du parcours client pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from utils.data_processing import safe_to_datetime, safe_to_numeric, adjust_dates_to_month_range, extract_conseiller
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link


def analyser_parcours_client(df_clients):
    """Analyse du parcours client.
    
    Args:
        df_clients (DataFrame): DataFrame contenant les données des clients
    """
    st.header("👣 Analyse du Parcours Client")
    
    # Vérification des données
    if df_clients is None or df_clients.empty:
        st.error("❌ Veuillez charger les données clients pour l'analyse du parcours client.")
        return
    
    # Vérification des colonnes requises
    colonnes_requises = ["Client", "Statut", "Date de création", "Dernière interaction"]
    colonnes_manquantes = [col for col in colonnes_requises if col not in df_clients.columns]
    if colonnes_manquantes:
        st.error(f"❌ Colonnes manquantes : {', '.join(colonnes_manquantes)}")
        st.write("Colonnes disponibles:", ", ".join(df_clients.columns))
        return
    
    # Prétraitement des données
    df_clients = extract_conseiller(df_clients)
    df_clients['Date de création'] = safe_to_datetime(df_clients['Date de création'])
    df_clients['Dernière interaction'] = safe_to_datetime(df_clients['Dernière interaction'])
    
    # Filtrer les données valides
    df_filtre = df_clients[df_clients['Date de création'].notna()]
    if df_filtre.empty:
        st.error("❌ Aucune donnée valide après filtrage.")
        return
    
    # Calculer la durée du parcours client
    df_filtre['Durée parcours (jours)'] = (df_filtre['Dernière interaction'] - df_filtre['Date de création']).dt.days
    
    # Remplacer les valeurs négatives ou NaN par 0
    df_filtre['Durée parcours (jours)'] = df_filtre['Durée parcours (jours)'].fillna(0)
    df_filtre.loc[df_filtre['Durée parcours (jours)'] < 0, 'Durée parcours (jours)'] = 0
    
    # Afficher les statistiques globales
    st.subheader("📈 Statistiques Globales")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nb_clients = len(df_filtre)
        st.metric("Nombre Total de Clients", nb_clients)
    
    with col2:
        duree_moyenne = df_filtre['Durée parcours (jours)'].mean()
        st.metric("Durée Moyenne du Parcours", f"{duree_moyenne:.1f} jours")
    
    with col3:
        nb_conseillers = df_filtre['Conseiller'].nunique()
        st.metric("Nombre de Conseillers", nb_conseillers)
    
    # Analyse par statut
    st.subheader("📊 Analyse par Statut")
    
    # Grouper par statut
    df_statut = df_filtre.groupby('Statut').agg(
        Nombre_Clients=('Client', 'count'),
        Duree_Moyenne=('Durée parcours (jours)', 'mean')
    ).reset_index()
    
    # Trier par nombre de clients décroissant
    df_statut = df_statut.sort_values('Nombre_Clients', ascending=False)
    
    # Créer le graphique en barres
    fig = px.bar(
        df_statut,
        x='Statut',
        y='Nombre_Clients',
        text='Nombre_Clients',
        title="Nombre de Clients par Statut",
        labels={
            'Statut': 'Statut',
            'Nombre_Clients': 'Nombre de Clients'
        },
        height=500,
        color='Duree_Moyenne',
        color_continuous_scale='Viridis',
        color_continuous_midpoint=df_statut['Duree_Moyenne'].median()
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        textposition='auto'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        coloraxis_colorbar=dict(
            title="Durée Moyenne (jours)"
        )
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau des données par statut
    st.write("### Détails par Statut")
    
    # Formater les colonnes pour l'affichage
    df_statut_display = df_statut.copy()
    df_statut_display['Duree_Moyenne'] = df_statut_display['Duree_Moyenne'].apply(lambda x: f"{x:.1f} jours")
    
    # Renommer les colonnes pour l'affichage
    df_statut_display = df_statut_display.rename(columns={
        'Statut': 'Statut',
        'Nombre_Clients': 'Nombre de Clients',
        'Duree_Moyenne': 'Durée Moyenne'
    })
    
    # Afficher le tableau
    st.write(df_statut_display)
    
    # Analyse par conseiller
    st.subheader("👨‍💼 Analyse par Conseiller")
    
    # Grouper par conseiller
    df_conseiller = df_filtre.groupby('Conseiller').agg(
        Nombre_Clients=('Client', 'count'),
        Duree_Moyenne=('Durée parcours (jours)', 'mean')
    ).reset_index()
    
    # Trier par nombre de clients décroissant
    df_conseiller = df_conseiller.sort_values('Nombre_Clients', ascending=False)
    
    # Créer le graphique en barres
    fig = px.bar(
        df_conseiller,
        x='Conseiller',
        y='Nombre_Clients',
        text='Nombre_Clients',
        title="Nombre de Clients par Conseiller",
        labels={
            'Conseiller': 'Conseiller',
            'Nombre_Clients': 'Nombre de Clients'
        },
        height=500,
        color='Duree_Moyenne',
        color_continuous_scale='Viridis',
        color_continuous_midpoint=df_conseiller['Duree_Moyenne'].median()
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        textposition='auto'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        coloraxis_colorbar=dict(
            title="Durée Moyenne (jours)"
        )
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau des données par conseiller
    st.write("### Détails par Conseiller")
    
    # Formater les colonnes pour l'affichage
    df_conseiller_display = df_conseiller.copy()
    df_conseiller_display['Duree_Moyenne'] = df_conseiller_display['Duree_Moyenne'].apply(lambda x: f"{x:.1f} jours")
    
    # Renommer les colonnes pour l'affichage
    df_conseiller_display = df_conseiller_display.rename(columns={
        'Conseiller': 'Conseiller',
        'Nombre_Clients': 'Nombre de Clients',
        'Duree_Moyenne': 'Durée Moyenne'
    })
    
    # Afficher le tableau
    st.write(df_conseiller_display)
    
    # Analyse de la distribution des durées de parcours
    st.subheader("📊 Distribution des Durées de Parcours")
    
    # Créer des tranches de durée
    bins = [0, 7, 14, 30, 60, 90, 180, 365, float('inf')]
    labels = ['0-7 jours', '7-14 jours', '14-30 jours', '1-2 mois', '2-3 mois', '3-6 mois', '6-12 mois', '> 12 mois']
    
    df_filtre['Tranche_Duree'] = pd.cut(df_filtre['Durée parcours (jours)'], bins=bins, labels=labels)
    
    # Grouper par tranche de durée
    df_duree = df_filtre.groupby('Tranche_Duree').size().reset_index(name='Nombre_Clients')
    
    # Créer le graphique en barres
    fig = px.bar(
        df_duree,
        x='Tranche_Duree',
        y='Nombre_Clients',
        text='Nombre_Clients',
        title="Distribution des Durées de Parcours Client",
        labels={
            'Tranche_Duree': 'Durée du Parcours',
            'Nombre_Clients': 'Nombre de Clients'
        },
        height=500,
        color='Nombre_Clients',
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
    
    # Analyse croisée statut-durée
    st.subheader("🔄 Analyse Croisée Statut-Durée")
    
    # Créer un tableau croisé dynamique
    pivot = pd.crosstab(
        index=df_filtre['Statut'],
        columns=df_filtre['Tranche_Duree'],
        margins=True,
        margins_name='Total'
    )
    
    # Trier par total décroissant
    pivot = pivot.sort_values('Total', ascending=False)
    
    # Afficher le tableau
    st.write("### Nombre de Clients par Statut et Durée de Parcours")
    st.write(pivot)
    
    # Exporter en CSV
    create_download_button(pivot, "parcours_client_statut_duree", "parcours_1")
    
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
    st.write("### Nombre de Clients par Conseiller et Statut")
    st.write(pivot)
    
    # Exporter en CSV
    create_download_button(pivot, "parcours_client_conseiller_statut", "parcours_2")
    
    # Filtrage des données
    st.subheader("🔍 Filtrage des Données")
    
    # Créer des filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        statuts = ['Tous'] + sorted(df_filtre['Statut'].unique().tolist())
        statut_filtre = st.selectbox("Statut", statuts)
    
    with col2:
        conseillers = ['Tous'] + sorted(df_filtre['Conseiller'].unique().tolist())
        conseiller_filtre = st.selectbox("Conseiller", conseillers)
    
    with col3:
        durees = ['Toutes'] + sorted(df_filtre['Tranche_Duree'].unique().tolist())
        duree_filtre = st.selectbox("Durée du Parcours", durees)
    
    # Appliquer les filtres
    df_filtered = df_filtre.copy()
    
    if statut_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['Statut'] == statut_filtre]
    
    if conseiller_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['Conseiller'] == conseiller_filtre]
    
    if duree_filtre != 'Toutes':
        df_filtered = df_filtered[df_filtered['Tranche_Duree'] == duree_filtre]
    
    # Afficher les résultats filtrés
    st.write(f"### Résultats ({len(df_filtered)} clients)")
    
    # Sélectionner les colonnes à afficher
    colonnes_affichage = ['Client', 'Conseiller', 'Statut', 'Date de création', 'Dernière interaction', 'Durée parcours (jours)', 'Tranche_Duree']
    df_filtered_display = df_filtered[colonnes_affichage]
    
    # Afficher le tableau
    st.write(df_filtered_display)
    
    # Exportation des données
    st.subheader("📤 Exportation des Données")
    create_download_button(df_filtered_display, "analyse_parcours_client", "parcours_3")
