"""
Fonctions d'analyse des paiements pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from utils.data_processing import safe_to_datetime, safe_to_numeric, adjust_dates_to_month_range, extract_conseiller
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link


def analyser_paiements(df_paiements):
    """Analyse des paiements.
    
    Args:
        df_paiements (DataFrame): DataFrame contenant les données de paiements
    """
    st.header("💰 Analyse des Paiements")
    
    # Vérification des données
    if df_paiements is None or df_paiements.empty:
        st.error("❌ Veuillez charger les données de paiements pour l'analyse.")
        return
    
    # Vérification des colonnes requises
    colonnes_requises = ["Date", "Montant", "Statut", "Client"]
    colonnes_manquantes = [col for col in colonnes_requises if col not in df_paiements.columns]
    if colonnes_manquantes:
        st.error(f"❌ Colonnes manquantes : {', '.join(colonnes_manquantes)}")
        st.write("Colonnes disponibles:", ", ".join(df_paiements.columns))
        
        # Permettre à l'utilisateur de sélectionner manuellement les colonnes manquantes
        st.subheader("Sélection manuelle des colonnes")
        col1, col2 = st.columns(2)
        
        date_col = "Date"
        if "Date" in colonnes_manquantes:
            with col1:
                manual_date_col = st.selectbox("Colonne Date", options=["-- Sélectionner --"] + list(df_paiements.columns), 
                                              key="paiements_manual_date_col")
                if manual_date_col != "-- Sélectionner --":
                    date_col = manual_date_col
                    st.success(f"Colonne de date définie manuellement: {date_col}")
                else:
                    return
        
        montant_col = "Montant"
        if "Montant" in colonnes_manquantes:
            with col2:
                manual_montant_col = st.selectbox("Colonne Montant", options=["-- Sélectionner --"] + list(df_paiements.columns), 
                                                 key="paiements_manual_montant_col")
                if manual_montant_col != "-- Sélectionner --":
                    montant_col = manual_montant_col
                    st.success(f"Colonne de montant définie manuellement: {montant_col}")
                else:
                    return
        
        statut_col = "Statut"
        if "Statut" in colonnes_manquantes:
            with col1:
                manual_statut_col = st.selectbox("Colonne Statut", options=["-- Sélectionner --"] + list(df_paiements.columns), 
                                                key="paiements_manual_statut_col")
                if manual_statut_col != "-- Sélectionner --":
                    statut_col = manual_statut_col
                    st.success(f"Colonne de statut définie manuellement: {statut_col}")
                else:
                    return
        
        client_col = "Client"
        if "Client" in colonnes_manquantes:
            with col2:
                manual_client_col = st.selectbox("Colonne Client", options=["-- Sélectionner --"] + list(df_paiements.columns), 
                                                key="paiements_manual_client_col")
                if manual_client_col != "-- Sélectionner --":
                    client_col = manual_client_col
                    st.success(f"Colonne de client définie manuellement: {client_col}")
                else:
                    return
        
        # Remplacer les noms de colonnes par ceux sélectionnés manuellement
        df_paiements = df_paiements.rename(columns={
            date_col: "Date" if date_col != "Date" else date_col,
            montant_col: "Montant" if montant_col != "Montant" else montant_col,
            statut_col: "Statut" if statut_col != "Statut" else statut_col,
            client_col: "Client" if client_col != "Client" else client_col
        })
    
    # Prétraitement des données
    df_paiements = extract_conseiller(df_paiements)
    df_paiements['Date'] = safe_to_datetime(df_paiements['Date'])
    df_paiements['Montant'] = safe_to_numeric(df_paiements['Montant'])
    
    # Filtrer les données valides
    df_filtre = df_paiements[(df_paiements['Date'].notna()) & (df_paiements['Montant'] > 0)]
    if df_filtre.empty:
        st.error("❌ Aucune donnée valide après filtrage.")
        return
    
    # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
    df_filtre = adjust_dates_to_month_range(df_filtre, 'Date')
    
    # Afficher les statistiques globales
    st.subheader("📈 Statistiques Globales")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        montant_total = df_filtre['Montant'].sum()
        st.metric("Montant Total", f"{montant_total:,.0f} €")
    
    with col2:
        nb_paiements = len(df_filtre)
        st.metric("Nombre de Paiements", nb_paiements)
    
    with col3:
        montant_moyen = df_filtre['Montant'].mean()
        st.metric("Montant Moyen", f"{montant_moyen:,.0f} €")
    
    with col4:
        nb_clients = df_filtre['Client'].nunique()
        st.metric("Nombre de Clients", nb_clients)
    
    # Analyse par mois
    st.subheader("📅 Analyse par Mois")
    
    # Grouper par mois
    df_mois = df_filtre.groupby('Mois').agg(
        Montant_Total=('Montant', 'sum'),
        Nombre_Paiements=('Montant', 'count'),
        Montant_Moyen=('Montant', 'mean')
    ).reset_index()
    
    # Trier par mois
    df_mois = df_mois.sort_values('Mois')
    
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
    
    # Ajouter la ligne pour le nombre de paiements
    fig.add_trace(go.Scatter(
        x=df_mois['Mois'],
        y=df_mois['Nombre_Paiements'],
        mode='lines+markers',
        name='Nombre de Paiements',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title="Montant Total et Nombre de Paiements par Mois",
        xaxis_title="Mois",
        yaxis_title="Montant (€)",
        yaxis2=dict(
            title="Nombre de Paiements",
            overlaying='y',
            side='right'
        ),
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
    df_mois_display['Montant_Moyen'] = df_mois_display['Montant_Moyen'].apply(lambda x: f"{x:,.0f} €")
    
    # Renommer les colonnes pour l'affichage
    df_mois_display = df_mois_display.rename(columns={
        'Mois': 'Mois',
        'Montant_Total': 'Montant Total',
        'Nombre_Paiements': 'Nombre de Paiements',
        'Montant_Moyen': 'Montant Moyen'
    })
    
    # Afficher le tableau
    st.write(df_mois_display)
    
    # Analyse par conseiller
    st.subheader("👨‍💼 Analyse par Conseiller")
    
    # Grouper par conseiller
    df_conseiller = df_filtre.groupby('Conseiller').agg(
        Montant_Total=('Montant', 'sum'),
        Nombre_Paiements=('Montant', 'count'),
        Montant_Moyen=('Montant', 'mean'),
        Nombre_Clients=('Client', 'nunique')
    ).reset_index()
    
    # Trier par montant total décroissant
    df_conseiller = df_conseiller.sort_values('Montant_Total', ascending=False)
    
    # Créer le graphique
    fig = px.bar(
        df_conseiller,
        x='Conseiller',
        y='Montant_Total',
        text='Montant_Total',
        title="Montant Total des Paiements par Conseiller",
        labels={
            'Conseiller': 'Conseiller',
            'Montant_Total': 'Montant Total (€)'
        },
        height=500,
        color='Nombre_Paiements',
        color_continuous_scale='Viridis'
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        texttemplate='%{text:,.0f} €',
        textposition='auto'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        coloraxis_colorbar=dict(
            title="Nombre de Paiements"
        )
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau des données par conseiller
    st.write("### Détails par Conseiller")
    
    # Formater les colonnes pour l'affichage
    df_conseiller_display = df_conseiller.copy()
    df_conseiller_display['Montant_Total'] = df_conseiller_display['Montant_Total'].apply(lambda x: f"{x:,.0f} €")
    df_conseiller_display['Montant_Moyen'] = df_conseiller_display['Montant_Moyen'].apply(lambda x: f"{x:,.0f} €")
    
    # Renommer les colonnes pour l'affichage
    df_conseiller_display = df_conseiller_display.rename(columns={
        'Conseiller': 'Conseiller',
        'Montant_Total': 'Montant Total',
        'Nombre_Paiements': 'Nombre de Paiements',
        'Montant_Moyen': 'Montant Moyen',
        'Nombre_Clients': 'Nombre de Clients'
    })
    
    # Afficher le tableau
    st.write(df_conseiller_display)
    
    # Analyse par statut
    st.subheader("📊 Analyse par Statut")
    
    # Grouper par statut
    df_statut = df_filtre.groupby('Statut').agg(
        Montant_Total=('Montant', 'sum'),
        Nombre_Paiements=('Montant', 'count'),
        Montant_Moyen=('Montant', 'mean')
    ).reset_index()
    
    # Trier par montant total décroissant
    df_statut = df_statut.sort_values('Montant_Total', ascending=False)
    
    # Créer le graphique en camembert
    fig = px.pie(
        df_statut,
        values='Montant_Total',
        names='Statut',
        title="Répartition du Montant Total par Statut",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+value+percent',
        hovertemplate='%{label}<br>Montant: %{value:,.0f} €<br>Pourcentage: %{percent}'
    )
    
    fig.update_layout(
        template="plotly_white",
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau des données par statut
    st.write("### Détails par Statut")
    
    # Formater les colonnes pour l'affichage
    df_statut_display = df_statut.copy()
    df_statut_display['Montant_Total'] = df_statut_display['Montant_Total'].apply(lambda x: f"{x:,.0f} €")
    df_statut_display['Montant_Moyen'] = df_statut_display['Montant_Moyen'].apply(lambda x: f"{x:,.0f} €")
    
    # Renommer les colonnes pour l'affichage
    df_statut_display = df_statut_display.rename(columns={
        'Statut': 'Statut',
        'Montant_Total': 'Montant Total',
        'Nombre_Paiements': 'Nombre de Paiements',
        'Montant_Moyen': 'Montant Moyen'
    })
    
    # Afficher le tableau
    st.write(df_statut_display)
    
    # Analyse de la distribution des montants
    st.subheader("📊 Distribution des Montants")
    
    # Créer des tranches de montant
    bins = [0, 100, 500, 1000, 5000, 10000, 50000, float('inf')]
    labels = ['0-100 €', '100-500 €', '500-1000 €', '1000-5000 €', '5000-10000 €', '10000-50000 €', '> 50000 €']
    
    df_filtre['Tranche_Montant'] = pd.cut(df_filtre['Montant'], bins=bins, labels=labels)
    
    # Grouper par tranche de montant
    df_tranche = df_filtre.groupby('Tranche_Montant').agg(
        Montant_Total=('Montant', 'sum'),
        Nombre_Paiements=('Montant', 'count')
    ).reset_index()
    
    # Créer le graphique en barres
    fig = px.bar(
        df_tranche,
        x='Tranche_Montant',
        y='Nombre_Paiements',
        text='Nombre_Paiements',
        title="Distribution des Montants de Paiement",
        labels={
            'Tranche_Montant': 'Tranche de Montant',
            'Nombre_Paiements': 'Nombre de Paiements'
        },
        height=500,
        color='Montant_Total',
        color_continuous_scale='Viridis'
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        textposition='auto'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        coloraxis_colorbar=dict(
            title="Montant Total (€)"
        )
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse croisée conseiller-statut
    st.subheader("🔄 Analyse Croisée Conseiller-Statut")
    
    # Créer un tableau croisé dynamique pour le nombre de paiements
    pivot_nombre = pd.crosstab(
        index=df_filtre['Conseiller'],
        columns=df_filtre['Statut'],
        values=df_filtre['Montant'],
        aggfunc='count',
        margins=True,
        margins_name='Total'
    )
    
    # Trier par total décroissant
    pivot_nombre = pivot_nombre.sort_values('Total', ascending=False)
    
    # Afficher le tableau
    st.write("### Nombre de Paiements par Conseiller et Statut")
    st.write(pivot_nombre)
    
    # Exporter en CSV
    create_download_button(pivot_nombre, "paiements_nombre_conseiller_statut", "paiements_1")
    
    # Créer un tableau croisé dynamique pour le montant total
    pivot_montant = pd.crosstab(
        index=df_filtre['Conseiller'],
        columns=df_filtre['Statut'],
        values=df_filtre['Montant'],
        aggfunc='sum',
        margins=True,
        margins_name='Total'
    )
    
    # Trier par total décroissant
    pivot_montant = pivot_montant.sort_values('Total', ascending=False)
    
    # Formater les montants
    pivot_montant_display = pivot_montant.applymap(lambda x: f"{x:,.0f} €" if pd.notna(x) else "")
    
    # Afficher le tableau
    st.write("### Montant Total des Paiements par Conseiller et Statut")
    st.write(pivot_montant_display)
    
    # Exporter en CSV
    create_download_button(pivot_montant, "paiements_montant_conseiller_statut", "paiements_2")
    
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
        tranches = ['Toutes'] + sorted(df_filtre['Tranche_Montant'].unique().tolist())
        tranche_filtre = st.selectbox("Tranche de Montant", tranches)
    
    # Appliquer les filtres
    df_filtered = df_filtre.copy()
    
    if statut_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['Statut'] == statut_filtre]
    
    if conseiller_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['Conseiller'] == conseiller_filtre]
    
    if tranche_filtre != 'Toutes':
        df_filtered = df_filtered[df_filtered['Tranche_Montant'] == tranche_filtre]
    
    # Afficher les résultats filtrés
    st.write(f"### Résultats ({len(df_filtered)} paiements)")
    
    # Sélectionner les colonnes à afficher
    colonnes_affichage = ['Date', 'Client', 'Conseiller', 'Montant', 'Statut', 'Tranche_Montant']
    df_filtered_display = df_filtered[colonnes_affichage]
    
    # Formater les montants
    df_filtered_display['Montant'] = df_filtered_display['Montant'].apply(lambda x: f"{x:,.0f} €")
    
    # Afficher le tableau
    st.write(df_filtered_display)
    
    # Exportation des données
    st.subheader("📤 Exportation des Données")
    create_download_button(df_filtered, "analyse_paiements", "paiements_3")
