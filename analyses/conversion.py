"""
Fonctions d'analyse des conversions pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from utils.data_processing import safe_to_datetime, safe_to_numeric, adjust_dates_to_month_range, extract_conseiller
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link


def analyser_conversion(df_entretiens, df_rdv, df_souscriptions):
    """Analyse des taux de conversion entre entretiens, RDV et souscriptions.
    
    Args:
        df_entretiens (DataFrame): DataFrame contenant les données d'entretiens
        df_rdv (DataFrame): DataFrame contenant les données de RDV
        df_souscriptions (DataFrame): DataFrame contenant les données de souscriptions
    """
    st.header("🔄 Analyse des Conversions")
    
    # Vérification des données
    if df_entretiens is None or df_rdv is None or df_souscriptions is None:
        st.error("❌ Veuillez charger tous les fichiers nécessaires pour l'analyse des conversions.")
        return
    
    # Prétraitement des données
    df_entretiens = pretraiter_dataframe(df_entretiens, 'entretiens')
    df_rdv = pretraiter_dataframe(df_rdv, 'rdv')
    df_souscriptions = pretraiter_dataframe(df_souscriptions, 'souscriptions')
    
    # Vérifier si les DataFrames sont valides après prétraitement
    if df_entretiens.empty or df_rdv.empty or df_souscriptions.empty:
        st.error("❌ Données insuffisantes pour l'analyse des conversions après prétraitement.")
        return
    
    # Analyse des conversions globales
    st.subheader("📈 Conversions Globales")
    
    # Calculer les métriques globales
    nb_entretiens = len(df_entretiens)
    nb_rdv = len(df_rdv)
    nb_souscriptions = len(df_souscriptions)
    
    # Calculer les taux de conversion
    taux_entretien_rdv = (nb_rdv / nb_entretiens * 100) if nb_entretiens > 0 else 0
    taux_rdv_souscription = (nb_souscriptions / nb_rdv * 100) if nb_rdv > 0 else 0
    taux_entretien_souscription = (nb_souscriptions / nb_entretiens * 100) if nb_entretiens > 0 else 0
    
    # Afficher les métriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Taux Entretien → RDV", f"{taux_entretien_rdv:.1f}%")
    
    with col2:
        st.metric("Taux RDV → Souscription", f"{taux_rdv_souscription:.1f}%")
    
    with col3:
        st.metric("Taux Entretien → Souscription", f"{taux_entretien_souscription:.1f}%")
    
    # Créer un graphique de l'entonnoir de conversion
    fig = go.Figure()
    
    # Ajouter les barres pour chaque étape
    fig.add_trace(go.Bar(
        x=['Entretiens', 'RDV', 'Souscriptions'],
        y=[nb_entretiens, nb_rdv, nb_souscriptions],
        text=[nb_entretiens, nb_rdv, nb_souscriptions],
        textposition='auto',
        marker_color=['royalblue', 'green', 'orange']
    ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title="Entonnoir de Conversion",
        xaxis_title="Étape",
        yaxis_title="Nombre",
        template="plotly_white",
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des conversions par mois
    st.subheader("📅 Conversions par Mois")
    
    # Grouper par mois
    entretiens_mois = df_entretiens.groupby('Mois').size().reset_index(name='Nombre_Entretiens')
    rdv_mois = df_rdv.groupby('Mois').size().reset_index(name='Nombre_RDV')
    souscriptions_mois = df_souscriptions.groupby('Mois').size().reset_index(name='Nombre_Souscriptions')
    
    # Fusionner les DataFrames
    df_mois = entretiens_mois.merge(rdv_mois, on='Mois', how='outer')
    df_mois = df_mois.merge(souscriptions_mois, on='Mois', how='outer')
    
    # Remplir les valeurs manquantes
    df_mois = df_mois.fillna(0)
    
    # Trier par mois
    df_mois = df_mois.sort_values('Mois')
    
    # Calculer les taux de conversion par mois
    df_mois['Taux_Entretien_RDV'] = (df_mois['Nombre_RDV'] / df_mois['Nombre_Entretiens'] * 100).round(1)
    df_mois['Taux_RDV_Souscription'] = (df_mois['Nombre_Souscriptions'] / df_mois['Nombre_RDV'] * 100).round(1)
    df_mois['Taux_Entretien_Souscription'] = (df_mois['Nombre_Souscriptions'] / df_mois['Nombre_Entretiens'] * 100).round(1)
    
    # Remplacer les valeurs infinies ou NaN
    df_mois = df_mois.replace([np.inf, -np.inf], np.nan)
    df_mois = df_mois.fillna(0)
    
    # Créer le graphique des nombres par mois
    fig = go.Figure()
    
    # Ajouter les lignes pour chaque métrique
    fig.add_trace(go.Scatter(
        x=df_mois['Mois'],
        y=df_mois['Nombre_Entretiens'],
        mode='lines+markers',
        name='Entretiens',
        line=dict(color='royalblue', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_mois['Mois'],
        y=df_mois['Nombre_RDV'],
        mode='lines+markers',
        name='RDV',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_mois['Mois'],
        y=df_mois['Nombre_Souscriptions'],
        mode='lines+markers',
        name='Souscriptions',
        line=dict(color='orange', width=2),
        marker=dict(size=8)
    ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title="Évolution des Entretiens, RDV et Souscriptions par Mois",
        xaxis_title="Mois",
        yaxis_title="Nombre",
        legend_title="Légende",
        template="plotly_white",
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Créer le graphique des taux de conversion par mois
    fig = go.Figure()
    
    # Ajouter les lignes pour chaque taux
    fig.add_trace(go.Scatter(
        x=df_mois['Mois'],
        y=df_mois['Taux_Entretien_RDV'],
        mode='lines+markers',
        name='Taux Entretien → RDV',
        line=dict(color='royalblue', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_mois['Mois'],
        y=df_mois['Taux_RDV_Souscription'],
        mode='lines+markers',
        name='Taux RDV → Souscription',
        line=dict(color='green', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_mois['Mois'],
        y=df_mois['Taux_Entretien_Souscription'],
        mode='lines+markers',
        name='Taux Entretien → Souscription',
        line=dict(color='orange', width=2),
        marker=dict(size=8)
    ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title="Évolution des Taux de Conversion par Mois",
        xaxis_title="Mois",
        yaxis_title="Taux de Conversion (%)",
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
    df_mois_display['Taux_Entretien_RDV'] = df_mois_display['Taux_Entretien_RDV'].apply(lambda x: f"{x:.1f}%")
    df_mois_display['Taux_RDV_Souscription'] = df_mois_display['Taux_RDV_Souscription'].apply(lambda x: f"{x:.1f}%")
    df_mois_display['Taux_Entretien_Souscription'] = df_mois_display['Taux_Entretien_Souscription'].apply(lambda x: f"{x:.1f}%")
    
    # Renommer les colonnes pour l'affichage
    df_mois_display = df_mois_display.rename(columns={
        'Mois': 'Mois',
        'Nombre_Entretiens': "Nombre d'Entretiens",
        'Nombre_RDV': 'Nombre de RDV',
        'Nombre_Souscriptions': 'Nombre de Souscriptions',
        'Taux_Entretien_RDV': 'Taux Entretien → RDV',
        'Taux_RDV_Souscription': 'Taux RDV → Souscription',
        'Taux_Entretien_Souscription': 'Taux Entretien → Souscription'
    })
    
    # Afficher le tableau
    st.write(df_mois_display)
    
    # Exporter en CSV
    create_download_button(df_mois, "conversions_par_mois", "conversion_1")
    
    # Analyse des conversions par conseiller
    st.subheader("👨‍💼 Conversions par Conseiller")
    
    # Grouper par conseiller
    entretiens_conseiller = df_entretiens.groupby('Conseiller').size().reset_index(name='Nombre_Entretiens')
    rdv_conseiller = df_rdv.groupby('Conseiller').size().reset_index(name='Nombre_RDV')
    souscriptions_conseiller = df_souscriptions.groupby('Conseiller').size().reset_index(name='Nombre_Souscriptions')
    
    # Fusionner les DataFrames
    df_conseiller = entretiens_conseiller.merge(rdv_conseiller, on='Conseiller', how='outer')
    df_conseiller = df_conseiller.merge(souscriptions_conseiller, on='Conseiller', how='outer')
    
    # Remplir les valeurs manquantes
    df_conseiller = df_conseiller.fillna(0)
    
    # Calculer les taux de conversion par conseiller
    df_conseiller['Taux_Entretien_RDV'] = (df_conseiller['Nombre_RDV'] / df_conseiller['Nombre_Entretiens'] * 100).round(1)
    df_conseiller['Taux_RDV_Souscription'] = (df_conseiller['Nombre_Souscriptions'] / df_conseiller['Nombre_RDV'] * 100).round(1)
    df_conseiller['Taux_Entretien_Souscription'] = (df_conseiller['Nombre_Souscriptions'] / df_conseiller['Nombre_Entretiens'] * 100).round(1)
    
    # Remplacer les valeurs infinies ou NaN
    df_conseiller = df_conseiller.replace([np.inf, -np.inf], np.nan)
    df_conseiller = df_conseiller.fillna(0)
    
    # Trier par nombre d'entretiens décroissant
    df_conseiller = df_conseiller.sort_values('Nombre_Entretiens', ascending=False)
    
    # Créer le graphique des taux de conversion par conseiller
    fig = px.bar(
        df_conseiller,
        x='Conseiller',
        y='Taux_Entretien_Souscription',
        text='Taux_Entretien_Souscription',
        title="Taux de Conversion Entretien → Souscription par Conseiller",
        labels={
            'Conseiller': 'Conseiller',
            'Taux_Entretien_Souscription': 'Taux de Conversion (%)'
        },
        height=500,
        color='Taux_Entretien_Souscription',
        color_continuous_scale='Viridis'
    )
    
    # Mise en forme du graphique
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='auto'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher le tableau des données par conseiller
    st.write("### Détails par Conseiller")
    
    # Formater les colonnes pour l'affichage
    df_conseiller_display = df_conseiller.copy()
    df_conseiller_display['Taux_Entretien_RDV'] = df_conseiller_display['Taux_Entretien_RDV'].apply(lambda x: f"{x:.1f}%")
    df_conseiller_display['Taux_RDV_Souscription'] = df_conseiller_display['Taux_RDV_Souscription'].apply(lambda x: f"{x:.1f}%")
    df_conseiller_display['Taux_Entretien_Souscription'] = df_conseiller_display['Taux_Entretien_Souscription'].apply(lambda x: f"{x:.1f}%")
    
    # Renommer les colonnes pour l'affichage
    df_conseiller_display = df_conseiller_display.rename(columns={
        'Conseiller': 'Conseiller',
        'Nombre_Entretiens': "Nombre d'Entretiens",
        'Nombre_RDV': 'Nombre de RDV',
        'Nombre_Souscriptions': 'Nombre de Souscriptions',
        'Taux_Entretien_RDV': 'Taux Entretien → RDV',
        'Taux_RDV_Souscription': 'Taux RDV → Souscription',
        'Taux_Entretien_Souscription': 'Taux Entretien → Souscription'
    })
    
    # Afficher le tableau
    st.write(df_conseiller_display)
    
    # Exporter en CSV
    create_download_button(df_conseiller, "conversions_par_conseiller", "conversion_2")


def pretraiter_dataframe(df, type_df):
    """Prétraite un DataFrame pour l'analyse des conversions.
    
    Args:
        df (DataFrame): DataFrame à prétraiter
        type_df (str): Type de DataFrame ('entretiens', 'rdv', 'souscriptions')
        
    Returns:
        DataFrame: DataFrame prétraité
    """
    # Vérifier si le DataFrame est valide
    if df is None or df.empty:
        st.error(f"❌ Le DataFrame {type_df} est vide ou non valide.")
        return pd.DataFrame()
    
    # Extraire le conseiller
    df = extract_conseiller(df)
    
    # Vérifier si la colonne Conseiller existe
    if 'Conseiller' not in df.columns:
        st.error(f"❌ Impossible de trouver la colonne Conseiller dans le DataFrame {type_df}.")
        st.write("Colonnes disponibles:", ", ".join(df.columns))
        
        # Recherche intelligente de colonnes potentielles pour le conseiller
        colonnes_potentielles = [col for col in df.columns if any(mot in col.lower() for mot in ['conseiller', 'agent', 'commercial', 'vendeur', 'user', 'nom'])]
        if colonnes_potentielles:
            st.info(f"ℹ️ Colonnes potentielles pour le conseiller : {', '.join(colonnes_potentielles)}")
        
        return pd.DataFrame()
    
    # Vérifier si la colonne Date existe
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break
    
    if date_col is None:
        st.error(f"❌ Impossible de trouver une colonne de date dans le DataFrame {type_df}.")
        return pd.DataFrame()
    
    # Convertir la date
    df['Date'] = safe_to_datetime(df[date_col])
    
    # Filtrer les données valides
    df_filtre = df[df['Date'].notna()]
    if df_filtre.empty:
        st.error(f"❌ Aucune donnée valide après filtrage des dates dans le DataFrame {type_df}.")
        return pd.DataFrame()
    
    # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
    df_filtre = adjust_dates_to_month_range(df_filtre, 'Date')
    
    return df_filtre
