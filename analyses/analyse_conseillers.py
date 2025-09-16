import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import os
import sys

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les fonctions utilitaires
from utils.data_processing import extract_conseiller, adjust_dates_to_month_range

# Définir les objectifs mensuels par type de conseiller
OBJECTIF_MENSUEL_SENIOR = 167000  # 167 000€ par mois pour un senior
OBJECTIF_MENSUEL_JUNIOR = 90000   # 90 000€ par mois pour un junior
OBJECTIF_ANNUEL_SENIOR = 2000000  # 2M€ pour 2025 pour un senior

# Définir les conseillers seniors et juniors
CONSEILLERS_SENIORS = [
    "Nissrine BEJAOUI", "Yassila LAMBATE", "Aicha NAILI", "Abdelkarim BOUTERA"
]
# Tous les autres conseillers sont considérés comme juniors

def analyser_performance_conseillers(df):
    """
    Analyse la performance des conseillers par rapport à leurs objectifs
    """
    st.title("📊 Analyse des Performances des Conseillers")
    
    if df.empty:
        st.warning("⚠️ Aucune donnée disponible pour l'analyse.")
        return
    
    # Préparation des données
    # S'assurer que la colonne Conseiller existe
    if 'Conseiller' not in df.columns:
        df['Conseiller'] = extract_conseiller(df)
        
    if 'Conseiller' not in df.columns or df['Conseiller'].isna().all():
        st.error("❌ Impossible de trouver ou d'extraire la colonne Conseiller dans les données.")
        return
    
    # S'assurer que la colonne Montant du placement existe
    if 'Montant du placement' not in df.columns:
        st.error("❌ Colonne 'Montant du placement' non trouvée dans les données.")
        return
    
    # S'assurer que la colonne Date de souscription existe
    if 'Date de souscription' not in df.columns:
        st.error("❌ Colonne 'Date de souscription' non trouvée dans les données.")
        return
    
    # Convertir la colonne Date de souscription en datetime
    df['Date de souscription'] = pd.to_datetime(df['Date de souscription'], errors='coerce')
    
    # Filtrer les lignes avec des dates valides
    df = df.dropna(subset=['Date de souscription'])
    
    # Ajouter une colonne pour le mois et l'année
    df['Mois'] = df['Date de souscription'].dt.strftime('%Y-%m')
    df['Année'] = df['Date de souscription'].dt.year
    
    # Ajouter une colonne pour le type de conseiller (Senior ou Junior)
    # Convertir la colonne Conseiller en string pour éviter les erreurs catégorielles
    df['Conseiller'] = df['Conseiller'].astype(str)
    df['Type Conseiller'] = df['Conseiller'].apply(lambda x: 'Senior' if x in CONSEILLERS_SENIORS else 'Junior')
    
    # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
    df = adjust_dates_to_month_range(df, 'Date de souscription')
    
    # Filtres pour l'analyse
    st.sidebar.header("📏 Filtres")
    
    # Filtre par année
    annees_disponibles = sorted(df['Année'].unique())
    annee_selectionnee = st.sidebar.selectbox(
        "Année",
        options=annees_disponibles,
        index=len(annees_disponibles)-1 if annees_disponibles else 0
    )
    
    # Filtre par mois
    mois_disponibles = sorted(df[df['Année'] == annee_selectionnee]['Date de souscription'].dt.month.unique())
    mois_options = [(str(m), calendar.month_name[m]) for m in mois_disponibles]
    mois_selectionne = st.sidebar.multiselect(
        "Mois",
        options=[m[0] for m in mois_options],
        default=[m[0] for m in mois_options],
        format_func=lambda x: dict(mois_options)[x]
    )
    
    # Filtre par type de conseiller
    type_conseiller = st.sidebar.multiselect(
        "Type de conseiller",
        options=['Senior', 'Junior', 'Tous'],
        default=['Tous']
    )
    
    # Filtrer les données en fonction des sélections
    df_filtre = df[df['Année'] == annee_selectionnee].copy()
    
    if mois_selectionne:
        df_filtre = df_filtre[df_filtre['Date de souscription'].dt.month.astype(str).isin(mois_selectionne)]
    
    if 'Tous' not in type_conseiller:
        df_filtre = df_filtre[df_filtre['Type Conseiller'].isin(type_conseiller)]
    
    # Afficher les informations sur les objectifs
    st.info("ℹ️ Objectifs mensuels: Seniors = 167 000€/mois (2M€ pour 2025), Juniors = 90 000€/mois")
    
    # Analyse des performances par conseiller
    st.header("📈 Performance par Conseiller")
    
    # Agrégation des données par conseiller
    performance_conseillers = df_filtre.groupby(['Conseiller', 'Type Conseiller']).agg(
        Montant_Total=('Montant du placement', 'sum'),
        Nombre_Souscriptions=('Montant du placement', 'count')
    ).reset_index()
    
    # Calculer l'objectif en fonction du type de conseiller et du nombre de mois sélectionnés
    nb_mois = len(mois_selectionne) if mois_selectionne else 0
    
    performance_conseillers['Objectif'] = performance_conseillers['Type Conseiller'].apply(
        lambda x: OBJECTIF_MENSUEL_SENIOR * nb_mois if x == 'Senior' else OBJECTIF_MENSUEL_JUNIOR * nb_mois
    )
    
    # Calculer l'écart par rapport à l'objectif
    performance_conseillers['Ecart'] = performance_conseillers['Montant_Total'] - performance_conseillers['Objectif']
    performance_conseillers['Pourcentage_Objectif'] = (performance_conseillers['Montant_Total'] / performance_conseillers['Objectif'] * 100).round(1)
    performance_conseillers['Statut'] = np.where(performance_conseillers['Ecart'] >= 0, '✅ Objectif Atteint', '❌ Sous Objectif')
    
    # Trier par montant décroissant
    performance_conseillers = performance_conseillers.sort_values('Montant_Total', ascending=False)
    
    # Créer le graphique de performance
    fig_performance = px.bar(
        performance_conseillers,
        x='Conseiller',
        y=['Montant_Total', 'Objectif'],
        barmode='group',
        color_discrete_map={'Montant_Total': '#1f77b4', 'Objectif': '#ff7f0e'},
        title=f"💰 Performance des Conseillers vs Objectifs ({annee_selectionnee})",
        labels={
            'value': 'Montant (€)',
            'variable': 'Mesure',
            'Conseiller': 'Conseiller'
        }
    )
    
    fig_performance.update_layout(
        height=500,
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_performance, use_container_width=True)
    
    # Tableau récapitulatif des performances
    st.subheader("📋 Tableau Récapitulatif des Performances")
    
    # Formater les colonnes pour l'affichage
    performance_display = performance_conseillers.copy()
    performance_display['Montant Total'] = performance_display['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
    performance_display['Objectif'] = performance_display['Objectif'].apply(lambda x: f"{x:,.0f}€")
    performance_display['Ecart'] = performance_display['Ecart'].apply(lambda x: f"{x:,.0f}€")
    performance_display['Pourcentage Objectif'] = performance_display['Pourcentage_Objectif'].apply(lambda x: f"{x:.1f}%")
    
    # Renommer les colonnes pour l'affichage
    performance_display = performance_display[['Conseiller', 'Type Conseiller', 'Montant Total', 'Objectif', 'Ecart', 'Pourcentage Objectif', 'Nombre_Souscriptions', 'Statut']]
    performance_display.columns = ['Conseiller', 'Type', 'Montant Total', 'Objectif', 'Écart', '% Objectif', 'Nb Souscriptions', 'Statut']
    
    st.dataframe(performance_display, use_container_width=True)
    
    # Téléchargement des données de performance
    csv_performance = performance_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données de performance (CSV)",
        data=csv_performance,
        file_name=f"performance_conseillers_{annee_selectionnee}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_performance"
    )
    
    # Analyse de l'évolution mensuelle par conseiller
    st.header("📉 Évolution Mensuelle par Conseiller")
    
    # Sélection du conseiller à analyser
    conseillers_disponibles = sorted(df_filtre['Conseiller'].unique())
    conseiller_selectionne = st.selectbox(
        "Sélectionnez un conseiller",
        options=conseillers_disponibles,
        key="conseiller_evolution"
    )
    
    if conseiller_selectionne:
        # Filtrer les données pour le conseiller sélectionné
        df_conseiller = df_filtre[df_filtre['Conseiller'] == conseiller_selectionne].copy()
        
        # Déterminer le type de conseiller et l'objectif mensuel
        type_conseiller_selectionne = 'Senior' if conseiller_selectionne in CONSEILLERS_SENIORS else 'Junior'
        objectif_mensuel = OBJECTIF_MENSUEL_SENIOR if type_conseiller_selectionne == 'Senior' else OBJECTIF_MENSUEL_JUNIOR
        
        # Agrégation des données par mois
        evolution_mensuelle = df_conseiller.groupby('Mois').agg(
            Montant_Total=('Montant du placement', 'sum'),
            Nombre_Souscriptions=('Montant du placement', 'count')
        ).reset_index()
        
        # Ajouter l'objectif mensuel
        evolution_mensuelle['Objectif'] = objectif_mensuel
        
        # Calculer l'écart par rapport à l'objectif
        evolution_mensuelle['Ecart'] = evolution_mensuelle['Montant_Total'] - evolution_mensuelle['Objectif']
        evolution_mensuelle['Pourcentage_Objectif'] = (evolution_mensuelle['Montant_Total'] / evolution_mensuelle['Objectif'] * 100).round(1)
        evolution_mensuelle['Statut'] = np.where(evolution_mensuelle['Ecart'] >= 0, '✅ Objectif Atteint', '❌ Sous Objectif')
        
        # Trier par mois
        evolution_mensuelle = evolution_mensuelle.sort_values('Mois')
        
        # Créer le graphique d'évolution mensuelle
        fig_evolution = px.bar(
            evolution_mensuelle,
            x='Mois',
            y=['Montant_Total', 'Objectif'],
            barmode='group',
            color_discrete_map={'Montant_Total': '#1f77b4', 'Objectif': '#ff7f0e'},
            title=f"💰 Évolution Mensuelle de {conseiller_selectionne} ({type_conseiller_selectionne})",
            labels={
                'value': 'Montant (€)',
                'variable': 'Mesure',
                'Mois': 'Mois'
            }
        )
        
        fig_evolution.update_layout(
            height=500,
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Tableau récapitulatif de l'évolution mensuelle
        st.subheader("📋 Évolution Mensuelle de " + conseiller_selectionne)
        
        # Formater les colonnes pour l'affichage
        evolution_display = evolution_mensuelle.copy()
        evolution_display['Montant Total'] = evolution_display['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
        evolution_display['Objectif'] = evolution_display['Objectif'].apply(lambda x: f"{x:,.0f}€")
        evolution_display['Ecart'] = evolution_display['Ecart'].apply(lambda x: f"{x:,.0f}€")
        evolution_display['Pourcentage Objectif'] = evolution_display['Pourcentage_Objectif'].apply(lambda x: f"{x:.1f}%")
        
        # Renommer les colonnes pour l'affichage
        evolution_display = evolution_display[['Mois', 'Montant Total', 'Objectif', 'Ecart', 'Pourcentage Objectif', 'Nombre_Souscriptions', 'Statut']]
        evolution_display.columns = ['Mois', 'Montant Total', 'Objectif', 'Écart', '% Objectif', 'Nb Souscriptions', 'Statut']
        
        st.dataframe(evolution_display, use_container_width=True)
        
        # Téléchargement des données d'évolution
        csv_evolution = evolution_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données d'évolution (CSV)",
            data=csv_evolution,
            file_name=f"evolution_mensuelle_{conseiller_selectionne}_{annee_selectionnee}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_evolution"
        )
        
        # Analyse du cumul annuel vs objectif annuel
        st.header("📈 Progression Annuelle vs Objectif")
        
        # Calculer le cumul progressif par mois
        evolution_mensuelle['Cumul'] = evolution_mensuelle['Montant_Total'].cumsum()
        
        # Calculer l'objectif cumulé
        mois_indices = {m: i+1 for i, m in enumerate(sorted(evolution_mensuelle['Mois'].unique()))}
        evolution_mensuelle['Mois_Index'] = evolution_mensuelle['Mois'].map(mois_indices)
        evolution_mensuelle['Objectif_Cumule'] = evolution_mensuelle['Mois_Index'] * objectif_mensuel
        
        # Calculer l'objectif annuel
        objectif_annuel = 12 * objectif_mensuel
        
        # Créer le graphique de progression annuelle
        fig_progression = go.Figure()
        
        # Ajouter la ligne de cumul réel
        fig_progression.add_trace(go.Scatter(
            x=evolution_mensuelle['Mois'],
            y=evolution_mensuelle['Cumul'],
            mode='lines+markers',
            name='Cumul Réel',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # Ajouter la ligne d'objectif cumulé
        fig_progression.add_trace(go.Scatter(
            x=evolution_mensuelle['Mois'],
            y=evolution_mensuelle['Objectif_Cumule'],
            mode='lines+markers',
            name='Objectif Cumulé',
            line=dict(color='#ff7f0e', width=3, dash='dash')
        ))
        
        # Ajouter une ligne horizontale pour l'objectif annuel
        fig_progression.add_hline(
            y=objectif_annuel,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Objectif Annuel: {objectif_annuel:,.0f}€"
        )
        
        fig_progression.update_layout(
            title=f"💰 Progression Annuelle de {conseiller_selectionne} vs Objectif ({annee_selectionnee})",
            xaxis_title="Mois",
            yaxis_title="Montant Cumulé (€)",
            height=500,
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig_progression, use_container_width=True)
        
        # Afficher le pourcentage de progression vers l'objectif annuel
        dernier_cumul = evolution_mensuelle['Cumul'].iloc[-1] if not evolution_mensuelle.empty else 0
        pourcentage_objectif_annuel = (dernier_cumul / objectif_annuel * 100).round(1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cumul Actuel", f"{dernier_cumul:,.0f}€")
        with col2:
            st.metric("Objectif Annuel", f"{objectif_annuel:,.0f}€")
        with col3:
            st.metric("Progression", f"{pourcentage_objectif_annuel}%")
    else:
        st.warning("⚠️ Aucun conseiller sélectionné ou disponible dans les données filtrées.")
    
    # Analyse comparative des types de conseillers
    st.header("📊 Analyse Comparative par Type de Conseiller")
    
    # Agrégation des données par type de conseiller
    performance_type = df_filtre.groupby('Type Conseiller').agg(
        Montant_Total=('Montant du placement', 'sum'),
        Nombre_Souscriptions=('Montant du placement', 'count'),
        Nombre_Conseillers=('Conseiller', 'nunique')
    ).reset_index()
    
    # Calculer le ticket moyen par type
    performance_type['Ticket_Moyen'] = performance_type['Montant_Total'] / performance_type['Nombre_Souscriptions']
    
    # Calculer la moyenne par conseiller
    performance_type['Moyenne_Par_Conseiller'] = performance_type['Montant_Total'] / performance_type['Nombre_Conseillers']
    
    # Créer le graphique comparatif
    fig_comparatif = px.bar(
        performance_type,
        x='Type Conseiller',
        y='Moyenne_Par_Conseiller',
        color='Type Conseiller',
        title=f"💰 Performance Moyenne par Type de Conseiller ({annee_selectionnee})",
        text='Moyenne_Par_Conseiller',
        labels={
            'Type Conseiller': 'Type de Conseiller',
            'Moyenne_Par_Conseiller': 'Montant Moyen par Conseiller (€)'
        }
    )
    
    fig_comparatif.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
    fig_comparatif.update_layout(height=500)
    
    st.plotly_chart(fig_comparatif, use_container_width=True)
    
    # Tableau récapitulatif par type de conseiller
    st.subheader("📋 Récapitulatif par Type de Conseiller")
    
    # Formater les colonnes pour l'affichage
    type_display = performance_type.copy()
    type_display['Montant Total'] = type_display['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
    type_display['Ticket Moyen'] = type_display['Ticket_Moyen'].apply(lambda x: f"{x:,.0f}€")
    type_display['Moyenne par Conseiller'] = type_display['Moyenne_Par_Conseiller'].apply(lambda x: f"{x:,.0f}€")
    
    # Renommer les colonnes pour l'affichage
    type_display = type_display[['Type Conseiller', 'Nombre_Conseillers', 'Montant Total', 'Nombre_Souscriptions', 'Ticket Moyen', 'Moyenne par Conseiller']]
    type_display.columns = ['Type de Conseiller', 'Nombre de Conseillers', 'Montant Total', 'Nombre de Souscriptions', 'Ticket Moyen', 'Moyenne par Conseiller']
    
    st.dataframe(type_display, use_container_width=True)
