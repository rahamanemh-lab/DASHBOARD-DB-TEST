"""
Fonctions d'analyse des dossiers immobiliers par groupe pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from utils.data_processing import safe_to_datetime, safe_to_numeric, adjust_dates_to_month_range, extract_conseiller
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link


# Définition des groupes
GROUPES_CONSEILLERS = {
    'IDR': ['Ikramah', 'Yassila'],
    'Mandataires': [
        'Hassan ISFOULA', 'Samuel BEAUMONT', 'Abdellah GOMES', 
        'Shafik BARHDADI', 'Karim BAHROUNE', 'Raouia BELAYADI',
        'Akram IBAR', 'Lahoucine AHARRA', 'Maxime DECOOPMAN', 'Myriam LATAR'
    ],
    'Sales internes': ['Aicha NAILI', 'Abdelkarim BOUTERA', 'Nissrine BEJAOUI']
}


def identifier_groupe_conseiller(conseiller_name):
    """
    Identifie le groupe d'un conseiller basé sur son nom.
    
    Args:
        conseiller_name (str): Nom du conseiller
        
    Returns:
        str: Nom du groupe ou 'Autre' si non trouvé
    """
    if pd.isna(conseiller_name) or conseiller_name == 'Inconnu':
        return 'Non défini'
    
    conseiller_name = str(conseiller_name).strip()
    
    # Recherche exacte d'abord
    for groupe, conseillers in GROUPES_CONSEILLERS.items():
        if conseiller_name in conseillers:
            return groupe
    
    # Recherche partielle (prénom/nom)
    for groupe, conseillers in GROUPES_CONSEILLERS.items():
        for conseiller in conseillers:
            # Vérifier si le nom contient une partie du conseiller ou vice versa
            if (conseiller_name.lower() in conseiller.lower() or 
                conseiller.lower() in conseiller_name.lower()):
                return groupe
    
    return 'Autre'


def analyser_groupes_dossiers_immo(df):
    """
    Analyse détaillée des dossiers immobiliers par groupe de conseillers.
    
    Args:
        df (DataFrame): DataFrame contenant les données immobilières
    """
    st.header("👥 Analyse par Groupe des Dossiers Immobiliers")
    
    # Vérification des données
    if df is None or df.empty:
        st.error("❌ Aucune donnée disponible pour l'analyse par groupe.")
        return
    
    # Extraction du conseiller avec la fonction améliorée
    df_with_conseiller = extract_conseiller(df.copy())
    
    if 'Conseiller' not in df_with_conseiller.columns:
        st.error("❌ Impossible d'extraire les informations de conseiller.")
        return
    
    # Ajouter la colonne groupe
    df_with_conseiller['Groupe'] = df_with_conseiller['Conseiller'].apply(identifier_groupe_conseiller)
    
    # Affichage des informations de débogage
    st.info("📋 Répartition des conseillers par groupe identifiée")
    
    # Créer des colonnes pour l'affichage
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Vue d'ensemble par groupe")
        
        # Statistiques générales par groupe
        stats_groupe = df_with_conseiller.groupby('Groupe').agg({
            'Conseiller': 'count',
            df_with_conseiller.select_dtypes(include=[np.number]).columns[0] if len(df_with_conseiller.select_dtypes(include=[np.number]).columns) > 0 else 'Conseiller': 'count'
        }).round(2)
        
        stats_groupe.columns = ['Nombre de dossiers', 'Total']
        if len(df_with_conseiller.select_dtypes(include=[np.number]).columns) > 0:
            montant_col = df_with_conseiller.select_dtypes(include=[np.number]).columns[0]
            stats_groupe = df_with_conseiller.groupby('Groupe').agg({
                'Conseiller': 'count',
                montant_col: ['sum', 'mean']
            }).round(2)
            stats_groupe.columns = ['Nombre de dossiers', 'Montant total', 'Montant moyen']
        
        st.dataframe(stats_groupe, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Répartition des groupes")
        
        # Graphique en secteurs
        groupe_counts = df_with_conseiller['Groupe'].value_counts()
        
        fig_pie = px.pie(
            values=groupe_counts.values,
            names=groupe_counts.index,
            title="Répartition des dossiers par groupe",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Analyse détaillée par conseiller dans chaque groupe
    st.subheader("📈 Performance détaillée par groupe")
    
    # Créer des onglets pour chaque groupe
    groupes_disponibles = df_with_conseiller['Groupe'].unique()
    if len(groupes_disponibles) > 1:
        groupe_tabs = st.tabs(list(groupes_disponibles))
        
        for i, groupe in enumerate(groupes_disponibles):
            with groupe_tabs[i]:
                analyser_groupe_specifique(df_with_conseiller, groupe)
    else:
        # Si un seul groupe, afficher directement
        analyser_groupe_specifique(df_with_conseiller, groupes_disponibles[0])
    
    # Comparaison inter-groupes
    st.subheader("⚖️ Comparaison inter-groupes")
    
    # Graphique de comparaison
    if len(df_with_conseiller.select_dtypes(include=[np.number]).columns) > 0:
        montant_col = df_with_conseiller.select_dtypes(include=[np.number]).columns[0]
        
        # Graphique en barres comparatif
        comparison_data = df_with_conseiller.groupby('Groupe').agg({
            'Conseiller': 'count',
            montant_col: 'sum'
        }).reset_index()
        
        fig_comparison = go.Figure()
        
        # Nombre de dossiers
        fig_comparison.add_trace(go.Bar(
            name='Nombre de dossiers',
            x=comparison_data['Groupe'],
            y=comparison_data['Conseiller'],
            yaxis='y',
            offsetgroup=1
        ))
        
        # Montant total (axe secondaire)
        fig_comparison.add_trace(go.Bar(
            name='Montant total',
            x=comparison_data['Groupe'],
            y=comparison_data[montant_col],
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig_comparison.update_layout(
            title='Comparaison des groupes : Volume vs Montant',
            xaxis_title='Groupe',
            yaxis=dict(title='Nombre de dossiers', side='left'),
            yaxis2=dict(title='Montant total (€)', side='right', overlaying='y'),
            barmode='group'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Export des données
    st.subheader("📥 Export des données")
    
    # Préparer les données pour l'export
    export_data = df_with_conseiller[['Conseiller', 'Groupe'] + 
                                   [col for col in df_with_conseiller.columns 
                                    if col not in ['Conseiller', 'Groupe']]]
    
    csv = export_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger l'analyse par groupe (CSV)",
        data=csv,
        file_name=f"analyse_groupes_immo_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def analyser_groupe_specifique(df, groupe_name):
    """
    Analyse spécifique d'un groupe de conseillers.
    
    Args:
        df (DataFrame): DataFrame avec les données et la colonne Groupe
        groupe_name (str): Nom du groupe à analyser
    """
    df_groupe = df[df['Groupe'] == groupe_name].copy()
    
    if df_groupe.empty:
        st.warning(f"Aucune donnée disponible pour le groupe {groupe_name}")
        return
    
    st.write(f"**Groupe : {groupe_name}** ({len(df_groupe)} dossiers)")
    
    # Statistiques par conseiller dans le groupe
    conseiller_stats = df_groupe.groupby('Conseiller').agg({
        'Conseiller': 'count'
    })
    conseiller_stats.columns = ['Nombre de dossiers']
    
    # Ajouter les montants si disponibles
    if len(df_groupe.select_dtypes(include=[np.number]).columns) > 0:
        montant_col = df_groupe.select_dtypes(include=[np.number]).columns[0]
        conseiller_stats = df_groupe.groupby('Conseiller').agg({
            'Conseiller': 'count',
            montant_col: ['sum', 'mean']
        }).round(2)
        conseiller_stats.columns = ['Nombre de dossiers', 'Montant total', 'Montant moyen']
    
    # Trier par nombre de dossiers décroissant
    conseiller_stats = conseiller_stats.sort_values('Nombre de dossiers', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(conseiller_stats, use_container_width=True)
    
    with col2:
        # Graphique des performances du groupe
        if len(conseiller_stats) > 1:
            fig_bar = px.bar(
                x=conseiller_stats.index,
                y=conseiller_stats['Nombre de dossiers'],
                title=f"Performance des conseillers - {groupe_name}",
                labels={'x': 'Conseiller', 'y': 'Nombre de dossiers'}
            )
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Un seul conseiller dans ce groupe")
    
    # Afficher les conseillers attendus vs trouvés
    if groupe_name in GROUPES_CONSEILLERS:
        conseillers_attendus = set(GROUPES_CONSEILLERS[groupe_name])
        conseillers_trouves = set(df_groupe['Conseiller'].unique())
        
        st.write("**Conseillers du groupe :**")
        col_attendus, col_trouves = st.columns(2)
        
        with col_attendus:
            st.write("*Attendus :*")
            for conseiller in conseillers_attendus:
                if conseiller in conseillers_trouves:
                    st.write(f"✅ {conseiller}")
                else:
                    st.write(f"❌ {conseiller} (non trouvé)")
        
        with col_trouves:
            st.write("*Trouvés dans les données :*")
            for conseiller in conseillers_trouves:
                if conseiller != 'Inconnu':
                    st.write(f"📊 {conseiller}")
