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
from utils.data_processing import safe_to_datetime, safe_to_numeric, extract_conseiller

def analyser_paiements_immo(df):
    """Analyse des paiements immobiliers."""
    st.header("💰 Analyse des Paiements Immobiliers")
    
    # Vérification des colonnes nécessaires
    colonnes_requises = ['Statut', 'Montant', 'Date de paiement']
    colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
    
    if colonnes_manquantes:
        st.error(f"❌ Colonnes manquantes pour l'analyse des paiements immobiliers: {', '.join(colonnes_manquantes)}")
        with st.expander("Colonnes disponibles"):
            st.write(df.columns.tolist())
        return
    
    # Préparation des données
    df_immo = df.copy()
    df_immo['Date de paiement'] = safe_to_datetime(df_immo['Date de paiement'])
    df_immo['Mois'] = df_immo['Date de paiement'].dt.to_period('M').astype(str)
    df_immo['Montant'] = safe_to_numeric(df_immo['Montant'])
    df_immo = extract_conseiller(df_immo)
    
    # Ajouter les colonnes Premier_Jour_Mois et Dernier_Jour_Mois pour l'analyse temporelle
    if 'Date de paiement' in df_immo.columns:
        # Créer la colonne Premier_Jour_Mois (premier jour du mois)
        df_immo['Premier_Jour_Mois'] = df_immo['Date de paiement'].dt.to_period('M').dt.to_timestamp()
        # Créer la colonne Dernier_Jour_Mois (dernier jour du mois)
        df_immo['Dernier_Jour_Mois'] = (df_immo['Premier_Jour_Mois'] + pd.DateOffset(months=1) - pd.DateOffset(days=1))
    
    # Analyse des statuts de paiement
    statuts_disponibles = df_immo['Statut'].dropna().unique().tolist()
    
    # Détection automatique des statuts "en attente" et "validé"
    statuts_attente = [s for s in statuts_disponibles if any(mot in s.lower() for mot in ['attente', 'pending', 'wait', 'progress'])]
    statuts_valides = [s for s in statuts_disponibles if any(mot in s.lower() for mot in ['valid', 'complet', 'confirm', 'done', 'finish', 'success'])]
    
    # Afficher les statuts détectés
    st.subheader("🔍 Analyse des Statuts de Paiement")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Statuts détectés:**")
        for statut in statuts_disponibles:
            if statut in statuts_attente:
                st.write(f"- {statut} ⏳ (En attente)")
            elif statut in statuts_valides:
                st.write(f"- {statut} ✅ (Validé)")
            else:
                st.write(f"- {statut} ❓ (Non classifié)")
    
    with col2:
        # Option pour sélectionner manuellement les statuts
        st.write("**Sélection manuelle des statuts:**")
        statuts_attente_selection = st.multiselect(
            "Statuts 'En attente':",
            options=statuts_disponibles,
            default=statuts_attente,
            key="statuts_attente_immo"
        )
        
        statuts_valides_selection = st.multiselect(
            "Statuts 'Validés':",
            options=statuts_disponibles,
            default=statuts_valides,
            key="statuts_valides_immo"
        )
        
        # Utiliser les sélections manuelles si elles sont différentes des détections automatiques
        if set(statuts_attente_selection) != set(statuts_attente) or set(statuts_valides_selection) != set(statuts_valides):
            statuts_attente = statuts_attente_selection
            statuts_valides = statuts_valides_selection
    
    # Créer des DataFrames filtrés
    df_attente = df_immo[df_immo['Statut'].isin(statuts_attente)].copy()
    df_valide = df_immo[df_immo['Statut'].isin(statuts_valides)].copy()
    
    # Métriques globales
    st.subheader("📊 Métriques Globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_paiements = df_immo['Montant'].sum()
        st.metric("💰 Total Paiements", f"{total_paiements:,.0f}€")
    
    with col2:
        nb_paiements = len(df_immo)
        st.metric("📝 Nombre de Paiements", f"{nb_paiements:,}")
    
    with col3:
        montant_moyen = df_immo['Montant'].mean() if len(df_immo) > 0 else 0
        st.metric("🎯 Montant Moyen", f"{montant_moyen:,.0f}€")
    
    with col4:
        nb_conseillers = df_immo['Conseiller'].nunique()
        st.metric("👥 Conseillers Actifs", f"{nb_conseillers}")
    
    # Comparaison des paiements en attente vs validés
    st.subheader("🔄 Paiements En Attente vs Validés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        montant_attente = df_attente['Montant'].sum()
        st.metric("⏳ Montant En Attente", f"{montant_attente:,.0f}€")
    
    with col2:
        montant_valide = df_valide['Montant'].sum()
        st.metric("✅ Montant Validé", f"{montant_valide:,.0f}€")
    
    with col3:
        ratio_validation = (montant_valide / total_paiements * 100) if total_paiements > 0 else 0
        st.metric("📊 Taux de Validation", f"{ratio_validation:.1f}%")
    
    with col4:
        nb_attente = len(df_attente)
        nb_valide = len(df_valide)
        st.write(f"⏳ **{nb_attente}** paiements en attente")
        st.write(f"✅ **{nb_valide}** paiements validés")
    
    # Filtres
    st.subheader("🔍 Filtres")
    col1, col2 = st.columns(2)
    
    df_filtre = df_immo.copy()
    
    with col1:
        mois_disponibles = sorted(df_immo['Mois'].dropna().unique())
        mois_selectionne = st.selectbox("📅 Mois", options=["Tous"] + mois_disponibles, key="mois_paiement_immo")
        if mois_selectionne != "Tous":
            df_filtre = df_filtre[df_filtre['Mois'] == mois_selectionne]
            df_attente = df_attente[df_attente['Mois'] == mois_selectionne]
            df_valide = df_valide[df_valide['Mois'] == mois_selectionne]
    
    with col2:
        conseillers_disponibles = sorted(df_immo['Conseiller'].dropna().unique())
        conseiller_selectionne = st.selectbox("👤 Conseiller", options=["Tous"] + conseillers_disponibles, key="conseiller_paiement_immo")
        if conseiller_selectionne != "Tous":
            df_filtre = df_filtre[df_filtre['Conseiller'] == conseiller_selectionne]
            df_attente = df_attente[df_attente['Conseiller'] == conseiller_selectionne]
            df_valide = df_valide[df_valide['Conseiller'] == conseiller_selectionne]
    
    # Graphiques
    if not df_filtre.empty:
        st.subheader("📈 Analyse Graphique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Répartition par statut
            fig_statut = px.pie(
                df_filtre, 
                names='Statut', 
                values='Montant',
                title="💰 Répartition des Paiements par Statut",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_statut, use_container_width=True)
        
        with col2:
            # Comparaison attente vs validé
            compare_data = pd.DataFrame({
                'Statut': ['En Attente', 'Validé'],
                'Montant': [df_attente['Montant'].sum(), df_valide['Montant'].sum()],
                'Nombre': [len(df_attente), len(df_valide)]
            })
            
            fig_compare = px.bar(
                compare_data,
                x='Statut',
                y='Montant',
                title="⚖️ Comparaison Attente vs Validé",
                text='Montant',
                color='Statut',
                color_discrete_map={'En Attente': '#FFA500', 'Validé': '#2E8B57'}
            )
            fig_compare.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
            st.plotly_chart(fig_compare, use_container_width=True)
        
        # Évolution mensuelle
        st.subheader("📅 Évolution Mensuelle")
        
        evolution_mensuelle = df_immo.groupby(['Mois', 'Statut'])['Montant'].sum().reset_index()
        fig_evolution = px.bar(
            evolution_mensuelle,
            x='Mois',
            y='Montant',
            color='Statut',
            title="📊 Évolution Mensuelle des Paiements par Statut",
            barmode='stack',
            text='Montant'
        )
        fig_evolution.update_traces(texttemplate='%{text:,.0f}€', textposition='inside')
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Top conseillers
        st.subheader("🏆 Performance des Conseillers")
        
        top_conseillers = df_immo.groupby('Conseiller')['Montant'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_top = px.bar(
            top_conseillers,
            x='Montant',
            y='Conseiller',
            orientation='h',
            title="🏆 Top 10 Conseillers - Montant Total",
            text='Montant',
            color='Montant',
            color_continuous_scale='Blues'
        )
        fig_top.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
        st.plotly_chart(fig_top, use_container_width=True)
    
    # Tableaux détaillés
    st.subheader("📋 Tableaux Détaillés")
    
    tab1, tab2, tab3 = st.tabs(["🔄 Tous les Paiements", "⏳ Paiements En Attente", "✅ Paiements Validés"])
    
    with tab1:
        if not df_filtre.empty:
            st.dataframe(df_filtre.sort_values('Date de paiement', ascending=False), use_container_width=True)
        else:
            st.info("Aucun paiement ne correspond aux filtres sélectionnés.")
    
    with tab2:
        if not df_attente.empty:
            st.dataframe(df_attente.sort_values('Date de paiement', ascending=False), use_container_width=True)
        else:
            st.info("Aucun paiement en attente ne correspond aux filtres sélectionnés.")
    
    with tab3:
        if not df_valide.empty:
            st.dataframe(df_valide.sort_values('Date de paiement', ascending=False), use_container_width=True)
        else:
            st.info("Aucun paiement validé ne correspond aux filtres sélectionnés.")

if __name__ == "__main__":
    st.set_page_config(page_title="Analyse Paiements Immobiliers", page_icon="💰", layout="wide")
    st.title("💰 Analyse des Paiements Immobiliers")
    
    uploaded_file = st.file_uploader("📁 Charger un fichier de paiements immobiliers", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        analyser_paiements_immo(df)
