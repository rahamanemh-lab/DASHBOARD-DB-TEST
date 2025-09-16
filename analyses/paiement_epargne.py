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

def analyser_paiements_epargne(df):
    """Analyse des paiements épargne."""
    st.header("💸 Analyse des Paiements Épargne")
    
    # Débogage initial - Vérification du DataFrame d'entrée
    st.write("🔍 **DÉBOGAGE: Vérification du DataFrame d'entrée**")
    st.write(f"Dimensions du DataFrame: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    
    # Afficher un échantillon des données
    st.write("Aperçu des premières lignes:")
    st.dataframe(df.head(3))
    
    # Afficher les colonnes disponibles pour le débogage
    st.write("Colonnes disponibles dans le fichier:")
    st.write(df.columns.tolist())
    
    # Détection intelligente des colonnes nécessaires
    st.write("🔍 **DÉBOGAGE: Détection des colonnes**")
    
    # Colonne Statut
    statut_cols = ['Statut', 'Status', 'État', 'Etat']
    statut_col = None
    for col in df.columns:
        if col in statut_cols or any(s.lower() in col.lower() for s in statut_cols):
            statut_col = col
            st.success(f"✅ Colonne de statut détectée: {statut_col}")
            break
    
    # Colonne Montant
    montant_cols = ['Montant', 'Montant(€)', 'Montant €', 'Montant en €', 'Amount', 'Somme']
    montant_col = None
    for col in df.columns:
        if col in montant_cols or any(m.lower() in col.lower() for m in montant_cols):
            montant_col = col
            st.success(f"✅ Colonne de montant détectée: {montant_col}")
            # Débogage des valeurs de montant
            st.write(f"Valeurs de montant (5 premières): {df[montant_col].head().tolist()}")
            st.write(f"Type de données de la colonne montant: {df[montant_col].dtype}")
            break
    
    # Colonne Date de paiement
    date_cols = ['Date de paiement', 'Date paiement', 'Payment Date', 'Date de règlement', 'Date règlement']
    date_col = None
    for col in df.columns:
        if col in date_cols or any(d.lower() in col.lower() for d in date_cols):
            date_col = col
            st.success(f"✅ Colonne de date détectée: {date_col}")
            # Débogage des valeurs de date
            st.write(f"Valeurs de date (5 premières): {df[date_col].head().tolist()}")
            st.write(f"Type de données de la colonne date: {df[date_col].dtype}")
            break
    
    # Vérification des colonnes nécessaires
    colonnes_manquantes = []
    if not statut_col: colonnes_manquantes.append("Statut")
    if not montant_col: colonnes_manquantes.append("Montant")
    if not date_col: colonnes_manquantes.append("Date de paiement")
    
    if colonnes_manquantes:
        st.error(f"❌ Colonnes manquantes pour l'analyse des paiements épargne: {', '.join(colonnes_manquantes)}")
        return
    
    # Débogage - Vérification avant transformation
    st.write("🔍 **DÉBOGAGE: Vérification avant transformation**")
    st.write(f"Nombre de lignes avant transformation: {len(df)}")
    
    # Préparation des données
    try:
        st.write("🔍 **DÉBOGAGE: Transformation des données**")
        df_epargne = df.copy()
        
        # Conversion des dates
        st.write("Conversion des dates...")
        df_epargne['Date de paiement'] = safe_to_datetime(df_epargne[date_col])
        st.write(f"Dates converties: {df_epargne['Date de paiement'].head().tolist()}")
        
        # Extraction du mois
        st.write("Extraction du mois...")
        df_epargne['Mois'] = df_epargne['Date de paiement'].dt.to_period('M').astype(str)
        st.write(f"Mois extraits: {df_epargne['Mois'].head().tolist()}")
        
        # Conversion des montants
        st.write("Conversion des montants...")
        df_epargne['Montant'] = safe_to_numeric(df_epargne[montant_col])
        st.write(f"Montants convertis: {df_epargne['Montant'].head().tolist()}")
        
        # Copie du statut
        df_epargne['Statut'] = df_epargne[statut_col]
        
        # Extraction du conseiller
        st.write("Extraction du conseiller...")
        df_epargne = extract_conseiller(df_epargne)
        st.write(f"Conseillers extraits: {df_epargne['Conseiller'].head().tolist() if 'Conseiller' in df_epargne.columns else 'Colonne Conseiller non créée'}")
        
        # Vérification finale du DataFrame transformé
        st.write("🔍 **DÉBOGAGE: Vérification après transformation**")
        st.write(f"Nombre de lignes après transformation: {len(df_epargne)}")
        st.write(f"Colonnes après transformation: {df_epargne.columns.tolist()}")
        st.write(f"Somme des montants: {df_epargne['Montant'].sum():,.2f}€")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la transformation des données: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return
    
    # Métriques globales
    st.subheader("📊 Métriques Globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_paiements = df_epargne['Montant'].sum()
        st.metric("💰 Total Paiements", f"{total_paiements:,.0f}€")
    
    with col2:
        nb_paiements = len(df_epargne)
        st.metric("📝 Nombre de Paiements", f"{nb_paiements:,}")
    
    with col3:
        montant_moyen = df_epargne['Montant'].mean() if len(df_epargne) > 0 else 0
        st.metric("🎯 Montant Moyen", f"{montant_moyen:,.0f}€")
    
    with col4:
        nb_conseillers = df_epargne['Conseiller'].nunique()
        st.metric("👥 Conseillers Actifs", f"{nb_conseillers}")
    
    # Analyse des statuts de paiement
    statuts_disponibles = df_epargne['Statut'].dropna().unique().tolist()
    
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
            key="statuts_attente_epargne"
        )
        
        statuts_valides_selection = st.multiselect(
            "Statuts 'Validés':",
            options=statuts_disponibles,
            default=statuts_valides,
            key="statuts_valides_epargne"
        )
        
        # Utiliser les sélections manuelles si elles sont différentes des détections automatiques
        if set(statuts_attente_selection) != set(statuts_attente) or set(statuts_valides_selection) != set(statuts_valides):
            statuts_attente = statuts_attente_selection
            statuts_valides = statuts_valides_selection
    
    # Créer des DataFrames filtrés
    df_attente = df_epargne[df_epargne['Statut'].isin(statuts_attente)].copy()
    df_valide = df_epargne[df_epargne['Statut'].isin(statuts_valides)].copy()
    
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
    
    df_filtre = df_epargne.copy()
    
    with col1:
        mois_disponibles = sorted(df_epargne['Mois'].dropna().unique())
        mois_selectionne = st.selectbox("📅 Mois", options=["Tous"] + mois_disponibles, key="mois_paiement_epargne")
        if mois_selectionne != "Tous":
            df_filtre = df_filtre[df_filtre['Mois'] == mois_selectionne]
            df_attente = df_attente[df_attente['Mois'] == mois_selectionne]
            df_valide = df_valide[df_valide['Mois'] == mois_selectionne]
    
    with col2:
        conseillers_disponibles = sorted(df_epargne['Conseiller'].dropna().unique())
        conseiller_selectionne = st.selectbox("👤 Conseiller", options=["Tous"] + conseillers_disponibles, key="conseiller_paiement_epargne")
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
                color_discrete_sequence=px.colors.qualitative.Set3
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
        
        evolution_mensuelle = df_epargne.groupby(['Mois', 'Statut'])['Montant'].sum().reset_index()
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
        
        top_conseillers = df_epargne.groupby('Conseiller')['Montant'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_top = px.bar(
            top_conseillers,
            x='Montant',
            y='Conseiller',
            orientation='h',
            title="🏆 Top 10 Conseillers - Montant Total",
            text='Montant',
            color='Montant',
            color_continuous_scale='Greens'
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
    st.set_page_config(page_title="Analyse Paiements Épargne", page_icon="💸", layout="wide")
    st.title("💸 Analyse des Paiements Épargne")
    
    uploaded_file = st.file_uploader("📁 Charger un fichier de paiements épargne", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        analyser_paiements_epargne(df)
