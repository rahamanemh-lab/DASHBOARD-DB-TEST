"""
Fonctions d'analyse des statuts des dossiers immobiliers pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from utils.data_processing import safe_to_datetime, safe_to_numeric, adjust_dates_to_month_range, extract_conseiller
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link


def analyser_statuts_dossiers_immo(df):
    """
    Analyse détaillée des dossiers immobiliers par étape.
    
    Cette fonction fournit une analyse approfondie des dossiers immobiliers
    en fonction de leur étape, avec des visualisations et des statistiques.
    
    Args:
        df (DataFrame): DataFrame contenant les données immobilières
    """
    st.header("📊 Analyse par Étape des Dossiers Immobiliers")
    
    # Vérification des données
    if df is None or df.empty:
        st.error("❌ Aucune donnée disponible pour l'analyse des étapes.")
        return
    
    # Vérifier que la colonne étape existe (recherche flexible)
    etape_col = None
    possible_etape_cols = ['étape', 'Étape', 'ÉTAPE', 'etape', 'Etape', 'ETAPE', 'Statut', 'statut', 'STATUT']
    
    for col in possible_etape_cols:
        if col in df.columns:
            etape_col = col
            break
    
    if etape_col is None:
        st.error("❌ Aucune colonne d'étape trouvée dans les données.")
        st.write("Colonnes disponibles:", ", ".join(df.columns))
        return
    
    st.info(f"📋 Utilisation de la colonne '{etape_col}' pour l'analyse des étapes.")
    
    # Nettoyage des données
    # Remplacer les valeurs NaN dans la colonne étape
    df[etape_col] = df[etape_col].fillna('Non défini')
    
    # 1. Vue d'ensemble des étapes
    st.subheader("Vue d'ensemble des étapes")
    
    # Compter le nombre de dossiers par étape
    df_statut = df.groupby(etape_col).size().reset_index(name='Nombre de dossiers')
    df_statut = df_statut.sort_values('Nombre de dossiers', ascending=False)
    
    # Calculer les pourcentages
    total_dossiers = df_statut['Nombre de dossiers'].sum()
    df_statut['Pourcentage'] = (df_statut['Nombre de dossiers'] / total_dossiers * 100).round(2)
    
    # Afficher le tableau des étapes
    st.write("#### Répartition des dossiers par étape")
    st.dataframe(df_statut)
    
    # Créer un graphique en barres
    fig_bar = px.bar(
        df_statut,
        x=etape_col,
        y='Nombre de dossiers',
        text='Nombre de dossiers',
        title="Nombre de dossiers par étape",
        color=etape_col,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_layout(xaxis_title="Étape", yaxis_title="Nombre de dossiers")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Créer un graphique en camembert
    fig_pie = px.pie(
        df_statut,
        values='Nombre de dossiers',
        names=etape_col,
        title="Répartition des dossiers par étape",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+value+percent'
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # 2. Évolution temporelle des étapes
    st.subheader("Évolution temporelle des étapes")
    
    # Vérifier si une colonne de date est disponible
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'création' in col.lower()]
    
    if not date_columns:
        st.warning("⚠️ Aucune colonne de date n'a été trouvée pour l'analyse temporelle.")
    else:
        # Utiliser la première colonne de date trouvée
        date_col = date_columns[0]
        st.info(f"Utilisation de la colonne '{date_col}' pour l'analyse temporelle.")
        
        # S'assurer que la colonne est au format datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Filtrer les lignes avec des dates valides
        df_with_dates = df.dropna(subset=[date_col])
        
        if df_with_dates.empty:
            st.warning(f"⚠️ Aucune date valide dans la colonne '{date_col}'.")
        else:
            # Ajouter une colonne pour le mois
            df_with_dates['Mois'] = df_with_dates[date_col].dt.strftime('%Y-%m')
            
            # Créer un tableau croisé dynamique: Mois x Étape
            pivot = pd.crosstab(
                index=df_with_dates['Mois'],
                columns=df_with_dates[etape_col],
                margins=True,
                margins_name='Total'
            )
            
            # Trier par mois
            pivot = pivot.sort_index()
            
            # Afficher le tableau
            st.write("#### Évolution mensuelle des étapes")
            st.dataframe(pivot)
            
            # Créer un graphique d'évolution
            # Préparer les données pour le graphique (sans la ligne Total)
            pivot_for_chart = pivot.drop('Total')
            
            # Convertir le pivot en format long pour Plotly
            df_long = pivot_for_chart.reset_index().melt(
                id_vars=['Mois'],
                var_name='Étape',
                value_name='Nombre de dossiers'
            )
            
            # Créer le graphique d'évolution
            fig_evolution = px.line(
                df_long,
                x='Mois',
                y='Nombre de dossiers',
                color='Étape',
                markers=True,
                title="Évolution mensuelle des étapes",
                line_shape='linear'
            )
            fig_evolution.update_layout(xaxis_title="Mois", yaxis_title="Nombre de dossiers")
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # Créer un graphique en barres empilées
            fig_stacked = px.bar(
                df_long,
                x='Mois',
                y='Nombre de dossiers',
                color='Étape',
                title="Répartition mensuelle des étapes",
                barmode='stack'
            )
            fig_stacked.update_layout(xaxis_title="Mois", yaxis_title="Nombre de dossiers")
            st.plotly_chart(fig_stacked, use_container_width=True)
            
            # Exporter en CSV
            create_download_button(pivot, "evolution_etapes_immo", "etapes_evolution")
    
    # 3. Analyse croisée Étape x Conseiller
    st.subheader("Analyse croisée Étape x Conseiller")
    
    # Vérifier si la colonne Conseiller existe et l'ajouter si nécessaire
    df_with_conseiller = extract_conseiller(df.copy())
    
    if 'Conseiller' not in df_with_conseiller.columns:
        st.warning("⚠️ Aucune colonne de conseiller n'a été trouvée pour l'analyse croisée.")
    else:
        # Créer un tableau croisé dynamique: Conseiller x Étape
        pivot_conseiller = pd.crosstab(
            index=df_with_conseiller['Conseiller'],
            columns=df_with_conseiller[etape_col],
            margins=True,
            margins_name='Total'
        )
        
        # Trier par nombre total de dossiers décroissant
        pivot_conseiller = pivot_conseiller.sort_values('Total', ascending=False)
        
        # Afficher le tableau
        st.write("#### Répartition des étapes par conseiller")
        st.dataframe(pivot_conseiller)
        
        # Sélectionner un conseiller pour analyse détaillée
        conseillers = ['Tous'] + sorted(df_with_conseiller['Conseiller'].unique().tolist())
        conseiller_selectionne = st.selectbox("Sélectionner un conseiller pour l'analyse détaillée", conseillers)
        
        if conseiller_selectionne != 'Tous':
            # Filtrer les données pour le conseiller sélectionné
            df_conseiller = df_with_conseiller[df_with_conseiller['Conseiller'] == conseiller_selectionne]
            
            # Compter le nombre de dossiers par étape pour ce conseiller
            df_statut_conseiller = df_conseiller.groupby(etape_col).size().reset_index(name='Nombre de dossiers')
            df_statut_conseiller = df_statut_conseiller.sort_values('Nombre de dossiers', ascending=False)
            
            # Créer un graphique en camembert pour ce conseiller
            fig_conseiller = px.pie(
                df_statut_conseiller,
                values='Nombre de dossiers',
                names=etape_col,
                title=f"Répartition des étapes pour {conseiller_selectionne}",
                hole=0.4
            )
            fig_conseiller.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hoverinfo='label+value+percent'
            )
            st.plotly_chart(fig_conseiller, use_container_width=True)
        
        # Exporter en CSV
        create_download_button(pivot_conseiller, "etapes_conseillers_immo", "etapes_conseillers")
    
    # 4. Analyse des transitions d'étape (si des dates de mise à jour sont disponibles)
    st.subheader("Analyse des délais par étape")
    
    # Vérifier si des colonnes de date de création et de mise à jour sont disponibles
    date_creation_cols = [col for col in df.columns if 'création' in col.lower() or 'creation' in col.lower()]
    date_maj_cols = [col for col in df.columns if 'maj' in col.lower() or 'update' in col.lower() or 'modif' in col.lower()]
    
    if not date_creation_cols or not date_maj_cols:
        st.warning("⚠️ Colonnes de dates insuffisantes pour l'analyse des délais.")
    else:
        # Utiliser les premières colonnes trouvées
        date_creation_col = date_creation_cols[0]
        date_maj_col = date_maj_cols[0]
        
        st.info(f"Utilisation des colonnes '{date_creation_col}' et '{date_maj_col}' pour l'analyse des délais.")
        
        # S'assurer que les colonnes sont au format datetime
        df[date_creation_col] = pd.to_datetime(df[date_creation_col], errors='coerce')
        df[date_maj_col] = pd.to_datetime(df[date_maj_col], errors='coerce')
        
        # Calculer le délai en jours
        df['Délai (jours)'] = (df[date_maj_col] - df[date_creation_col]).dt.days
        
        # Filtrer les lignes avec des délais valides
        df_with_delays = df.dropna(subset=['Délai (jours)'])
        
        if df_with_delays.empty:
            st.warning("⚠️ Impossible de calculer des délais valides.")
        else:
            # Calculer le délai moyen par étape
            delay_by_status = df_with_delays.groupby(etape_col)['Délai (jours)'].agg(['mean', 'median', 'min', 'max']).reset_index()
            delay_by_status['mean'] = delay_by_status['mean'].round(1)
            delay_by_status['median'] = delay_by_status['median'].round(1)
            
            # Renommer les colonnes
            delay_by_status.columns = [etape_col, 'Délai moyen (jours)', 'Délai médian (jours)', 'Délai minimum (jours)', 'Délai maximum (jours)']
            
            # Trier par délai moyen décroissant
            delay_by_status = delay_by_status.sort_values('Délai moyen (jours)', ascending=False)
            
            # Afficher le tableau
            st.write("#### Délais moyens par étape")
            st.dataframe(delay_by_status)
            
            # Créer un graphique en barres pour les délais moyens
            fig_delays = px.bar(
                delay_by_status,
                x=etape_col,
                y='Délai moyen (jours)',
                text='Délai moyen (jours)',
                title="Délai moyen par étape (jours)",
                color=etape_col
            )
            fig_delays.update_layout(xaxis_title="Étape", yaxis_title="Délai moyen (jours)")
            st.plotly_chart(fig_delays, use_container_width=True)
            
            # Exporter en CSV
            create_download_button(delay_by_status, "delais_etapes_immo", "delais_etapes")
    
    # 5. Filtrage et exploration des dossiers par étape
    st.subheader("Exploration des dossiers par étape")
    
    # Sélectionner une étape pour explorer les dossiers
    etapes = ['Tous'] + sorted(df[etape_col].unique().tolist())
    etape_selectionnee = st.selectbox("Sélectionner une étape à explorer", etapes)
    
    if etape_selectionnee != 'Tous':
        # Filtrer les données pour l'étape sélectionnée
        df_filtered = df[df[etape_col] == etape_selectionnee]
        
        # Afficher le nombre de dossiers
        st.write(f"#### {len(df_filtered)} dossiers avec l'étape '{etape_selectionnee}'")
        
        # Afficher les dossiers
        st.dataframe(df_filtered)
        
        # Exporter en CSV
        create_download_button(df_filtered, f"dossiers_{etape_selectionnee.lower().replace(' ', '_')}_immo", f"dossiers_{etape_selectionnee}")
    else:
        st.info("Sélectionnez une étape spécifique pour voir les dossiers correspondants.")
