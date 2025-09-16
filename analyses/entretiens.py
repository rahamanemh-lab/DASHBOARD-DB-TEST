"""
Fonctions d'analyse des entretiens pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from utils.data_processing import safe_to_datetime, safe_to_numeric, adjust_dates_to_month_range, extract_conseiller
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link


def analyser_entretiens(df, key_suffix=""):
    """Analyse des entretiens.
    
    Args:
        df (DataFrame): DataFrame contenant les données d'entretiens
    """
    st.header("🗣️ Analyse des Entretiens")
    
    # Vérification si le DataFrame est None
    if df is None:
        st.error("❌ Veuillez charger un fichier de données d'entretiens.")
        return
    
    # Vérification et prétraitement des données
    df = df.copy()
    
    # Vérifier et gérer les colonnes de date
    date_cols = ['Date de souscription', 'Date', 'date', 'Date de création', 'date_de_creation']
    date_col = None
    
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        st.warning("⚠️ Aucune colonne de date trouvée dans le fichier d'entretiens.")
        st.info(f"Colonnes disponibles: {', '.join(df.columns)}")
        return
    
    df[date_col] = safe_to_datetime(df[date_col])
    df['Mois'] = df[date_col].dt.to_period('M').astype(str)
    
    # Extraire les conseillers
    df = extract_conseiller(df)
    
    # Détection de la colonne de statut
    etape_cols = ['Statut', 'Status', 'Etape', 'Étape', 'Stage', 'Phase', 'Etat', 'État']
    etape_col = None
    for col in etape_cols:
        if col in df.columns:
            etape_col = col
            break
    
    # Filtres globaux
    st.subheader("🔍 Filtres")
    col1, col2, col3 = st.columns(3)
    
    # Filtre par mois
    with col1:
        mois_disponibles = sorted(df['Mois'].unique())
        mois_selectionne = st.selectbox(
            "📅 Sélectionner un mois",
            options=["Tous"] + mois_disponibles,
            index=0,
            key=f"mois_filter{key_suffix}"
        )
    
    # Filtre par conseiller
    with col2:
        conseillers_disponibles = sorted(df['Conseiller'].unique())
        conseiller_selectionne = st.selectbox(
            "👤 Sélectionner un conseiller",
            options=["Tous"] + conseillers_disponibles,
            index=0,
            key=f"conseiller_filter{key_suffix}"
        )
    
    # Filtre par statut si disponible
    if etape_col:
        with col3:
            statuts_disponibles = sorted(df[etape_col].unique())
            statut_selectionne = st.selectbox(
                "💡 Sélectionner un statut",
                options=["Tous"] + statuts_disponibles,
                index=0,
                key=f"statut_filter{key_suffix}"
            )
    else:
        statut_selectionne = "Tous"
    
    # Appliquer les filtres
    df_filtre = df.copy()
    if mois_selectionne != "Tous":
        df_filtre = df_filtre[df_filtre['Mois'] == mois_selectionne]
    if conseiller_selectionne != "Tous":
        df_filtre = df_filtre[df_filtre['Conseiller'] == conseiller_selectionne]
    if etape_col and statut_selectionne != "Tous":
        df_filtre = df_filtre[df_filtre[etape_col] == statut_selectionne]
    
    if df_filtre.empty:
        st.error("❌ Aucune donnée valide après filtrage.")
        return
    
    # Métriques globales
    st.subheader("📊 Métriques Globales")
    
    # Calculer les métriques
    total_entretiens = len(df_filtre)
    nb_conseillers_actifs = df_filtre['Conseiller'].nunique()
    
    # Entretiens terminés (si colonne de statut disponible)
    entretiens_termines = 0
    taux_completion = 0
    if etape_col:
        statuts_termines = ['Terminé', 'Completed', 'Done', 'Fini', 'Clôturé', 'Closed']
        mask_termine = df_filtre[etape_col].str.contains('|'.join(statuts_termines), case=False, na=False)
        entretiens_termines = mask_termine.sum()
        taux_completion = round(entretiens_termines / total_entretiens * 100, 2) if total_entretiens > 0 else 0
    
    # Calculer le mois en cours
    mois_en_cours = datetime.now().strftime('%Y-%m')
    
    # Compter les entretiens du mois en cours
    entretiens_mois_en_cours = df_filtre[df_filtre['Mois'] == mois_en_cours].shape[0]
    
    # Calculer la moyenne mensuelle
    mois_uniques = df_filtre['Mois'].nunique()
    moyenne_mensuelle = total_entretiens / mois_uniques if mois_uniques > 0 else 0
    
    # Calculer la différence par rapport à la moyenne
    diff_pourcentage = ((entretiens_mois_en_cours - moyenne_mensuelle) / moyenne_mensuelle * 100) if moyenne_mensuelle > 0 else 0
    
    # Afficher les métriques dans des colonnes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total des entretiens", total_entretiens)
    with col2:
        st.metric("Conseillers actifs", nb_conseillers_actifs)
    with col3:
        st.metric(f"Entretiens {mois_en_cours}", entretiens_mois_en_cours, f"{diff_pourcentage:.1f}%")
    if etape_col:
        with col4:
            st.metric("Taux de complétion", f"{taux_completion}%")
    
    # Évolution des entretiens dans le temps
    st.subheader("📈 Évolution des Entretiens dans le Temps")
    
    # Préparer les données pour le graphique d'évolution
    df_evolution = df_filtre.copy()
    df_evolution['Date'] = safe_to_datetime(df_evolution[date_col])
    df_evolution['Date_Jour'] = df_evolution['Date'].dt.date
    
    # Compter les entretiens par jour
    evolution_quotidienne = df_evolution.groupby('Date_Jour').size().reset_index(name='Nombre_Entretiens')
    evolution_quotidienne = evolution_quotidienne.sort_values('Date_Jour')
    
    # Créer le graphique d'évolution
    fig_evolution = px.line(
        evolution_quotidienne,
        x='Date_Jour',
        y='Nombre_Entretiens',
        title="Évolution du nombre d'entretiens",
        labels={
            'Date_Jour': 'Date',
            'Nombre_Entretiens': "Nombre d'entretiens"
        },
        markers=True
    )
    
    # Personnaliser le graphique
    fig_evolution.update_layout(
        xaxis_title="Date",
        yaxis_title="Nombre d'entretiens",
        hovermode="x unified"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig_evolution, use_container_width=True)
    
    # Analyse par mois
    st.subheader("📅 Analyse par Mois")
    
    # Grouper par mois
    df_mois = df_filtre.groupby('Mois').size().reset_index(name='Nombre_Entretiens')
    
    # Trier par mois
    df_mois = df_mois.sort_values('Mois')
    
    # Créer le graphique
    fig = px.bar(
        df_mois,
        x='Mois',
        y='Nombre_Entretiens',
        text='Nombre_Entretiens',
        title="Nombre d'Entretiens par Mois",
        labels={
            'Mois': 'Mois',
            'Nombre_Entretiens': "Nombre d'Entretiens"
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
    df_conseiller = df_filtre.groupby('Conseiller').size().reset_index(name='Nombre_Entretiens')
    
    # Trier par nombre d'entretiens décroissant
    df_conseiller = df_conseiller.sort_values('Nombre_Entretiens', ascending=False)
    
    # Créer le graphique
    fig = px.bar(
        df_conseiller,
        x='Conseiller',
        y='Nombre_Entretiens',
        text='Nombre_Entretiens',
        title="Nombre d'Entretiens par Conseiller",
        labels={
            'Conseiller': 'Conseiller',
            'Nombre_Entretiens': "Nombre d'Entretiens"
        },
        height=500,
        color='Nombre_Entretiens',
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
    
    # Analyse de performance par conseiller
    st.subheader("🏆 Performance par Conseiller")
    
    # Grouper par conseiller
    performance_conseillers = df_filtre.groupby('Conseiller').size().reset_index(name='Nombre_Entretiens')
    
    # Ajouter des colonnes pour les entretiens avec/sans souscription
    if 'Dernière souscription' in df_filtre.columns:
        # Utiliser la regex pour identifier les souscriptions
        pattern = r'\d+-\w+-[A-Z]+'
        df_filtre['Avec_Souscription'] = df_filtre['Dernière souscription'].str.contains(pattern, na=False, regex=True)
        
        # Compter les entretiens avec souscription par conseiller
        souscriptions_par_conseiller = df_filtre[df_filtre['Avec_Souscription']].groupby('Conseiller').size().reset_index(name='Nombre_Souscriptions')
        
        # Fusionner avec le DataFrame de performance
        performance_conseillers = performance_conseillers.merge(souscriptions_par_conseiller, on='Conseiller', how='left')
        performance_conseillers['Nombre_Souscriptions'] = performance_conseillers['Nombre_Souscriptions'].fillna(0).astype(int)
        
        # Calculer le taux de conversion
        performance_conseillers['Taux_Conversion'] = (performance_conseillers['Nombre_Souscriptions'] / performance_conseillers['Nombre_Entretiens'] * 100).round(2)
    
    # Trier par nombre d'entretiens décroissant
    performance_conseillers = performance_conseillers.sort_values('Nombre_Entretiens', ascending=False)
    
    # Prendre les 10 premiers conseillers pour le graphique
    top_conseillers = performance_conseillers.head(10)
    
    # Créer le graphique des top 10 conseillers
    fig_top_conseillers = px.bar(
        top_conseillers,
        x='Conseiller',
        y='Nombre_Entretiens',
        title="Top 10 des conseillers par nombre d'entretiens",
        labels={
            'Conseiller': 'Conseiller',
            'Nombre_Entretiens': "Nombre d'entretiens"
        },
        color='Nombre_Entretiens',
        color_continuous_scale='Viridis'
    )
    
    # Personnaliser le graphique
    fig_top_conseillers.update_layout(
        xaxis_title="Conseiller",
        yaxis_title="Nombre d'entretiens",
        xaxis={'categoryorder':'total descending'}
    )
    
    # Afficher le graphique
    st.plotly_chart(fig_top_conseillers, use_container_width=True)
    
    # Afficher le tableau de performance détaillé
    st.write("### Tableau détaillé de performance par conseiller")
    
    # Préparer le tableau pour l'affichage
    tableau_performance = performance_conseillers.copy()
    
    # Renommer les colonnes pour l'affichage
    colonnes_affichage = {'Conseiller': 'Conseiller', 'Nombre_Entretiens': "Nombre d'entretiens"}
    
    if 'Nombre_Souscriptions' in tableau_performance.columns:
        colonnes_affichage.update({
            'Nombre_Souscriptions': 'Nombre de souscriptions',
            'Taux_Conversion': 'Taux de conversion (%)'
        })
    
    tableau_performance = tableau_performance.rename(columns=colonnes_affichage)
    
    # Afficher le tableau
    st.dataframe(tableau_performance, use_container_width=True)
    
    # Bouton d'export
    create_download_button(tableau_performance, f"performance_conseillers{key_suffix}", f"performance_conseillers{key_suffix}")
    
    # Analyse des entretiens avec/sans souscription si la colonne est disponible
    if 'Dernière souscription' in df_filtre.columns:
        st.subheader("💰 Analyse des Entretiens avec/sans Souscription")
        
        # Utiliser la regex pour identifier les souscriptions
        pattern = r'\d+-\w+-[A-Z]+'
        df_filtre['Avec_Souscription'] = df_filtre['Dernière souscription'].str.contains(pattern, na=False, regex=True)
        
        # Compter les entretiens avec et sans souscription
        nb_avec_souscription = df_filtre['Avec_Souscription'].sum()
        nb_sans_souscription = len(df_filtre) - nb_avec_souscription
        
        # Créer un DataFrame pour le graphique
        df_souscriptions = pd.DataFrame({
            'Catégorie': ['Avec souscription', 'Sans souscription'],
            'Nombre': [nb_avec_souscription, nb_sans_souscription]
        })
        
        # Calculer le taux de conversion global
        taux_conversion = round(nb_avec_souscription / len(df_filtre) * 100, 2) if len(df_filtre) > 0 else 0
        
        # Afficher le taux de conversion global
        st.metric("Taux de conversion global", f"{taux_conversion}%")
        
        # Créer le graphique en camembert
        fig = px.pie(
            df_souscriptions,
            values='Nombre',
            names='Catégorie',
            title="Répartition des Entretiens avec/sans Souscription",
            hole=0.4,
            color='Catégorie',
            color_discrete_map={
                'Avec souscription': '#4CAF50',  # Vert
                'Sans souscription': '#FFA500'  # Orange
            }
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
        
        # Analyse par conseiller avec/sans souscription
        st.write("### Répartition des entretiens avec/sans souscription par conseiller")
        
        # Grouper par conseiller
        entretiens_par_conseiller = df_filtre.groupby('Conseiller').agg({
            'Avec_Souscription': ['sum', 'count']
        })
        
        # Renommer les colonnes
        entretiens_par_conseiller.columns = ['Avec_Souscription', 'Total_Entretiens']
        
        # Calculer les entretiens sans souscription
        entretiens_par_conseiller['Sans_Souscription'] = entretiens_par_conseiller['Total_Entretiens'] - entretiens_par_conseiller['Avec_Souscription']
        
        # Calculer le taux de conversion
        entretiens_par_conseiller['Taux_Conversion'] = (entretiens_par_conseiller['Avec_Souscription'] / entretiens_par_conseiller['Total_Entretiens'] * 100).round(2)
        
        # Réorganiser le DataFrame
        analyse_souscriptions = entretiens_par_conseiller.reset_index()
        
        # Trier par nombre total d'entretiens décroissant
        analyse_souscriptions = analyse_souscriptions.sort_values('Total_Entretiens', ascending=False)
        
        # Créer un graphique en barres pour la répartition par conseiller
        fig_barres = px.bar(
            analyse_souscriptions.head(10),  # Top 10 conseillers
            x='Conseiller',
            y=['Avec_Souscription', 'Sans_Souscription'],
            title="Répartition des Entretiens avec/sans Souscription par Conseiller (Top 10)",
            labels={
                'Conseiller': 'Conseiller',
                'value': "Nombre d'Entretiens",
                'variable': 'Type'
            },
            barmode='stack',
            color_discrete_map={
                'Avec_Souscription': '#4CAF50',  # Vert
                'Sans_Souscription': '#FFA500'  # Orange
            }
        )
        
        # Renommer les légendes
        fig_barres.update_layout(
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_tickangle=-45
        )
        
        # Renommer les catégories dans la légende
        newnames = {'Avec_Souscription': 'Avec souscription', 'Sans_Souscription': 'Sans souscription'}
        fig_barres.for_each_trace(lambda t: t.update(name = newnames[t.name]))
        
        # Afficher le graphique
        st.plotly_chart(fig_barres, use_container_width=True)
        
        # Afficher le tableau des données
        st.write("### Détail par conseiller")
        tableau_souscriptions = analyse_souscriptions[['Conseiller', 'Total_Entretiens', 'Avec_Souscription', 'Sans_Souscription', 'Taux_Conversion']]
        tableau_souscriptions.columns = ['Conseiller', "Total des entretiens", 'Avec souscription', 'Sans souscription', 'Taux de conversion (%)']  
        tableau_souscriptions = tableau_souscriptions.sort_values('Total des entretiens', ascending=False)
        st.dataframe(tableau_souscriptions, use_container_width=True)
        
        # Bouton d'export
        create_download_button(tableau_souscriptions, f"analyse_souscriptions_entretiens{key_suffix}", f"souscriptions_entretiens{key_suffix}")
    
    # Afficher les données filtrées
    st.subheader("📃 Données détaillées")
    
    # Afficher les données
    if not df_filtre.empty:
        with st.expander("Voir les données détaillées"):
            st.dataframe(df_filtre)
            
            # Exporter en CSV
            create_download_button(df_filtre, "entretiens_filtres", f"entretiens_filtres{key_suffix}")
    else:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés.")

def analyser_entretiens_epargne(df_entretiens, df_souscriptions=None, df_rdv=None):
    """Analyse détaillée des entretiens épargne avec calcul des taux de conversion et de RDV.
    
    Args:
        df_entretiens (DataFrame): DataFrame contenant les données d'entretiens épargne
        df_souscriptions (DataFrame, optional): DataFrame contenant les données de souscriptions épargne
        df_rdv (DataFrame, optional): DataFrame contenant les données de rendez-vous
    """
    st.header("🗣️💰 Analyse Détaillée des Entretiens Épargne")
    
    if df_entretiens is None or df_entretiens.empty:
        st.error("❌ Aucune donnée d'entretien épargne disponible.")
        return
    
    # Prétraitement des données
    df = df_entretiens.copy()
    
    # Vérifier et gérer les colonnes de date
    date_cols = ['Date de souscription', 'Date', 'date', 'Date de création', 'date_de_creation']
    date_col = None
    
    for col in date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        st.warning("⚠️ Aucune colonne de date trouvée dans le fichier d'entretiens épargne.")
        st.info(f"Colonnes disponibles: {', '.join(df.columns)}")
        return
    
    df[date_col] = safe_to_datetime(df[date_col])
    df['Mois'] = df[date_col].dt.to_period('M').astype(str)
    
    # Extraire les conseillers
    df = extract_conseiller(df)
    
    # Détection de la colonne de statut
    etape_cols = ['Statut', 'Status', 'Etape', 'Étape', 'Stage', 'Phase', 'Etat', 'État']
    etape_col = None
    for col in etape_cols:
        if col in df.columns:
            etape_col = col
            break
    
    # Filtres globaux
    st.subheader("🔍 Filtres")
    col1, col2, col3 = st.columns(3)
    
    # Filtre par mois
    with col1:
        mois_disponibles = sorted(df['Mois'].unique())
        mois_selectionne = st.selectbox(
            "📅 Sélectionner un mois",
            options=["Tous"] + mois_disponibles,
            index=0
        )
    
    # Filtre par conseiller
    with col2:
        conseillers_disponibles = sorted(df['Conseiller'].unique())
        conseiller_selectionne = st.selectbox(
            "👤 Sélectionner un conseiller",
            options=["Tous"] + conseillers_disponibles,
            index=0
        )
    
    # Filtre par statut si disponible
    if etape_col:
        with col3:
            statuts_disponibles = sorted(df[etape_col].unique())
            statut_selectionne = st.selectbox(
                "💡 Sélectionner un statut",
                options=["Tous"] + statuts_disponibles,
                index=0
            )
    else:
        statut_selectionne = "Tous"
    
    # Appliquer les filtres
    df_filtre = df.copy()
    if mois_selectionne != "Tous":
        df_filtre = df_filtre[df_filtre['Mois'] == mois_selectionne]
    if conseiller_selectionne != "Tous":
        df_filtre = df_filtre[df_filtre['Conseiller'] == conseiller_selectionne]
    if etape_col and statut_selectionne != "Tous":
        df_filtre = df_filtre[df_filtre[etape_col] == statut_selectionne]
    
    # Métriques globales
    st.subheader("📊 Métriques Globales")
    
    # Calculer les métriques
    total_entretiens = len(df_filtre)
    nb_conseillers_actifs = df_filtre['Conseiller'].nunique()
    
    # Entretiens terminés (si colonne de statut disponible)
    entretiens_termines = 0
    taux_completion = 0
    if etape_col:
        statuts_termines = ['Terminé', 'Completed', 'Done', 'Fini', 'Clôturé', 'Closed']
        mask_termine = df_filtre[etape_col].str.contains('|'.join(statuts_termines), case=False, na=False)
        entretiens_termines = mask_termine.sum()
        taux_completion = round(entretiens_termines / total_entretiens * 100, 2) if total_entretiens > 0 else 0
    
    # Afficher les métriques dans des colonnes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total des entretiens", total_entretiens)
    with col2:
        st.metric("Conseillers actifs", nb_conseillers_actifs)
    if etape_col:
        with col3:
            st.metric("Entretiens terminés", entretiens_termines)
        with col4:
            st.metric("Taux de complétion", f"{taux_completion}%")
    
    if df_filtre.empty:
        st.error("❌ Aucune donnée d'entretien valide après filtrage.")
        return
    
    # Évolution des entretiens dans le temps
    st.subheader("📈 Évolution des Entretiens dans le Temps")
    
    # Préparer les données pour le graphique d'évolution
    df_evolution = df_filtre.copy()
    df_evolution['Date'] = safe_to_datetime(df_evolution[date_col])
    df_evolution['Date_Jour'] = df_evolution['Date'].dt.date
    
    # Compter les entretiens par jour
    evolution_quotidienne = df_evolution.groupby('Date_Jour').size().reset_index(name='Nombre_Entretiens')
    evolution_quotidienne = evolution_quotidienne.sort_values('Date_Jour')
    
    # Créer le graphique d'évolution
    fig_evolution = px.line(
        evolution_quotidienne,
        x='Date_Jour',
        y='Nombre_Entretiens',
        title="Évolution du nombre d'entretiens",
        labels={
            'Date_Jour': 'Date',
            'Nombre_Entretiens': "Nombre d'entretiens"
        },
        markers=True
    )
    
    # Personnaliser le graphique
    fig_evolution.update_layout(
        xaxis_title="Date",
        yaxis_title="Nombre d'entretiens",
        hovermode="x unified"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig_evolution, use_container_width=True)
    
    # Analyse de performance par conseiller
    st.subheader("🏆 Performance par Conseiller")
    
    # Grouper par conseiller
    performance_conseillers = df_filtre.groupby('Conseiller').size().reset_index(name='Nombre_Entretiens')
    
    # Ajouter des colonnes pour les entretiens avec/sans souscription
    if 'Dernière souscription' in df_filtre.columns:
        # Utiliser la regex pour identifier les souscriptions
        pattern = r'\d+-\w+-[A-Z]+'
        df_filtre['Avec_Souscription'] = df_filtre['Dernière souscription'].str.contains(pattern, na=False, regex=True)
        
        # Compter les entretiens avec souscription par conseiller
        souscriptions_par_conseiller = df_filtre[df_filtre['Avec_Souscription']].groupby('Conseiller').size().reset_index(name='Nombre_Souscriptions')
        
        # Fusionner avec le DataFrame de performance
        performance_conseillers = performance_conseillers.merge(souscriptions_par_conseiller, on='Conseiller', how='left')
        performance_conseillers['Nombre_Souscriptions'] = performance_conseillers['Nombre_Souscriptions'].fillna(0).astype(int)
        
        # Calculer le taux de conversion
        performance_conseillers['Taux_Conversion'] = (performance_conseillers['Nombre_Souscriptions'] / performance_conseillers['Nombre_Entretiens'] * 100).round(2)
    
    # Trier par nombre d'entretiens décroissant
    performance_conseillers = performance_conseillers.sort_values('Nombre_Entretiens', ascending=False)
    
    # Prendre les 10 premiers conseillers pour le graphique
    top_conseillers = performance_conseillers.head(10)
    
    # Créer le graphique des top 10 conseillers
    fig_top_conseillers = px.bar(
        top_conseillers,
        x='Conseiller',
        y='Nombre_Entretiens',
        title="Top 10 des conseillers par nombre d'entretiens",
        labels={
            'Conseiller': 'Conseiller',
            'Nombre_Entretiens': "Nombre d'entretiens"
        },
        color='Nombre_Entretiens',
        color_continuous_scale='Viridis'
    )
    
    # Personnaliser le graphique
    fig_top_conseillers.update_layout(
        xaxis_title="Conseiller",
        yaxis_title="Nombre d'entretiens",
        xaxis={'categoryorder':'total descending'}
    )
    
    # Afficher le graphique
    st.plotly_chart(fig_top_conseillers, use_container_width=True)
    
    # Afficher le tableau de performance détaillé
    st.write("### Tableau détaillé de performance par conseiller")
    
    # Préparer le tableau pour l'affichage
    tableau_performance = performance_conseillers.copy()
    
    # Renommer les colonnes pour l'affichage
    colonnes_affichage = {'Conseiller': 'Conseiller', 'Nombre_Entretiens': "Nombre d'entretiens"}
    
    if 'Nombre_Souscriptions' in tableau_performance.columns:
        colonnes_affichage.update({
            'Nombre_Souscriptions': 'Nombre de souscriptions',
            'Taux_Conversion': 'Taux de conversion (%)'
        })
    
    tableau_performance = tableau_performance.rename(columns=colonnes_affichage)
    
    # Afficher le tableau
    st.dataframe(tableau_performance, use_container_width=True)
    
    # Bouton d'export
    create_download_button(tableau_performance, "performance_conseillers_epargne", "performance_conseillers")
    
    # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
    df_filtre = adjust_dates_to_month_range(df_filtre, 'Date')
    
    # Afficher les statistiques globales
    st.subheader("📈 Statistiques Globales")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nb_entretiens = len(df_filtre)
        st.metric("Nombre Total d'Entretiens Épargne", nb_entretiens)
    
    with col2:
        nb_conseillers = df_filtre['Conseiller'].nunique()
        st.metric("Nombre de Conseillers", nb_conseillers)
    
    with col3:
        # Calculer le mois en cours
        mois_en_cours = datetime.now().strftime('%Y-%m')
        
        # Compter les entretiens du mois en cours
        entretiens_mois_en_cours = df_filtre[df_filtre['Mois'] == mois_en_cours].shape[0]
        
        # Calculer la moyenne mensuelle
        mois_uniques = df_filtre['Mois'].nunique()
        moyenne_mensuelle = nb_entretiens / mois_uniques if mois_uniques > 0 else 0
        
        # Calculer la différence par rapport à la moyenne
        diff_pourcentage = ((entretiens_mois_en_cours - moyenne_mensuelle) / moyenne_mensuelle * 100) if moyenne_mensuelle > 0 else 0
        
        st.metric(
            f"Entretiens du Mois ({mois_en_cours})",
            entretiens_mois_en_cours,
            f"{diff_pourcentage:.1f}% par rapport à la moyenne"
        )
    
    # Analyse par mois
    st.subheader("📅 Analyse par Mois")
    
    # Grouper par mois
    df_mois = df_filtre.groupby('Mois').size().reset_index(name='Nombre_Entretiens')
    
    # Trier par mois
    df_mois = df_mois.sort_values('Mois')
    
    # Créer le graphique
    fig = px.bar(
        df_mois,
        x='Mois',
        y='Nombre_Entretiens',
        text='Nombre_Entretiens',
        title="Nombre d'Entretiens par Mois",
        labels={
            'Mois': 'Mois',
            'Nombre_Entretiens': "Nombre d'Entretiens"
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
    df_conseiller = df_filtre.groupby('Conseiller').size().reset_index(name='Nombre_Entretiens')
    
    # Trier par nombre d'entretiens décroissant
    df_conseiller = df_conseiller.sort_values('Nombre_Entretiens', ascending=False)
    
    # Créer le graphique
    fig = px.bar(
        df_conseiller,
        x='Conseiller',
        y='Nombre_Entretiens',
        text='Nombre_Entretiens',
        title="Nombre d'Entretiens par Conseiller",
        labels={
            'Conseiller': 'Conseiller',
            'Nombre_Entretiens': "Nombre d'Entretiens"
        },
        height=500,
        color='Nombre_Entretiens',
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
    
    
    # Analyse par statut
    st.subheader("📊 Analyse par Statut")
    
    # Grouper par statut
    df_statut = df_filtre.groupby('Statut').size().reset_index(name='Nombre_Entretiens')
    
    # Trier par nombre d'entretiens décroissant
    df_statut = df_statut.sort_values('Nombre_Entretiens', ascending=False)
    
    # Créer le graphique en camembert
    fig = px.pie(
        df_statut,
        values='Nombre_Entretiens',
        names='Statut',
        title="Répartition des Entretiens par Statut",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Mise en forme du graphique
    fig.update_layout(
        template="plotly_white"
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Créer un DataFrame pour l'analyse mensuelle
    analyse_mensuelle = pd.DataFrame()
    analyse_mensuelle['Mois'] = sorted(df_filtre['Mois'].unique())
    
    # Compter les entretiens par mois
    entretiens_par_mois = df_filtre.groupby('Mois').size().reset_index(name='Nombre_Entretiens')
    analyse_mensuelle = analyse_mensuelle.merge(entretiens_par_mois, on='Mois', how='left')
    
    # Remplacer les NaN par 0
    analyse_mensuelle = analyse_mensuelle.fillna(0)
    
    # Afficher le tableau des données
    st.write("### Données mensuelles des entretiens")
    tableau_entretiens = analyse_mensuelle[['Mois', 'Nombre_Entretiens']]
    tableau_entretiens.columns = ['Mois', "Nombre d'Entretiens"]
    st.dataframe(tableau_entretiens, use_container_width=True)
    
    # Bouton d'export
    create_download_button(tableau_entretiens, "entretiens_mensuels_epargne", "entretiens_epargne")
    
    # Analyse des taux d'entretiens avec souscription (si les données de souscription sont disponibles)
    if df_souscriptions is not None:
        st.subheader("💰 Taux d'Entretiens avec Souscription par Mois")
        
        # Prétraitement des données de souscriptions
        date_cols_souscriptions = ['Date de souscription', 'Date', 'date', 'Date de création', 'date_de_creation']
        date_col_souscription = None
        
        for col in date_cols_souscriptions:
            if col in df_souscriptions.columns:
                date_col_souscription = col
                break
        
        if date_col_souscription is None:
            st.warning("⚠️ Aucune colonne de date trouvée dans le fichier des souscriptions.")
        else:
            # Standardiser les colonnes de date et conseiller
            df_souscriptions[date_col_souscription] = safe_to_datetime(df_souscriptions[date_col_souscription])
            df_souscriptions = extract_conseiller(df_souscriptions)
            
            # Ajouter une colonne Mois aux souscriptions
            df_souscriptions['Mois'] = df_souscriptions[date_col_souscription].dt.to_period('M').astype(str)
            
            # Compter les souscriptions par mois
            souscriptions_par_mois = df_souscriptions.groupby('Mois').size().reset_index(name='Nombre_Souscriptions')
            analyse_mensuelle = analyse_mensuelle.merge(souscriptions_par_mois, on='Mois', how='left')
            
            # Remplacer les NaN par 0
            analyse_mensuelle['Nombre_Souscriptions'] = analyse_mensuelle['Nombre_Souscriptions'].fillna(0)
            
            # Calculer le taux de conversion
            analyse_mensuelle['Taux_Conversion'] = (analyse_mensuelle['Nombre_Souscriptions'] / analyse_mensuelle['Nombre_Entretiens'] * 100).round(2)
            analyse_mensuelle['Taux_Conversion'] = analyse_mensuelle['Taux_Conversion'].fillna(0)
            
            # Calculer les entretiens sans souscription
            analyse_mensuelle['Entretiens_Sans_Souscription'] = analyse_mensuelle['Nombre_Entretiens'] - analyse_mensuelle['Nombre_Souscriptions']
            
            # Créer le graphique de taux de conversion
            fig_conversion = px.line(
                analyse_mensuelle,
                x='Mois',
                y='Taux_Conversion',
                title="Taux de Conversion Mensuel (Entretiens → Souscriptions)",
                labels={
                    'Mois': 'Mois',
                    'Taux_Conversion': 'Taux de Conversion (%)',
                },
                markers=True,
            )
            
            # Ajouter des annotations pour les valeurs
            fig_conversion.update_traces(
                text=analyse_mensuelle['Taux_Conversion'].apply(lambda x: f"{x:.2f}%"),
                textposition="top center"
            )
            
            # Afficher le graphique
            st.plotly_chart(fig_conversion, use_container_width=True)
            
            # Créer un graphique pour montrer les entretiens avec et sans souscription
            fig_entretiens = px.bar(
                analyse_mensuelle,
                x='Mois',
                y=['Nombre_Souscriptions', 'Entretiens_Sans_Souscription'],
                title="Répartition Mensuelle des Entretiens avec/sans Souscription",
                labels={
                    'Mois': 'Mois',
                    'value': "Nombre d'Entretiens",
                    'variable': 'Type'
                },
                barmode='stack',
                color_discrete_map={
                    'Nombre_Souscriptions': '#2E8B57',  # Vert
                    'Entretiens_Sans_Souscription': '#DC143C'  # Rouge
                }
            )
            
            # Renommer les légendes
            fig_entretiens.update_layout(
                legend=dict(
                    title="",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Renommer les catégories dans la légende
            newnames = {'Nombre_Souscriptions': 'Avec Souscription', 'Entretiens_Sans_Souscription': 'Sans Souscription'}
            fig_entretiens.for_each_trace(lambda t: t.update(name = newnames[t.name]))
            
            # Afficher le graphique
            st.plotly_chart(fig_entretiens, use_container_width=True)
            
            # Afficher le tableau des données
            st.write("### Données mensuelles de conversion")
            tableau_conversion = analyse_mensuelle[['Mois', 'Nombre_Entretiens', 'Nombre_Souscriptions', 'Taux_Conversion']]
            tableau_conversion.columns = ['Mois', "Nombre d'Entretiens", 'Nombre de Souscriptions', 'Taux de Conversion (%)']  
            st.dataframe(tableau_conversion, use_container_width=True)
            
            # Bouton d'export
            create_download_button(tableau_conversion, "conversion_mensuelle_epargne", "conversion_epargne")
    
    # Analyse des entretiens avec ou sans souscription
    st.subheader("💰 Analyse des Entretiens avec/sans Souscription")
    
    # Vérifier si la colonne "Dernière souscription" existe
    derniere_souscription_cols = ["Dernière souscription", "Derniere souscription", "derniere souscription", "derniere_souscription", "DERNIERE SOUSCRIPTION"]
    col_derniere_souscription = next((col for col in derniere_souscription_cols if col in df_filtre.columns), None)
    
    if col_derniere_souscription:
        # Définir un motif regex pour identifier les chaînes qui ressemblent à une souscription
        # Le motif recherche : un numéro, suivi d'un tiret, d'un mot, d'un tiret, de lettres majuscules
        souscription_pattern = r'\d+-\w+-[A-Z]+'
        
        # Vérifier si les valeurs correspondent au motif
        df_filtre['Avec_Souscription'] = df_filtre[col_derniere_souscription].astype(str).str.contains(souscription_pattern, regex=True, na=False)
        
        # Compter les entretiens par conseiller et par statut de souscription
        entretiens_par_conseiller = df_filtre.groupby(['Conseiller', 'Avec_Souscription']).size().reset_index(name='Nombre')
        
        # Pivoter pour avoir les colonnes 'Avec souscription' et 'Sans souscription'
        analyse_souscriptions = entretiens_par_conseiller.pivot_table(
            index='Conseiller', 
            columns='Avec_Souscription', 
            values='Nombre', 
            fill_value=0
        ).reset_index()
        
        # Renommer les colonnes
        analyse_souscriptions.columns = ['Conseiller', 'Sans_Souscription', 'Avec_Souscription'] if False in analyse_souscriptions.columns and True in analyse_souscriptions.columns else ['Conseiller', 'Sans_Souscription'] if False in analyse_souscriptions.columns else ['Conseiller', 'Avec_Souscription']
        
        # Ajouter les colonnes manquantes si nécessaire
        if 'Avec_Souscription' not in analyse_souscriptions.columns:
            analyse_souscriptions['Avec_Souscription'] = 0
        if 'Sans_Souscription' not in analyse_souscriptions.columns:
            analyse_souscriptions['Sans_Souscription'] = 0
        
        # Calculer le total des entretiens par conseiller
        analyse_souscriptions['Total_Entretiens'] = analyse_souscriptions['Avec_Souscription'] + analyse_souscriptions['Sans_Souscription']
        
        # Calculer le taux de conversion
        analyse_souscriptions['Taux_Conversion'] = (analyse_souscriptions['Avec_Souscription'] / analyse_souscriptions['Total_Entretiens'] * 100).round(2)
        
        # Calculer les totaux
        total_entretiens = analyse_souscriptions['Total_Entretiens'].sum()
        total_avec_souscription = analyse_souscriptions['Avec_Souscription'].sum()
        total_sans_souscription = analyse_souscriptions['Sans_Souscription'].sum()
        taux_conversion_global = (total_avec_souscription / total_entretiens * 100).round(2) if total_entretiens > 0 else 0
        
        # Afficher les métriques globales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total des entretiens", total_entretiens)
        
        with col2:
            st.metric("Avec souscription", total_avec_souscription, f"{taux_conversion_global}%")
        
        with col3:
            st.metric("Sans souscription", total_sans_souscription)
        
        # Créer un graphique en camembert pour la répartition globale
        donnees_camembert = pd.DataFrame({
            'Catégorie': ['Avec souscription', 'Sans souscription'],
            'Nombre': [total_avec_souscription, total_sans_souscription]
        })
        
        fig = px.pie(
            donnees_camembert,
            values='Nombre',
            names='Catégorie',
            title="Répartition des Entretiens avec/sans Souscription",
            hole=0.4,
            color_discrete_map={
                'Avec souscription': '#4CAF50',  # Vert
                'Sans souscription': '#FFA500'  # Orange
            }
        )
        
        # Mise en forme du graphique
        fig.update_layout(
            template="plotly_white"
        )
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Créer un graphique en barres pour la répartition par conseiller
        fig_barres = px.bar(
            analyse_souscriptions.sort_values('Total_Entretiens', ascending=False),
            x='Conseiller',
            y=['Avec_Souscription', 'Sans_Souscription'],
            title="Répartition des Entretiens avec/sans Souscription par Conseiller",
            labels={
                'Conseiller': 'Conseiller',
                'value': "Nombre d'Entretiens",
                'variable': 'Type'
            },
            barmode='stack',
            color_discrete_map={
                'Avec_Souscription': '#4CAF50',  # Vert
                'Sans_Souscription': '#FFA500'  # Orange
            }
        )
        
        # Renommer les légendes
        fig_barres.update_layout(
            legend=dict(
                title="",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_tickangle=-45
        )
        
        # Renommer les catégories dans la légende
        newnames = {'Avec_Souscription': 'Avec souscription', 'Sans_Souscription': 'Sans souscription'}
        fig_barres.for_each_trace(lambda t: t.update(name = newnames[t.name]))
        
        # Afficher le graphique
        st.plotly_chart(fig_barres, use_container_width=True)
        
        # Afficher le tableau des données
        st.write("### Détail par conseiller")
        tableau_souscriptions = analyse_souscriptions[['Conseiller', 'Total_Entretiens', 'Avec_Souscription', 'Sans_Souscription', 'Taux_Conversion']]
        tableau_souscriptions.columns = ['Conseiller', "Total des entretiens", 'Avec souscription', 'Sans souscription', 'Taux de conversion (%)']  
        tableau_souscriptions = tableau_souscriptions.sort_values('Total des entretiens', ascending=False)
        st.dataframe(tableau_souscriptions, use_container_width=True)
        
        # Bouton d'export
        create_download_button(tableau_souscriptions, "analyse_souscriptions_entretiens", "souscriptions_entretiens")
    else:
        st.warning("⚠️ La colonne 'Dernière souscription' n'a pas été trouvée dans les données d'entretiens. Impossible de réaliser l'analyse des entretiens avec/sans souscription.")
        st.info("Colonnes disponibles: " + ", ".join(df_filtre.columns))
    
    # Prétraitement des données de RDV si disponibles
    if df_rdv is not None and 'Mois' not in df_rdv.columns:
        date_cols_rdv = ['Date', 'date', 'Date de création', 'date_de_creation', 'Date du RDV', 'date_rdv']
        date_col_rdv = next((col for col in date_cols_rdv if col in df_rdv.columns), None)
        
        if date_col_rdv:
            df_rdv[date_col_rdv] = safe_to_datetime(df_rdv[date_col_rdv])
            df_rdv = extract_conseiller(df_rdv)
            df_rdv['Mois'] = df_rdv[date_col_rdv].dt.to_period('M').astype(str)
    
    # Prétraitement des données de souscriptions si disponibles
    if df_souscriptions is not None and 'Mois' not in df_souscriptions.columns:
        date_cols_souscriptions = ['Date de souscription', 'Date', 'date', 'Date de création', 'date_de_creation']
        date_col_souscription = next((col for col in date_cols_souscriptions if col in df_souscriptions.columns), None)
        
        if date_col_souscription:
            df_souscriptions[date_col_souscription] = safe_to_datetime(df_souscriptions[date_col_souscription])
            df_souscriptions = extract_conseiller(df_souscriptions)
            df_souscriptions['Mois'] = df_souscriptions[date_col_souscription].dt.to_period('M').astype(str)
