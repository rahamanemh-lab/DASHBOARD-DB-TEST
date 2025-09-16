"""
Fonctions d'analyse des souscriptions immobilières pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from utils.data_processing import adjust_dates_to_month_range, extract_conseiller
from utils.data_processing_debug import safe_to_datetime_debug as safe_to_datetime, safe_to_numeric_debug as safe_to_numeric
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link

# Définir les constantes d'objectif localement
OBJECTIF_MENSUEL_IMMO = 20  # Objectif mensuel de dossiers immobiliers
OBJECTIF_ANNUEL_IMMO = 220  # Objectif annuel de dossiers immobiliers

# Fonction pour assurer que les clés existent dans un DataFrame pivot
def ensure_pivot_keys(df, keys):
    """Assure que les clés spécifiées existent dans un DataFrame pivot/crosstab.
    
    Cette fonction est utilisée pour éviter les KeyError lors de l'accès à des
    catégories qui pourraient ne pas être présentes dans un tableau croisé.
    
    Args:
        df (DataFrame): DataFrame pivot/crosstab à vérifier
        keys (list): Liste des clés à vérifier/ajouter
        
    Returns:
        DataFrame: DataFrame avec les clés assurées
    """
    if df is None:
        # Retourner un DataFrame vide avec les clés demandées
        return pd.DataFrame(index=keys)
        
    # Pour chaque clé demandée
    for key in keys:
        # Si la clé n'existe pas dans l'index
        if key not in df.index:
            # Ajouter une ligne avec des valeurs NaN
            df.loc[key] = np.nan
    
    return df


@st.cache_data
def ensure_column_types(df):
    """Assure la cohérence des types de colonnes pour éviter les erreurs de conversion Arrow.
    
    Args:
        df: DataFrame à traiter
        
    Returns:
        DataFrame avec les types de colonnes corrigés
    """
    # Créer une copie pour éviter de modifier l'original
    df = df.copy()
    
    # Liste des colonnes problématiques connues
    problematic_columns = ['Dernière Souscription', 'Dernière souscription', 'Derniere souscription']
    
    # Convertir les colonnes problématiques en chaînes de caractères
    for col in problematic_columns:
        if col in df.columns:
            # Convertir explicitement en string pour éviter les erreurs Arrow
            df[col] = df[col].astype(str)
    
    return df


def analyser_souscriptions_immo(df):
    """Analyse des souscriptions IMMO.
    
    Args:
        df (DataFrame): DataFrame contenant les données de souscriptions immobilières
    """
    st.header("🏠 Analyse des Souscriptions Immobilières")
    
    # Ajouter un bouton pour vider le cache
    if st.button("🔄 Vider le cache et rafraîchir les données", key="clear_cache_button"):
        st.cache_data.clear()
        st.success("✅ Cache vidé avec succès ! Les données seront rechargées.")
        st.rerun()
    
    # Vérification des colonnes requises
    colonnes_requises = ["Client", "Statut", "Conseiller affecté", "Date de création"]
    colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
    if colonnes_manquantes:
        st.error(f"❌ Colonnes manquantes : {', '.join(colonnes_manquantes)}")
        st.write("Colonnes disponibles:", ", ".join(df.columns))
        return
        
    # Jauge d'objectif annuel
    total_dossiers = len(df)
    pourcentage_objectif = min(100, total_dossiers / OBJECTIF_ANNUEL_IMMO * 100)
    
    st.subheader("🏁 Objectif Annuel de Dossiers Immobiliers")
    st.info(f"Objectif annuel: {OBJECTIF_ANNUEL_IMMO} dossiers immobiliers")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Dossiers créés", f"{total_dossiers:,}", f"{pourcentage_objectif:.1f}% de l'objectif")
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=total_dossiers,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': OBJECTIF_ANNUEL_IMMO, 'position': "top"},
            gauge={
                'axis': {'range': [0, OBJECTIF_ANNUEL_IMMO], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, OBJECTIF_ANNUEL_IMMO * 0.3], 'color': "red"},
                    {'range': [OBJECTIF_ANNUEL_IMMO * 0.3, OBJECTIF_ANNUEL_IMMO * 0.7], 'color': "orange"},
                    {'range': [OBJECTIF_ANNUEL_IMMO * 0.7, OBJECTIF_ANNUEL_IMMO], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': OBJECTIF_ANNUEL_IMMO
                }
            },
            title={'text': "Progression vers l'objectif annuel"}
        ))
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Prétraitement des données
    df = extract_conseiller(df)
    
    # Créer une colonne Date si elle n'existe pas
    if 'Date' not in df.columns and 'Date de création' in df.columns:
        df['Date'] = safe_to_datetime(df['Date de création'])
    
    # Filtrer les données valides
    df_filtre = df[df['Date'].notna()] if 'Date' in df.columns else df
    if df_filtre.empty:
        st.error("❌ Aucune donnée valide après filtrage.")
        return
    
    # Créer une colonne Mois pour l'analyse mensuelle
    if 'Mois' not in df_filtre.columns and 'Date' in df_filtre.columns:
        df_filtre['Mois'] = pd.to_datetime(df_filtre['Date']).dt.strftime('%Y-%m')
    
    # Ajouter les colonnes Premier_Jour_Mois et Dernier_Jour_Mois pour l'analyse temporelle
    if 'Date' in df_filtre.columns:
        # Convertir la colonne Date en datetime si ce n'est pas déjà fait
        df_filtre['Date'] = pd.to_datetime(df_filtre['Date'])
        # Créer la colonne Premier_Jour_Mois (premier jour du mois)
        df_filtre['Premier_Jour_Mois'] = df_filtre['Date'].dt.to_period('M').dt.to_timestamp()
        # Créer la colonne Dernier_Jour_Mois (dernier jour du mois)
        df_filtre['Dernier_Jour_Mois'] = (df_filtre['Premier_Jour_Mois'] + pd.DateOffset(months=1) - pd.DateOffset(days=1))
    
    # Analyse des souscriptions immobilières par mois
    st.subheader("📈 Analyse des Souscriptions Immobilières par Mois")
    
    # Grouper par mois
    if 'Mois' in df_filtre.columns:
        souscriptions_par_mois = df_filtre.groupby('Mois').size().reset_index(name='Nombre')
        souscriptions_par_mois = souscriptions_par_mois.sort_values('Mois')
        
        # Graphique des souscriptions par mois
        fig_mois = px.bar(
            souscriptions_par_mois,
            x='Mois',
            y='Nombre',
            title="Nombre de Dossiers Immobiliers par Mois",
            text='Nombre'
        )
        fig_mois.update_traces(texttemplate='%{text}', textposition='outside')
        fig_mois.update_layout(xaxis_title="Mois", yaxis_title="Nombre de dossiers")
        
        # Ajouter une ligne horizontale pour l'objectif mensuel
        fig_mois.add_hline(
            y=OBJECTIF_MENSUEL_IMMO,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Objectif mensuel: {OBJECTIF_MENSUEL_IMMO}",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig_mois, use_container_width=True)
        
        # Tableau des souscriptions par mois
        st.write("### Détail des Dossiers par Mois")
        st.dataframe(souscriptions_par_mois, use_container_width=True)
    else:
        st.warning("⚠️ Impossible d'analyser par mois : colonne 'Mois' manquante")
    
    # Analyse globale de tous les dossiers pour tous les mois et tous les conseillers
    st.header("🌐 Analyse Globale - Tous Conseillers / Tous Mois")
    
    if 'Mois' in df_filtre.columns and 'Conseiller' in df_filtre.columns:
        # Approche alternative sans utiliser pivot_table pour éviter les problèmes de type
        # Créer un DataFrame pour l'analyse croisée mois/conseiller
        df_global = df_filtre.copy()
        
        # Compter les occurrences par mois et conseiller
        df_count = df_global.groupby(['Mois', 'Conseiller']).size().reset_index(name='Nombre')
        
        # Obtenir les listes uniques de mois et conseillers
        mois_list = sorted(df_count['Mois'].unique())
        conseillers_list = sorted(df_count['Conseiller'].unique())
        
        # Créer un tableau croisé dynamique manuellement
        pivot_data = {}
        
        # Initialiser le dictionnaire avec des zéros
        for mois in mois_list:
            pivot_data[mois] = {conseiller: 0 for conseiller in conseillers_list}
        
        # Remplir avec les données réelles
        for _, row in df_count.iterrows():
            pivot_data[row['Mois']][row['Conseiller']] = row['Nombre']
        
        # Convertir en DataFrame
        pivot_df = pd.DataFrame(pivot_data).T
        
        # Ajouter une colonne de total
        pivot_df['Total'] = pivot_df.sum(axis=1)
        
        # Afficher le tableau croisé dynamique
        st.write("### Tableau croisé dynamique - Dossiers par Mois et par Conseiller")
        st.dataframe(pivot_df, use_container_width=True)
        
        # Créer un DataFrame séparé pour les totaux par conseiller
        totaux_conseillers = pd.DataFrame({conseiller: pivot_df[conseiller].sum() for conseiller in conseillers_list}, index=['Total'])
        totaux_conseillers['Total'] = totaux_conseillers.sum(axis=1)
        
        # Afficher les totaux par conseiller
        st.write("### Totaux par Conseiller")
        st.dataframe(totaux_conseillers, use_container_width=True)
        
        # Graphique de l'évolution mensuelle globale
        evolution_data = {'Mois': mois_list, 'Nombre de dossiers': [pivot_df.loc[mois, 'Total'] for mois in mois_list]}
        evolution_mensuelle = pd.DataFrame(evolution_data)
        
        fig_evolution_globale = px.line(
            evolution_mensuelle,
            x='Mois',
            y='Nombre de dossiers',
            title="Évolution mensuelle globale des dossiers immobiliers",
            markers=True
        )
        
        # Ajouter une ligne d'objectif mensuel
        if 'OBJECTIF_MENSUEL_IMMO' in globals():
            fig_evolution_globale.add_hline(
                y=OBJECTIF_MENSUEL_IMMO,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Objectif: {OBJECTIF_MENSUEL_IMMO}",
                annotation_position="top right"
            )
        
        st.plotly_chart(fig_evolution_globale, use_container_width=True)
        
        # Heatmap des performances par conseiller et par mois
        st.write("### Heatmap des performances par conseiller et par mois")
        
        # Créer une matrice pour la heatmap
        heatmap_data = pivot_df.drop('Total', axis=1).values
        
        fig_heatmap = px.imshow(
            heatmap_data,
            x=conseillers_list,
            y=mois_list,
            color_continuous_scale='Viridis',
            title="Heatmap des dossiers par conseiller et par mois",
            labels=dict(x="Conseiller", y="Mois", color="Nombre de dossiers")
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Téléchargement des données
        csv_global = pivot_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Télécharger l'analyse globale (CSV)",
            data=csv_global,
            file_name=f"analyse_globale_immo_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_global"
        )
    else:
        st.warning("⚠️ Impossible d'effectuer l'analyse globale : colonnes 'Mois' ou 'Conseiller' manquantes")
    
    # Analyse par conseiller
    st.header("👨‍💼 Analyse par Conseiller")
    
    if 'Conseiller' in df_filtre.columns:
        # Grouper par conseiller
        souscriptions_par_conseiller = df_filtre.groupby('Conseiller').size().reset_index(name='Nombre')
        souscriptions_par_conseiller = souscriptions_par_conseiller.sort_values('Nombre', ascending=False)
        
        # Graphique des souscriptions par conseiller
        fig_conseiller = px.bar(
            souscriptions_par_conseiller.head(10),  # Top 10 conseillers
            x='Conseiller',
            y='Nombre',
            title="Top 10 Conseillers - Nombre de Dossiers Immobiliers",
            text='Nombre'
        )
        fig_conseiller.update_traces(texttemplate='%{text}', textposition='outside')
        fig_conseiller.update_layout(xaxis_title="Conseiller", yaxis_title="Nombre de dossiers")
        st.plotly_chart(fig_conseiller, use_container_width=True)
        
        # Tableau des souscriptions par conseiller
        st.write("### Détail des Dossiers par Conseiller")
        st.dataframe(souscriptions_par_conseiller, use_container_width=True)
    else:
        st.warning("⚠️ Impossible d'analyser par conseiller : colonne 'Conseiller' manquante")
        
    # Analyse par conseiller par mois
    st.header("📅 Analyse par Conseiller par Mois")
    
    if 'Conseiller' in df_filtre.columns and 'Mois' in df_filtre.columns:
        # Afficher la liste des conseillers disponibles
        conseillers_disponibles = sorted(df_filtre['Conseiller'].unique())
        st.write(f"### Conseillers disponibles ({len(conseillers_disponibles)})")
        
        # Afficher les noms des conseillers dans un format compact
        col1, col2, col3 = st.columns(3)
        conseillers_par_colonne = len(conseillers_disponibles) // 3 + 1
        
        for i, conseiller in enumerate(conseillers_disponibles):
            if i < conseillers_par_colonne:
                col1.write(f"• {conseiller}")
            elif i < 2 * conseillers_par_colonne:
                col2.write(f"• {conseiller}")
            else:
                col3.write(f"• {conseiller}")
        
        # Option pour sélectionner un ou plusieurs conseillers
        option_selection = st.radio(
            "Mode de sélection des conseillers",
            ["Un seul conseiller", "Plusieurs conseillers"],
            key="mode_selection_conseillers"
        )
        
        if option_selection == "Un seul conseiller":
            # Sélection d'un seul conseiller (comportement actuel)
            conseiller_selectionne = st.selectbox(
                "Sélectionner un conseiller", 
                options=conseillers_disponibles, 
                key="select_conseiller_immo_mois"
            )
            
            # Filtrer les données pour le conseiller sélectionné
            df_conseiller = df_filtre[df_filtre['Conseiller'] == conseiller_selectionne].copy()
            
            if not df_conseiller.empty:
                # Afficher les métriques clés pour le conseiller sélectionné
                nb_total = len(df_conseiller)
                st.metric(f"Nombre total de dossiers - {conseiller_selectionne}", f"{nb_total:,}")
                
                # Évolution mensuelle du conseiller
                evolution_conseiller = df_conseiller.groupby('Mois').size().reset_index(name='Nombre')
                
                # Trier par mois
                evolution_conseiller = evolution_conseiller.sort_values('Mois')
                
                # Graphique d'évolution mensuelle
                fig_evolution = px.line(
                    evolution_conseiller,
                    x='Mois',
                    y='Nombre',
                    title=f"Évolution Mensuelle des Dossiers - {conseiller_selectionne}",
                    markers=True,
                    line_shape='linear'
                )
                fig_evolution.update_traces(line=dict(width=3), marker=dict(size=10))
                fig_evolution.update_layout(yaxis_title="Nombre de dossiers")
                
                # Ajouter une ligne horizontale pour l'objectif mensuel
                fig_evolution.add_hline(
                    y=OBJECTIF_MENSUEL_IMMO / 10,  # Diviser l'objectif mensuel par 10 pour le rendre plus réaliste par conseiller
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Objectif mensuel par conseiller",
                    annotation_position="top right"
                )
                
                st.plotly_chart(fig_evolution, use_container_width=True)
                
                # Tableau détaillé par mois
                st.subheader(f"📋 Détail Mensuel pour {conseiller_selectionne}")
                st.dataframe(evolution_conseiller, use_container_width=True)
                
                # Téléchargement des données mensuelles
                csv_mensuel = evolution_conseiller.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Télécharger l'évolution mensuelle de {conseiller_selectionne} (CSV)",
                    data=csv_mensuel,
                    file_name=f"evolution_mensuelle_{conseiller_selectionne}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_mensuel"
                )
            else:
                st.warning(f"Aucune donnée disponible pour le conseiller {conseiller_selectionne}")
        else:
            # Sélection de plusieurs conseillers
            conseillers_selectionnes = st.multiselect(
                "Sélectionner un ou plusieurs conseillers",
                options=conseillers_disponibles,
                default=conseillers_disponibles[:min(3, len(conseillers_disponibles))],
                key="select_multiple_conseillers_immo_mois"
            )
            
            if conseillers_selectionnes:
                # Filtrer les données pour les conseillers sélectionnés
                df_conseillers = df_filtre[df_filtre['Conseiller'].isin(conseillers_selectionnes)].copy()
                
                if not df_conseillers.empty:
                    # Afficher les métriques clés pour les conseillers sélectionnés
                    nb_total = len(df_conseillers)
                    st.metric(f"Nombre total de dossiers - {len(conseillers_selectionnes)} conseillers", f"{nb_total:,}")
                    
                    # Évolution mensuelle des conseillers sélectionnés
                    evolution_conseillers = df_conseillers.groupby(['Mois', 'Conseiller']).size().reset_index(name='Nombre')
                    
                    # Trier par mois
                    evolution_conseillers = evolution_conseillers.sort_values(['Mois', 'Conseiller'])
                    
                    # Graphique d'évolution mensuelle comparatif
                    fig_evolution_multi = px.line(
                        evolution_conseillers,
                        x='Mois',
                        y='Nombre',
                        color='Conseiller',
                        title=f"Évolution Mensuelle Comparative des Dossiers",
                        markers=True,
                        line_shape='linear'
                    )
                    fig_evolution_multi.update_traces(marker=dict(size=8))
                    fig_evolution_multi.update_layout(
                        yaxis_title="Nombre de dossiers",
                        legend_title="Conseillers",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    # Ajouter une ligne horizontale pour l'objectif mensuel
                    fig_evolution_multi.add_hline(
                        y=OBJECTIF_MENSUEL_IMMO / 10,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Objectif mensuel par conseiller",
                        annotation_position="top right"
                    )
                    
                    st.plotly_chart(fig_evolution_multi, use_container_width=True)
                    
                    # Tableau détaillé par mois et conseiller
                    st.subheader(f"📋 Détail Mensuel pour les Conseillers Sélectionnés")
                    
                    # Créer un tableau croisé dynamique pour une meilleure visualisation
                    pivot_conseillers = evolution_conseillers.pivot_table(
                        index='Mois',
                        columns='Conseiller',
                        values='Nombre',
                        fill_value=0
                    ).reset_index()
                    
                    st.dataframe(pivot_conseillers, use_container_width=True)
                    
                    # Téléchargement des données mensuelles
                    csv_mensuel_multi = evolution_conseillers.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Télécharger l'évolution mensuelle des conseillers sélectionnés (CSV)",
                        data=csv_mensuel_multi,
                        file_name=f"evolution_mensuelle_conseillers_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="download_mensuel_multi"
                    )
                else:
                    st.warning(f"Aucune donnée disponible pour les conseillers sélectionnés")
            else:
                st.warning("Veuillez sélectionner au moins un conseiller")
    
        # Comparaison entre conseillers par mois
        st.subheader("🔍 Comparaison entre Conseillers par Mois")
        
        # Créer un sélecteur de mois
        mois_disponibles = sorted(df_filtre['Mois'].unique())
        mois_selectionne = st.selectbox("Sélectionner un mois", options=mois_disponibles, key="select_mois_immo")
        
        # Filtrer les données pour le mois sélectionné
        df_mois = df_filtre[df_filtre['Mois'] == mois_selectionne].copy()
        
        if not df_mois.empty:
            # Calculer les statistiques par conseiller pour le mois sélectionné
            stats_mois = df_mois.groupby('Conseiller').size().reset_index(name='Nombre')
            
            # Trier par nombre total décroissant
            stats_mois = stats_mois.sort_values('Nombre', ascending=False)
            
            # Graphique des conseillers du mois
            fig_conseillers_mois = px.bar(
                stats_mois.head(10),  # Top 10 conseillers
                x='Conseiller',
                y='Nombre',
                title=f"Top 10 Conseillers - Nombre de Dossiers - {mois_selectionne}",
                text='Nombre'
            )
            fig_conseillers_mois.update_traces(texttemplate='%{text}', textposition='outside')
            fig_conseillers_mois.update_layout(xaxis_title="Conseiller", yaxis_title="Nombre de dossiers")
            st.plotly_chart(fig_conseillers_mois, use_container_width=True)
            
            # Tableau détaillé par conseiller du mois
            st.dataframe(stats_mois, use_container_width=True)
            
            # Téléchargement des données du mois
            csv_mois = stats_mois.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Télécharger les données de {mois_selectionne} (CSV)",
                data=csv_mois,
                file_name=f"analyse_conseillers_{mois_selectionne}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_mois"
            )
        else:
            st.warning(f"Aucune donnée disponible pour le mois {mois_selectionne}")
    else:
        st.error("Impossible d'effectuer l'analyse par conseiller par mois : colonnes 'Conseiller' ou 'Mois' manquantes")
        st.write(f"Colonnes disponibles : {df_filtre.columns.tolist()}")



def analyse_dossiers_immo(df, title):
    """Analyse du nombre de dossiers IMMO par mois, conseiller et statut.
    
    Args:
        df (DataFrame): DataFrame contenant les données
        title (str): Titre du graphique
    """
    st.subheader(title)
    
    # Vérifier les colonnes requises
    if not all(col in df.columns for col in ['Date', 'Statut', 'Conseiller']):
        st.error("❌ Colonnes manquantes pour l'analyse des dossiers IMMO.")
        return
    
    # Statistiques globales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nb_dossiers = len(df)
        st.metric("Nombre Total de Dossiers", nb_dossiers)
    
    with col2:
        nb_conseillers = df['Conseiller'].nunique()
        st.metric("Nombre de Conseillers", nb_conseillers)
    
    with col3:

        
        # Calculer le mois en cours
        mois_en_cours = datetime.now().strftime('%Y-%m')
        
        # Compter les dossiers du mois en cours
        dossiers_mois_en_cours = df[df['Mois'] == mois_en_cours].shape[0]
        
        # Calculer le pourcentage de l'objectif
        pourcentage_objectif = (dossiers_mois_en_cours / OBJECTIF_DOSSIERS_IMMO) * 100
        
        st.metric(
            f"Dossiers du Mois ({mois_en_cours})",
            f"{dossiers_mois_en_cours} / {OBJECTIF_DOSSIERS_IMMO}",
            f"{pourcentage_objectif:.1f}%"
        )
    
    # Analyse par mois
    st.write("### Dossiers par Mois")
    
    # Grouper par mois
    df_mois = df.groupby('Mois').size().reset_index(name='Nombre_Dossiers')
    
    # Trier par mois
    df_mois = df_mois.sort_values('Mois')
    
    # Ajouter l'objectif mensuel
    from dashboard_souscriptions import OBJECTIF_DOSSIERS_IMMO
    df_mois['Objectif'] = OBJECTIF_DOSSIERS_IMMO
    
    # Créer le graphique
    fig = go.Figure()
    
    # Ajouter les barres pour le nombre de dossiers
    fig.add_trace(go.Bar(
        x=df_mois['Mois'],
        y=df_mois['Nombre_Dossiers'],
        name='Nombre de Dossiers',
        marker_color='royalblue',
        text=df_mois['Nombre_Dossiers'],
        textposition='auto'
    ))
    
    # Ajouter la ligne d'objectif
    fig.add_trace(go.Scatter(
        x=df_mois['Mois'],
        y=df_mois['Objectif'],
        mode='lines+markers',
        name=f'Objectif ({OBJECTIF_DOSSIERS_IMMO})',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Mise en forme du graphique
    fig.update_layout(
        title="Nombre de Dossiers IMMO par Mois",
        xaxis_title="Mois",
        yaxis_title="Nombre de Dossiers",
        legend_title="Légende",
        template="plotly_white",
        height=500
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse par statut
    st.write("### Dossiers par Statut")
    
    # Grouper par statut
    df_statut = df.groupby('Statut').size().reset_index(name='Nombre_Dossiers')
    
    # Trier par nombre de dossiers décroissant
    df_statut = df_statut.sort_values('Nombre_Dossiers', ascending=False)
    
    # Créer le graphique en camembert
    fig = px.pie(
        df_statut,
        values='Nombre_Dossiers',
        names='Statut',
        title="Répartition des Dossiers par Statut",
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
    
    # Analyse globale par mois
    st.write("### Analyse Globale des Souscriptions par Mois")
    
    # Vérifier si la colonne Mois existe, sinon la créer
    if 'Mois' not in df.columns and 'Date' in df.columns:
        df['Mois'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
    
    if 'Mois' in df.columns:
        # Grouper par mois
        df_mois = df.groupby('Mois').agg(
            Nombre_Dossiers=('Conseiller', 'count'),
            Nombre_Souscriptions=('A souscrit', lambda x: x.sum() if 'A souscrit' in df.columns else 0)
        ).reset_index()
        
        # Trier par mois
        df_mois = df_mois.sort_values('Mois')
        
        # Calculer le taux de conversion
        df_mois['Taux_Conversion'] = (df_mois['Nombre_Souscriptions'] / df_mois['Nombre_Dossiers'] * 100).fillna(0)
        
        # Créer un graphique d'évolution mensuelle
        fig = px.line(
            df_mois,
            x='Mois',
            y=['Nombre_Dossiers', 'Nombre_Souscriptions'],
            title="Évolution Mensuelle des Dossiers et Souscriptions Immobilières",
            labels={
                'value': 'Nombre',
                'Mois': 'Mois',
                'variable': 'Type'
            },
            color_discrete_map={
                'Nombre_Dossiers': '#1f77b4',
                'Nombre_Souscriptions': '#2ca02c'
            }
        )
        
        # Ajouter une ligne horizontale pour l'objectif mensuel
        fig.add_hline(
            y=OBJECTIF_MENSUEL_IMMO,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Objectif mensuel: {OBJECTIF_MENSUEL_IMMO}",
            annotation_position="top right"
        )
        
        # Mise en forme du graphique
        fig.update_layout(
            template="plotly_white",
            height=500,
            legend_title="Légende"
        )
        
        # Afficher le graphique
        st.plotly_chart(fig, use_container_width=True)
        
        # Afficher le tableau des données
        df_mois_display = df_mois.copy()
        df_mois_display['Taux_Conversion'] = df_mois_display['Taux_Conversion'].apply(lambda x: f"{x:.1f}%")
        df_mois_display.columns = ['Mois', 'Nombre de Dossiers', 'Nombre de Souscriptions', 'Taux de Conversion']
        st.dataframe(df_mois_display, use_container_width=True)
        
        # Téléchargement des données
        csv_mois = df_mois_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger l'analyse mensuelle (CSV)",
            data=csv_mois,
            file_name=f"analyse_mensuelle_immo_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_analyse_mensuelle"
        )
    else:
        st.error("❌ Impossible d'effectuer l'analyse par mois : colonne 'Mois' manquante")
    
    # Analyse par conseiller
    st.write("### Dossiers par Conseiller")
    
    # Grouper par conseiller
    df_conseiller = df.groupby('Conseiller').size().reset_index(name='Nombre_Dossiers')
    
    # Trier par nombre de dossiers décroissant
    # S'assurer que Nombre_Dossiers est numérique avant le tri
    try:
        df_conseiller['Nombre_Dossiers'] = pd.to_numeric(df_conseiller['Nombre_Dossiers'], errors='coerce')
        # Ajouter un log pour déboguer
        st.write(f"DEBUG - Types dans df_conseiller['Nombre_Dossiers']: {df_conseiller['Nombre_Dossiers'].apply(type).value_counts()}")
        df_conseiller = df_conseiller.sort_values('Nombre_Dossiers', ascending=False)
    except Exception as e:
        st.error(f"Erreur lors du tri de df_conseiller: {e}")
        st.write(f"Colonnes disponibles: {df_conseiller.columns.tolist()}")
        # Essayer un tri simple sans conversion
        try:
            df_conseiller = df_conseiller.reset_index()
        except Exception as e2:
            st.error(f"Impossible de réinitialiser l'index: {e2}")
    
    # Créer le graphique
    fig = px.bar(
        df_conseiller,
        x='Conseiller',
        y='Nombre_Dossiers',
        text='Nombre_Dossiers',
        title="Nombre de Dossiers par Conseiller",
        labels={
            'Conseiller': 'Conseiller',
            'Nombre_Dossiers': 'Nombre de Dossiers'
        },
        height=500,
        color='Nombre_Dossiers',
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
    
    # Analyse croisée conseiller-statut
    st.write("### Analyse Croisée Conseiller-Statut")
    
    # Créer un tableau croisé dynamique
    pivot = pd.crosstab(
        index=df['Conseiller'],
        columns=df['Statut'],
        margins=True,
        margins_name='Total'
    )
    
    # Trier par total décroissant
    pivot = pivot.sort_values('Total', ascending=False)
    
    # Sécuriser l'accès aux clés du pivot
    pivot = ensure_pivot_keys(pivot, ['Avec RDV', 'Sans RDV', 'Total'])
    
    # Afficher le tableau
    st.write(pivot)
    
    # Exporter en CSV
    create_download_button(pivot, "dossiers_immo_conseiller_statut", "immo_1")
    
    # Analyse par mois et statut
    st.write("### Analyse par Mois et Statut")
    
    # Créer un tableau croisé dynamique
    pivot_mois_statut = pd.crosstab(
        index=df['Mois'],
        columns=df['Statut'],
        margins=True,
        margins_name='Total'
    )
    
    # Trier par mois
    pivot_mois_statut = pivot_mois_statut.sort_index()
    
    # Afficher le tableau
    st.write(pivot_mois_statut)
    
    # Exporter en CSV
    create_download_button(pivot_mois_statut, "dossiers_immo_mois_statut", "immo_2")


def analyser_suivi_immo(df):
    """
    Analyse des souscriptions immobilières par mois et par conseiller.
    
    Permet d'analyser les souscriptions immobilières, avec une jauge d'objectif annuel,
    et les statistiques par conseiller et par mois.
    
    Args:
        df (DataFrame): DataFrame contenant les données immobilières
    """
    st.header("🏢 Analyse des Souscriptions Immobilières")
    
    # Vérification si le DataFrame est None
    if df is None:
        st.warning("⚠️ Aucune donnée immobilière disponible.")
        return
        
    # Jauge d'objectif annuel
    total_dossiers = len(df)
    pourcentage_objectif = min(100, total_dossiers / OBJECTIF_ANNUEL_IMMO * 100)
    
    st.subheader("🏁 Objectif Annuel de Dossiers Immobiliers")
    st.info(f"Objectif annuel: {OBJECTIF_ANNUEL_IMMO} dossiers immobiliers")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Dossiers créés", f"{total_dossiers:,}", f"{pourcentage_objectif:.1f}% de l'objectif")
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=total_dossiers,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': OBJECTIF_ANNUEL_IMMO, 'position': "top"},
            gauge={
                'axis': {'range': [0, OBJECTIF_ANNUEL_IMMO], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, OBJECTIF_ANNUEL_IMMO * 0.3], 'color': "red"},
                    {'range': [OBJECTIF_ANNUEL_IMMO * 0.3, OBJECTIF_ANNUEL_IMMO * 0.7], 'color': "orange"},
                    {'range': [OBJECTIF_ANNUEL_IMMO * 0.7, OBJECTIF_ANNUEL_IMMO], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': OBJECTIF_ANNUEL_IMMO
                }
            },
            title={'text': "Progression vers l'objectif annuel"}
        ))
        
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Vérification des colonnes disponibles et adaptation de l'analyse
    colonnes_requises = ["Client", "Statut", "Conseiller affecté", "Date de création"]
    
    # Vérifier quelles colonnes sont disponibles
    colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
    colonnes_disponibles = [col for col in colonnes_requises if col in df.columns]
    
    if colonnes_manquantes:
        st.warning(f"⚠️ Certaines colonnes sont manquantes : {', '.join(colonnes_manquantes)}")
        st.info("L'analyse sera adaptée en fonction des colonnes disponibles.")
        
        # Vérifier les colonnes minimales nécessaires
        colonnes_minimales = ["Statut", "Conseiller"]
        colonnes_minimales_manquantes = [col for col in colonnes_minimales if col not in df.columns and col.lower() not in [c.lower() for c in df.columns]]
        
        if colonnes_minimales_manquantes:
            st.error(f"❌ Impossible de poursuivre l'analyse : colonnes essentielles manquantes ({', '.join(colonnes_minimales_manquantes)})")
            st.write("Colonnes disponibles:", ", ".join(df.columns))
            return
    
    # Prétraitement des données
    # Utiliser la colonne Conseiller si disponible, sinon chercher Conseiller affecté
    if "Conseiller affecté" in df.columns:
        df = extract_conseiller(df, "Conseiller affecté")
    else:
        df = extract_conseiller(df)  # La fonction extract_conseiller cherchera parmi les noms de colonnes possibles
    
    # Convertir les dates si les colonnes existent
    if "Date de création" in df.columns:
        df['Date de création'] = safe_to_datetime(df['Date de création'])
    elif "Date de souscription" in df.columns:
        df['Date de création'] = safe_to_datetime(df['Date de souscription'])
        
    # Créer la colonne 'Date' si elle n'existe pas
    if 'Date' not in df.columns:
        if 'Date de création' in df.columns:
            df['Date'] = df['Date de création']
        elif 'Date de souscription' in df.columns:
            df['Date'] = safe_to_datetime(df['Date de souscription'])
        else:
            # Créer une colonne Date par défaut avec la date du jour pour éviter les erreurs
            df['Date'] = datetime.now()
    
    # Nous n'utilisons plus la notion de RDV dans l'analyse immobilière
    # Mais nous créons une colonne par défaut pour éviter les erreurs
    df['A eu un RDV'] = False
        
    # Si la colonne État n'existe pas, utiliser Statut ou Étape comme alternative
    if "État" not in df.columns and "Statut" in df.columns:
        df['État'] = df['Statut']
    elif "État" not in df.columns and "Étape" in df.columns:
        df['État'] = df['Étape']
    elif "État" not in df.columns:
        df['État'] = "Non spécifié"
    
    # Gérer la détection des souscriptions
    if "Dernière souscription" in df.columns:
        # Une souscription existe si la colonne "Dernière souscription" contient une chaîne
        # qui correspond à un pattern spécifique (ex: "4549-Financement-IMMO-n°2139990,00 € - 20/06/2025")
        pattern = r'\d+-\w+-\w+-[n°]\d+'
        
        # Vérifier si les valeurs correspondent au pattern
        df['A souscrit'] = df['Dernière souscription'].astype(str).str.contains(pattern, regex=True, na=False)
    elif "Produit" in df.columns and "Montant" in df.columns:
        # Alternative: si un produit et un montant sont spécifiés, on considère qu'il y a eu souscription
        # Forcer la conversion de Montant en numérique avant la comparaison
        df['Montant'] = pd.to_numeric(df['Montant'], errors='coerce')
        # Ajouter un log pour déboguer
        st.write(f"DEBUG - Types dans df['Montant']: {df['Montant'].apply(type).value_counts()}")
        df['A souscrit'] = (df['Produit'].notna() & df['Montant'].notna() & (df['Montant'] > 0))
    elif "Statut" in df.columns:
        # Autre alternative: considérer comme souscrit si le statut contient des mots-clés comme "validé", "signé", etc.
        # Convertir la colonne Statut en string pour éviter les erreurs catégorielles
        df['Statut'] = df['Statut'].astype(str)
        souscription_keywords = ['valid', 'sign', 'souscri', 'finalis', 'act', 'clôtur']
        df['A souscrit'] = df['Statut'].str.lower().apply(
            lambda x: any(keyword in x for keyword in souscription_keywords)
        )
    else:
        # Si aucune information n'est disponible, créer une colonne par défaut
        df['A souscrit'] = False
        
    # Afficher des informations de diagnostic sur la détection des souscriptions
    with st.expander("Détails de détection des souscriptions"):
        st.write("Nombre de souscriptions détectées :", df['A souscrit'].sum())
        st.write("Pourcentage de souscriptions :", f"{(df['A souscrit'].mean() * 100):.1f}%")
    
    # Analyse par conseiller par mois
    st.subheader("📈 Analyse par conseiller par mois")
    
    if all(col in df.columns for col in ['Conseiller', 'Date']):
        # Préparer les données
        df_filtre = df.copy()
        df_filtre['Mois'] = df_filtre['Date'].dt.strftime('%Y-%m')
        
        # Grouper par conseiller et mois
        stats_conseiller_mois = df_filtre.groupby(['Conseiller', 'Mois']).size().reset_index(name='Nombre de dossiers')
        
        # Sélection du conseiller
        conseillers = ['Tous les conseillers'] + sorted(df_filtre['Conseiller'].unique())
        conseiller_selectionne = st.selectbox("Sélectionner un conseiller", conseillers, index=0)
        
        # Filtrer pour le conseiller sélectionné ou afficher tous les conseillers
        if conseiller_selectionne == 'Tous les conseillers':
            stats_conseiller = stats_conseiller_mois
            title = "Evolution mensuelle des dossiers pour tous les conseillers"
        else:
            stats_conseiller = stats_conseiller_mois[stats_conseiller_mois['Conseiller'] == conseiller_selectionne]
            title = f"Evolution mensuelle des dossiers pour {conseiller_selectionne}"
        
        if not stats_conseiller.empty:
            # Graphique d'évolution pour le(s) conseiller(s) sélectionné(s)
            if conseiller_selectionne == 'Tous les conseillers':
                fig = px.line(stats_conseiller, x='Mois', y='Nombre de dossiers', color='Conseiller',
                              title=title)
            else:
                fig = px.line(stats_conseiller, x='Mois', y='Nombre de dossiers',
                              title=title)
            
            # Ajouter une ligne horizontale pour l'objectif mensuel
            fig.add_hline(y=OBJECTIF_MENSUEL_IMMO, line_dash="dash", line_color="red",
                          annotation_text=f"Objectif mensuel: {OBJECTIF_MENSUEL_IMMO}", 
                          annotation_position="top right")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des statistiques
            st.write(f"Statistiques mensuelles pour {conseiller_selectionne}:")
            st.dataframe(stats_conseiller.sort_values('Mois'))
        else:
            st.warning(f"Aucune donnée disponible pour {conseiller_selectionne}")
        
        # Analyse par mois pour tous les conseillers
        st.subheader("Comparaison des conseillers par mois")
        mois_disponibles = ['Tous les mois'] + sorted(df_filtre['Mois'].unique())
        mois_selectionne = st.selectbox("Sélectionner un mois", mois_disponibles, index=0)
        
        # Filtrer selon le mois sélectionné ou afficher tous les mois
        if mois_selectionne == 'Tous les mois':
            stats_mois = stats_conseiller_mois.groupby('Conseiller')['Nombre de dossiers'].sum().reset_index().sort_values('Nombre de dossiers', ascending=False)
        else:
            stats_mois = stats_conseiller_mois[stats_conseiller_mois['Mois'] == mois_selectionne].sort_values('Nombre de dossiers', ascending=False)
        
        if not stats_mois.empty:
            # Adapter le titre selon l'option sélectionnée
            if mois_selectionne == 'Tous les mois':
                title = "Nombre total de dossiers par conseiller (tous mois confondus)"
                stats_title = "Statistiques pour tous les mois:"
            else:
                title = f"Nombre de dossiers par conseiller en {mois_selectionne}"
                stats_title = f"Statistiques pour {mois_selectionne}:"
            
            fig = px.bar(stats_mois, x='Conseiller', y='Nombre de dossiers', title=title)
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des statistiques
            st.write(stats_title)
            st.dataframe(stats_mois)
            
            # Bouton de téléchargement
            csv_mois = stats_mois.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Télécharger les données de {mois_selectionne} (CSV)",
                data=csv_mois,
                file_name=f"analyse_conseillers_immo_{mois_selectionne}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_mois_immo"
            )
        else:
            st.warning(f"Aucune donnée disponible pour le mois {mois_selectionne}")
    else:
        st.error("Impossible d'effectuer l'analyse par conseiller par mois : colonnes 'Conseiller' ou 'Date' manquantes")
        st.write(f"Colonnes disponibles : {df.columns.tolist()}")
    
    # Filtrage des Données
    st.subheader("🔍 Filtrage des Données")
    
    # Créer des filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        statuts = ['Tous'] + sorted(df['Statut'].unique().tolist())
        statut_filtre = st.selectbox("Statut", statuts)
    
    with col2:
        conseillers = ['Tous'] + sorted(df['Conseiller'].unique().tolist())
        conseiller_filtre = st.selectbox("Conseiller", conseillers)
    
    with col3:
        etats = ['Tous'] + sorted(df['État'].unique().tolist())
        etat_filtre = st.selectbox("État", etats)
    
    # Appliquer les filtres
    df_filtered = df.copy()
    
    if statut_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['Statut'] == statut_filtre]
    
    if conseiller_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['Conseiller'] == conseiller_filtre]
    
    if etat_filtre != 'Tous':
        df_filtered = df_filtered[df_filtered['État'] == etat_filtre]
    
    # Afficher les résultats filtrés
    st.write(f"### Résultats ({len(df_filtered)} clients)")
    # Assurer la cohérence des types de colonnes avant l'affichage
    df_display = ensure_column_types(df_filtered)
    st.write(df_display)
    
    # Exportation des données
    st.subheader("📤 Exportation des Données")
    create_download_button(df_filtered, "analyse_immo_suivi", "immo_7")
    
    # Analyse par conseiller par mois
    st.header("📅 Analyse par Conseiller par Mois")
    
    # Créer une colonne Mois si elle n'existe pas
    if 'Mois' not in df.columns and 'Date de création' in df.columns:
        df['Mois'] = pd.to_datetime(df['Date de création']).dt.strftime('%Y-%m')
    elif 'Mois' not in df.columns and 'Date' in df.columns:
        df['Mois'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')
    
    if 'Conseiller' in df.columns and 'Mois' in df.columns:
        # Créer un sélecteur de conseiller
        conseillers_disponibles = sorted(df['Conseiller'].unique())
        conseiller_selectionne = st.selectbox("Sélectionner un conseiller", options=conseillers_disponibles, key="select_conseiller_immo_mois")
        
        # Filtrer les données pour le conseiller sélectionné
        df_conseiller = df[df['Conseiller'] == conseiller_selectionne].copy()
        
        if not df_conseiller.empty:
            # Afficher les métriques clés pour le conseiller sélectionné
            col1, col2, col3 = st.columns(3)
            with col1:
                nb_total = len(df_conseiller)
                st.metric(f"Nombre total de dossiers - {conseiller_selectionne}", f"{nb_total:,}")
            with col2:
                nb_souscrit = df_conseiller['A souscrit'].sum() if 'A souscrit' in df_conseiller.columns else 0
                st.metric("Nombre de souscriptions", f"{nb_souscrit:,}")
            with col3:
                # Forcer la conversion en entier avant la comparaison
                nb_total = int(nb_total) if isinstance(nb_total, (int, float, str)) and str(nb_total).strip() else 0
                # Ajouter un log pour déboguer
                st.write(f"DEBUG - Type de nb_total: {type(nb_total)}, Valeur: {nb_total}")
                taux_conversion = (nb_souscrit / nb_total * 100) if nb_total > 0 else 0
                st.metric("Taux de conversion", f"{taux_conversion:.1f}%")
            
            # Évolution mensuelle du conseiller
            evolution_conseiller = df_conseiller.groupby('Mois').agg(
                Nombre_Total=('Conseiller', 'count'),
                Nombre_Souscriptions=('A souscrit', lambda x: x.sum() if 'A souscrit' in df_conseiller.columns else 0)
            ).reset_index()
            
            # Trier par mois
            evolution_conseiller = evolution_conseiller.sort_values('Mois')
            
            # Graphique d'évolution mensuelle
            fig_evolution = px.line(
                evolution_conseiller,
                x='Mois',
                y=['Nombre_Total', 'Nombre_Souscriptions'],
                title=f"Évolution Mensuelle des Dossiers - {conseiller_selectionne}",
                markers=True,
                line_shape='linear'
            )
            fig_evolution.update_traces(line=dict(width=3), marker=dict(size=10))
            fig_evolution.update_layout(yaxis_title="Nombre de dossiers", legend_title="Type")
            fig_evolution.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            # Calculer le taux de conversion par mois
            evolution_conseiller['Taux_Conversion'] = evolution_conseiller['Nombre_Souscriptions'] / evolution_conseiller['Nombre_Total'] * 100
            evolution_conseiller['Taux_Conversion'] = evolution_conseiller['Taux_Conversion'].fillna(0)
            
            # Graphique du taux de conversion
            fig_conversion = px.bar(
                evolution_conseiller,
                x='Mois',
                y='Taux_Conversion',
                title=f"Taux de Conversion Mensuel - {conseiller_selectionne}",
                color='Taux_Conversion',
                color_continuous_scale='RdYlGn'
            )
            fig_conversion.update_layout(yaxis_title="Taux de conversion (%)")
            fig_conversion.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            st.plotly_chart(fig_conversion, use_container_width=True)
            
            # Tableau détaillé par mois
            st.subheader(f"📋 Détail Mensuel pour {conseiller_selectionne}")
            
            # Formatage pour l'affichage
            evolution_display = evolution_conseiller.copy()
            evolution_display['Taux_Conversion'] = evolution_display['Taux_Conversion'].apply(lambda x: f"{x:.1f}%")
            evolution_display.columns = ['Mois', 'Nombre Total de Dossiers', 'Nombre de Souscriptions', 'Taux de Conversion']
            
            st.dataframe(evolution_display, use_container_width=True)
            
            # Téléchargement des données mensuelles
            csv_mensuel = evolution_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Télécharger l'évolution mensuelle de {conseiller_selectionne} (CSV)",
                data=csv_mensuel,
                file_name=f"evolution_mensuelle_{conseiller_selectionne}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_mensuel"
            )
        else:
            st.warning(f"Aucune donnée disponible pour le conseiller {conseiller_selectionne}")
    
        # Comparaison entre conseillers par mois
        st.subheader("🔍 Comparaison entre Conseillers par Mois")
        
        # Créer un sélecteur de mois
        mois_disponibles = sorted(df['Mois'].unique())
        mois_selectionne = st.selectbox("Sélectionner un mois", options=mois_disponibles, key="select_mois_immo")
        
        # Filtrer les données pour le mois sélectionné
        df_mois = df[df['Mois'] == mois_selectionne].copy()
        
        if not df_mois.empty:
            # Calculer les statistiques par conseiller pour le mois sélectionné
            stats_mois = df_mois.groupby('Conseiller').agg(
                Nombre_Total=('Conseiller', 'count'),
                Nombre_Souscriptions=('A souscrit', lambda x: x.sum() if 'A souscrit' in df_mois.columns else 0)
            ).reset_index()
            
            # Calculer le taux de conversion
            stats_mois['Taux_Conversion'] = stats_mois['Nombre_Souscriptions'] / stats_mois['Nombre_Total'] * 100
            stats_mois['Taux_Conversion'] = stats_mois['Taux_Conversion'].fillna(0)
            
            # Trier par nombre total décroissant
            stats_mois = stats_mois.sort_values('Nombre_Total', ascending=False)
            
            # Graphique des conseillers du mois
            fig_conseillers_mois = px.bar(
                stats_mois.head(10),  # Top 10 conseillers
                x='Conseiller',
                y='Nombre_Total',
                title=f"Top 10 Conseillers - Nombre de Dossiers - {mois_selectionne}",
                text='Nombre_Total',
                color='Taux_Conversion',
                color_continuous_scale='RdYlGn'
            )
            fig_conseillers_mois.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig_conseillers_mois, use_container_width=True)
            
            # Tableau détaillé par conseiller du mois
            stats_mois_display = stats_mois.copy()
            stats_mois_display['Taux_Conversion'] = stats_mois_display['Taux_Conversion'].apply(lambda x: f"{x:.1f}%")
            stats_mois_display.columns = ['Conseiller', 'Nombre Total de Dossiers', 'Nombre de Souscriptions', 'Taux de Conversion']
            
            st.dataframe(stats_mois_display, use_container_width=True)
            
            # Téléchargement des données du mois
            csv_mois = stats_mois_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Télécharger les données de {mois_selectionne} (CSV)",
                data=csv_mois,
                file_name=f"analyse_conseillers_{mois_selectionne}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_mois"
            )
        else:
            st.warning(f"Aucune donnée disponible pour le mois {mois_selectionne}")
    else:
        st.error("Impossible d'effectuer l'analyse par conseiller par mois : colonnes 'Conseiller' ou 'Mois' manquantes")
        st.write(f"Colonnes disponibles : {df.columns.tolist()}")
    
    # Section de débogage
    st.header("🔧 Section de Débogage")
    with st.expander("Informations de débogage"):
        # Messages d'adaptation et de diagnostic
        st.subheader("Messages d'adaptation et de diagnostic")
        
        # Informations sur les colonnes manquantes et adaptations
        st.info("""
        Certaines colonnes sont manquantes : Conseiller affecté, Date de création

        L'analyse sera adaptée en fonction des colonnes disponibles.
        """)
        
        
        # Avertissement sur le format de date
        st.warning("""
        ⚠️ Le format de date principal (jj/mm/aaaa) n'a pas été détecté dans votre fichier. 
        Pour de meilleurs résultats, utilisez le format jj/mm/aaaa (exemple: 15-07-2025).
        """)
        
        # Informations sur les colonnes utilisées
        st.success("""
        Utilisation de 'Date de souscription' comme date de création.

        Utilisation de 'Date de création' pour l'analyse par mois.

        Utilisation de 'Statut' comme état.

        Détection des souscriptions basée sur la présence d'un produit et d'un montant.
        """)
        
        # Informations sur le DataFrame
        st.subheader("Informations sur le DataFrame")
        st.write(f"Nombre total de lignes : {len(df)}")
        st.write(f"Nombre total de colonnes : {len(df.columns)}")
        
        # Liste des colonnes avec leurs types
        st.subheader("Liste des colonnes et types")
        df_types = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Valeurs non nulles': df.count().values,
            'Pourcentage non null': (df.count() / len(df) * 100).values
        })
        st.dataframe(df_types)
        
        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        # Sélectionner uniquement les colonnes numériques pour les statistiques
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            st.write(df[numeric_cols].describe())
        else:
            st.info("Aucune colonne numérique disponible pour les statistiques descriptives")
        
        # Exemples de lignes
        st.subheader("Exemples de lignes")
        st.write("Premières lignes :")
        st.dataframe(df.head(5))
        
        st.write("Dernières lignes :")
        st.dataframe(df.tail(5))
        
        # Valeurs uniques pour les colonnes catégorielles importantes
        st.subheader("Valeurs uniques pour les colonnes clés")
        categorical_cols = ['Conseiller', 'Statut', 'État', 'Produit']
        for col in categorical_cols:
            if col in df.columns:
                unique_values = df[col].value_counts().reset_index()
                unique_values.columns = [col, 'Nombre']
                st.write(f"Valeurs uniques pour {col}:")
                st.dataframe(unique_values)
        
        # Informations sur les dates
        st.subheader("Informations sur les dates")
        date_cols = ['Date', 'Date de création', 'Date de souscription']
        for col in date_cols:
            if col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    # S'assurer que les valeurs sont des dates avant d'appeler min() et max()
                    try:
                        min_date = df[col].min()
                        max_date = df[col].max()
                        st.write(f"Plage de dates pour {col}: {min_date} à {max_date}")
                    except TypeError as e:
                        st.error(f"Erreur lors du calcul min/max pour {col}: {e}")
                        st.write(f"Types dans {col}: {df[col].apply(type).value_counts()}")
                        # Essayer de convertir en datetime si possible
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            min_date = df[col].min()
                            max_date = df[col].max()
                            st.write(f"Après conversion: Plage de dates pour {col}: {min_date} à {max_date}")
                        except Exception as e2:
                            st.error(f"Impossible de convertir {col} en dates: {e2}")
                    # Distribution par mois
                    monthly_counts = df[col].dt.strftime('%Y-%m').value_counts().sort_index()
                    st.write(f"Distribution par mois pour {col}:")
                    st.bar_chart(monthly_counts)
                else:
                    st.warning(f"La colonne {col} n'est pas au format date")
        
        # Téléchargement des données complètes pour analyse externe
        st.subheader("Téléchargement des données complètes")
        csv_debug = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données complètes (CSV)",
            data=csv_debug,
            file_name=f"debug_data_immo_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_debug_data"
        )
