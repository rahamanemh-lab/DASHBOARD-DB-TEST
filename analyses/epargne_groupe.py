"""
Fonctions d'analyse des souscriptions épargne par groupe pour le dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

from utils.data_processing import safe_to_datetime, safe_to_numeric, adjust_dates_to_month_range, extract_conseiller
from utils.export import create_download_button, export_to_pdf, create_pdf_download_link


# Définition des groupes (même structure que pour l'immobilier)
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


def analyser_groupes_epargne(df):
    """
    Analyse détaillée des souscriptions épargne par groupe de conseillers.
    
    Args:
        df (DataFrame): DataFrame contenant les données épargne
    """
    st.header("👥 Analyse par Groupe - Épargne")
    
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
    
    # Identifier la colonne montant
    montant_col = None
    possible_montant_cols = ['Montant', 'montant', 'MONTANT', 'Montant de la souscription', 'Collecte']
    for col in possible_montant_cols:
        if col in df_with_conseiller.columns:
            montant_col = col
            break
    
    if montant_col is None:
        # Prendre la première colonne numérique disponible
        numeric_cols = df_with_conseiller.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            montant_col = numeric_cols[0]
            st.info(f"📋 Utilisation de la colonne '{montant_col}' comme montant.")
        else:
            st.warning("⚠️ Aucune colonne de montant trouvée. Analyse basée sur le nombre de souscriptions uniquement.")
    
    # Affichage des informations de débogage
    st.info("📋 Répartition des conseillers par groupe identifiée")
    
    # Vue d'ensemble par groupe
    st.subheader("📊 Vue d'ensemble par groupe")
    
    # Statistiques générales par groupe
    if montant_col:
        # Vérifier que la colonne montant est bien numérique
        df_with_conseiller[montant_col] = pd.to_numeric(df_with_conseiller[montant_col], errors='coerce')
        
        # Supprimer les lignes avec des valeurs NaN dans la colonne montant
        df_with_conseiller = df_with_conseiller.dropna(subset=[montant_col])
        
        if not df_with_conseiller.empty:
            stats_groupe = df_with_conseiller.groupby('Groupe').agg({
                'Conseiller': 'count',
                montant_col: ['sum', 'mean']
            }).round(2)
            stats_groupe.columns = ['Nombre de souscriptions', 'Montant total (€)', 'Montant moyen (€)']
            
            # Formater les montants
            stats_groupe['Montant total (€)'] = stats_groupe['Montant total (€)'].apply(lambda x: f"{x:,.0f}€")
            stats_groupe['Montant moyen (€)'] = stats_groupe['Montant moyen (€)'].apply(lambda x: f"{x:,.0f}€")
        else:
            st.error("❌ Aucune donnée numérique valide trouvée dans la colonne montant.")
            return
    else:
        stats_groupe = df_with_conseiller.groupby('Groupe').agg({
            'Conseiller': 'count'
        })
        stats_groupe.columns = ['Nombre de souscriptions']
    
    st.dataframe(stats_groupe, use_container_width=True)
    
    # Espacement entre le tableau et le graphique
    st.write("")
    
    # Graphique de répartition des groupes sur une ligne complète pour plus de visibilité
    st.subheader("🎯 Répartition des groupes")
    
    # Graphique en secteurs
    if montant_col:
        # Répartition par montant
        groupe_montants = df_with_conseiller.groupby('Groupe')[montant_col].sum()
        fig_pie = px.pie(
            values=groupe_montants.values,
            names=groupe_montants.index,
            title="Répartition des montants par groupe",
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=500
        )
    else:
        # Répartition par nombre
        groupe_counts = df_with_conseiller['Groupe'].value_counts()
        fig_pie = px.pie(
            values=groupe_counts.values,
            names=groupe_counts.index,
            title="Répartition des souscriptions par groupe",
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=500
        )
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Espacement entre les sections
    st.markdown("---")
    st.write("")
    
    # Analyse détaillée par conseiller dans chaque groupe
    st.subheader("📈 Performance détaillée par groupe")
    
    # Créer des onglets pour chaque groupe
    groupes_disponibles = df_with_conseiller['Groupe'].unique()
    if len(groupes_disponibles) > 1:
        groupe_tabs = st.tabs(list(groupes_disponibles))
        
        for i, groupe in enumerate(groupes_disponibles):
            with groupe_tabs[i]:
                analyser_groupe_specifique_epargne(df_with_conseiller, groupe, montant_col)
    else:
        # Si un seul groupe, afficher directement
        analyser_groupe_specifique_epargne(df_with_conseiller, groupes_disponibles[0], montant_col)
    
    # Espacement entre les sections
    st.markdown("---")
    st.write("")
    
    # Comparaison inter-groupes
    st.subheader("⚖️ Comparaison inter-groupes")
    
    if montant_col:
        # Graphique de comparaison avec montants
        comparison_data = df_with_conseiller.groupby('Groupe').agg({
            'Conseiller': 'count',
            montant_col: 'sum'
        }).reset_index()
        
        fig_comparison = go.Figure()
        
        # Nombre de souscriptions
        fig_comparison.add_trace(go.Bar(
            name='Nombre de souscriptions',
            x=comparison_data['Groupe'],
            y=comparison_data['Conseiller'],
            yaxis='y',
            offsetgroup=1
        ))
        
        # Montant total (axe secondaire)
        fig_comparison.add_trace(go.Bar(
            name='Montant total (€)',
            x=comparison_data['Groupe'],
            y=comparison_data[montant_col],
            yaxis='y2',
            offsetgroup=2
        ))
        
        fig_comparison.update_layout(
            title='Comparaison des groupes : Volume vs Montant',
            xaxis_title='Groupe',
            yaxis=dict(title='Nombre de souscriptions', side='left'),
            yaxis2=dict(title='Montant total (€)', side='right', overlaying='y'),
            barmode='group'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Espacement entre les sections
    st.markdown("---")
    st.write("")
    
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
        file_name=f"analyse_groupes_epargne_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def analyser_groupe_specifique_epargne(df, groupe_name, montant_col):
    """
    Analyse spécifique d'un groupe de conseillers pour l'épargne.
    
    Args:
        df (DataFrame): DataFrame avec les données et la colonne Groupe
        groupe_name (str): Nom du groupe à analyser
        montant_col (str): Nom de la colonne montant
    """
    df_groupe = df[df['Groupe'] == groupe_name].copy()
    
    if df_groupe.empty:
        st.warning(f"Aucune donnée disponible pour le groupe {groupe_name}")
        return
    
    st.write(f"**Groupe : {groupe_name}** ({len(df_groupe)} souscriptions)")
    
    # Statistiques par conseiller dans le groupe
    if montant_col:
        # Vérifier que la colonne montant est bien numérique
        df_groupe[montant_col] = pd.to_numeric(df_groupe[montant_col], errors='coerce')
        
        # Supprimer les lignes avec des valeurs NaN dans la colonne montant
        df_groupe = df_groupe.dropna(subset=[montant_col])
        
        if not df_groupe.empty:
            conseiller_stats = df_groupe.groupby('Conseiller').agg({
                'Conseiller': 'count',
                montant_col: ['sum', 'mean']
            }).round(2)
            conseiller_stats.columns = ['Nombre de souscriptions', 'Montant total (€)', 'Montant moyen (€)']
        else:
            st.warning(f"Aucune donnée numérique valide pour le groupe {groupe_name}")
            return
    else:
        conseiller_stats = df_groupe.groupby('Conseiller').agg({
            'Conseiller': 'count'
        })
        conseiller_stats.columns = ['Nombre de souscriptions']
    
    # Trier par nombre de souscriptions décroissant
    conseiller_stats = conseiller_stats.sort_values('Nombre de souscriptions', ascending=False)
    
    # Première ligne : Tableau des statistiques
    st.dataframe(conseiller_stats, use_container_width=True)
    
    # Espacement entre les lignes
    st.write("")
    
    # Deuxième ligne : Graphique des performances du groupe
    if len(conseiller_stats) > 1:
        if montant_col:
            # Graphique des montants
            fig_bar = px.bar(
                x=conseiller_stats.index,
                y=conseiller_stats['Montant total (€)'] if isinstance(conseiller_stats['Montant total (€)'].iloc[0], (int, float)) else conseiller_stats['Nombre de souscriptions'],
                title=f"Performance des conseillers - {groupe_name}",
                labels={'x': 'Conseiller', 'y': 'Montant total (€)' if montant_col else 'Nombre de souscriptions'}
            )
        else:
            fig_bar = px.bar(
                x=conseiller_stats.index,
                y=conseiller_stats['Nombre de souscriptions'],
                title=f"Performance des conseillers - {groupe_name}",
                labels={'x': 'Conseiller', 'y': 'Nombre de souscriptions'}
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


def analyser_performance_conseillers_epargne(df):
    """
    Analyse détaillée de la performance par conseiller pour l'épargne.
    
    Args:
        df (DataFrame): DataFrame contenant les données épargne
    """
    st.header("👤 Analyse par Conseiller - Épargne")
    
    # Vérification des données
    if df is None or df.empty:
        st.error("❌ Aucune donnée disponible pour l'analyse par conseiller.")
        return
    
    # Extraction du conseiller avec la fonction améliorée
    df_with_conseiller = extract_conseiller(df.copy())
    
    if 'Conseiller' not in df_with_conseiller.columns:
        st.error("❌ Impossible d'extraire les informations de conseiller.")
        return
    
    # Identifier la colonne montant
    montant_col = None
    possible_montant_cols = ['Montant', 'montant', 'MONTANT', 'Montant de la souscription', 'Collecte']
    for col in possible_montant_cols:
        if col in df_with_conseiller.columns:
            montant_col = col
            break
    
    if montant_col is None:
        # Prendre la première colonne numérique disponible
        numeric_cols = df_with_conseiller.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            montant_col = numeric_cols[0]
            st.info(f"📋 Utilisation de la colonne '{montant_col}' comme montant.")
    
    # Identifier la colonne de date pour le filtre par mois
    date_col = None
    possible_date_cols = ['Date', 'date', 'DATE', 'Date de souscription', 'Date souscription', 'Timestamp']
    for col in possible_date_cols:
        if col in df_with_conseiller.columns:
            date_col = col
            break
    
    # Si une colonne de date est trouvée, la convertir en datetime
    if date_col:
        df_with_conseiller[date_col] = pd.to_datetime(df_with_conseiller[date_col], errors='coerce')
        df_with_conseiller = df_with_conseiller.dropna(subset=[date_col])
        df_with_conseiller['Mois'] = df_with_conseiller[date_col].dt.to_period('M').astype(str)
    
    # Section des filtres
    st.subheader("🔍 Filtres de sélection")
    
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        # Filtre par conseiller
        conseillers_disponibles = sorted([c for c in df_with_conseiller['Conseiller'].unique() if c != 'Inconnu'])
        conseillers_selectionnes = st.multiselect(
            "Sélectionner les conseillers :",
            options=conseillers_disponibles,
            default=conseillers_disponibles,
            key="filtre_conseillers_epargne"
        )
    
    with col_filter2:
        # Filtre par mois (si colonne de date disponible)
        if date_col and 'Mois' in df_with_conseiller.columns:
            mois_disponibles = sorted(df_with_conseiller['Mois'].unique())
            mois_selectionnes = st.multiselect(
                "Sélectionner les mois :",
                options=mois_disponibles,
                default=mois_disponibles,
                key="filtre_mois_epargne"
            )
        else:
            st.info("ℹ️ Aucune colonne de date trouvée pour le filtre par mois")
            mois_selectionnes = None
    
    # Appliquer les filtres
    if conseillers_selectionnes:
        df_filtered = df_with_conseiller[df_with_conseiller['Conseiller'].isin(conseillers_selectionnes)]
    else:
        df_filtered = df_with_conseiller.copy()
    
    if mois_selectionnes and date_col:
        df_filtered = df_filtered[df_filtered['Mois'].isin(mois_selectionnes)]
    
    # Vérifier qu'il reste des données après filtrage
    if df_filtered.empty:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    st.info(f"📊 Analyse basée sur {len(df_filtered)} souscriptions après filtrage")
    
    # Espacement après les filtres
    st.markdown("---")
    
    # Statistiques par conseiller
    if montant_col:
        # Vérifier que la colonne montant est bien numérique
        df_filtered[montant_col] = pd.to_numeric(df_filtered[montant_col], errors='coerce')
        
        # Supprimer les lignes avec des valeurs NaN dans la colonne montant
        df_filtered = df_filtered.dropna(subset=[montant_col])
        
        if not df_filtered.empty:
            conseiller_stats = df_filtered.groupby('Conseiller').agg({
                'Conseiller': 'count',
                montant_col: ['sum', 'mean', 'std']
            }).round(2)
            conseiller_stats.columns = ['Nombre de souscriptions', 'Montant total (€)', 'Montant moyen (€)', 'Écart-type (€)']
            
            # Calculer des métriques additionnelles
            conseiller_stats['Part du total (%)'] = (conseiller_stats['Montant total (€)'] / conseiller_stats['Montant total (€)'].sum() * 100).round(2)
        else:
            st.error("❌ Aucune donnée numérique valide trouvée dans la colonne montant.")
            return
    else:
        conseiller_stats = df_filtered.groupby('Conseiller').agg({
            'Conseiller': 'count'
        })
        conseiller_stats.columns = ['Nombre de souscriptions']
        conseiller_stats['Part du total (%)'] = (conseiller_stats['Nombre de souscriptions'] / conseiller_stats['Nombre de souscriptions'].sum() * 100).round(2)
    
    # Trier par performance décroissante
    sort_col = 'Montant total (€)' if montant_col else 'Nombre de souscriptions'
    conseiller_stats = conseiller_stats.sort_values(sort_col, ascending=False)
    
    # Affichage des résultats
    st.subheader("📊 Performance détaillée par conseiller")
    st.dataframe(conseiller_stats, use_container_width=True)
    
    # Espacement entre le tableau et le graphique Top performers
    st.write("")
    
    # Graphique Top performers sur une ligne complète pour plus de visibilité
    st.subheader("🏆 Top performers")
    top_5 = conseiller_stats.head(5)
    
    if montant_col:
        fig_top = px.bar(
            x=top_5['Montant total (€)'],
            y=top_5.index,
            orientation='h',
            title="Top 5 - Montant total",
            labels={'x': 'Montant total (€)', 'y': 'Conseiller'},
            height=400
        )
    else:
        fig_top = px.bar(
            x=top_5['Nombre de souscriptions'],
            y=top_5.index,
            orientation='h',
            title="Top 5 - Nombre de souscriptions",
            labels={'x': 'Nombre de souscriptions', 'y': 'Conseiller'},
            height=400
        )
    
    st.plotly_chart(fig_top, use_container_width=True)
    
    # Graphique principal : Montant Total des Souscriptions par Conseiller
    st.subheader("💰 Montant Total des Souscriptions par Conseiller")
    
    if montant_col and len(conseiller_stats) > 1:
        # Préparer les données pour le graphique principal
        df_for_chart = conseiller_stats.reset_index()
        df_for_chart = df_for_chart.sort_values('Montant total (€)', ascending=False)
        
        # Créer le graphique principal
        fig_main = px.bar(
            df_for_chart.head(10),  # Top 10 conseillers
            x='Conseiller',
            y='Montant total (€)',
            text='Montant total (€)',
            color='Nombre de souscriptions',
            color_continuous_scale='Viridis',
            title="Montant Total des Souscriptions par Conseiller (Top 10)",
            labels={
                'Conseiller': 'Conseiller',
                'Montant total (€)': 'Montant Total (€)',
                'Nombre de souscriptions': 'Nombre de Souscriptions'
            },
            height=500
        )
        
        # Mise en forme du graphique principal
        fig_main.update_traces(
            texttemplate='%{text:,.0f} €',
            textposition='auto'
        )
        
        fig_main.update_layout(
            xaxis_tickangle=45,
            template="plotly_white"
        )
        
        # Afficher le graphique principal
        st.plotly_chart(fig_main, use_container_width=True)
    
    # Graphiques de performance complémentaires
    st.subheader("📈 Visualisations de performance complémentaires")
    
    if montant_col and len(conseiller_stats) > 1:
        # Ajouter un espacement avant les graphiques
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Premier graphique : Montant moyen par conseiller
        st.markdown("#### 📊 Montant moyen par conseiller")
        fig_montant_moyen = px.bar(
            x=conseiller_stats.index,
            y=conseiller_stats['Montant moyen (€)'],
            title="Montant moyen par conseiller",
            labels={'x': 'Conseiller', 'y': 'Montant moyen (€)'},
            height=500
        )
        fig_montant_moyen.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_montant_moyen, use_container_width=True)
        
        # Espacement entre les graphiques
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Deuxième graphique : Répartition des montants
        st.markdown("#### 🥧 Répartition des montants par conseiller")
        fig_pie_conseiller = px.pie(
            values=conseiller_stats['Montant total (€)'],
            names=conseiller_stats.index,
            title="Répartition des montants par conseiller",
            height=500
        )
        st.plotly_chart(fig_pie_conseiller, use_container_width=True)
        
        # Ajouter un espacement après les graphiques
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
    
    # Analyse du taux de conversion par conseiller
    st.subheader("🎯 Taux de Conversion par Conseiller")
    
    # Vérifier si nous avons les données nécessaires pour calculer le taux de conversion
    statut_cols = ['Statut', 'statut', 'STATUT', 'Étape', 'etape', 'ETAPE']
    statut_col = None
    
    for col in statut_cols:
        if col in df_filtered.columns:
            statut_col = col
            break
    
    if statut_col:
        # Calculer le taux de conversion par conseiller
        conversion_data = []
        
        for conseiller in df_filtered['Conseiller'].unique():
            if conseiller == 'Inconnu':
                continue
                
            df_conseiller = df_filtered[df_filtered['Conseiller'] == conseiller]
            
            # Compter les différents statuts
            total_dossiers = len(df_conseiller)
            
            # Identifier les statuts de conversion (ajustez selon vos données)
            statuts_convertis = ['Validé', 'Souscrit', 'Finalisé', 'Accepté', 'Converti']
            dossiers_convertis = 0
            
            for statut in statuts_convertis:
                dossiers_convertis += len(df_conseiller[df_conseiller[statut_col].str.contains(statut, case=False, na=False)])
            
            if total_dossiers >= 3:  # Minimum 3 dossiers pour un taux fiable
                taux_conversion = (dossiers_convertis / total_dossiers) * 100
                conversion_data.append({
                    'Conseiller': conseiller,
                    'Total_Dossiers': total_dossiers,
                    'Dossiers_Convertis': dossiers_convertis,
                    'Taux_Conversion': taux_conversion
                })
        
        if conversion_data:
            df_conversion = pd.DataFrame(conversion_data)
            df_conversion = df_conversion.sort_values('Taux_Conversion', ascending=False)
            
            # Ajouter un espacement avant la section
            st.markdown("---")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Afficher le tableau des taux de conversion
            st.markdown("#### 📋 Tableau des taux de conversion")
            display_conversion = df_conversion.copy()
            display_conversion['Taux_Conversion'] = display_conversion['Taux_Conversion'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_conversion, use_container_width=True)
            
            # Espacement entre le tableau et le graphique
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Graphique des taux de conversion
            if len(df_conversion) > 0:
                st.markdown("#### 📊 Graphique des taux de conversion")
                fig_conversion = px.bar(
                    df_conversion.head(10),  # Top 10 conseillers
                    x='Conseiller',
                    y='Taux_Conversion',
                    text='Taux_Conversion',
                    color='Total_Dossiers',
                    color_continuous_scale='Viridis',
                    title="Taux de Conversion par Conseiller (Top 10)",
                    labels={
                        'Conseiller': 'Conseiller',
                        'Taux_Conversion': 'Taux de Conversion (%)',
                        'Total_Dossiers': 'Nombre de Dossiers'
                    },
                    height=500
                )
                
                # Mise en forme du graphique
                fig_conversion.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='auto'
                )
                
                fig_conversion.update_layout(
                    xaxis_tickangle=45,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_conversion, use_container_width=True)
        else:
            st.info("ℹ️ Aucun conseiller n'a au moins 3 dossiers pour calculer un taux de conversion fiable.")
    else:
        st.info("ℹ️ Colonne de statut non trouvée. Impossible de calculer les taux de conversion.")
    
    # Export des données
    st.subheader("📥 Export des données")
    csv = conseiller_stats.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Télécharger l'analyse par conseiller (CSV)",
        data=csv,
        file_name=f"analyse_conseillers_epargne_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
