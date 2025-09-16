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
from utils.data_processing import safe_to_datetime, safe_to_numeric

# Définition des colonnes strictement autorisées
# Cette liste est la SEULE référence pour toutes les opérations
COLONNES_AUTORISEES = {
    # Colonnes strictes du fichier source (seules ces colonnes seront importées)
    "source": [
        "Nom de l'opportunité",
        "Produit",
        "Statut",
        "Étape",
        "Montant",
        "Montant des frais",
        "Type d'investissement",
        "Conseiller",
        "Date de souscription",
        "Date de validation",
   ],
    # Colonnes dérivées nécessaires au fonctionnement
    "derivees": [
        "Montant du placement",
        "Date de souscription",
        "Mois",
        "Premier_Jour_Mois",
        "Dernier_Jour_Mois"
    ]
}

# Mapping des colonnes alternatives vers les noms standard
COLONNE_MAPPING = {
    "Conseiller": "Conseiller",
    "Date de souscription": "Date de souscription",
    "Date de validation": "Date de validation"
}

# Importer les fonctions adaptées pour l'analyse sans colonne Produit
from analyses.utils_analyse import analyse_collecte_produit_conseiller_fallback, analyse_performance_conseiller_fallback, extract_conseiller

# Constantes
OBJECTIF_MENSUEL_IMMO = 2000000  # Objectif mensuel pour l'immobilier (2M€)

def adjust_dates_to_month_range(df, date_column):
    """Ajoute les colonnes de premier et dernier jour du mois pour chaque entrée."""
    df = df.copy()
    
    # Extraire le premier jour du mois
    df['Premier_Jour_Mois'] = df[date_column].dt.to_period('M').dt.start_time
    
    # Extraire le dernier jour du mois
    df['Dernier_Jour_Mois'] = df[date_column].dt.to_period('M').dt.end_time
    
    return df

def analyser_souscriptions_immo(df_original):
    """Analyse des souscriptions immobilières avec un contrôle strict des colonnes.
    
    Cette fonction applique un contrôle strict des colonnes utilisées dans l'analyse:
    1. Seules les colonnes explicitement listées dans COLONNES_AUTORISEES["source"] sont importées
    2. Aucune colonne parasite n'est autorisée
    3. Les colonnes dérivées sont créées uniquement à partir des colonnes sources autorisées
    
    Args:
        df_original: DataFrame contenant les données brutes
        
    Returns:
        DataFrame traité ou None en cas d'erreur
    """
    
    # DIAGNOSTIC: Vérifier les colonnes dans le DataFrame original
    st.write("### DIAGNOSTIC INITIAL: Colonnes dans le DataFrame original")
    st.write(f"Colonnes disponibles dans df_original: {df_original.columns.tolist()}")
    st.write(f"'Conseiller' existe dans df_original: {'Conseiller' in df_original.columns}")
    if 'Conseiller' in df_original.columns:
        st.write("Exemples de valeurs 'Conseiller':", df_original['Conseiller'].head(3).tolist())
    
    st.header("🏠 Analyse des Souscriptions Immobilières")
    
    # Afficher un message d'information sur les colonnes attendues
    st.info("""
    📝 **Colonnes strictement requises dans le fichier source**:
    - "Nom de l'opportunité" (texte)
    - "Produit" (texte)
    - "Statut" (texte)
    - "Étape" (texte)
    - "Montant" (nombre)
    - "Montant des frais" (nombre)
    - "Type d'investissement" (texte)
    - "Conseiller" (texte)
    - "Date de souscription" (date au format jj/mm/aaaa)
    - "Date de validation" (date au format jj/mm/aaaa)
    
    ℹ️ **Important**: Les noms des colonnes doivent correspondre **exactement** à ceux listés ci-dessus.
    Toute colonne non listée sera ignorée. Toute colonne manquante sera signalée.
    """)
    
    # 1. DIAGNOSTIC DES COLONNES DU FICHIER SOURCE
    colonnes_disponibles = df_original.columns.tolist()
    colonnes_requises = COLONNES_AUTORISEES["source"]
    colonnes_manquantes = [col for col in colonnes_requises if col not in colonnes_disponibles]
    
    if colonnes_manquantes:
        st.error(f"❌ ERREUR: Colonnes manquantes dans le fichier source: {', '.join(colonnes_manquantes)}")
        with st.expander("Détails des colonnes disponibles"):
            st.write(colonnes_disponibles)
        return None
    
    # 2. EXTRACTION DES COLONNES AUTORISÉES UNIQUEMENT
    df = df_original[colonnes_requises].copy()
    
    # 3. PRÉTRAITEMENT DES DONNÉES
    # Conversion des dates
    df['Date de souscription'] = safe_to_datetime(df['Date de souscription'])
    df['Date de validation'] = safe_to_datetime(df['Date de validation'])
    
    # Ajout des colonnes de mois
    df['Mois'] = df['Date de souscription'].dt.to_period('M').astype(str)
    
    # Calcul du montant du placement
    if 'Montant' in df.columns and 'Montant des frais' in df.columns:
        df['Montant'] = safe_to_numeric(df['Montant'])
        df['Montant des frais'] = safe_to_numeric(df['Montant des frais'])
        df['Montant du placement'] = df['Montant'] - df['Montant des frais']
    elif 'Montant' in df.columns:
        df['Montant'] = safe_to_numeric(df['Montant'])
        df['Montant du placement'] = df['Montant']
    else:
        st.error("❌ ERREUR: Impossible de calculer le montant du placement. Vérifiez les colonnes du fichier source.")
        return None
    
    # Extraction du conseiller
    df = extract_conseiller(df)
    
    # 4. FILTRAGE DES DONNÉES INVALIDES
    # Vérifier les montants à 0 ou négatifs
    invalid_count = ((df['Montant du placement'] <= 0) | df['Montant du placement'].isna()).sum()
    if invalid_count > 0:
        st.warning(f"⚠️ {invalid_count} souscriptions immobilières avec un montant invalide (0, négatif ou manquant) détectées.")
    
    # Filtrer les montants valides
    df_valid = df[df['Montant du placement'] > 0].copy()
    
    # Exclure les souscriptions annulées
    if 'Statut' in df_valid.columns:
        annulation_keywords = ['annul', 'cancel', 'rejet', 'refus', 'abandon']
        mask_annule = df_valid['Statut'].astype(str).str.lower().apply(
            lambda x: any(keyword in x for keyword in annulation_keywords)
        )
        nb_annule = mask_annule.sum()
        if nb_annule > 0:
            st.warning(f"⚠️ {nb_annule} souscriptions immobilières annulées détectées et exclues de l'analyse.")
            df_valid = df_valid[~mask_annule].copy()
    
    if df_valid.empty:
        st.warning("⚠️ Aucune souscription immobilière valide après filtrage.")
        return None
    
    # Utiliser le DataFrame filtré pour l'analyse
    df_immo_valid = df_valid.copy()
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_collecte = df_immo_valid['Montant du placement'].sum()
        st.metric("💰 Collecte Totale Immobilier", f"{total_collecte:,.0f}€")
    with col2:
        nb_souscriptions = len(df_immo_valid)
        st.metric("📝 Nombre de Souscriptions", f"{nb_souscriptions:,}")
    with col3:
        ticket_moyen = df_immo_valid['Montant du placement'].mean()
        st.metric("🎯 Ticket Moyen", f"{ticket_moyen:,.0f}€")
    with col4:
        nb_conseillers = df_immo_valid['Conseiller'].nunique()
        st.metric("👥 Nombre de Conseillers", f"{nb_conseillers}")
    
    # Filtres
    st.subheader("🔍 Filtres")
    col1, col2 = st.columns(2)
    df_filtre = df_immo_valid.copy()
    with col1:
        mois_disponibles = sorted(df_immo_valid['Mois'].dropna().unique())
        mois_selectionne = st.selectbox("📅 Mois", options=["Tous"] + mois_disponibles, key="mois_immo")
        if mois_selectionne != "Tous":
            df_filtre = df_filtre[df_filtre['Mois'] == mois_selectionne]
    with col2:
        conseillers_disponibles = sorted(df_immo_valid['Conseiller'].dropna().unique())
        conseiller_selectionne = st.selectbox("👤 Conseiller", options=["Tous"] + conseillers_disponibles, key="conseiller_immo")
        if conseiller_selectionne != "Tous":
            df_filtre = df_filtre[df_filtre['Conseiller'] == conseiller_selectionne]
    
    if not df_filtre.empty:
        # Analyse du Pipe de Collecte Immobilier (souscriptions en cours)
        st.subheader("💰 Pipe de Collecte Immobilier")
        
        # Exclure les souscriptions annulées du pipe de collecte
        if 'Statut' in df_immo_valid.columns and 'Étape' in df_immo_valid.columns:
            # Filtrer pour ne garder que les souscriptions en cours (non annulées)
            annulation_keywords = ['annul', 'cancel', 'rejet', 'refus', 'abandon']
            mask_en_cours = ~df_immo_valid['Statut'].astype(str).str.lower().apply(
                lambda x: any(keyword in x for keyword in annulation_keywords)
            )
            
            # Créer une copie du DataFrame filtré
            df_pipe = df_immo_valid[mask_en_cours].copy()
            
            # Afficher le nombre de souscriptions en cours
            nb_en_cours = len(df_pipe)
            montant_en_cours = df_pipe['Montant du placement'].sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💰 Montant Total en Pipe", f"{montant_en_cours:,.0f}€")
            with col2:
                st.metric("📃 Nombre de Dossiers en Pipe", f"{nb_en_cours:,}")
        
        # Analyse par conseiller et par produit
        st.subheader("📈 Analyse de la Collecte par Conseiller")
        
        # Utiliser la fonction d'analyse de collecte adaptée
        analyse_collecte_produit_conseiller_fallback(df_immo_valid, "Immobilier")
        
        # Analyse de performance des conseillers
        st.subheader("🏆 Performance des Conseillers")
        analyse_performance_conseiller_fallback(df_immo_valid, 'Montant du placement', 'Conseiller', "Immobilier", min_operations=3)
        
        # Identifier la colonne de statut ou d'étape
        statut_col = None
        for col in df_filtre.columns:
            if col.lower() in ['statut', 'status', 'état', 'etat', 'state']:
                statut_col = col
                break
        
        etape_col = None
        for col in df_filtre.columns:
            if col.lower() in ['étape', 'etape', 'step', 'phase', 'stage']:
                etape_col = col
                break
        
        # Utiliser la colonne étape si disponible, sinon utiliser la colonne statut
        filtre_col = etape_col if etape_col else statut_col
        
        if filtre_col and 'Montant du placement' in df_filtre.columns:
            # Filtrer pour n'inclure que les souscriptions en cours (pipe)
            # Exclure les étapes/statuts finalisés (acté, validé, clôturé) et les statuts annulés
            statuts_finalises = ['acté', 'validé', 'cloturé', 'clôturé', 'acte', 'valide', 'cloture']
            
            # Identifier les statuts annulés
            statuts_annules = [s.lower() for s in df_filtre[filtre_col].unique() if 'annul' in str(s).lower()]
            
            # Combiner les statuts à exclure
            statuts_a_exclure = [s.lower() for s in statuts_finalises] + statuts_annules
            
            # Filtrer le dataframe pour exclure les statuts finalisés et annulés
            df_pipe = df_filtre[~df_filtre[filtre_col].str.lower().isin(statuts_a_exclure)].copy()
            
            if not df_pipe.empty:
                # Créer des options de période pour l'analyse
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sélection de la période d'analyse
                    periode_options = ['Mois', 'Semaine', 'Trimestre']
                    periode_selectionnee = st.selectbox("📅 Période d'analyse", options=periode_options, key="periode_pipe_immo")
                
                with col2:
                    # Sélection du type de graphique
                    chart_options = ['Histogramme', 'Camembert']
                    chart_type = st.selectbox("📆 Type de graphique", options=chart_options, key="chart_pipe_immo")
                
                # Agrégation par période et conseiller
                if periode_selectionnee == 'Mois':
                    df_pipe['Periode'] = df_pipe['Date de souscription'].dt.strftime('%Y-%m')
                elif periode_selectionnee == 'Semaine':
                    df_pipe['Periode'] = df_pipe['Date de souscription'].dt.strftime('%Y-%U')
                else:  # Trimestre
                    df_pipe['Periode'] = df_pipe['Date de souscription'].dt.to_period('Q').astype(str)
                
                # Agrégation par période et conseiller
                pipe_agg = df_pipe.groupby(['Periode', 'Conseiller'])['Montant du placement'].sum().reset_index()
                
                # Créer le graphique approprié
                if chart_type == 'Histogramme':
                    fig_pipe = px.bar(
                        pipe_agg,
                        x='Periode',
                        y='Montant du placement',
                        color='Conseiller',
                        title=f"💰 Pipe de Collecte par {periode_selectionnee} et par Conseiller",
                        labels={'Montant du placement': 'Montant (€)', 'Periode': periode_selectionnee}
                    )
                else:  # Camembert
                    pipe_conseiller = df_pipe.groupby('Conseiller')['Montant du placement'].sum().reset_index()
                    fig_pipe = px.pie(
                        pipe_conseiller,
                        values='Montant du placement',
                        names='Conseiller',
                        title="💰 Répartition du Pipe de Collecte par Conseiller"
                    )
                    fig_pipe.update_traces(textposition='inside', textinfo='percent+label')
                
                st.plotly_chart(fig_pipe, use_container_width=True)
                
                # Tableau récapitulatif du pipe
                st.subheader("📋 Tableau Récapitulatif du Pipe")
                
                # Agrégation par conseiller pour le tableau
                pipe_table = df_pipe.groupby('Conseiller').agg(
                    Nb_Dossiers=('Montant du placement', 'count'),
                    Montant_Total=('Montant du placement', 'sum'),
                    Ticket_Moyen=('Montant du placement', 'mean')
                ).reset_index().sort_values('Montant_Total', ascending=False)
                
                # Formatage pour l'affichage
                pipe_display = pipe_table.copy()
                pipe_display['Montant_Total'] = pipe_display['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
                pipe_display['Ticket_Moyen'] = pipe_display['Ticket_Moyen'].apply(lambda x: f"{x:,.0f}€")
                
                # Renommer les colonnes pour l'affichage
                pipe_display.columns = ['Conseiller', 'Nombre de Dossiers', 'Montant Total', 'Ticket Moyen']
                
                st.dataframe(pipe_display, use_container_width=True)
                
                # Téléchargement des données du pipe
                csv_pipe = pipe_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les données du pipe (CSV)",
                    data=csv_pipe,
                    file_name=f"pipe_collecte_immo_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_pipe_immo"
                )
            else:
                st.info("ℹ️ Aucune souscription en cours dans le pipe de collecte immobilier.")
        else:
            st.warning("⚠️ Impossible d'analyser le pipe de collecte : colonne de statut ou étape non trouvée.")
        
        st.subheader("📈 Évolution de la Collecte Immobilière (du 1er au dernier jour)")
        
        # Ajuster les dates pour avoir une plage complète du 1er au dernier jour du mois
        df_filtre = adjust_dates_to_month_range(df_filtre, 'Date de souscription')
        
        # Créer une agrégation mensuelle avec les dates de début et fin
        evolution_mensuelle = df_filtre.groupby('Mois').agg(
            Montant_Total=('Montant du placement', 'sum'),
            Premier_Jour=('Premier_Jour_Mois', 'first'),
            Dernier_Jour=('Dernier_Jour_Mois', 'first')
        ).reset_index()
        
        # Trier par date de début
        evolution_mensuelle = evolution_mensuelle.sort_values('Premier_Jour')
        
        # Créer des étiquettes personnalisées pour l'axe X avec les plages de dates
        evolution_mensuelle['Période'] = evolution_mensuelle.apply(
            lambda row: f"{row['Mois']} ({row['Premier_Jour'].strftime('%d/%m') if pd.notna(row['Premier_Jour']) else '??'} - {row['Dernier_Jour'].strftime('%d/%m') if pd.notna(row['Dernier_Jour']) else '??'})", 
            axis=1
        )
        
        # Calculer l'écart par rapport à l'objectif
        evolution_mensuelle['Écart Objectif'] = evolution_mensuelle['Montant_Total'] - OBJECTIF_MENSUEL_IMMO
        evolution_mensuelle['Statut'] = np.where(evolution_mensuelle['Écart Objectif'] >= 0, '✅ Atteint', '❌ Sous Objectif')
        
        # Créer un DataFrame pour l'affichage et l'export
        display_df = pd.DataFrame({
            'Mois': evolution_mensuelle['Mois'],
            'Période': evolution_mensuelle['Période'],
            'Montant Total': evolution_mensuelle['Montant_Total'],
            'Écart Objectif': evolution_mensuelle['Écart Objectif'],
            'Statut': evolution_mensuelle['Statut']
        })
        
        # Créer le graphique avec les périodes complètes
        fig_mensuel = px.bar(
            evolution_mensuelle,
            x='Période',
            y='Montant_Total',
            title=f"📊 Évolution Mensuelle de la Collecte Immobilière (Objectif: {OBJECTIF_MENSUEL_IMMO:,.0f}€)",
            text='Montant_Total',
            color='Statut',
            color_discrete_map={'✅ Atteint': '#2E8B57', '❌ Sous Objectif': '#DC143C'}
        )
        fig_mensuel.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
        
        # Ajouter une ligne pour l'objectif
        fig_mensuel.add_shape(
            type="line",
            x0=0,
            x1=len(evolution_mensuelle['Période'])-1,
            y0=OBJECTIF_MENSUEL_IMMO,
            y1=OBJECTIF_MENSUEL_IMMO,
            line=dict(color="black", width=2, dash="dash")
        )
        
        # Ajouter une annotation pour l'objectif
        fig_mensuel.add_annotation(
            x=len(evolution_mensuelle['Période'])-1,
            y=OBJECTIF_MENSUEL_IMMO,
            text=f"Objectif: {OBJECTIF_MENSUEL_IMMO:,.0f}€",
            showarrow=False,
            yshift=10
        )
        
        st.plotly_chart(fig_mensuel, use_container_width=True)
        
        # Tableau récapitulatif
        st.subheader("📋 Récapitulatif Mensuel")
        
        # Formatage pour l'affichage
        display_df['Montant Total'] = display_df['Montant Total'].apply(lambda x: f"{x:,.0f}€")
        display_df['Écart Objectif'] = display_df['Écart Objectif'].apply(lambda x: f"{x:,.0f}€")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Analyse par conseiller
        st.subheader("👥 Analyse par Conseiller")
        
        # Calculer les statistiques par conseiller
        analyse_conseiller = df_filtre.groupby('Conseiller').agg(
            Nb_Souscriptions=('Montant du placement', 'count'),
            Montant_Total=('Montant du placement', 'sum'),
            Ticket_Moyen=('Montant du placement', 'mean')
        ).reset_index()
        
        # Trier par montant total
        analyse_conseiller = analyse_conseiller.sort_values('Montant_Total', ascending=False)
        
        # Graphiques par conseiller
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 conseillers par montant
            fig_top_montant = px.bar(
                analyse_conseiller.head(10),
                x='Montant_Total',
                y='Conseiller',
                orientation='h',
                title="🏆 Top 10 Conseillers - Collecte Immobilière",
                text='Montant_Total',
                color='Montant_Total',
                color_continuous_scale='Blues'
            )
            fig_top_montant.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
            st.plotly_chart(fig_top_montant, use_container_width=True)
        
        with col2:
            # Top 10 conseillers par ticket moyen
            top_ticket = analyse_conseiller[analyse_conseiller['Nb_Souscriptions'] >= 3].sort_values('Ticket_Moyen', ascending=False).head(10)
            fig_top_ticket = px.bar(
                top_ticket,
                x='Ticket_Moyen',
                y='Conseiller',
                orientation='h',
                title="🎯 Top 10 Conseillers - Ticket Moyen (min 3 souscriptions)",
                text='Ticket_Moyen',
                color='Ticket_Moyen',
                color_continuous_scale='Greens'
            )
            fig_top_ticket.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
            st.plotly_chart(fig_top_ticket, use_container_width=True)
        
        # Tableau détaillé par conseiller
        st.subheader("📊 Tableau Détaillé par Conseiller")
        
        # Formatage pour l'affichage
        analyse_display = analyse_conseiller.copy()
        analyse_display['Montant_Total'] = analyse_display['Montant_Total'].apply(lambda x: f"{x:,.0f}€")
        analyse_display['Ticket_Moyen'] = analyse_display['Ticket_Moyen'].apply(lambda x: f"{x:,.0f}€")
        
        # Renommer les colonnes pour l'affichage
        analyse_display.columns = ['Conseiller', 'Nombre de Souscriptions', 'Montant Total', 'Ticket Moyen']
        
        st.dataframe(analyse_display, use_container_width=True)
        
        # Téléchargement des données
        csv = analyse_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données (CSV)",
            data=csv,
            file_name=f"analyse_souscriptions_immo_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Analyse par conseiller par mois
        st.header("📅 Analyse par Conseiller par Mois")
        
        if 'Conseiller' in df_filtre.columns and 'Mois' in df_filtre.columns:
            # Créer un sélecteur de conseiller
            conseillers_disponibles = sorted(df_filtre['Conseiller'].unique())
            conseiller_selectionne = st.selectbox("Sélectionner un conseiller", options=conseillers_disponibles, key="select_conseiller_immo_mois")
            
            # Filtrer les données pour le conseiller sélectionné
            df_conseiller = df_filtre[df_filtre['Conseiller'] == conseiller_selectionne].copy()
            
            if not df_conseiller.empty:
                # Afficher les métriques clés pour le conseiller sélectionné
                col1, col2, col3 = st.columns(3)
                with col1:
                    collecte_conseiller = df_conseiller['Montant du placement'].sum()
                    st.metric(f"Collecte Totale - {conseiller_selectionne}", f"{collecte_conseiller:,.0f}€")
                with col2:
                    nb_souscriptions = len(df_conseiller)
                    st.metric("Nombre de Souscriptions", f"{nb_souscriptions:,}")
                with col3:
                    ticket_moyen = df_conseiller['Montant du placement'].mean()
                    st.metric("Ticket Moyen", f"{ticket_moyen:,.0f}€")
                
                # Évolution mensuelle du conseiller
                evolution_conseiller = df_conseiller.groupby('Mois').agg(
                    Collecte=('Montant du placement', 'sum'),
                    Nombre=('Montant du placement', 'count')
                ).reset_index()
                
                # Trier par mois
                evolution_conseiller = evolution_conseiller.sort_values('Mois')
                
                # Graphique d'évolution mensuelle
                fig_evolution = px.line(
                    evolution_conseiller,
                    x='Mois',
                    y='Collecte',
                    title=f"Évolution Mensuelle de la Collecte - {conseiller_selectionne}",
                    markers=True,
                    line_shape='linear'
                )
                fig_evolution.update_traces(line=dict(width=3), marker=dict(size=10))
                fig_evolution.update_layout(yaxis_title="Collecte (€)")
                st.plotly_chart(fig_evolution, use_container_width=True)
                
                # Graphique du nombre de souscriptions
                fig_nombre = px.bar(
                    evolution_conseiller,
                    x='Mois',
                    y='Nombre',
                    title=f"Nombre de Souscriptions Mensuelles - {conseiller_selectionne}",
                    color_discrete_sequence=['#2E86C1']
                )
                fig_nombre.update_layout(yaxis_title="Nombre de Souscriptions")
                st.plotly_chart(fig_nombre, use_container_width=True)
                
                # Tableau détaillé par mois
                st.subheader(f"📋 Détail Mensuel pour {conseiller_selectionne}")
                
                # Formatage pour l'affichage
                evolution_display = evolution_conseiller.copy()
                evolution_display['Collecte'] = evolution_display['Collecte'].apply(lambda x: f"{x:,.0f}€")
                evolution_display.columns = ['Mois', 'Collecte', 'Nombre de Souscriptions']
                
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
        else:
            st.error("Impossible d'effectuer l'analyse par conseiller par mois : colonnes 'Conseiller' ou 'Mois' manquantes")
            st.write(f"Colonnes disponibles : {df_filtre.columns.tolist()}")
        
        # Comparaison entre conseillers par mois
        st.subheader("🔍 Comparaison entre Conseillers par Mois")
        
        if 'Conseiller' in df_filtre.columns and 'Mois' in df_filtre.columns:
            # Créer un sélecteur de mois
            mois_disponibles = sorted(df_filtre['Mois'].unique())
            mois_selectionne = st.selectbox("Sélectionner un mois", options=mois_disponibles, key="select_mois_immo")
            
            # Filtrer les données pour le mois sélectionné
            df_mois = df_filtre[df_filtre['Mois'] == mois_selectionne].copy()
            
            if not df_mois.empty:
                # Calculer les statistiques par conseiller pour le mois sélectionné
                stats_mois = df_mois.groupby('Conseiller').agg(
                    Collecte=('Montant du placement', 'sum'),
                    Nombre=('Montant du placement', 'count'),
                    Ticket_Moyen=('Montant du placement', 'mean')
                ).reset_index()
                
                # Trier par collecte décroissante
                stats_mois = stats_mois.sort_values('Collecte', ascending=False)
                
                # Graphique des conseillers du mois
                fig_conseillers_mois = px.bar(
                    stats_mois.head(10),  # Top 10 conseillers
                    x='Conseiller',
                    y='Collecte',
                    title=f"Top 10 Conseillers - {mois_selectionne}",
                    text='Collecte',
                    color='Collecte',
                    color_continuous_scale='Viridis'
                )
                fig_conseillers_mois.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
                st.plotly_chart(fig_conseillers_mois, use_container_width=True)
                
                # Tableau détaillé par conseiller du mois
                stats_mois_display = stats_mois.copy()
                stats_mois_display['Collecte'] = stats_mois_display['Collecte'].apply(lambda x: f"{x:,.0f}€")
                stats_mois_display['Ticket_Moyen'] = stats_mois_display['Ticket_Moyen'].apply(lambda x: f"{x:,.0f}€")
                stats_mois_display.columns = ['Conseiller', 'Collecte', 'Nombre de Souscriptions', 'Ticket Moyen']
                
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
            st.error("Impossible d'effectuer la comparaison entre conseillers par mois : colonnes 'Conseiller' ou 'Mois' manquantes")
            st.write(f"Colonnes disponibles : {df_filtre.columns.tolist()}")
        


if __name__ == "__main__":
    st.set_page_config(page_title="Analyse Souscriptions Immobilières", page_icon="🏠", layout="wide")
    st.title("🏠 Analyse des Souscriptions Immobilières")
    
    uploaded_file = st.file_uploader("📁 Charger un fichier de souscriptions immobilières", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        analyser_souscriptions_immo(df)
