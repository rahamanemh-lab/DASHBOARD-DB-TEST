import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import re
import calendar

# Importer les fonctions de capture des messages pour debug
try:
    from utils.data_processing_debug import capture_success, capture_warning, capture_info, capture_error
except ImportError:
    # Fallback si le module debug n'est pas disponible
    def capture_success(msg): st.success(f"✅ {msg}")
    def capture_warning(msg): st.warning(f"⚠️ {msg}")
    def capture_info(msg): st.info(f"ℹ️ {msg}")
    def capture_error(msg): st.error(f"❌ {msg}")

def export_dataframe(df, prefix):
    """Exporte un DataFrame au format CSV."""
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Télécharger les données (CSV)",
        data=csv,
        file_name=f"{prefix}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def extract_conseiller(df, column=None):
    """Extrait le nom du conseiller à partir d'une colonne spécifiée ou recherche parmi plusieurs colonnes possibles.
    
    Args:
        df: DataFrame contenant les données
        column: Nom de la colonne contenant l'information du conseiller (optionnel)
        
    Returns:
        DataFrame avec une colonne 'Conseiller' standardisée
    """
    df = df.copy()
    
    # Capturer les messages de débogage au lieu de les afficher directement
    capture_info(f"Débogage extraction conseiller")
    capture_info(f"Colonnes disponibles: {df.columns.tolist()}")
    
    # Si la colonne Conseiller existe déjà et contient des valeurs non nulles, on la garde
    if 'Conseiller' in df.columns and df['Conseiller'].notna().any() and (df['Conseiller'] != 'Inconnu').any():
        capture_success("Colonne 'Conseiller' déjà présente et valide")
        return df
    
    # Si une colonne spécifique est fournie, on l'utilise
    if column and column in df.columns:
        colonne_trouvee = column
        capture_info(f"Utilisation de la colonne spécifiée: '{column}'")
    else:
        # Sinon, on cherche parmi les colonnes possibles
        colonne_trouvee = None
        # Élargir la liste des colonnes possibles
        colonnes_possibles = [
            'Conseiller', 'conseiller', 'Conseiller', 'Conseiller', 'Staff', 'staff',
            'Vendeur', 'vendeur', 'Commercial', 'commercial', 'Agent', 'agent',
            'Responsable', 'responsable', 'Consultant', 'consultant', 'Représentant', 'représentant',
            'Souscription', 'souscription'
        ]
        
        # Recherche intelligente parmi toutes les colonnes
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['conseiller', 'Conseiller', 'agent', 'staff', 'vendeur', 'commercial']):
                colonne_trouvee = col
                capture_info(f"Colonne trouvée par recherche intelligente: '{col}'")
                break
        
        # Si rien n'est trouvé, chercher dans les colonnes standard
        if colonne_trouvee is None:
            for col in colonnes_possibles:
                if col in df.columns and df[col].notna().any():
                    colonne_trouvee = col
                    capture_info(f"Colonne trouvée dans la liste standard: '{col}'")
                    break
    
    if colonne_trouvee:
        # Capturer quelques exemples de valeurs pour le débogage
        capture_info(f"Exemples de valeurs dans la colonne '{colonne_trouvee}': {df[colonne_trouvee].head(3).tolist()}")
        
        # Extraire le nom du conseiller avec plusieurs patterns possibles
        patterns = [
            r"Conseiller\s*['\"]([^'\"]+)['\"]?",  # Conseiller 'Nom' ou Conseiller "Nom"
            r"Conseiller\s*['\"]([^'\"]+)['\"]?",     # Conseiller 'Nom' ou Conseiller "Nom"
            r"Staff\s*['\"]([^'\"]+)['\"]?",       # Staff 'Nom' ou Staff "Nom"
            r"Agent:\s*([^,;]+)",                    # Agent: Nom
            r"Par:\s*([^,;]+)"                       # Par: Nom
        ]
        
        # Appliquer chaque pattern et garder la première correspondance
        df['Conseiller'] = df[colonne_trouvee]
        
        # Compteur pour le débogage
        matches_count = 0
        
        for pattern in patterns:
            mask = df['Conseiller'].str.contains(pattern, na=False, regex=True)
            if mask.any():
                matches_count += mask.sum()
                df.loc[mask, 'Conseiller'] = df.loc[mask, colonne_trouvee].str.extract(pattern, expand=False)
                capture_success(f"Pattern '{pattern}' a trouvé {mask.sum()} correspondances")
        
        # Si aucun pattern n'a fonctionné, utiliser la valeur brute
        if matches_count == 0:
            capture_warning("Aucun pattern n'a fonctionné, utilisation des valeurs brutes")
        
        # Nettoyer les valeurs (supprimer espaces en début/fin)
        df['Conseiller'] = df['Conseiller'].str.strip()
        
        # Remplacer les valeurs vides ou NaN par 'Inconnu'
        df['Conseiller'] = df['Conseiller'].fillna('Inconnu')
        df.loc[df['Conseiller'] == '', 'Conseiller'] = 'Inconnu'
        
        # Capturer quelques exemples de résultats
        capture_info("Exemples de conseillers extraits:")
        capture_info(f"{df[['Conseiller']].head(5).to_dict()}")
        
        return df
    else:
        # Si aucune colonne de conseiller n'est trouvée, créer une colonne par défaut
        capture_warning("Aucune colonne de conseiller trouvée (Conseiller, Conseiller, Staff, etc.). Utilisation d'une valeur par défaut.")
        df['Conseiller'] = 'Inconnu'
        return df

@st.cache_data
def adjust_dates_to_month_range(df, date_column):
    """Ajuste les dates pour avoir une plage complète du 1er au dernier jour du mois.
    
    Args:
        df (DataFrame): DataFrame contenant les dates à ajuster
        date_column (str): Nom de la colonne contenant les dates
        
    Returns:
        DataFrame: DataFrame avec les colonnes ajoutées:
            - 'Mois': Format 'YYYY-MM'
            - 'Premier_Jour_Mois': Premier jour du mois (date)
            - 'Dernier_Jour_Mois': Dernier jour du mois (date)
    """
    if date_column not in df.columns:
        return df
    
    # Créer une copie pour éviter de modifier l'original
    result_df = df.copy()
    
    # Filtrer les dates valides
    valid_dates = result_df[date_column].notna()
    
    if valid_dates.sum() == 0:
        # Aucune date valide
        result_df['Mois'] = 'Date inconnue'
        result_df['Premier_Jour_Mois'] = pd.NaT
        result_df['Dernier_Jour_Mois'] = pd.NaT
        return result_df
    
    # Extraire l'année et le mois
    result_df.loc[valid_dates, 'Mois'] = result_df.loc[valid_dates, date_column].dt.strftime('%Y-%m')
    result_df.loc[~valid_dates, 'Mois'] = 'Date inconnue'
    
    # Calculer le premier jour du mois
    result_df.loc[valid_dates, 'Premier_Jour_Mois'] = result_df.loc[valid_dates, date_column].dt.to_period('M').dt.start_time
    
    # Calculer le dernier jour du mois
    result_df.loc[valid_dates, 'Dernier_Jour_Mois'] = result_df.loc[valid_dates, date_column].dt.to_period('M').dt.end_time
    
    return result_df

def analyse_collecte_produit_conseiller_fallback(df, title):
    """Analyse de la répartition de la collecte par conseiller, avec gestion optionnelle de la colonne Produit.
    
    Cette fonction est une version adaptée qui fonctionne même sans la colonne 'Produit'.
    Si la colonne 'Produit' est présente, elle fonctionne comme l'original.
    Sinon, elle crée une colonne 'Produit' avec une valeur par défaut.
    """
    st.subheader(f"📊 Répartition Collecte par Conseiller - {title}")
    
    if df.empty:
        st.warning(f"⚠️ Aucune donnée trouvée pour {title}.")
        return
    
    # Vérifier si la colonne Montant du placement existe
    if 'Montant du placement' not in df.columns:
        st.error("❌ Colonne 'Montant du placement' non trouvée dans les données.")
        with st.expander("Colonnes disponibles"):
            st.write(df.columns.tolist())
        return
    
    # Créer une copie du DataFrame pour éviter de modifier l'original
    df_valid = df[df['Montant du placement'] > 0].copy()
    if df_valid.empty:
        st.warning(f"⚠️ Aucune souscription avec un montant supérieur à 0 pour {title}.")
        return
    
    # Vérifier si la colonne Produit existe
    has_product_column = 'Produit' in df_valid.columns
    
    if not has_product_column:
        # Créer une colonne Produit par défaut
        capture_info("Colonne 'Produit' non trouvée. Utilisation d'une catégorie par défaut 'Épargne'.")
        df_valid['Produit'] = 'Épargne'
    
    # Créer le tableau pivot
    pivot_table = df_valid.pivot_table(
        values='Montant du placement',
        index='Conseiller',
        columns='Produit',
        aggfunc='sum',
        fill_value=0
    ).round(0)
    
    # Vérifier les conseillers sans collecte
    conseillers_sans_collecte = df_valid['Conseiller'].value_counts()
    if len(conseillers_sans_collecte) < df['Conseiller'].nunique():
        capture_warning(f"Certains conseillers n'ont aucune souscription valide (montant > 0).")
    
    # Si nous avons plusieurs produits, afficher la heatmap
    if has_product_column and len(df_valid['Produit'].unique()) > 1:
        fig_heatmap = px.imshow(
            pivot_table,
            labels=dict(x="Produit", y="Conseiller", color="Collecte (€)"),
            title="Collecte par Produit et Conseiller (Heatmap)",
            color_continuous_scale='Blues',
            text_auto='.0f'
        )
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Graphique à barres empilées
        pivot_table_reset = pivot_table.reset_index().melt(id_vars='Conseiller', var_name='Produit', value_name='Collecte')
        fig_stacked = px.bar(
            pivot_table_reset,
            x='Conseiller',
            y='Collecte',
            color='Produit',
            title="Collecte par Conseiller et Produit (Barres Empilées)",
            text_auto='.0f'
        )
        fig_stacked.update_layout(height=600, xaxis_tickangle=45)
        st.plotly_chart(fig_stacked, use_container_width=True)
    else:
        # Afficher un graphique à barres simple pour un seul produit
        collecte_par_conseiller = df_valid.groupby('Conseiller')['Montant du placement'].sum().sort_values(ascending=False)
        fig_bar = px.bar(
            collecte_par_conseiller,
            title=f"Collecte par Conseiller - {title}",
            labels={'value': 'Collecte (€)', 'index': 'Conseiller'},
            text_auto='.0f',
            color=collecte_par_conseiller.values,
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(height=600, xaxis_tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Afficher le tableau de données
    pivot_table_display = pivot_table.copy()
    pivot_table_display['Total'] = pivot_table_display.sum(axis=1)
    pivot_table_display = pivot_table_display.sort_values('Total', ascending=False)
    pivot_table_display = pivot_table_display.apply(lambda x: x.apply(lambda y: f"{y:,.0f}€" if pd.notnull(y) else "0€"))
    st.dataframe(pivot_table_display, use_container_width=True)
    export_dataframe(pivot_table_display, f"collecte_produit_conseiller_{title.lower().replace(' ', '_')}")

def analyse_performance_conseiller_fallback(df, montant_col, groupby_col, title, min_operations=5):
    """Analyse générique de la performance des conseillers.
    Version adaptée qui fonctionne sans dépendance à des colonnes spécifiques.
    """
    if montant_col not in df.columns:
        st.error(f"❌ Colonne '{montant_col}' non trouvée dans les données.")
        with st.expander("Colonnes disponibles"):
            st.write(df.columns.tolist())
        return
    
    df_valid = df[df[montant_col] > 0].copy()
    if df_valid.empty:
        st.warning(f"⚠️ Aucune donnée avec montant > 0 pour {title}.")
        return
    
    agg_dict = {montant_col: ['sum', 'count', 'mean', 'std']}
    perf = df_valid.groupby(groupby_col).agg(agg_dict).round(0)
    perf.columns = ['Collecte Totale', 'Nb Opérations', 'Ticket Moyen', 'Écart-Type']
    perf = perf.reset_index().sort_values('Collecte Totale', ascending=False)
    
    perf_filtered = perf[perf['Nb Opérations'] >= min_operations].head(10)
    
    st.subheader(f"🏆 Top 10 Conseillers - {title}")
    fig = px.bar(
        perf_filtered,
        x='Collecte Totale',
        y=groupby_col,
        orientation='h',
        title=f"Collecte Totale par Conseiller",
        text='Collecte Totale',
        color='Collecte Totale',
        color_continuous_scale='Blues'
    )
    fig.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    perf_display = perf.copy()
    perf_display['Collecte Totale'] = perf_display['Collecte Totale'].apply(lambda x: f"{x:,.0f}€")
    perf_display['Ticket Moyen'] = perf_display['Ticket Moyen'].apply(lambda x: f"{x:,.0f}€")
    perf_display['Écart-Type'] = perf_display['Écart-Type'].apply(lambda x: f"{x:,.0f}€")
    
    st.dataframe(perf_display, use_container_width=True)
    export_dataframe(perf_display, title.replace(" ", "_").lower())
