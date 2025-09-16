import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

OBJECTIF_MENSUEL = 1_800_000  # Objectif de collecte mensuelle en euros
OBJECTIF_HEBDO = 450_000     # Objectif hebdomadaire en euros

# Ajout d'une fonction d'évolution mensuelle, collecte par conseiller et étapes commerciales détaillées

def afficher_graphiques_complementaires(df):
    st.header("📈 Graphiques Complémentaires")

    df['Date de souscription'] = pd.to_datetime(df['Date de souscription'], errors='coerce')
    df['Mois'] = df['Date de souscription'].dt.to_period('M').astype(str)
    df['Semaine'] = df['Date de souscription'].dt.to_period('W').astype(str)

    mois_disponibles = sorted(df['Mois'].dropna().unique())
    mois_selectionne = st.selectbox("📆 Sélectionnez un mois pour filtrer les graphiques", options=["Tous"] + mois_disponibles)

    if mois_selectionne != "Tous":
        df = df[df['Mois'] == mois_selectionne]

    df_immo = df[df["Type de l'opportunité"] == "IMMO"]
    df_epargne = df[df["Type de l'opportunité"] != "IMMO"]

    if not df_immo.empty:
        st.subheader("🏠 Opportunités IMMO (hors collecte épargne)")
        st.write(f"Nombre d'opportunités IMMO : {len(df_immo)}")

        repartition_etapes_immo = df_immo['Étape'].value_counts().reset_index()
        repartition_etapes_immo.columns = ['Étape', 'Nombre']
        fig_immo = px.pie(repartition_etapes_immo, names='Étape', values='Nombre', title="Répartition des étapes pour les opportunités IMMO", hole=0.3)
        st.plotly_chart(fig_immo, use_container_width=True)

        st.subheader("📊 Détail des opportunités IMMO par conseiller")
        count_etapes_immo = df_immo.groupby(['Conseiller', 'Étape']).size().reset_index(name='Nombre')
        fig_immo2 = px.bar(count_etapes_immo, x='Nombre', y='Conseiller', color='Étape', title="Opportunités IMMO par conseiller et par étape", orientation='h')
        st.plotly_chart(fig_immo2, use_container_width=True)

    df = df_epargne

    if 'Date de souscription' in df.columns and 'Montant du placement' in df.columns:
        st.subheader("📅 Évolution mensuelle de la collecte (Objectif: 1,8M€ par mois)")
        evolution = df.groupby('Mois')['Montant du placement'].sum().reset_index()
        evolution['Écart à l\'objectif (€)'] = evolution['Montant du placement'] - OBJECTIF_MENSUEL
        evolution['Statut'] = evolution['Écart à l\'objectif (€)'].apply(lambda x: '✅ Objectif atteint' if x >= 0 else '❌ En retard')

        fig1 = px.bar(evolution, x='Mois', y='Montant du placement', text='Montant du placement', color='Statut',
                     color_discrete_map={'✅ Objectif atteint': 'green', '❌ En retard': 'red'},
                     title="Collecte mensuelle vs Objectif (1,8M€/mois)",
                     labels={'Montant du placement': 'Collecte (€)', 'Mois': 'Période'})
        fig1.add_hline(y=OBJECTIF_MENSUEL, line_dash="dash", line_color="blue", annotation_text="Objectif 1,8M€")
        fig1.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
        fig1.update_layout(yaxis_title="Collecte mensuelle (€)", height=500)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📊 Progression actuelle (mois en cours)")
        mois_actuel = datetime.now().strftime("%Y-%m")
        collecte_mois_actuel = df[df['Mois'] == mois_actuel]['Montant du placement'].sum()
        progress = min(collecte_mois_actuel / OBJECTIF_MENSUEL, 1.0)
        st.progress(progress)
        st.write(f"Collecte actuelle : {collecte_mois_actuel:,.0f} € / {OBJECTIF_MENSUEL:,.0f} €")

        st.subheader("📉 Retard cumulé depuis avril")
        evolution['Retard cumulé'] = evolution['Écart à l\'objectif (€)'].cumsum()
        fig_retard = px.line(evolution, x='Mois', y='Retard cumulé', title="Retard cumulé depuis avril", markers=True)
        fig_retard.update_layout(yaxis_title="Cumul du retard (€)", height=400)
        st.plotly_chart(fig_retard, use_container_width=True)

        st.subheader("🧮 Scénario de compensation du retard")
        total_retard = evolution['Écart à l\'objectif (€)'].apply(lambda x: x if x < 0 else 0).sum()
        st.write(f"📉 Retard global estimé : {abs(total_retard):,.0f} €")

        moyenne_souscription = df['Montant du placement'].mean()
        petit_ticket = 2000
        gros_ticket = 10000

        if moyenne_souscription > 0:
            st.write("### 🧪 Simulations de rattrapage")
            col1, col2, col3 = st.columns(3)
            col1.metric("Souscriptions nécessaires (ticket moyen)", f"{int(np.ceil(abs(total_retard) / moyenne_souscription))}")
            col2.metric("Avec petit ticket (2K€)", f"{int(np.ceil(abs(total_retard) / petit_ticket))}")
            col3.metric("Avec gros ticket (10K€)", f"{int(np.ceil(abs(total_retard) / gros_ticket))}")

        st.subheader("🧭 Répartition du rattrapage par conseiller")
        total_par_conseiller = df.groupby('Conseiller')['Montant du placement'].sum().reset_index()
        total_collecte = total_par_conseiller['Montant du placement'].sum()
        total_par_conseiller['Poids (%)'] = total_par_conseiller['Montant du placement'] / total_collecte
        total_par_conseiller['Part à rattraper (€)'] = total_par_conseiller['Poids (%)'] * abs(total_retard)
        total_par_conseiller['Souscriptions estimées (ticket moyen)'] = total_par_conseiller['Part à rattraper (€)'] / moyenne_souscription

        st.dataframe(total_par_conseiller[['Conseiller', 'Part à rattraper (€)', 'Souscriptions estimées (ticket moyen)']].round(0), use_container_width=True)

        st.subheader("📉 Nombre de souscriptions par mois")
        count_souscriptions = df.groupby('Mois').size().reset_index(name='Nombre')
        fig1b = px.bar(count_souscriptions, x='Mois', y='Nombre', title="Nombre de souscriptions par mois", labels={'Nombre': "Nombre de souscriptions"})
        st.plotly_chart(fig1b, use_container_width=True)

        st.subheader("📊 Collecte moyenne")
        moyenne_globale = df['Montant du placement'].mean()
        moyenne_par_conseiller = df.groupby('Conseiller')['Montant du placement'].mean().reset_index()
        moyenne_par_conseiller.columns = ['Conseiller', 'Montant moyen (€)']
        st.metric("Montant moyen global", f"{moyenne_globale:,.0f} €")
        st.dataframe(moyenne_par_conseiller.sort_values('Montant moyen (€)', ascending=False), use_container_width=True)

        st.subheader("👤 Collecte par conseiller")
        collecte_conseiller = df.groupby('Conseiller')['Montant du placement'].sum().sort_values().reset_index()
        fig2 = px.bar(collecte_conseiller, x='Montant du placement', y='Conseiller', orientation='h',
                     title="Collecte totale par conseiller", labels={'Montant du placement': 'Collecte (€)', 'Conseiller': 'Conseiller'},
                     text='Montant du placement')
        fig2.update_traces(texttemplate='%{text:,.0f} €', textposition='outside')
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📋 Nombre de souscriptions par conseiller")
        count_conseillers = df.groupby('Conseiller').size().reset_index(name='Nombre')
        fig2b = px.bar(count_conseillers, x='Nombre', y='Conseiller', orientation='h',
                      title="Volume de souscriptions par conseiller", color='Nombre', color_continuous_scale='Viridis')
        st.plotly_chart(fig2b, use_container_width=True)

        if 'Étape' in df.columns:
            st.subheader("🔁 Répartition des étapes commerciales")
            repartition_etapes = df['Étape'].value_counts().reset_index()
            repartition_etapes.columns = ['Étape', 'Nombre']
            fig3 = px.pie(repartition_etapes, names='Étape', values='Nombre', title="Répartition des opportunités par étape", hole=0.3)
            st.plotly_chart(fig3, use_container_width=True)

        if 'Type de l\'opportunité' in df.columns:
            st.subheader("📦 Répartition de la collecte par type de produit")
            collecte_produit = df.groupby("Type de l'opportunité")["Montant du placement"].sum().sort_values(ascending=False).reset_index()
            fig4 = px.pie(collecte_produit, names="Type de l'opportunité", values="Montant du placement", title="Part des produits dans la collecte", hole=0.4)
            st.plotly_chart(fig4, use_container_width=True)

        if 'Conseiller' in df.columns and 'Étape' in df.columns:
            st.subheader("📊 Opportunités par conseiller selon les étapes")
            count_etapes = df.groupby(['Conseiller', 'Étape']).size().reset_index(name='Nombre')
            fig5 = px.bar(count_etapes, x='Nombre', y='Conseiller', color='Étape', title="Détail des opportunités par conseiller et par étape", orientation='h')
            st.plotly_chart(fig5, use_container_width=True)
            
            
            
# Intégration dans la fonction main()

def main():
    st.title("📊 Dashboard Souscriptions - Données Réelles")

    uploaded_file = st.file_uploader("Chargez votre fichier Excel", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=1)
        df['Montant du placement'] = pd.to_numeric(df['Montant du placement'], errors='coerce')
        df['Montant des frais'] = pd.to_numeric(df['Montant des frais'], errors='coerce')
        df['Conseiller'] = df['Conseiller'].str.extract(r"Conseiller '([^']+)'", expand=False).fillna(df['Conseiller'])

        afficher_graphiques_complementaires(df)
        st.subheader("🔍 Aperçu des données")
        st.dataframe(df.head(50))

if __name__ == "__main__":
    main()
