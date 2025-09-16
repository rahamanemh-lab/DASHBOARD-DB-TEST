def analyser_paiements_immo(df):
    """Analyse des paiements immobiliers reçus, leurs statuts, par conseiller et par mois."""
    st.header("🏠 Analyse des Paiements Immobiliers")
    
    # Mapper les colonnes du nouveau format aux colonnes attendues
    colonnes_mapping = {
        'Montant(€)': 'Montant',
        'Date de paiement': 'Date de paiement',
        'Statut': 'Statut',
        'Conseiller': 'Conseiller',
        'Méthode': 'Méthode de paiement',
        'Type': 'Type de paiement',
        'Frais(€)': 'Frais',
        'Remise(%)': 'Remise',
        'Produit': 'Produit',
        'Souscription': 'Souscription',
        'Bien immobilier': 'Bien immobilier',
        'Adresse': 'Adresse',
        'Prix du bien': 'Prix du bien'
    }
    
    # Import des modules nécessaires s'ils ne sont pas déjà importés
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    from datetime import datetime
    
    # Import des fonctions utilitaires si nécessaire
    try:
        from utils.data_processing import safe_to_datetime, safe_to_numeric, extract_conseiller
    except ImportError:
        # Fonctions de secours si les imports échouent
        def safe_to_datetime(series):
            return pd.to_datetime(series, errors='coerce')
            
        def safe_to_numeric(series):
            return pd.to_numeric(series, errors='coerce')
            
        def extract_conseiller(df):
            return df
    
    # Renommer les colonnes si elles existent
    for col_source, col_dest in colonnes_mapping.items():
        if col_source in df.columns and col_dest not in df.columns:
            df[col_dest] = df[col_source]
    
    # Vérifier les colonnes nécessaires
    colonnes_requises = ['Date de paiement', 'Montant', 'Statut']
    colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
    
    # Vérifier si les colonnes existent ou si des alternatives sont disponibles
    if 'Date de paiement' not in df.columns and 'Date de création' in df.columns:
        df['Date de paiement'] = df['Date de création']
        colonnes_manquantes.remove('Date de paiement') if 'Date de paiement' in colonnes_manquantes else None
    
    # Extraire le conseiller depuis la colonne Souscription si disponible
    if 'Conseiller' not in df.columns:
        if 'Souscription' in df.columns:
            df = extract_conseiller(df)
        else:
            df['Conseiller'] = 'Non spécifié'
            st.warning("⚠️ Colonne 'Conseiller' non trouvée et aucune alternative disponible.")
    
    if colonnes_manquantes:
        st.error(f"❌ Colonnes manquantes pour l'analyse des paiements immobiliers: {', '.join(colonnes_manquantes)}")
        with st.expander("Colonnes disponibles"):
            st.write(df.columns.tolist())
        return
    
    # Préparation des données pour l'analyse
    df_immo = df.copy()
    
    # Convertir les dates et montants
    if 'Date de paiement' in df_immo.columns:
        df_immo['Date de paiement'] = safe_to_datetime(df_immo['Date de paiement'])
        df_immo['Mois'] = df_immo['Date de paiement'].dt.strftime('%Y-%m')
        
        # Ajouter les colonnes Premier_Jour_Mois et Dernier_Jour_Mois pour l'analyse temporelle
        df_immo['Premier_Jour_Mois'] = df_immo['Date de paiement'].dt.to_period('M').dt.to_timestamp()
        df_immo['Dernier_Jour_Mois'] = (df_immo['Premier_Jour_Mois'] + pd.DateOffset(months=1) - pd.DateOffset(days=1))
    
    if 'Montant' in df_immo.columns:
        df_immo['Montant'] = safe_to_numeric(df_immo['Montant'])
    
    # Filtrer pour ne garder que les transactions immobilières
    if 'Type de transaction' in df.columns:
        df_immo = df[df['Type de transaction'] == 'IMMO'].copy()
    elif 'Type de produit' in df.columns:
        df_immo = df[df['Type de produit'].str.contains('IMMO|Immobilier', case=False, na=False)].copy()
    elif 'Produit' in df.columns:
        df_immo = df[df['Produit'].str.contains('IMMO|Immobilier|Bien|Appartement|Maison', case=False, na=False)].copy()
    else:
        df_immo = df.copy()
        st.warning("⚠️ Aucune colonne permettant de filtrer les transactions immobilières n'a été trouvée. Toutes les transactions sont affichées.")
    
    if df_immo.empty:
        st.warning("⚠️ Aucune transaction immobilière trouvée dans les données.")
        return
    
    # Afficher les informations sur les plages de dates
    with st.expander("📅 Informations sur les périodes mensuelles"):
        st.info("📆 Les données sont regroupées par mois complet (du 1er au dernier jour du mois)")
        
        # Créer un tableau des périodes
        if 'Premier_Jour_Mois' in df_immo.columns and 'Dernier_Jour_Mois' in df_immo.columns:
            periodes = df_immo.dropna(subset=['Premier_Jour_Mois', 'Dernier_Jour_Mois']).drop_duplicates(['Mois'])
            if not periodes.empty:
                periodes_df = pd.DataFrame({
                    'Mois': periodes['Mois'],
                    'Début': periodes['Premier_Jour_Mois'].dt.strftime('%d/%m/%Y'),
                    'Fin': periodes['Dernier_Jour_Mois'].dt.strftime('%d/%m/%Y')
                }).sort_values('Mois')
                st.dataframe(periodes_df, use_container_width=True)
    
    # Ajout d'informations spécifiques à l'immobilier
    if 'Bien immobilier' in df_immo.columns:
        st.subheader("🏢 Répartition par Type de Bien")
        type_bien_counts = df_immo.groupby('Bien immobilier').size().reset_index(name='Nombre')
        fig_type_bien = px.pie(type_bien_counts, values='Nombre', names='Bien immobilier', title="Types de Biens Immobiliers", hole=0.4)
        fig_type_bien.update_traces(textinfo='percent+label+value')
        st.plotly_chart(fig_type_bien, use_container_width=True)
    
    if 'Prix du bien' in df_immo.columns:
        df_immo['Prix du bien'] = safe_to_numeric(df_immo['Prix du bien'])
        st.subheader("💰 Répartition par Prix du Bien")
        fig_prix = px.histogram(df_immo, x='Prix du bien', nbins=20, title="Distribution des Prix des Biens")
        st.plotly_chart(fig_prix, use_container_width=True)
    
    if 'Méthode de paiement' in df_immo.columns:
        st.subheader("💳 Répartition par Méthode de Paiement")
        methode_counts = df_immo.groupby('Méthode de paiement').size().reset_index(name='Nombre')
        fig_methode = px.pie(methode_counts, values='Nombre', names='Méthode de paiement', title="Méthodes de Paiement", hole=0.4)
        fig_methode.update_traces(textinfo='percent+label+value')
        st.plotly_chart(fig_methode, use_container_width=True)
    
    # Filtrer les montants valides
    df_valid = df_immo[df_immo['Montant'] > 0].copy()
    if df_valid.empty:
        st.warning("⚠️ Aucun paiement immobilier avec un montant supérieur à 0.")
        return
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_paiements = df_valid['Montant'].sum()
        st.metric("💰 Total des Paiements", f"{total_paiements:,.0f}€")
    
    with col2:
        nb_transactions = len(df_valid)
        st.metric("🏠 Nombre de Transactions", f"{nb_transactions}")
    
    with col3:
        if 'Prix du bien' in df_valid.columns:
            prix_moyen = df_valid['Prix du bien'].mean()
            st.metric("🏢 Prix Moyen des Biens", f"{prix_moyen:,.0f}€")
        else:
            montant_moyen = df_valid['Montant'].mean()
            st.metric("💸 Montant Moyen", f"{montant_moyen:,.0f}€")
    
    with col4:
        if 'Conseiller' in df_valid.columns:
            nb_conseillers = df_valid['Conseiller'].nunique()
            st.metric("👥 Nombre de Conseillers", f"{nb_conseillers}")
        else:
            if 'Statut' in df_valid.columns:
                nb_statuts = df_valid['Statut'].nunique()
                st.metric("📊 Nombre de Statuts", f"{nb_statuts}")
    
    # Analyse temporelle
    if 'Mois' in df_valid.columns:
        st.subheader("📈 Évolution Mensuelle des Paiements Immobiliers")
        evolution_mensuelle = df_valid.groupby('Mois')['Montant'].agg(['sum', 'count']).reset_index()
        evolution_mensuelle.columns = ['Mois', 'Montant Total', 'Nombre de Transactions']
        
        # Graphique d'évolution des montants
        fig_evolution = px.line(evolution_mensuelle, x='Mois', y='Montant Total', 
                              title="Évolution des Paiements Immobiliers par Mois",
                              markers=True)
        fig_evolution.update_layout(xaxis_title="Mois", yaxis_title="Montant Total (€)")
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Graphique d'évolution du nombre de transactions
        fig_transactions = px.bar(evolution_mensuelle, x='Mois', y='Nombre de Transactions',
                                title="Nombre de Transactions Immobilières par Mois")
        fig_transactions.update_layout(xaxis_title="Mois", yaxis_title="Nombre de Transactions")
        st.plotly_chart(fig_transactions, use_container_width=True)
    
    # Analyse par conseiller
    if 'Conseiller' in df_valid.columns:
        st.subheader("👥 Performance par Conseiller")
        performance_conseillers = df_valid.groupby('Conseiller')['Montant'].agg(['sum', 'count']).reset_index()
        performance_conseillers.columns = ['Conseiller', 'Montant Total', 'Nombre de Transactions']
        performance_conseillers = performance_conseillers.sort_values('Montant Total', ascending=False)
        
        # Graphique des montants par conseiller
        fig_conseillers = px.bar(performance_conseillers, x='Conseiller', y='Montant Total',
                               title="Montant Total des Transactions par Conseiller",
                               text='Montant Total')
        fig_conseillers.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
        fig_conseillers.update_layout(xaxis_title="Conseiller", yaxis_title="Montant Total (€)")
        st.plotly_chart(fig_conseillers, use_container_width=True)
        
        # Graphique du nombre de transactions par conseiller
        fig_nb_transactions = px.bar(performance_conseillers, x='Conseiller', y='Nombre de Transactions',
                                   title="Nombre de Transactions par Conseiller",
                                   text='Nombre de Transactions')
        fig_nb_transactions.update_traces(texttemplate='%{text}', textposition='outside')
        fig_nb_transactions.update_layout(xaxis_title="Conseiller", yaxis_title="Nombre de Transactions")
        st.plotly_chart(fig_nb_transactions, use_container_width=True)
    
    # Analyse par statut
    if 'Statut' in df_valid.columns:
        st.subheader("📊 Analyse par Statut")
        statut_counts = df_valid.groupby('Statut').agg({
            'Montant': ['sum', 'mean', 'count']
        }).reset_index()
        statut_counts.columns = ['Statut', 'Montant Total', 'Montant Moyen', 'Nombre de Transactions']
        
        # Graphique des montants par statut
        fig_statut = px.bar(statut_counts, x='Statut', y='Montant Total',
                          title="Montant Total par Statut",
                          text='Montant Total')
        fig_statut.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
        fig_statut.update_layout(xaxis_title="Statut", yaxis_title="Montant Total (€)")
        st.plotly_chart(fig_statut, use_container_width=True)
        
        # Graphique du nombre de transactions par statut
        fig_nb_statut = px.bar(statut_counts, x='Statut', y='Nombre de Transactions',
                             title="Nombre de Transactions par Statut",
                             text='Nombre de Transactions')
        fig_nb_statut.update_traces(texttemplate='%{text}', textposition='outside')
        fig_nb_statut.update_layout(xaxis_title="Statut", yaxis_title="Nombre de Transactions")
        st.plotly_chart(fig_nb_statut, use_container_width=True)
        
        # Analyse spécifique des paiements en attente de validation et validés
        st.subheader("🔍 Analyse des Paiements en Attente vs Validés")
        
        # Identifier les statuts liés à la validation
        statuts_attente = [s for s in df_valid['Statut'].unique() if any(mot in s.lower() for mot in ['attente', 'pending', 'en cours', 'à valider', 'a valider'])]
        statuts_valides = [s for s in df_valid['Statut'].unique() if any(mot in s.lower() for mot in ['valid', 'approuv', 'confirm', 'accepté', 'accepte'])]
        
        if not statuts_attente and not statuts_valides:
            # Si aucun statut ne correspond aux mots-clés, essayer de deviner
            unique_statuts = df_valid['Statut'].unique()
            if len(unique_statuts) >= 2:
                # Supposer que certains statuts pourraient être liés à l'attente ou à la validation
                st.info("📝 Aucun statut clairement identifié comme 'en attente' ou 'validé'. Veuillez sélectionner les statuts correspondants:")
                col1, col2 = st.columns(2)
                with col1:
                    statuts_attente = st.multiselect("Statuts 'En Attente'", options=unique_statuts, default=[])
                with col2:
                    statuts_valides = st.multiselect("Statuts 'Validés'", options=unique_statuts, default=[])
        
        # Filtrer les données selon les statuts
        if statuts_attente or statuts_valides:
            df_attente = df_valid[df_valid['Statut'].isin(statuts_attente)] if statuts_attente else pd.DataFrame()
            df_valides = df_valid[df_valid['Statut'].isin(statuts_valides)] if statuts_valides else pd.DataFrame()
            
            # Métriques comparatives
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⏳ Paiements en Attente")
                if not df_attente.empty:
                    montant_attente = df_attente['Montant'].sum()
                    nb_attente = len(df_attente)
                    montant_moyen_attente = df_attente['Montant'].mean() if nb_attente > 0 else 0
                    
                    st.metric("💰 Montant Total en Attente", f"{montant_attente:,.0f}€")
                    st.metric("🏠 Nombre de Transactions en Attente", f"{nb_attente}")
                    st.metric("💸 Montant Moyen en Attente", f"{montant_moyen_attente:,.0f}€")
                    
                    # Détail des transactions en attente
                    with st.expander("Détail des transactions en attente"):
                        st.dataframe(df_attente[['Date de paiement', 'Montant', 'Conseiller', 'Statut'] + 
                                     [col for col in ['Bien immobilier', 'Adresse', 'Prix du bien'] if col in df_attente.columns]],
                                     use_container_width=True)
                else:
                    st.info("Aucun paiement en attente trouvé.")
            
            with col2:
                st.subheader("✅ Paiements Validés")
                if not df_valides.empty:
                    montant_valides = df_valides['Montant'].sum()
                    nb_valides = len(df_valides)
                    montant_moyen_valides = df_valides['Montant'].mean() if nb_valides > 0 else 0
                    
                    st.metric("💰 Montant Total Validé", f"{montant_valides:,.0f}€")
                    st.metric("🏠 Nombre de Transactions Validées", f"{nb_valides}")
                    st.metric("💸 Montant Moyen Validé", f"{montant_moyen_valides:,.0f}€")
                    
                    # Détail des transactions validées
                    with st.expander("Détail des transactions validées"):
                        st.dataframe(df_valides[['Date de paiement', 'Montant', 'Conseiller', 'Statut'] + 
                                     [col for col in ['Bien immobilier', 'Adresse', 'Prix du bien'] if col in df_valides.columns]],
                                     use_container_width=True)
                else:
                    st.info("Aucun paiement validé trouvé.")
            
            # Comparaison graphique
            if not df_attente.empty or not df_valides.empty:
                st.subheader("📊 Comparaison Attente vs Validés")
                
                # Préparer les données pour la comparaison
                data_comp = []
                if not df_attente.empty:
                    data_comp.append({
                        'Catégorie': 'En Attente',
                        'Montant Total': df_attente['Montant'].sum(),
                        'Nombre de Transactions': len(df_attente)
                    })
                if not df_valides.empty:
                    data_comp.append({
                        'Catégorie': 'Validés',
                        'Montant Total': df_valides['Montant'].sum(),
                        'Nombre de Transactions': len(df_valides)
                    })
                
                if data_comp:
                    df_comp = pd.DataFrame(data_comp)
                    
                    # Graphique comparatif des montants
                    fig_comp_montant = px.bar(df_comp, x='Catégorie', y='Montant Total',
                                           title="Comparaison des Montants: En Attente vs Validés",
                                           text='Montant Total',
                                           color='Catégorie',
                                           color_discrete_map={'En Attente': 'orange', 'Validés': 'green'})
                    fig_comp_montant.update_traces(texttemplate='%{text:,.0f}€', textposition='outside')
                    st.plotly_chart(fig_comp_montant, use_container_width=True)
                    
                    # Graphique comparatif du nombre de transactions
                    fig_comp_nb = px.bar(df_comp, x='Catégorie', y='Nombre de Transactions',
                                       title="Comparaison du Nombre de Transactions: En Attente vs Validés",
                                       text='Nombre de Transactions',
                                       color='Catégorie',
                                       color_discrete_map={'En Attente': 'orange', 'Validés': 'green'})
                    fig_comp_nb.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig_comp_nb, use_container_width=True)
                    
                # Analyse temporelle des validations si les dates sont disponibles
                if 'Date de paiement' in df_valid.columns and 'Mois' in df_valid.columns:
                    if not df_attente.empty and not df_valides.empty:
                        st.subheader("📈 Évolution Mensuelle: Attente vs Validés")
                        
                        # Évolution mensuelle des paiements en attente
                        evolution_attente = df_attente.groupby('Mois')['Montant'].sum().reset_index()
                        evolution_attente['Catégorie'] = 'En Attente'
                        
                        # Évolution mensuelle des paiements validés
                        evolution_valides = df_valides.groupby('Mois')['Montant'].sum().reset_index()
                        evolution_valides['Catégorie'] = 'Validés'
                        
                        # Combiner les données
                        evolution_combinee = pd.concat([evolution_attente, evolution_valides])
                        
                        # Graphique d'évolution combinée
                        fig_evolution_combinee = px.line(evolution_combinee, x='Mois', y='Montant', 
                                                      color='Catégorie',
                                                      title="Évolution Mensuelle: Paiements en Attente vs Validés",
                                                      markers=True,
                                                      color_discrete_map={'En Attente': 'orange', 'Validés': 'green'})
                        fig_evolution_combinee.update_layout(xaxis_title="Mois", yaxis_title="Montant Total (€)")
                        st.plotly_chart(fig_evolution_combinee, use_container_width=True)
        else:
            st.info("📝 Aucun statut identifié comme 'en attente' ou 'validé' dans les données.")
    else:
        st.warning("⚠️ La colonne 'Statut' est nécessaire pour analyser les paiements en attente et validés.")

    
    # Tableau croisé dynamique
    st.subheader("📋 Tableau Croisé Dynamique")
    
    # Sélection des dimensions
    col1, col2 = st.columns(2)
    with col1:
        dimensions_disponibles = ['Mois', 'Conseiller', 'Statut']
        dimensions_disponibles += [col for col in ['Méthode de paiement', 'Type de paiement', 'Bien immobilier'] 
                                  if col in df_valid.columns]
        dimension_ligne = st.selectbox("Dimension en ligne", options=dimensions_disponibles, index=0)
    
    with col2:
        dimension_colonne = st.selectbox("Dimension en colonne", 
                                       options=[dim for dim in dimensions_disponibles if dim != dimension_ligne],
                                       index=min(1, len(dimensions_disponibles)-1))
    
    # Création du tableau croisé
    if dimension_ligne in df_valid.columns and dimension_colonne in df_valid.columns:
        pivot_table = pd.pivot_table(df_valid, 
                                    values='Montant',
                                    index=dimension_ligne,
                                    columns=dimension_colonne,
                                    aggfunc=['sum', 'count'],
                                    margins=True,
                                    margins_name='Total')
        
        # Formater le tableau pour l'affichage
        pivot_display = pivot_table.copy()
        pivot_display.columns = [f"{agg} - {col}" if col != 'Total' else f"{agg} - Total" 
                               for agg, col in pivot_display.columns]
        
        # Afficher le tableau
        st.dataframe(pivot_display, use_container_width=True)
        
        # Option de téléchargement
        csv = pivot_display.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="tableau_croise_paiements_immo.csv">Télécharger le tableau (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # Filtres avancés
    st.subheader("🔍 Filtres Avancés")
    with st.expander("Afficher/Masquer les filtres"):
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Mois' in df_valid.columns:
                mois_disponibles = sorted(df_valid['Mois'].unique())
                mois_selectionnes = st.multiselect("Mois", options=mois_disponibles, default=[])
            
            if 'Conseiller' in df_valid.columns:
                conseillers_disponibles = sorted(df_valid['Conseiller'].unique())
                conseillers_selectionnes = st.multiselect("Conseillers", options=conseillers_disponibles, default=[])
        
        with col2:
            if 'Statut' in df_valid.columns:
                statuts_disponibles = sorted(df_valid['Statut'].unique())
                statuts_selectionnes = st.multiselect("Statuts", options=statuts_disponibles, default=[])
            
            if 'Bien immobilier' in df_valid.columns:
                biens_disponibles = sorted(df_valid['Bien immobilier'].unique())
                biens_selectionnes = st.multiselect("Types de biens", options=biens_disponibles, default=[])
        
        # Appliquer les filtres
        df_filtre = df_valid.copy()
        
        if 'Mois' in df_valid.columns and mois_selectionnes:
            df_filtre = df_filtre[df_filtre['Mois'].isin(mois_selectionnes)]
        
        if 'Conseiller' in df_valid.columns and conseillers_selectionnes:
            df_filtre = df_filtre[df_filtre['Conseiller'].isin(conseillers_selectionnes)]
        
        if 'Statut' in df_valid.columns and statuts_selectionnes:
            df_filtre = df_filtre[df_filtre['Statut'].isin(statuts_selectionnes)]
        
        if 'Bien immobilier' in df_valid.columns and biens_selectionnes:
            df_filtre = df_filtre[df_filtre['Bien immobilier'].isin(biens_selectionnes)]
        
        # Afficher les résultats filtrés
        if not df_filtre.empty:
            st.subheader("📋 Données Filtrées")
            st.dataframe(df_filtre, use_container_width=True)
            
            # Métriques des données filtrées
            col1, col2, col3 = st.columns(3)
            with col1:
                total_filtre = df_filtre['Montant'].sum()
                st.metric("💰 Total Filtré", f"{total_filtre:,.0f}€")
            
            with col2:
                nb_transactions_filtre = len(df_filtre)
                st.metric("🏠 Transactions Filtrées", f"{nb_transactions_filtre}")
            
            with col3:
                if nb_transactions_filtre > 0:
                    montant_moyen_filtre = df_filtre['Montant'].mean()
                    st.metric("💸 Montant Moyen Filtré", f"{montant_moyen_filtre:,.0f}€")
        else:
            st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés.")
