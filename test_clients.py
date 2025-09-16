"""
Test rapide de l'onglet analyse clients
Lancer avec : streamlit run test_clients.py
"""
import streamlit as st
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Test Analyse Clients",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Test - Analyse Clients")

# Import du module clients
try:
    from analyses.clients import analyser_clients
    st.success("✅ Module clients importé avec succès")
except ImportError as e:
    st.error(f"❌ Erreur d'import: {e}")
    st.stop()

# Upload de fichier
uploaded_file = st.file_uploader(
    "📁 Chargez votre fichier clients Excel",
    type=['xlsx', 'xls'],
    help="Fichier avec colonnes: Nom & Prénom, Email, Date entretien, Nb Souscriptions, VR, Métier, Secteur, Revenus, etc."
)

if uploaded_file is not None:
    try:
        # Lecture du fichier
        df = pd.read_excel(uploaded_file)
        st.info(f"📊 Fichier chargé: {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Appel de l'analyse
        analyser_clients(df)
        
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement: {e}")
        st.exception(e)
else:
    st.info("👆 Veuillez charger un fichier Excel pour commencer l'analyse")
    
    # Exemple de structure attendue
    st.markdown("""
    ### 📋 Structure attendue du fichier Excel
    
    **Colonnes recommandées :**
    - Nom & Prénom
    - Email  
    - Date de l'entretien
    - Nb Souscriptions
    - Dernière Souscription
    - VR (Valeur Rachetable)
    - Métier
    - Secteur d'activité
    - Revenus
    - Type de contrat
    - Éligibilité
    - TMI (Tranche Marginale d'Imposition)
    - Profil épargnant
    - Épargne disponible
    - Situation familiale
    - Nb d'enfants
    - Date d'inscription
    - Conseiller
    
    *Note: Toutes les colonnes ne sont pas obligatoires. L'analyse s'adapte aux colonnes présentes.*
    """)