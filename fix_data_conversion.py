import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime

# Fonctions améliorées pour la conversion de données
def safe_to_datetime_improved(df, col, format=None, errors='coerce'):
    """Version améliorée de safe_to_datetime avec meilleure gestion des erreurs."""
    if col not in df.columns:
        return df
    try:
        df[col] = pd.to_datetime(df[col], format=format, errors=errors)
        return df
    except Exception as e:
        st.warning(f"Erreur lors de la conversion de la colonne {col} en datetime: {e}")
        return df

def safe_to_numeric_improved(df, col, errors='coerce'):
    """Version améliorée de safe_to_numeric avec meilleure gestion des erreurs."""
    if col not in df.columns:
        return df
    try:
        df[col] = pd.to_numeric(df[col], errors=errors)
        df[col] = df[col].fillna(0).astype(float)  # Assurer que c'est un float et remplacer NaN par 0
        return df
    except Exception as e:
        st.warning(f"Erreur lors de la conversion de la colonne {col} en numérique: {e}")
        return df

def extract_conseiller_improved(df):
    """Version améliorée de extract_conseiller avec recherche plus robuste."""
    # Liste des noms possibles pour la colonne conseiller
    conseiller_columns = [
        'Conseiller', 'conseiller', 'CONSEILLER', 'Advisor', 'advisor', 'ADVISOR',
        'Staff', 'staff', 'STAFF', 'Vendeur', 'vendeur', 'VENDEUR',
        'Commercial', 'commercial', 'COMMERCIAL', 'Agent', 'agent', 'AGENT',
        'Représentant', 'représentant', 'REPRÉSENTANT', 'Nom', 'nom', 'NOM'
    ]
    
    # Chercher la première colonne qui existe
    for col in conseiller_columns:
        if col in df.columns and df[col].notna().any():
            # Nettoyer les valeurs
            df['Conseiller'] = df[col].astype(str).str.strip()
            # Extraire le nom si format "Conseiller: Nom" ou similaire
            df['Conseiller'] = df['Conseiller'].str.replace(r'^.*[":;]\s*', '', regex=True)
            # Remplacer les valeurs vides par 'Inconnu'
            df['Conseiller'] = df['Conseiller'].replace(['', 'nan', 'None', 'NaN'], 'Inconnu')
            return df
    
    # Si aucune colonne trouvée, créer une colonne par défaut
    df['Conseiller'] = 'Inconnu'
    return df

def ensure_arrow_compatibility_improved(df):
    """Assure que le DataFrame est compatible avec Arrow pour Streamlit."""
    if df is None or df.empty:
        return df
        
    # Créer une copie pour éviter de modifier l'original
    df_safe = df.copy()
    
    # Convertir toutes les colonnes numériques en float pour éviter les problèmes de conversion
    for col in df_safe.select_dtypes(include=['number']).columns:
        df_safe[col] = df_safe[col].astype(float)
    
    # Convertir toutes les colonnes object en string
    for col in df_safe.select_dtypes(include=['object']).columns:
        df_safe[col] = df_safe[col].astype(str)
    
    # Traitement spécial pour les colonnes de montant
    montant_columns = ['Montant', 'montant', 'MONTANT', 'Montant_Total', 'Montant Total', 'Montant_Moyen']
    for col in df_safe.columns:
        if any(montant_name in col for montant_name in montant_columns):
            df_safe[col] = pd.to_numeric(df_safe[col], errors='coerce').fillna(0).astype(float)
            
    return df_safe

def main():
    st.set_page_config(page_title="Correction des Données", page_icon="🛠️", layout="wide")
    st.title("🛠️ Outil de Correction des Données")
    
    st.write("""
    Cet outil vous permet de diagnostiquer et corriger les problèmes de conversion de données
    dans votre fichier Excel avant de l'utiliser dans le dashboard principal.
    """)
    
    uploaded_file = st.file_uploader("📁 Charger un fichier Excel", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            # Charger les données
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Fichier chargé avec succès: {uploaded_file.name}")
            
            # Afficher un aperçu des données
            st.subheader("📊 Aperçu des données")
            st.write(f"Nombre de lignes: {len(df)}")
            st.write(f"Colonnes: {df.columns.tolist()}")
            st.dataframe(df.head(5))
            
            # Analyse des colonnes
            st.subheader("🔍 Analyse des colonnes")
            
            # Vérifier les colonnes importantes
            colonnes_importantes = [
                "Date de souscription", 
                "Montant du placement", 
                "Montant des frais", 
                "Type d'investissement",
                "Conseiller",
                "Conseiller"
            ]
            
            colonnes_presentes = [col for col in colonnes_importantes if col in df.columns]
            colonnes_manquantes = [col for col in colonnes_importantes if col not in df.columns]
            
            if colonnes_manquantes:
                st.warning(f"⚠️ Colonnes importantes manquantes: {', '.join(colonnes_manquantes)}")
            
            # Traitement des colonnes présentes
            if colonnes_presentes:
                st.success(f"✅ Colonnes importantes présentes: {', '.join(colonnes_presentes)}")
                
                # Traitement des dates
                if "Date de souscription" in df.columns:
                    st.subheader("📅 Traitement des dates")
                    st.write("Conversion de la colonne de date Date de souscription avec safe_to_datetime_improved...")
                    
                    # Afficher quelques exemples de dates avant conversion
                    st.write("Exemples de dates avant conversion:")
                    st.write(df["Date de souscription"].head(5))
                    
                    # Convertir les dates
                    dates_converties = safe_to_datetime_improved(df["Date de souscription"])
                    
                    # Afficher quelques exemples après conversion
                    st.write("Exemples de dates après conversion:")
                    st.write(dates_converties.head(5))
                    
                    # Mettre à jour le DataFrame
                    df["Date de souscription"] = dates_converties
                
                # Traitement des montants
                if "Montant du placement" in df.columns:
                    st.subheader("💰 Traitement des montants")
                    st.write("Conversion de la colonne de montant Montant du placement avec safe_to_numeric_improved...")
                    
                    # Afficher quelques exemples de montants avant conversion
                    st.write("Exemples de montants avant conversion:")
                    st.write(df["Montant du placement"].head(5))
                    
                    # Convertir les montants
                    montants_convertis = safe_to_numeric_improved(df["Montant du placement"])
                    
                    # Afficher quelques exemples après conversion
                    st.write("Exemples de montants après conversion:")
                    st.write(montants_convertis.head(5))
                    
                    # Mettre à jour le DataFrame
                    df["Montant du placement"] = montants_convertis
                
                # Traitement des frais
                if "Montant des frais" in df.columns:
                    st.write("Conversion de la colonne de frais Montant des frais avec safe_to_numeric_improved...")
                    df["Montant des frais"] = safe_to_numeric_improved(df["Montant des frais"])
                
                # Extraction des conseillers
                st.subheader("👤 Extraction des conseillers")
                df = extract_conseiller_improved(df)
                
                # Afficher les conseillers extraits
                st.write("Conseillers extraits:")
                st.write(df["Conseiller"].value_counts())
                
                # Assurer la compatibilité Arrow
                st.subheader("🔄 Compatibilité Arrow")
                st.write("Conversion des types pour assurer la compatibilité avec Arrow...")
                df = ensure_arrow_compatibility_improved(df)
                
                # Afficher le DataFrame final
                st.subheader("✅ Données corrigées")
                st.dataframe(df.head(10))
                
                # Téléchargement du fichier corrigé
                st.subheader("💾 Télécharger les données corrigées")
                
                # Convertir en CSV ou Excel selon le format d'origine
                if uploaded_file.name.endswith('.csv'):
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger CSV corrigé",
                        data=csv,
                        file_name=f"corrige_{uploaded_file.name}",
                        mime="text/csv"
                    )
                else:
                    # Utiliser BytesIO pour créer un fichier Excel en mémoire
                    from io import BytesIO
                    
                    # Créer un buffer en mémoire
                    output = BytesIO()
                    
                    # Écrire le DataFrame dans le buffer
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Données')
                        
                        # Ajuster automatiquement la largeur des colonnes
                        worksheet = writer.sheets['Données']
                        for i, col in enumerate(df.columns):
                            # Trouver la largeur maximale
                            max_len = max(
                                df[col].astype(str).map(len).max(),
                                len(str(col))
                            ) + 2  # Ajouter un peu d'espace
                            worksheet.set_column(i, i, max_len)
                    
                    # Récupérer les données du buffer
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📥 Télécharger Excel corrigé",
                        data=excel_data,
                        file_name=f"corrige_{uploaded_file.name}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.success("✅ Traitement terminé! Vous pouvez télécharger le fichier corrigé et l'utiliser dans le dashboard principal.")
        
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
