"""
Fonctions d'exportation pour le dashboard de souscriptions.
"""
import streamlit as st
import pandas as pd
import io
import base64
import tempfile
import os
from PIL import Image
import uuid
from datetime import datetime


def export_dataframe(df, filename):
    """Permet de convertir un DataFrame en CSV et retourne les données.
    
    Args:
        df: DataFrame à convertir
        filename: Nom du fichier (sans extension)
        
    Returns:
        Données CSV au format BytesIO
    """
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding='utf-8')
    buffer.seek(0)
    return buffer


import uuid

def create_download_button(df, filename, key_prefix):
    """Crée un bouton de téléchargement avec une clé unique.
    
    Args:
        df: DataFrame à télécharger
        filename: Nom du fichier (sans extension)
        key_prefix: Préfixe pour la clé unique
        
    Returns:
        Le bouton de téléchargement Streamlit
    """
    # Créer une copie du DataFrame pour éviter de modifier l'original
    df_safe = df.copy()
    
    # Convertir toutes les colonnes numériques en float pour éviter les problèmes de conversion
    for col in df_safe.select_dtypes(include=['number']).columns:
        df_safe[col] = df_safe[col].astype(float)
    
    # Convertir toutes les colonnes object en string
    for col in df_safe.select_dtypes(include=['object']).columns:
        df_safe[col] = df_safe[col].astype(str)
    
    # Générer un CSV à partir du DataFrame sécurisé
    csv_data = export_dataframe(df_safe, filename)
    
    # Générer un identifiant unique pour chaque bouton
    unique_id = str(uuid.uuid4())[:8]
    
    # Créer un lien de téléchargement HTML au lieu d'utiliser st.download_button
    # pour éviter les problèmes de conversion Arrow
    csv = df_safe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv">📥 Télécharger ce tableau (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)


@st.cache_data
def export_to_pdf(title, fig_list=None, df_list=None, df_titles=None):
    """Génère un PDF contenant des graphiques et des tableaux de données.
    
    Args:
        title (str): Titre du PDF
        fig_list (list, optional): Liste des figures Plotly à inclure
        df_list (list, optional): Liste des DataFrames à inclure
        df_titles (list, optional): Liste des titres pour les DataFrames
    
    Returns:
        bytes: Contenu du PDF en bytes pour le téléchargement
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from io import BytesIO
    except ImportError:
        st.error("❌ Les bibliothèques nécessaires pour générer des PDF ne sont pas installées.")
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    # Titre
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=12
    )
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # Ajouter les graphiques
    if fig_list:
        for i, fig in enumerate(fig_list):
            img_bytes = fig.to_image(format="png")
            img_file = BytesIO(img_bytes)
            img = RLImage(img_file, width=6*inch, height=4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.25*inch))
    
    # Ajouter les tableaux
    if df_list:
        for i, df in enumerate(df_list):
            # Ajouter un titre pour le tableau si disponible
            if df_titles and i < len(df_titles):
                elements.append(Paragraph(df_titles[i], styles['Heading2']))
                elements.append(Spacer(1, 0.1*inch))
            
            # Convertir le DataFrame en tableau ReportLab
            data = [df.columns.tolist()] + df.values.tolist()
            table = Table(data)
            
            # Styliser le tableau
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 0.25*inch))
    
    # Générer le PDF
    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data


def create_pdf_download_link(pdf_bytes, filename):
    """Crée un bouton de téléchargement pour un PDF."""
    return st.download_button(
        label="📥 Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf"
    )
