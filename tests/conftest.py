"""
Configuration pytest pour le projet Dashboard Commercial
"""
import pytest
import sys
from pathlib import Path

# Ajouter le répertoire du projet au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_data():
    """Fixture pour des données de test basiques"""
    import pandas as pd
    import numpy as np
    
    return pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=10),
        'Montant': np.random.uniform(100, 1000, 10),
        'Type': ['A'] * 5 + ['B'] * 5
    })