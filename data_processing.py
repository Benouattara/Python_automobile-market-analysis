"""
Module de traitement de données pour l'analyse du marché automobile.

Ce module contient des fonctions utilitaires pour le nettoyage,
la transformation et l'analyse des données automobiles.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime


def load_vehicle_data(filepath: str, file_format: str = 'csv') -> pd.DataFrame:
    """
    Charge les données de véhicules depuis un fichier.
    
    Args:
        filepath: Chemin vers le fichier
        file_format: Format du fichier ('csv' ou 'parquet')
    
    Returns:
        DataFrame contenant les données
    """
    if file_format == 'csv':
        df = pd.read_csv(filepath)
    elif file_format == 'parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Format non supporté: {file_format}")
    
    # Conversion de la date si présente
    if 'date_immatriculation' in df.columns:
        df['date_immatriculation'] = pd.to_datetime(df['date_immatriculation'])
    
    return df


def calculate_depreciation(df: pd.DataFrame, current_year: int = 2024) -> pd.DataFrame:
    """
    Calcule la dépréciation des véhicules.
    
    Args:
        df: DataFrame des véhicules
        current_year: Année de référence
    
    Returns:
        DataFrame avec colonnes de dépréciation ajoutées
    """
    df = df.copy()
    df['age_vehicule'] = current_year - df['annee']
    df['depreciation_annuelle'] = df['prix_euro'] / (df['age_vehicule'] + 1)
    
    return df


def create_price_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des segments de prix.
    
    Args:
        df: DataFrame des véhicules
    
    Returns:
        DataFrame avec colonne 'price_segment' ajoutée
    """
    df = df.copy()
    
    bins = [0, 15000, 25000, 40000, float('inf')]
    labels = ['Budget', 'Économique', 'Moyen', 'Premium']
    
    df['price_segment'] = pd.cut(df['prix_euro'], bins=bins, labels=labels)
    
    return df


def calculate_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule des métriques d'efficience.
    
    Args:
        df: DataFrame des véhicules
    
    Returns:
        DataFrame avec métriques d'efficience ajoutées
    """
    df = df.copy()
    
    # Efficience énergétique
    df['efficience'] = df['puissance_cv'] / (df['consommation'] + 0.1)
    
    # Prix par CV
    df['prix_par_cv'] = df['prix_euro'] / df['puissance_cv']
    
    # Kilométrage annuel moyen
    df['km_par_an'] = df['kilometrage'] / (2024 - df['annee'] + 1)
    
    # Score environnemental (inverse des émissions CO2)
    df['score_eco'] = 100 - (df['co2_g_km'] / df['co2_g_km'].max() * 100)
    
    return df


def identify_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """
    Identifie les valeurs aberrantes dans une colonne.
    
    Args:
        df: DataFrame
        column: Nom de la colonne à analyser
        method: Méthode de détection ('iqr' ou 'zscore')
    
    Returns:
        Series booléenne indiquant les outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = pd.Series(False, index=df.index)
        outliers.loc[z_scores.index] = z_scores > 3
    
    else:
        raise ValueError(f"Méthode non supportée: {method}")
    
    return outliers


def get_market_share(df: pd.DataFrame, 
                     group_by: str, 
                     year: int = None) -> pd.DataFrame:
    """
    Calcule les parts de marché.
    
    Args:
        df: DataFrame des véhicules
        group_by: Colonne pour le regroupement (ex: 'marque', 'carburant')
        year: Année spécifique (optionnel)
    
    Returns:
        DataFrame avec parts de marché
    """
    if year:
        df_filtered = df[df['annee'] == year].copy()
    else:
        df_filtered = df.copy()
    
    market_share = df_filtered.groupby(group_by).size().reset_index(name='count')
    market_share['part_marche_%'] = (market_share['count'] / market_share['count'].sum() * 100).round(2)
    market_share = market_share.sort_values('part_marche_%', ascending=False)
    
    return market_share


def calculate_emissions_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'évolution des émissions CO2 par année.
    
    Args:
        df: DataFrame des véhicules
    
    Returns:
        DataFrame avec tendance des émissions
    """
    trend = df.groupby('annee').agg({
        'co2_g_km': ['mean', 'median', 'std'],
        'id': 'count'
    }).round(2)
    
    trend.columns = ['_'.join(col).strip() for col in trend.columns.values]
    trend = trend.reset_index()
    
    # Calcul de la variation annuelle
    trend['variation_%'] = trend['co2_g_km_mean'].pct_change() * 100
    
    return trend


def prepare_ml_features(df: pd.DataFrame, 
                        target: str = 'prix_euro',
                        categorical_encoding: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prépare les features pour le machine learning.
    
    Args:
        df: DataFrame des véhicules
        target: Variable cible
        categorical_encoding: Type d'encodage ('label' ou 'onehot')
    
    Returns:
        Tuple (X, y) avec features et target
    """
    from sklearn.preprocessing import LabelEncoder
    
    df = df.copy()
    
    # Features numériques de base
    numeric_features = [
        'annee', 'puissance_cv', 'consommation', 'co2_g_km',
        'kilometrage', 'portes', 'nombre_proprietaires'
    ]
    
    # Features catégorielles
    categorical_features = ['marque', 'carburant', 'categorie', 'transmission', 'pays']
    
    # Encodage
    if categorical_encoding == 'label':
        for col in categorical_features:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            numeric_features.append(f'{col}_encoded')
    
    elif categorical_encoding == 'onehot':
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        numeric_features = [col for col in df.columns 
                          if col not in [target, 'id', 'modele', 'date_immatriculation', 'couleur']]
    
    # Séparation X et y
    X = df[numeric_features]
    y = df[target]
    
    return X, y


def generate_summary_report(df: pd.DataFrame) -> Dict:
    """
    Génère un rapport résumé du dataset.
    
    Args:
        df: DataFrame des véhicules
    
    Returns:
        Dictionnaire avec statistiques clés
    """
    report = {
        'nombre_total_vehicules': len(df),
        'periode': f"{df['annee'].min()} - {df['annee'].max()}",
        'nombre_marques': df['marque'].nunique(),
        'nombre_modeles': df['modele'].nunique(),
        'prix_moyen': f"{df['prix_euro'].mean():,.0f}€",
        'prix_median': f"{df['prix_euro'].median():,.0f}€",
        'kilometrage_moyen': f"{df['kilometrage'].mean():,.0f} km",
        'emissions_co2_moyennes': f"{df['co2_g_km'].mean():.0f} g/km",
        'puissance_moyenne': f"{df['puissance_cv'].mean():.0f} CV",
        'repartition_carburant': df['carburant'].value_counts().to_dict(),
        'top_3_marques': df['marque'].value_counts().head(3).to_dict(),
        'taux_electrification': f"{(df['carburant'].isin(['Electrique', 'Hybride', 'Hybride rechargeable']).sum() / len(df) * 100):.1f}%"
    }
    
    return report


def export_analysis_results(df: pd.DataFrame, 
                           output_dir: str = '../data/processed/',
                           formats: List[str] = ['csv', 'excel']) -> None:
    """
    Exporte les résultats d'analyse dans différents formats.
    
    Args:
        df: DataFrame à exporter
        output_dir: Répertoire de sortie
        formats: Liste des formats d'export
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f'analysis_results_{timestamp}'
    
    for fmt in formats:
        if fmt == 'csv':
            filepath = os.path.join(output_dir, f'{base_filename}.csv')
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"✅ Exporté en CSV: {filepath}")
        
        elif fmt == 'excel':
            filepath = os.path.join(output_dir, f'{base_filename}.xlsx')
            df.to_excel(filepath, index=False, engine='openpyxl')
            print(f"✅ Exporté en Excel: {filepath}")
        
        elif fmt == 'parquet':
            filepath = os.path.join(output_dir, f'{base_filename}.parquet')
            df.to_parquet(filepath, index=False)
            print(f"✅ Exporté en Parquet: {filepath}")


# Fonctions utilitaires additionnelles

def validate_data_quality(df: pd.DataFrame) -> Dict:
    """Vérifie la qualité des données."""
    quality_report = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    return quality_report


def calculate_correlation_matrix(df: pd.DataFrame, 
                                 numerical_only: bool = True) -> pd.DataFrame:
    """Calcule la matrice de corrélation."""
    if numerical_only:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].corr()
    return df.corr()


if __name__ == "__main__":
    # Test du module
    print("Module data_processing chargé avec succès!")
    print("\nFonctions disponibles:")
    functions = [func for func in dir() if callable(globals()[func]) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")
