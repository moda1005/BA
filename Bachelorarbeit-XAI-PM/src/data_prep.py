"""
Datenaufbereitung
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(config, logger):
    """Lädt die Rohdaten, bereitet sie vor und teilt sie in Trainings- und Testdaten auf"""
    
    # Daten laden
    data_path = Path(config['paths']['data']['raw']) / config['data']['filename']
    df = pd.read_csv(data_path, sep=';', encoding='utf-8')
    logger.info(f"Daten geladen: {df.shape}")
    
    # Zielvariable kodieren
    df['Ausfall'] = df['Ausfall'].map({'ja': 1, 'nein': 0})
    
    # Unnötige Spalten entfernen
    df = df.drop(columns=['Messungsnr', 'KatTemp'], errors='ignore')
    
    # Merkmale und Zielvariable trennen
    X = df.drop('Ausfall', axis=1)
    y = df['Ausfall']
    feature_names = list(X.columns)
    
    # Aufteilung in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Merkmals-Skalierung
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info(f"Trainingsdaten: {len(X_train)}, Testdaten: {len(X_test)}, Merkmale: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler
