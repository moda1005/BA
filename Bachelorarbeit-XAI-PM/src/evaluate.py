"""
Modellbewertung
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, logger):
    """Bewertet ein Modell auf den Trainings- und Testdaten"""
    
    # Vorhersagen für Trainingsdaten
    y_train_pred = model.predict(X_train)
    train_results = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'confusion_matrix': confusion_matrix(y_train, y_train_pred)
    }
    
    # Vorhersagen für Testdaten
    y_test_pred = model.predict(X_test)
    test_results = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred)
    }
    
    logger.info(f"[Modellbewertung] {model_name} Testgenauigkeit: {test_results['accuracy']:.3f}")
    
    return {'train': train_results, 'test': test_results}


def create_comparison(rf_results, dt_results, logger):
    """Erstellt eine Vergleichstabelle zwischen Random Forest und Entscheidungsbaum"""
    
    comparison = pd.DataFrame({
        'Metrik': ['Genauigkeit', 'Präzision', 'Recall', 'F1-Score'],
        'Random Forest': [
            rf_results['test']['accuracy'],
            rf_results['test']['precision'],
            rf_results['test']['recall'],
            rf_results['test']['f1']
        ],
        'Entscheidungsbaum': [
            dt_results['test']['accuracy'],
            dt_results['test']['precision'],
            dt_results['test']['recall'],
            dt_results['test']['f1']
        ]
    })
    
    logger.info(f"\n[Modellvergleich]\n{comparison.to_string(index=False)}")
    
    return comparison
