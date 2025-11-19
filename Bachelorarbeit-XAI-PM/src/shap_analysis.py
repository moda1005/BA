"""
SHAP-Analyse
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path


def analyze_with_shap(model, X_train, X_test, feature_names, config, logger):
    """Führt eine SHAP-Analyse zur Erklärbarkeit der Modellvorhersagen durch"""
    
    # Erzeugt den SHAP-Erklärer
    explainer = shap.TreeExplainer(model)
    
    # Berechnet die SHAP-Werte
    shap_values = explainer.shap_values(X_test)
    
    # Behandlung binärer Klassifikation
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    
    logger.info(f"[SHAP-Analyse] SHAP-Werte berechnet: {shap_values.shape}")
    
    # Ausgabeverzeichnis erstellen
    output_dir = Path(config['paths']['shap_plots'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Zusammenfassungsdiagramm (Summary Plot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Balkendiagramm der mittleren Beitragshöhen
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    #waterfall-Diagramm
    try:
        sample_idx = 0  # Beispielinstanz
        
        shap_vec = shap_values[sample_idx]
        
        # Base Value für Klasse 1
        base_val = (
            explainer.expected_value[1]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )
        
        # Feature-Werte
        sample_data = X_test[sample_idx]

        # SHAP Explanation Objekt
        expl = shap.Explanation(
            values=shap_vec,
            base_values=base_val,
            data=sample_data,
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(expl, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("[SHAP-Analyse] Waterfall-Plot gespeichert (sample_idx=0).")

    except Exception as e:
        logger.error(f"[SHAP-Analyse] Fehler beim Erstellen des Waterfall-Plots: {e}")
    # Berechnung der mittleren Merkmalsbedeutung
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)
    
    logger.info("[SHAP-Analyse] Wichtigste 5 Merkmale:")
    for i, (feat, imp) in enumerate(importance[:5], 1):
        logger.info(f"  {i}. {feat}: {imp:.4f}")
    
    return {
        'shap_values': shap_values,
        'feature_importance': [{'merkmal': f, 'bedeutung': float(i)} for f, i in importance]
    }
