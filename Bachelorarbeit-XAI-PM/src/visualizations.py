"""
Erstellung aller Visualisierungen
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from pathlib import Path


def create_all_visualizations(rf_model, dt_model, X_test, y_test, feature_names, 
                             shap_results, comparison, config, logger):
    """Erstellt alle erforderlichen Visualisierungen der Modelle und Ergebnisse"""
    
    output_dir = Path('artifacts/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Balkendiagramm zum Modellvergleich
    metrics = comparison['Metrik'].values
    rf_vals = comparison['Random Forest'].values
    dt_vals = comparison['Entscheidungsbaum'].values
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, rf_vals, width, label='Random Forest')
    ax.bar(x + width / 2, dt_vals, width, label='Entscheidungsbaum')
    ax.set_xlabel('Bewertungsmetriken')
    ax.set_ylabel('Wert')
    ax.set_title('Modellvergleich: Random Forest vs. Entscheidungsbaum')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300)
    plt.close()
    
    # 2. Visualisierung eines einzelnen Baums des Random Forest
    plt.figure(figsize=(20, 10))
    plot_tree(
        rf_model.estimators_[0],
        feature_names=feature_names,
        class_names=['Kein Ausfall', 'Ausfall'],
        filled=True,
        rounded=True,
        max_depth=3
    )
    plt.title('Random Forest – Einzelbaum (Tiefe 3)')
    plt.tight_layout()
    plt.savefig(output_dir / 'random_forest_tree.png', dpi=300)
    plt.close()
    
    # 3. Visualisierung des Entscheidungsbaums
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=['Kein Ausfall', 'Ausfall'],
        filled=True,
        rounded=True,
        max_depth=3
    )
    plt.title('Entscheidungsbaum (Tiefe 3)')
    plt.tight_layout()
    plt.savefig(output_dir / 'decision_tree.png', dpi=300)
    plt.close()
    
    logger.info(f"[Visualisierung] Diagramme wurden in {output_dir} gespeichert.")
