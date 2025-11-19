"""
XAI für Predictive Maintenance - Bachelorarbeit
"""

import argparse
from pathlib import Path
from datetime import datetime

from src.utils import load_config, setup_logging, save_model, save_results
from src.data_prep import load_and_prepare_data
from src.train_model import train_random_forest, train_decision_tree
from src.evaluate import evaluate_model, create_comparison
from src.shap_analysis import analyze_with_shap
from src.visualizations import create_all_visualizations


def main(args):
    # Konfiguration laden und Logging einrichten
    config = load_config(args.config)
    logger = setup_logging(config)
    
    logger.info("=" * 70)
    logger.info("XAI FÜR PREDICTIVE MAINTENANCE - BACHELORARBEIT")
    logger.info("=" * 70)
    
    # 1. Datenaufbereitung
    logger.info("\n[1/6] Starte Datenaufbereitung ...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare_data(config, logger)
    
    # 2. Training des Random Forest
    logger.info("\n[2/6] Trainiere Random Forest ...")
    rf_model, rf_cv = train_random_forest(X_train, y_train, config, logger)
    
    # 3. Training des Entscheidungsbaums
    logger.info("\n[3/6] Trainiere Entscheidungsbaum ...")
    dt_model, dt_cv = train_decision_tree(X_train, y_train, config, logger)
    
    # 4. Bewertung beider Modelle
    logger.info("\n[4/6] Bewerte Modelle ...")
    rf_results = evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest", logger)
    dt_results = evaluate_model(dt_model, X_train, y_train, X_test, y_test, "Entscheidungsbaum", logger)
    
    # 5. Modellvergleich
    logger.info("\n[5/6] Erstelle Modellvergleich ...")
    comparison = create_comparison(rf_results, dt_results, logger)
    
    # 6. SHAP-Analyse (nur für Random Forest)
    logger.info("\n[6/6] Führe SHAP-Analyse durch ...")
    shap_results = analyze_with_shap(rf_model, X_train, X_test, feature_names, config, logger)
    
    # 7. Visualisierungen erzeugen
    logger.info("\nErstelle Visualisierungen ...")
    create_all_visualizations(
        rf_model, dt_model, X_test, y_test, feature_names,
        shap_results, comparison, config, logger
    )
    
    # 8. Ergebnisse speichern
    logger.info("\nSpeichere Ergebnisse ...")
    save_model(rf_model, "random_forest_model", config)
    save_model(dt_model, "decision_tree_model", config)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'random_forest': rf_results,
        'decision_tree': dt_results,
        'comparison': comparison.to_dict('records'),
        'shap': shap_results
    }
    save_results(results, "final_results", config)
    
    # Zusammenfassung
    logger.info("\n" + "=" * 70)
    logger.info("AUSFÜHRUNG ABGESCHLOSSEN!")
    logger.info(f"Random Forest Genauigkeit: {rf_results['test']['accuracy']:.3f}")
    logger.info(f"Entscheidungsbaum Genauigkeit: {dt_results['test']['accuracy']:.3f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    main(args)
