"""
Hilfsfunktionen
"""

import os
import yaml
import logging
import json
import pickle
from pathlib import Path
from datetime import datetime


def setup_logging(config):
    """Richtet das Logging-System ein"""
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"lauf_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path="config/settings.yaml"):
    """Lädt die YAML-Konfigurationsdatei"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_model(model, name, config):
    """Speichert ein Modell als Pickle-Datei"""
    path = Path(config['paths']['models']) / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return str(path)


def save_results(results, name, config):
    """Speichert Ergebnisse im JSON-Format"""
    path = Path(config['paths']['results']) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return str(path)
