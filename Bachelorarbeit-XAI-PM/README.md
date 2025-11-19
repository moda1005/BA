# XAI für Predictive Maintenance

Dieses Projekt wurde im Rahmen der Bachelorarbeit *„XAI für Predictive Maintenance“* an der Hochschule Kaiserslautern im Studiengang Elektrotechnik mit Schwerpunkt Automatisierungs- und Informationstechnik entwickelt.  
Ziel des Projekts ist die Entwicklung, Implementierung und Evaluation von Explainable-AI-Methoden zur transparenten Analyse von Machine-Learning-Modellen im Bereich der vorausschauenden Instandhaltung (Predictive Maintenance).

---

## 1. Projektziel

Das Projekt untersucht den Einsatz von **Shapley Values (SHAP)** zur Erklärbarkeit von Klassifikationsmodellen, die den Zustand technischer Systeme bewerten und mögliche Ausfälle vorhersagen.  
Hierzu werden insbesondere **Entscheidungsbäume (Decision Tree)** und **Random-Forest-Modelle** auf einem industriellen Datensatz trainiert und ihre Vorhersagen mittels SHAP nachvollziehbar gemacht.  

Die Arbeit umfasst die folgenden Hauptaufgaben:
1. Datenaufbereitung und Feature-Skalierung  
2. Training und Vergleich der Modelle (Decision Tree und Random Forest)  
3. Modellbewertung anhand gängiger Metriken  
4. Analyse der Modellentscheidungen mit dem SHAP-Framework  
5. Visualisierung der Ergebnisse und Protokollierung im Log-System  

---

## 2. Projektstruktur

```plaintext
Bachelorarbeit-XAI-PM-better
├─ artifacts
│  ├─ models                # Gespeicherte Modelle (.pkl)
│  ├─ plots                 # Generierte Diagramme (Modelle, Vergleiche)
│  └─ results               # Ergebnisse (z. B. final_results.json)
├─ config
│  └─ settings.yaml         # Konfigurationsdatei
├─ data
│  ├─ processed             # (Optional) Vorverarbeitete Daten
│  └─ raw                   # Rohdaten (automotive_data_train_140.csv)
├─ logs                     # Log-Dateien aller Programmläufe
├─ notebooks
│  ├─ complete_analysis.ipynb   # Explorative Datenanalyse und Modellbewertung
│  └─ interactive_dashboard.ipynb (optional)
├─ src
│  ├─ data_prep.py          # Datenaufbereitung
│  ├─ train_model.py        # Modelltraining
│  ├─ evaluate.py           # Evaluation und Vergleich
│  ├─ shap_analysis.py      # SHAP-Analyse
│  ├─ visualizations.py     # Diagrammerstellung
│  ├─ utils.py              # Hilfsfunktionen (Logging, Speicherung)
│  └─ __init__.py
├─ main.py                  # Hauptskript zur Ausführung der Pipeline
├─ requirements.txt         # Python-Abhängigkeiten
└─ README.md                # Projektdokumentation
````

---

## 3. Installation und Setup

### Voraussetzungen

* **Python 3.10 oder höher**
* Empfohlen: Virtuelle Umgebung (z. B. `venv` oder `conda`)

### Installation

```bash
cd Bachelorarbeit-XAI-PM-better
python -m venv .venv
source .venv/bin/activate     # unter Linux/Mac
.venv\Scripts\activate        # unter Windows
pip install -r requirements.txt
```

---

## 4. Konfiguration

Die Konfiguration erfolgt über die Datei [`config/settings.yaml`](config/settings.yaml).
Sie definiert Pfade, Dateinamen, Modellparameter und SHAP-Einstellungen:

```yaml
paths:
  data:
    raw: "data/raw"
  models: "artifacts/models"
  results: "artifacts/results"
  shap_plots: "artifacts/shap_plots"
  logs: "logs"

data:
  filename: "automotive_data_train_140.csv"

model:
  type: "random_forest"
  random_state: 42

shap:
  sample_size: 100
  background_samples: 50
  dpi: 300
  plot_format: "png"
```

---

## 5. Nutzung

Das gesamte Projekt wird über das Hauptskript `main.py` ausgeführt.
Standardmäßig wird die Konfiguration aus `config/settings.yaml` geladen.

### Ausführung

```bash
python main.py --config config/settings.yaml
```

### Ablauf der Pipeline

1. **Datenaufbereitung:** Laden, Kodierung und Skalierung der Eingabedaten
2. **Training:** Aufbau und Cross-Validation von Decision Tree und Random Forest
3. **Evaluation:** Berechnung von Accuracy, Precision, Recall und F1-Score
4. **Vergleich:** Gegenüberstellung der beiden Modelle
5. **SHAP-Analyse:** Berechnung der Shapley Values für den Random Forest
6. **Visualisierung:** Speicherung von Baumdiagrammen, Vergleichsgrafiken und SHAP-Plots
7. **Speicherung:** Modelle und Ergebnisse werden in `/artifacts` abgelegt
8. **Logging:** Alle Ausgaben werden in `/logs` protokolliert

---

## 6. Ergebnisse und Artefakte

| Ordner / Datei          | Inhalt                                             |
| ----------------------- | -------------------------------------------------- |
| `artifacts/models/`     | Trainierte Modelle (`.pkl`)                        |
| `artifacts/results/`    | JSON-Ergebnisse (Metriken, Vergleiche, SHAP-Daten) |
| `artifacts/plots/`      | Diagramme (Modellvergleiche, Entscheidungsbäume)   |
| `artifacts/shap_plots/` | SHAP-Summary- und Bar-Plots                        |
| `logs/`                 | Laufzeit- und Ergebnisprotokolle                   |

Beispielhafte SHAP-Grafiken:

* **`shap_summary.png`**: Einfluss der Features auf einzelne Vorhersagen
* **`shap_bar.png`**: Durchschnittliche Feature-Wichtigkeit über alle Testdaten

---

## 7. Notebooks

Das Notebook [`notebooks/complete_analysis.ipynb`](notebooks/complete_analysis.ipynb) enthält eine ausführliche Analyse der Daten, Modellbewertung und beispielhafte Visualisierungen.
Es dient der interaktiven Untersuchung und Validierung der im Python-Pipeline erzeugten Ergebnisse.

---

## 8. Verwendete Technologien

* **Programmiersprache:** Python 3.10
* **Machine Learning:** scikit-learn
* **Explainable AI:** SHAP
* **Visualisierung:** Matplotlib, Seaborn
* **Protokollierung:** Python `logging`
* **Konfiguration:** YAML

---

© 2025 Mohamed Darguech – Hochschule Kaiserslautern
