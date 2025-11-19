"""
Modelltraining
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def train_random_forest(X_train, y_train, config, logger):
    """Trainiert ein Random-Forest-Modell"""
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Kreuzvalidierung
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    logger.info(f"[Modelltraining] Random Forest Kreuzvalidierung: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Endgültiges Modell trainieren
    model.fit(X_train, y_train)
    
    return model, {'mean': cv_scores.mean(), 'std': cv_scores.std()}


def train_decision_tree(X_train, y_train, config, logger):
    """Trainiert ein Entscheidungsbaum-Modell"""
    
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=0.1,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    )
    
    # Kreuzvalidierung
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    logger.info(f"[Modelltraining] Entscheidungsbaum Kreuzvalidierung: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Endgültiges Modell trainieren
    model.fit(X_train, y_train)
    
    return model, {'mean': cv_scores.mean(), 'std': cv_scores.std()}
