from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_models():
    return {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, solver="liblinear"),
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0]
            },
            "search": "grid"
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "model__max_depth": [3, 5, 7, None],
                "model__min_samples_split": [2, 5, 10]
            },
            "search": "grid"
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [5, 10, None]
            },
            "search": "grid"
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.01, 0.1],
                "model__max_depth": [3, 5]
            },
            "search": "grid"
        }
    }
