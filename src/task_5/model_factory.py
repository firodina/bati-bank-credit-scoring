from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_models(random_state=42):
    return {
        "LogisticRegression": {
            "model": LogisticRegression(solver="liblinear", random_state=random_state),
            "search": "grid",
            "params": {
                "C": [0.1, 1, 10],
                "penalty": ["l1", "l2"]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=random_state),
            "search": "grid",
            "params": {
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5, 10]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=random_state),
            "search": "random",
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, None]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "search": "random",
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        }
    }
