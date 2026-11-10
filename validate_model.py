"""Educational WDBC model selection with an untouched test partition."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path('breast+cancer+wisconsin+diagnostic/wdbc.data')


def load_wdbc(path=DATA):
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] != 32:
        raise ValueError('WDBC input must contain ID, diagnosis and 30 features')
    labels = raw.iloc[:, 1].map({'M': 1, 'B': 0})
    if labels.isna().any() or raw.iloc[:, 0].duplicated().any():
        raise ValueError('Invalid diagnosis or duplicate patient ID')
    return raw.iloc[:, 2:].astype(float), labels.astype(int)


def split_data(X, y):
    x_dev, x_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def run(output_dir=Path('reports')):
    X, y = load_wdbc()
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = split_data(X, y)
    candidates = {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
    }
    validation = {}
    for name, model in candidates.items():
        model.fit(train_x, train_y)
        validation[name] = float(f1_score(val_y, model.predict(val_x)))
    selected = max(validation, key=validation.get)
    model = candidates[selected]
    pred = model.predict(test_x)
    score = model.predict_proba(test_x)[:, 1]
    result = {
        'selected_on_validation_f1': selected,
        'validation_f1': validation,
        'test_rows': len(test_y),
        'test': {
            'accuracy': float(accuracy_score(test_y, pred)),
            'precision_malignant': float(precision_score(test_y, pred)),
            'recall_malignant': float(recall_score(test_y, pred)),
            'f1_malignant': float(f1_score(test_y, pred)),
            'roc_auc_malignant': float(roc_auc_score(test_y, score)),
        },
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'validation_results.json').write_text(json.dumps(result, indent=2))
    joblib.dump(model, out / 'selected_model.joblib')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
