# src/model.py

import pickle
from typing import List, Dict, Optional, Any
import sklearn_crfsuite
from sklearn_crfsuite import metrics

def build_crf_model(
    algorithm: str = 'lbfgs',
    c1: float = 0.1,
    c2: float = 0.1,
    max_iterations: int = 100,
    all_possible_transitions: bool = True
) -> sklearn_crfsuite.CRF:
    return sklearn_crfsuite.CRF(
        algorithm=algorithm,
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=all_possible_transitions
    )

def train_crf(
    crf: sklearn_crfsuite.CRF,
    X_train: List[List[Dict[str, Any]]],
    y_train: List[List[str]],
    save_path: Optional[str] = None
) -> sklearn_crfsuite.CRF:
    """
    CRF modelini yalnızca eğitim verisiyle eğitir.
    """
    crf.fit(X_train, y_train)

    if save_path:
        with open(save_path, 'wb') as fout:
            pickle.dump(crf, fout)
    return crf

def load_crf_model(model_path: str) -> sklearn_crfsuite.CRF:
    with open(model_path, 'rb') as fin:
        return pickle.load(fin)

def predict_tags(
    crf: sklearn_crfsuite.CRF,
    X: List[List[Dict[str, Any]]]
) -> List[List[str]]:
    return crf.predict(X)

def evaluate_model(
    crf: sklearn_crfsuite.CRF,
    X_test: List[List[Dict[str, Any]]],
    y_test: List[List[str]],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    y_pred = predict_tags(crf, X_test)
    if labels is None:
        labels = list(crf.classes_)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    report = metrics.flat_classification_report(
        y_test, y_pred,
        labels=sorted_labels,
        digits=4,
        output_dict=True
    )
    return report
