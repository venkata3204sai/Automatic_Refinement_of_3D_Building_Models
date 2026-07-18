"""
Diagnostic: what's causing hipped misclassification?

Trains a Random Forest on four combinations — 9 tiles vs. 30 tiles, with vs.
without neighbor features — and compares confusion matrices, to separate two
possible causes: neighbor features hurting (A vs B, C vs D) from more tiles
hurting (A vs C, B vs D).

Usage:
    Run from the repository root (so the relative ``data/`` paths resolve):
        python scripts/diagnostic.py
"""

import sys
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')
# This script lives in scripts/; add the repository root (its parent) to the
# import path so the pipeline modules can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cityjson_parser import CityJSONParser
from feature_extraction import process_tile, filter_buildings, compute_neighbor_features
from ml_models_revised import ALL_FEATURES

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def load_and_filter(filepaths):
    """Load tiles, extract features, filter to 4-vertex buildings."""
    all_data = []
    for fp in filepaths:
        parser = CityJSONParser(fp)
        data = process_tile(parser)
        for d in data:
            d['_parser'] = parser
        all_data.extend(data)
    filtered = filter_buildings(all_data)
    return filtered


def train_and_evaluate(building_data, feature_list, test_name):
    """Train RF classifier and print confusion matrix."""
    # Build feature matrix
    X_rows = []
    y_labels = []
    for bd in building_data:
        features = bd['features']
        row = []
        valid = True
        for key in feature_list:
            val = features.get(key)
            if val is None:
                valid = False
                break
            row.append(float(val))
        if valid:
            X_rows.append(row)
            y_labels.append(bd['roof_type'])

    X = np.array(X_rows)
    y = np.array(y_labels)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, stratify=y_enc, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    print(f"  Dataset: {len(X)} buildings")
    print(f"  Classes: { {c: int(np.sum(y == c)) for c in np.unique(y)} }")
    print(f"  Features: {len(feature_list)}")
    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)}")
    print(f"  Confusion Matrix:")
    print(f"  Predicted →  {' '.join(f'{c:>8}' for c in le.classes_)}")
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:<10}  {' '.join(f'{v:>8}' for v in row)}")

    # Hipped-specific metrics
    hipped_idx = list(le.classes_).index('hipped') if 'hipped' in le.classes_ else None
    if hipped_idx is not None:
        hipped_total = sum(cm[hipped_idx])
        hipped_correct = cm[hipped_idx][hipped_idx]
        hipped_as_gabled = cm[hipped_idx][list(le.classes_).index('gabled')]
        print(f"\n  HIPPED DETAIL:")
        print(f"    Total hipped in test: {hipped_total}")
        print(f"    Correctly classified: {hipped_correct} ({hipped_correct/hipped_total*100:.1f}%)")
        print(f"    Misclassified as gabled: {hipped_as_gabled} ({hipped_as_gabled/hipped_total*100:.1f}%)")


FEATURES_NO_NEIGHBOR = [f for f in ALL_FEATURES if not f.startswith('neighbor_') and f != 'n_neighbors']
FEATURES_WITH_NEIGHBOR = ALL_FEATURES

ORIGINAL_9 = [
    "data/9-564-628.city.json", "data/9-564-632.city.json", "data/9-564-636.city.json",
    "data/9-568-628.city.json", "data/9-568-632.city.json", "data/9-568-636.city.json",
    "data/9-572-628.city.json", "data/9-572-632.city.json", "data/9-572-636.city.json",
]

ALL_30 = ORIGINAL_9 + [
    "data/9-560-624.city.json", "data/9-560-632.city.json", "data/9-560-636.city.json",
    "data/9-560-644.city.json", "data/9-564-640.city.json", "data/9-564-644.city.json",
    "data/9-568-624.city.json", "data/9-568-640.city.json", "data/9-568-644.city.json",
    "data/9-572-624.city.json", "data/9-572-644.city.json",
    "data/9-576-632.city.json", "data/9-576-636.city.json", "data/9-576-644.city.json",
    "data/9-580-632.city.json", "data/9-580-640.city.json", "data/9-580-644.city.json",
    "data/10-560-640.city.json", "data/10-562-642.city.json",
    "data/10-564-624.city.json", "data/10-564-626.city.json",
    "data/10-566-624.city.json", "data/10-574-642.city.json",
    "data/8-576-624.city.json",
]


if __name__ == "__main__":
    # Test A: 9 tiles, no neighbor features
    print("\nLoading original 9 tiles...")
    data_9 = load_and_filter(ORIGINAL_9)
    train_and_evaluate(data_9, FEATURES_NO_NEIGHBOR, "TEST A: 9 tiles, NO neighbor features")

    # Test B: 9 tiles, with neighbor features
    print("\nAdding neighbor features to 9-tile data...")
    data_9_neighbor = compute_neighbor_features(data_9, radius=50.0, use_ground_truth=True)
    train_and_evaluate(data_9_neighbor, FEATURES_WITH_NEIGHBOR, "TEST B: 9 tiles, WITH neighbor features")

    # Test C: 30 tiles, no neighbor features
    print("\nLoading all 30 tiles...")
    # Check which files exist
    existing_30 = [f for f in ALL_30 if os.path.exists(f)]
    print(f"  Found {len(existing_30)} of {len(ALL_30)} tiles")
    data_30 = load_and_filter(existing_30)
    train_and_evaluate(data_30, FEATURES_NO_NEIGHBOR, "TEST C: 30 tiles, NO neighbor features")

    # Test D: 30 tiles, with neighbor features
    print("\nAdding neighbor features to 30-tile data...")
    data_30_neighbor = compute_neighbor_features(data_30, radius=50.0, use_ground_truth=True)
    train_and_evaluate(data_30_neighbor, FEATURES_WITH_NEIGHBOR, "TEST D: 30 tiles, WITH neighbor features")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Compare hipped recall across tests")
    print(f"{'='*60}")
    print(f"  Test A (9 tiles, no neighbors):  → check hipped recall above")
    print(f"  Test B (9 tiles, + neighbors):   → check hipped recall above")
    print(f"  Test C (30 tiles, no neighbors): → check hipped recall above")
    print(f"  Test D (30 tiles, + neighbors):  → check hipped recall above")
    print(f"\n  If A→B worsens hipped: neighbor features hurt")
    print(f"  If A→C worsens hipped: more tiles hurt")
    print(f"  If both: combined effect")