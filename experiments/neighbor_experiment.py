"""
Neighbor Strategy Experiment

Sweeps different neighbor-selection strategies to see which one helps
hipped-roof classification most: a no-neighbors baseline, then radius
(30m/50m), weighting (equal vs. distance-weighted), and max-K variants.
All configs compute the same 6 neighbor features (n_neighbors,
neighbor_mean_height, neighbor_mean_area, neighbor_frac_flat/gabled/hipped)
— only how neighbors are selected and weighted changes between them.

Usage:
    Run from the repository root (so the relative ``data/`` paths resolve):
        python experiments/neighbor_experiment.py
"""

import sys
import os
import math
import warnings
import numpy as np

warnings.filterwarnings('ignore')
# This script lives in experiments/; add the repository root (its parent) to
# the import path so the pipeline modules can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cityjson_parser import CityJSONParser
from feature_extraction import process_tile, filter_buildings
from ml_models_revised import ALL_FEATURES

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# --- configurable neighbor-feature computation ---

def compute_neighbor_features_configurable(
    building_data,
    radius=50.0,
    max_k=None,
    distance_weighted=False,
    use_ground_truth=True,
    silent=False
):
    """
    Compute neighbor features with configurable strategy.

    Args:
        building_data: list of building data dicts
        radius: search radius in meters
        max_k: if set, keep only the K nearest neighbors within radius
        distance_weighted: if True, weight neighbor contributions by 1/distance
        use_ground_truth: True for training (use roof_type), False for inference
        silent: suppress print output

    Returns:
        Same list with neighbor features added to each building's feature dict.
    """
    n = len(building_data)
    if n == 0:
        return building_data

    # Extract centroids and roof types
    centroids = []
    roof_types = []
    for bd in building_data:
        f = bd['features']
        cx = f.get('centroid_x', 0)
        cy = f.get('centroid_y', 0)
        centroids.append((cx, cy))

        if use_ground_truth:
            roof_types.append(bd.get('roof_type', 'unknown'))
        else:
            roof_types.append(bd.get('predicted_roof_type', 'unknown'))

    neighbor_counts = 0

    for i in range(n):
        cx_i, cy_i = centroids[i]

        # Find all neighbors within radius, with their distances
        neighbor_dists = []
        for j in range(n):
            if i == j:
                continue
            cx_j, cy_j = centroids[j]
            dist = math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            if dist <= radius and dist > 0.1:  # skip overlapping buildings
                neighbor_dists.append((j, dist))

        # Sort by distance (nearest first)
        neighbor_dists.sort(key=lambda x: x[1])

        # Apply max_k cap if specified
        if max_k is not None and len(neighbor_dists) > max_k:
            neighbor_dists = neighbor_dists[:max_k]

        n_neighbors = len(neighbor_dists)
        neighbor_counts += n_neighbors

        if n_neighbors > 0:
            # Compute weights
            if distance_weighted:
                weights = [1.0 / d for _, d in neighbor_dists]
            else:
                weights = [1.0] * n_neighbors

            total_weight = sum(weights)

            # Weighted geometric features
            neighbor_heights = [building_data[j]['features'].get('building_height', 0)
                                for j, _ in neighbor_dists]
            neighbor_areas = [building_data[j]['features'].get('footprint_area', 0)
                              for j, _ in neighbor_dists]

            w_mean_height = sum(h * w for h, w in zip(neighbor_heights, weights)) / total_weight
            w_mean_area = sum(a * w for a, w in zip(neighbor_areas, weights)) / total_weight

            # Weighted roof type fractions
            neighbor_rt = [roof_types[j] for j, _ in neighbor_dists]
            frac_flat = sum(w for rt, w in zip(neighbor_rt, weights) if rt == 'flat') / total_weight
            frac_gabled = sum(w for rt, w in zip(neighbor_rt, weights) if rt == 'gabled') / total_weight
            frac_hipped = sum(w for rt, w in zip(neighbor_rt, weights) if rt == 'hipped') / total_weight

            building_data[i]['features']['n_neighbors'] = n_neighbors
            building_data[i]['features']['neighbor_mean_height'] = w_mean_height
            building_data[i]['features']['neighbor_mean_area'] = w_mean_area
            building_data[i]['features']['neighbor_frac_flat'] = frac_flat
            building_data[i]['features']['neighbor_frac_gabled'] = frac_gabled
            building_data[i]['features']['neighbor_frac_hipped'] = frac_hipped
        else:
            # No neighbors — neutral values
            building_data[i]['features']['n_neighbors'] = 0
            building_data[i]['features']['neighbor_mean_height'] = building_data[i]['features'].get('building_height', 0)
            building_data[i]['features']['neighbor_mean_area'] = building_data[i]['features'].get('footprint_area', 0)
            building_data[i]['features']['neighbor_frac_flat'] = 0.33
            building_data[i]['features']['neighbor_frac_gabled'] = 0.33
            building_data[i]['features']['neighbor_frac_hipped'] = 0.33

    if not silent:
        avg_neighbors = neighbor_counts / n if n > 0 else 0
        no_neighbor_count = sum(1 for bd in building_data
                                if bd['features'].get('n_neighbors', 0) == 0)
        print(f"    Avg neighbors/building: {avg_neighbors:.1f}, "
              f"No neighbors: {no_neighbor_count}")

    return building_data


# --- train + evaluate helper ---

def train_and_evaluate(building_data, feature_list, test_name, n_splits=5):
    """
    Train RF classifier with cross-validation and print results.
    Uses CV for more stable estimates than a single train/test split.
    """
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

    # --- Cross-validated evaluation for stable metrics ---
    clf_cv = RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf_cv, X, y_enc, cv=cv, scoring='accuracy')

    # --- Single holdout for confusion matrix and per-class detail ---
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

    print(f"\n{'='*65}")
    print(f"  {test_name}")
    print(f"{'='*65}")
    print(f"  Dataset: {len(X)} buildings, Features: {len(feature_list)}")
    print(f"  CV Accuracy ({n_splits}-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Holdout Accuracy: {acc:.4f}, Holdout F1: {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)}")

    print(f"  Confusion Matrix:")
    print(f"  Predicted →  {' '.join(f'{c:>8}' for c in le.classes_)}")
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:<10}  {' '.join(f'{v:>8}' for v in row)}")

    # Hipped-specific metrics
    result = {'name': test_name, 'cv_acc': cv_scores.mean(), 'cv_std': cv_scores.std(),
              'holdout_acc': acc, 'holdout_f1': f1}

    if 'hipped' in le.classes_:
        hipped_idx = list(le.classes_).index('hipped')
        gabled_idx = list(le.classes_).index('gabled')
        hipped_total = sum(cm[hipped_idx])
        hipped_correct = cm[hipped_idx][hipped_idx]
        hipped_as_gabled = cm[hipped_idx][gabled_idx]

        hipped_recall = hipped_correct / hipped_total if hipped_total > 0 else 0
        hipped_misgabled = hipped_as_gabled / hipped_total if hipped_total > 0 else 0

        print(f"\n  HIPPED DETAIL:")
        print(f"    Total hipped in test: {hipped_total}")
        print(f"    Correctly classified: {hipped_correct} ({hipped_recall*100:.1f}%)")
        print(f"    Misclassified as gabled: {hipped_as_gabled} ({hipped_misgabled*100:.1f}%)")

        result['hipped_recall'] = hipped_recall
        result['hipped_as_gabled'] = hipped_misgabled
        result['hipped_total'] = hipped_total
    else:
        result['hipped_recall'] = None

    return result


# compute_neighbor_features mutates in-place, so each config needs its own copy

def deep_copy_building_data(building_data):
    """Deep copy building data so neighbor features don't bleed between tests."""
    copied = []
    for bd in building_data:
        new_bd = {}
        for k, v in bd.items():
            if k == 'features':
                new_bd[k] = dict(v)  # shallow copy of feature dict is sufficient
            elif k == '_parser':
                new_bd[k] = v  # don't copy parser, just reference
            else:
                new_bd[k] = v
        copied.append(new_bd)
    return copied


def strip_neighbor_features(building_data):
    """Remove any existing neighbor features from building data."""
    neighbor_keys = ['n_neighbors', 'neighbor_mean_height', 'neighbor_mean_area',
                     'neighbor_frac_flat', 'neighbor_frac_gabled', 'neighbor_frac_hipped']
    for bd in building_data:
        for key in neighbor_keys:
            bd['features'].pop(key, None)


# --- tile list ---

ORIGINAL_9 = [
    "data/9-564-628.city.json", "data/9-564-632.city.json", "data/9-564-636.city.json",
    "data/9-568-628.city.json", "data/9-568-632.city.json", "data/9-568-636.city.json",
    "data/9-572-628.city.json", "data/9-572-632.city.json", "data/9-572-636.city.json",
]


# --- feature lists ---

FEATURES_NO_NEIGHBOR = [f for f in ALL_FEATURES
                        if not f.startswith('neighbor_') and f != 'n_neighbors']
FEATURES_WITH_NEIGHBOR = ALL_FEATURES


# --- main experiment ---

if __name__ == "__main__":
    # Load data once
    print("Loading 9 tiles...")
    all_data = []
    existing = [f for f in ORIGINAL_9 if os.path.exists(f)]
    print(f"  Found {len(existing)} of {len(ORIGINAL_9)} tiles")

    for fp in existing:
        parser = CityJSONParser(fp)
        data = process_tile(parser)
        for d in data:
            d['_parser'] = parser
        all_data.extend(data)

    base_data = filter_buildings(all_data)
    print(f"  Total buildings after filtering: {len(base_data)}")

    # Count roof types
    type_counts = {}
    for bd in base_data:
        t = bd['roof_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Roof types: {type_counts}")

    # Define experiment configurations
    configs = [
        {
            'name': 'BASELINE: No neighbor features',
            'use_neighbors': False,
        },
        {
            'name': 'CONFIG A: 50m radius, equal weight (current)',
            'use_neighbors': True,
            'radius': 50.0,
            'max_k': None,
            'distance_weighted': False,
        },
        {
            'name': 'CONFIG B: 30m radius, equal weight',
            'use_neighbors': True,
            'radius': 30.0,
            'max_k': None,
            'distance_weighted': False,
        },
        {
            'name': 'CONFIG C: 50m radius, distance-weighted',
            'use_neighbors': True,
            'radius': 50.0,
            'max_k': None,
            'distance_weighted': True,
        },
        {
            'name': 'CONFIG D: 30m radius, distance-weighted, K=6',
            'use_neighbors': True,
            'radius': 30.0,
            'max_k': 6,
            'distance_weighted': True,
        },
        {
            'name': 'CONFIG E: 30m radius, distance-weighted, K=4',
            'use_neighbors': True,
            'radius': 30.0,
            'max_k': 4,
            'distance_weighted': True,
        },
    ]

    # Run experiments
    results = []

    for cfg in configs:
        print(f"\n{'━'*65}")
        print(f"  Running: {cfg['name']}")
        print(f"{'━'*65}")

        # Deep copy so neighbor features don't leak between tests
        data_copy = deep_copy_building_data(base_data)
        strip_neighbor_features(data_copy)

        if cfg['use_neighbors']:
            compute_neighbor_features_configurable(
                data_copy,
                radius=cfg['radius'],
                max_k=cfg.get('max_k'),
                distance_weighted=cfg.get('distance_weighted', False),
                use_ground_truth=True,
            )
            feature_list = FEATURES_WITH_NEIGHBOR
        else:
            feature_list = FEATURES_NO_NEIGHBOR

        result = train_and_evaluate(data_copy, feature_list, cfg['name'])
        results.append(result)

    # Summary comparison
    print(f"\n\n{'█'*65}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'█'*65}")
    print(f"\n  {'Configuration':<48} {'CV Acc':>8} {'Hipped':>8} {'H→G':>8}")
    print(f"  {'─'*48} {'─'*8} {'─'*8} {'─'*8}")

    for r in results:
        h_recall = f"{r['hipped_recall']*100:.1f}%" if r.get('hipped_recall') is not None else "N/A"
        h_gabled = f"{r['hipped_as_gabled']*100:.1f}%" if r.get('hipped_as_gabled') is not None else "N/A"
        print(f"  {r['name']:<48} {r['cv_acc']*100:>7.2f}% {h_recall:>8} {h_gabled:>8}")

    print(f"\n  CV Acc = 5-fold cross-validated accuracy")
    print(f"  Hipped = Hipped recall (correctly classified)")
    print(f"  H→G    = Hipped misclassified as gabled")
    print(f"\n  Best config = highest hipped recall without major accuracy drop")