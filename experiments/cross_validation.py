"""
Leave-One-Tile-Out Cross-Validation

For each tile in the dataset, trains the full pipeline on all the other
tiles and evaluates on that one. Every tile gets evaluated with the maximum
amount of training data available for it — this complements the held-out
split, which instead measures generalization to geographically distant
tiles. Hyperparameters are fixed (best-known values from prior GridSearchCV
runs) rather than re-tuned per fold, to keep runtime manageable: ~1-2
minutes per tile instead of 10+.

Usage:
    Run from the repository root (so the relative ``data/`` paths resolve):
        python experiments/cross_validation.py data/tile1.city.json data/tile2.city.json ...
        python experiments/cross_validation.py data/*.city.json
"""

import sys
import os
import json
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
# This script lives in experiments/; add the repository root (its parent) to
# the import path so the pipeline modules can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cityjson_parser import CityJSONParser
from feature_extraction import process_tile, filter_buildings, compute_neighbor_features
from ml_models_revised import (
    FeatureSelector, RoofVertexPredictor, ALL_FEATURES
)
from roof_construction import run_construction_pipeline
from evaluation import evaluate_tile, compute_aggregate_metrics, print_evaluation_report

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# Per-fold model comparison with pre-tuned hyperparameters, taken from prior
# GridSearchCV runs on the full dataset (thesis, Table 3.2). Fixing them
# avoids redundant per-fold tuning while still comparing several model
# families each fold.

CLASSIFIER_CONFIGS = {
    'Random Forest': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 20,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
        },
    },
    'Gradient Boosted Trees': {
        'class': GradientBoostingClassifier,
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 7,
            'random_state': 42,
        },
    },
    'K-Nearest Neighbors': {
        'class': KNeighborsClassifier,
        'params': {
            'n_neighbors': 5,
            'weights': 'distance',
            'n_jobs': -1,
        },
        'needs_scaling': True,
    },
}

VERTEX_PREDICTOR_CONFIGS = {
    'Random Forest': {
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': 200,
            'max_depth': 20,
            'random_state': 42,
            'n_jobs': -1,
        },
    },
    'Gradient Boosted Trees': {
        'class': GradientBoostingRegressor,
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': 42,
        },
    },
}


def train_and_select_classifier(X, y_enc, selected_features_idx):
    """
    Train multiple classifiers with pre-tuned hyperparameters and select the
    best by cross-validated F1 on the training set.

    Returns the best-performing classifier (refit on full training set)
    and the name of the winning model.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_reduced = X[:, selected_features_idx]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for speed
    best_score = -1
    best_model = None
    best_name = None

    for name, cfg in CLASSIFIER_CONFIGS.items():
        model = cfg['class'](**cfg['params'])

        # Wrap in pipeline if scaling is needed
        if cfg.get('needs_scaling', False):
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        else:
            pipe = model

        try:
            scores = cross_val_score(pipe, X_reduced, y_enc, cv=cv,
                                     scoring='f1_weighted', n_jobs=1)
            mean_score = scores.mean()
        except Exception as e:
            print(f"    {name} failed: {e}")
            continue

        if mean_score > best_score:
            best_score = mean_score
            best_model = pipe
            best_name = name

    # Refit best model on full training data
    best_model.fit(X_reduced, y_enc)
    return best_model, best_name, best_score


def train_and_select_vertex_predictors(training_data):
    """
    Train multiple vertex predictor families per roof type with pre-tuned
    hyperparameters and select the best by held-out MAE.

    Returns a RoofVertexPredictor with the winning models.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    predictor = RoofVertexPredictor(random_state=42)
    predictor.models = {}
    predictor._best_needs_scaling = {}
    predictor.scalers_X = {}
    predictor.scalers_y = {}

    datasets = predictor.prepare_vertex_data(training_data)

    winners = {}
    for roof_type in ['gabled', 'hipped']:
        X = datasets[roof_type]['X']
        y = datasets[roof_type]['y']

        if len(X) < 10:
            predictor.models[roof_type] = None
            winners[roof_type] = None
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        best_mae = float('inf')
        best_model = None
        best_name = None

        for name, cfg in VERTEX_PREDICTOR_CONFIGS.items():
            # GBT needs MultiOutputRegressor for multi-dim targets
            if cfg['class'] is GradientBoostingRegressor and y_tr.ndim > 1 and y_tr.shape[1] > 1:
                from sklearn.multioutput import MultiOutputRegressor
                base = cfg['class'](**cfg['params'])
                model = MultiOutputRegressor(base)
            else:
                model = cfg['class'](**cfg['params'])

            try:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                mae = mean_absolute_error(y_te, y_pred)
            except Exception as e:
                print(f"    {roof_type} {name} failed: {e}")
                continue

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_name = name

        # Refit winner on full training data
        if best_model is not None:
            # Recreate clean model (GBT wrapper might need resetting)
            cfg = VERTEX_PREDICTOR_CONFIGS[best_name]
            if cfg['class'] is GradientBoostingRegressor and y.ndim > 1 and y.shape[1] > 1:
                from sklearn.multioutput import MultiOutputRegressor
                final_model = MultiOutputRegressor(cfg['class'](**cfg['params']))
            else:
                final_model = cfg['class'](**cfg['params'])
            final_model.fit(X, y)
            predictor.models[roof_type] = final_model
            predictor._best_needs_scaling[roof_type] = False
            winners[roof_type] = (best_name, best_mae)
        else:
            predictor.models[roof_type] = None
            winners[roof_type] = None

    return predictor, winners


# --- load and prepare all tiles once ---

def load_all_tiles(tile_paths):
    """Load all tiles and return per-tile building data + parser."""
    tile_data = {}  # tile_name -> {'parser', 'buildings'}

    for fp in tile_paths:
        if not os.path.exists(fp):
            print(f"  WARNING: {fp} not found, skipping")
            continue

        parser = CityJSONParser(fp)
        data = process_tile(parser)
        for d in data:
            d['_parser'] = parser

        tile_name = os.path.basename(fp).replace('.city.json', '')
        tile_data[tile_name] = {
            'parser': parser,
            'filepath': fp,
            'raw_buildings': data,
        }

    return tile_data


def build_training_set(tile_data, exclude_tile):
    """Build training data from all tiles except exclude_tile."""
    all_buildings = []
    for tile_name, td in tile_data.items():
        if tile_name == exclude_tile:
            continue
        all_buildings.extend(td['raw_buildings'])

    filtered = filter_buildings(all_buildings)
    # Compute neighbor features using ground-truth (training mode)
    filtered = compute_neighbor_features(filtered, radius=50.0, use_ground_truth=True)
    return filtered


def run_loo_cv(tile_paths, output_path="output/cv_results.json"):
    """Run leave-one-tile-out cross-validation."""

    # Load all tiles once
    print(f"\n{'█'*70}")
    print(f"  LOADING ALL TILES")
    print(f"{'█'*70}")
    tile_data = load_all_tiles(tile_paths)
    print(f"  Loaded {len(tile_data)} tiles")

    # Feature selection once, using all tiles, so the selected features
    # stay consistent across folds.
    # (Per-fold selection would add noise without clear benefit.)
    print(f"\n{'█'*70}")
    print(f"  FEATURE SELECTION (using all data)")
    print(f"{'█'*70}")
    all_buildings_data = []
    for td in tile_data.values():
        all_buildings_data.extend(td['raw_buildings'])
    all_filtered = filter_buildings(all_buildings_data)
    all_filtered = compute_neighbor_features(all_filtered, radius=50.0,
                                              use_ground_truth=True)

    # Build feature matrix for selection
    X_rows = []
    y_labels = []
    for bd in all_filtered:
        features = bd['features']
        row = []
        valid = True
        for key in ALL_FEATURES:
            val = features.get(key)
            if val is None:
                valid = False
                break
            row.append(float(val))
        if valid:
            X_rows.append(row)
            y_labels.append(bd['roof_type'])

    X_all = np.array(X_rows)
    y_all = np.array(y_labels)

    selector = FeatureSelector()
    selection_results = selector.analyze(X_all, y_all, ALL_FEATURES)
    selected_features = selection_results['selected_features']
    selected_indices = selection_results['selected_indices']

    print(f"\n  Selected {len(selected_features)} features: {selected_features}")

    # Run CV loop
    print(f"\n{'█'*70}")
    print(f"  LEAVE-ONE-TILE-OUT CROSS-VALIDATION")
    print(f"{'█'*70}")

    all_per_building = []
    tile_results = []
    total_time = 0

    for i, (tile_name, td) in enumerate(tile_data.items(), 1):
        print(f"\n{'━'*70}")
        print(f"  [{i}/{len(tile_data)}] Testing on: {tile_name}")
        print(f"{'━'*70}")

        start = time.time()

        # Build training data (all tiles except this one)
        print(f"  Building training set (excluding {tile_name})...")
        training_data = build_training_set(tile_data, tile_name)

        # Build feature matrix for training
        X_train_rows = []
        y_train_labels = []
        for bd in training_data:
            features = bd['features']
            row = []
            valid = True
            for key in ALL_FEATURES:
                val = features.get(key)
                if val is None:
                    valid = False
                    break
                row.append(float(val))
            if valid:
                X_train_rows.append(row)
                y_train_labels.append(bd['roof_type'])

        X_train = np.array(X_train_rows)
        y_train = np.array(y_train_labels)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        # Train classifiers — compare RF, GBT, KNN, pick best
        print(f"  Training classifier (comparing 3 families)...")
        clf, clf_name, clf_cv_score = train_and_select_classifier(
            X_train, y_train_enc, selected_indices
        )
        print(f"    Selected: {clf_name} (CV F1={clf_cv_score:.4f})")

        # Train vertex predictors — compare RF, GBT, pick best per roof type
        print(f"  Training vertex predictors (comparing 2 families per type)...")
        vertex_predictor, vertex_winners = train_and_select_vertex_predictors(training_data)
        for rt, winner in vertex_winners.items():
            if winner:
                name, mae = winner
                print(f"    {rt}: {name} (MAE={mae:.4f})")

        # Wrap classifier so roof_construction can use it (expects .predict & .predict_proba)
        # The pipeline already provides these, just use directly
        clf_for_construction = clf

        # Run construction on held-out tile
        print(f"  Running construction on {tile_name}...")
        construction_result = run_construction_pipeline(
            parser=td['parser'],
            classifier=clf,
            vertex_predictor=vertex_predictor,
            label_encoder=le,
            selected_features=selected_features,
            output_path=None,
        )

        # Evaluate against ground truth
        eval_result = evaluate_tile(td['parser'], construction_result['buildings'])
        n_eval = eval_result['n_evaluated']
        metrics = eval_result['metrics']
        acc = metrics['classification']['accuracy']

        elapsed = time.time() - start
        total_time += elapsed

        # Extract key metrics
        tile_summary = {
            'tile': tile_name,
            'n_evaluated': n_eval,
            'accuracy': acc,
            'time_seconds': elapsed,
            'classifier_winner': clf_name,
            'classifier_cv_f1': clf_cv_score,
            'vertex_gabled_winner': vertex_winners.get('gabled'),
            'vertex_hipped_winner': vertex_winners.get('hipped'),
        }

        for rtype in ['flat', 'gabled', 'hipped']:
            rt = metrics.get(rtype, {})
            rh = rt.get('ridge_height', {})
            tile_summary[f'{rtype}_n'] = rt.get('n_buildings', 0)
            tile_summary[f'{rtype}_ridge_mae'] = rh.get('mae')
            tile_summary[f'{rtype}_ridge_median'] = rh.get('median_error')

        tile_results.append(tile_summary)
        all_per_building.extend(eval_result['per_building'])

        # Progress log
        print(f"  Result: {n_eval} buildings, accuracy={acc:.3f}, time={elapsed:.0f}s")
        eta = (len(tile_data) - i) * (total_time / i)
        print(f"  Progress: {i}/{len(tile_data)} tiles done, ETA: {eta/60:.1f} min")

        # Save intermediate results after each tile (safety)
        with open(output_path, 'w') as f:
            serializable = {
                'tile_results': tile_results,
                'tiles_completed': i,
                'tiles_total': len(tile_data),
                'selected_features': selected_features,
            }
            json.dump(serializable, f, indent=2, default=str)

    # Combined metrics across all tiles
    print(f"\n{'█'*70}")
    print(f"  COMBINED CV RESULTS")
    print(f"{'█'*70}")

    combined_metrics = compute_aggregate_metrics(all_per_building)
    combined = {
        'n_evaluated': len(all_per_building),
        'metrics': combined_metrics,
        'per_building': all_per_building,
    }
    print_evaluation_report(combined, "LEAVE-ONE-TILE-OUT CV")

    # Per-tile summary table
    print(f"\n{'█'*70}")
    print(f"  PER-TILE SUMMARY")
    print(f"{'█'*70}")
    print(f"\n  {'Tile':<20} {'N':>5} {'Acc':>7} {'Flat MAE':>9} {'Gabled MAE':>11} {'Hipped MAE':>11}")
    print(f"  {'─'*20} {'─'*5} {'─'*7} {'─'*9} {'─'*11} {'─'*11}")

    for tr in tile_results:
        flat_str = f"{tr.get('flat_ridge_mae'):.3f}" if tr.get('flat_ridge_mae') else "N/A"
        gab_str = f"{tr.get('gabled_ridge_mae'):.3f}" if tr.get('gabled_ridge_mae') else "N/A"
        hip_str = f"{tr.get('hipped_ridge_mae'):.3f}" if tr.get('hipped_ridge_mae') else "N/A"
        print(f"  {tr['tile']:<20} {tr['n_evaluated']:>5} "
              f"{tr['accuracy']:>6.1%} {flat_str:>9} {gab_str:>11} {hip_str:>11}")

    # Compute summary statistics across tiles
    accuracies = [tr['accuracy'] for tr in tile_results]
    print(f"\n  Per-tile accuracy:")
    print(f"    Mean:   {np.mean(accuracies):.3f}")
    print(f"    Median: {np.median(accuracies):.3f}")
    print(f"    Min:    {min(accuracies):.3f} ({tile_results[np.argmin(accuracies)]['tile']})")
    print(f"    Max:    {max(accuracies):.3f} ({tile_results[np.argmax(accuracies)]['tile']})")

    # Model winner summary
    clf_winners = {}
    for tr in tile_results:
        w = tr.get('classifier_winner')
        if w:
            clf_winners[w] = clf_winners.get(w, 0) + 1
    print(f"\n  Classifier selected per fold:")
    for name, count in sorted(clf_winners.items(), key=lambda x: -x[1]):
        print(f"    {name:<30}: {count}/{len(tile_results)} folds")

    gabled_winners = {}
    hipped_winners = {}
    for tr in tile_results:
        gw = tr.get('vertex_gabled_winner')
        hw = tr.get('vertex_hipped_winner')
        if gw:
            name = gw[0] if isinstance(gw, (list, tuple)) else gw
            gabled_winners[name] = gabled_winners.get(name, 0) + 1
        if hw:
            name = hw[0] if isinstance(hw, (list, tuple)) else hw
            hipped_winners[name] = hipped_winners.get(name, 0) + 1

    print(f"\n  Gabled vertex predictor selected per fold:")
    for name, count in sorted(gabled_winners.items(), key=lambda x: -x[1]):
        print(f"    {name:<30}: {count}/{len(tile_results)} folds")

    print(f"\n  Hipped vertex predictor selected per fold:")
    for name, count in sorted(hipped_winners.items(), key=lambda x: -x[1]):
        print(f"    {name:<30}: {count}/{len(tile_results)} folds")

    print(f"\n  Total CV time: {total_time/60:.1f} minutes")

    # Save final results
    with open(output_path, 'w') as f:
        serializable = {
            'tile_results': tile_results,
            'combined_metrics': combined_metrics,
            'selected_features': selected_features,
            'total_time_seconds': total_time,
            'n_tiles': len(tile_data),
        }
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cross_validation.py <tile1.city.json> [tile2.city.json ...]")
        print("\nTip: use 'data/*.city.json' to include all tiles")
        sys.exit(1)

    tile_paths = sys.argv[1:]
    print(f"Running leave-one-tile-out CV on {len(tile_paths)} tiles")

    os.makedirs("output", exist_ok=True)
    run_loo_cv(tile_paths)