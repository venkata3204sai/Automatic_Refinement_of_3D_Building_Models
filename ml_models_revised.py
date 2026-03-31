"""
=============================================================================
Module 4 (Revised): Improved ML Pipeline
=============================================================================
Changes based on supervisor feedback:

1. FEATURE SELECTION for classification:
   - ONLY features derivable from LOD1.3 geometry (no LiDAR/LOD2.2 data)
   - Started with 13 pure LOD1.3 features, selection reduces further
   - LOD2.2 data is ONLY used as ground-truth labels, never as input

2. ROOF VERTEX PREDICTION using Neural Networks:
   - Input: normalized LOD1.3 geometry (footprint vertices + building dims)
   - Output: 3D coordinates of roof vertices to add
   - All input features are LOD1.3-only

3. DATA INTEGRITY:
   - Input features: LOD1.3 geometry only (footprint shape, height, volume)
   - Ground truth labels: from LOD2.2 (roof type, ridge vertex positions)
   - This ensures the model can work on any LOD1.3 building without LiDAR
=============================================================================
"""

import json
import math
import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: Only features derivable from LOD1.3 geometry are used as input.
# Features from 3DBAG attributes (h_dak_*, vol_lod22, vol_ratio, etc.) are
# excluded because they are computed from LiDAR/LOD2.2 data, which would not
# be available in a real deployment scenario.
#
# LOD2.2 data is ONLY used as ground-truth labels for training, never as input.
# ─────────────────────────────────────────────────────────────────────────────

# Features computed purely from LOD1.3 geometry (13 features)
ALL_FEATURES = [
    'footprint_area', 'footprint_perimeter', 'n_footprint_vertices',
    'aspect_ratio', 'compactness', 'rectangularity',
    'mbr_length', 'mbr_width', 'edge_length_ratio',
    'building_height',
    'orientation', 'longest_edge_length',
    'vol_lod13'
]

# Features that were REMOVED because they leak LOD2.2/LiDAR information:
# h_dak_min, h_dak_max, h_dak_50p, h_dak_70p  (roof height stats from LiDAR)
# height_range, height_std_proxy               (derived from h_dak_* above)
# vol_lod22, vol_ratio, vol_difference         (use LOD2.2 volume)


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: FEATURE SELECTION FOR CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

class FeatureSelector:
    """
    Identifies which features are necessary for roof type classification
    and which can be dropped without hurting performance.

    Uses three methods:
    1. Tree-based feature importance (from Random Forest)
    2. Permutation importance (model-agnostic)
    3. Mutual information (statistical dependency)

    Then validates by comparing full vs reduced feature set accuracy.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}

    def analyze(self, X: np.ndarray, y: np.ndarray,
                feature_names: list) -> dict:
        """
        Run all feature selection analyses.

        Returns dict with selected features and comparison results.
        """
        print(f"\n{'='*70}")
        print("FEATURE SELECTION ANALYSIS")
        print(f"{'='*70}")
        print(f"Starting features: {len(feature_names)}")

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, stratify=y_enc,
            random_state=self.random_state
        )

        # ─── Method 1: Tree-based importance ───
        print(f"\n--- Method 1: Random Forest Feature Importance ---")
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=self.random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        rf_importances = rf.feature_importances_
        rf_ranking = np.argsort(rf_importances)[::-1]

        print(f"{'Rank':<6} {'Feature':<25} {'Importance':>12}")
        print("-" * 45)
        for rank, idx in enumerate(rf_ranking):
            marker = " ✓" if rf_importances[idx] >= 0.03 else " ✗"
            print(f"  {rank+1:<4} {feature_names[idx]:<25} {rf_importances[idx]:>10.4f}{marker}")

        # ─── Method 2: Permutation importance ───
        print(f"\n--- Method 2: Permutation Importance ---")
        perm_result = permutation_importance(
            rf, X_test, y_test, n_repeats=20,
            random_state=self.random_state, n_jobs=-1
        )
        perm_importances = perm_result.importances_mean
        perm_ranking = np.argsort(perm_importances)[::-1]

        print(f"{'Rank':<6} {'Feature':<25} {'Importance':>12} {'Std':>10}")
        print("-" * 55)
        for rank, idx in enumerate(perm_ranking):
            marker = " ✓" if perm_importances[idx] >= 0.01 else " ✗"
            print(f"  {rank+1:<4} {feature_names[idx]:<25} "
                  f"{perm_importances[idx]:>10.4f} "
                  f"{perm_result.importances_std[idx]:>8.4f}{marker}")

        # ─── Method 3: Mutual information ───
        print(f"\n--- Method 3: Mutual Information ---")
        mi_scores = mutual_info_classif(
            X_train, y_train, random_state=self.random_state
        )
        mi_ranking = np.argsort(mi_scores)[::-1]

        print(f"{'Rank':<6} {'Feature':<25} {'MI Score':>12}")
        print("-" * 45)
        for rank, idx in enumerate(mi_ranking):
            marker = " ✓" if mi_scores[idx] >= 0.05 else " ✗"
            print(f"  {rank+1:<4} {feature_names[idx]:<25} {mi_scores[idx]:>10.4f}{marker}")

        # ─── Consensus: Select features that appear important in ≥2 methods ───
        print(f"\n--- Consensus Feature Selection ---")

        # Score each feature: how many methods rank it in the top half
        n_features = len(feature_names)
        half = n_features // 2
        consensus_scores = {}

        for i, name in enumerate(feature_names):
            score = 0
            rf_rank = list(rf_ranking).index(i)
            perm_rank = list(perm_ranking).index(i)
            mi_rank = list(mi_ranking).index(i)

            if rf_rank < half:
                score += 1
            if perm_rank < half:
                score += 1
            if mi_rank < half:
                score += 1

            # Also check absolute thresholds
            if rf_importances[i] >= 0.03:
                score += 1
            if perm_importances[i] >= 0.01:
                score += 1
            if mi_scores[i] >= 0.05:
                score += 1

            consensus_scores[name] = {
                'total_score': score,
                'rf_importance': float(rf_importances[i]),
                'perm_importance': float(perm_importances[i]),
                'mi_score': float(mi_scores[i]),
                'rf_rank': int(rf_rank) + 1,
                'perm_rank': int(perm_rank) + 1,
                'mi_rank': int(mi_rank) + 1,
            }

        # Select features with score >= 3 (important in at least 2+ methods)
        selected = [name for name, info in consensus_scores.items()
                    if info['total_score'] >= 3]
        dropped = [name for name in feature_names if name not in selected]

        print(f"\nSelected features ({len(selected)}):")
        for name in selected:
            info = consensus_scores[name]
            print(f"  ✓ {name:<25} score={info['total_score']} "
                  f"(RF rank:{info['rf_rank']}, Perm rank:{info['perm_rank']}, MI rank:{info['mi_rank']})")

        print(f"\nDropped features ({len(dropped)}):")
        for name in dropped:
            info = consensus_scores[name]
            print(f"  ✗ {name:<25} score={info['total_score']} "
                  f"(RF rank:{info['rf_rank']}, Perm rank:{info['perm_rank']}, MI rank:{info['mi_rank']})")

        # ─── Validate: Compare full vs reduced feature set ───
        print(f"\n--- Validation: Full vs Reduced Feature Set ---")

        selected_indices = [feature_names.index(name) for name in selected]
        X_train_reduced = X_train[:, selected_indices]
        X_test_reduced = X_test[:, selected_indices]

        # Train on full features
        rf_full = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=self.random_state, n_jobs=-1
        )
        rf_full.fit(X_train, y_train)
        acc_full = accuracy_score(y_test, rf_full.predict(X_test))
        f1_full = f1_score(y_test, rf_full.predict(X_test), average='weighted')

        # Train on selected features
        rf_reduced = RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=self.random_state, n_jobs=-1
        )
        rf_reduced.fit(X_train_reduced, y_train)
        acc_reduced = accuracy_score(y_test, rf_reduced.predict(X_test_reduced))
        f1_reduced = f1_score(y_test, rf_reduced.predict(X_test_reduced), average='weighted')

        print(f"\n  Full features ({len(feature_names)}): Accuracy={acc_full:.4f}, F1={f1_full:.4f}")
        print(f"  Selected features ({len(selected)}):  Accuracy={acc_reduced:.4f}, F1={f1_reduced:.4f}")
        diff = acc_reduced - acc_full
        print(f"  Difference: {diff:+.4f} ({'better' if diff >= 0 else 'slightly worse'})")

        if diff >= -0.02:
            print(f"  → Recommendation: USE REDUCED SET (simpler model, similar performance)")
        else:
            print(f"  → Recommendation: KEEP FULL SET (reduced set loses significant accuracy)")

        self.results = {
            'selected_features': selected,
            'dropped_features': dropped,
            'selected_indices': selected_indices,
            'consensus_scores': consensus_scores,
            'full_accuracy': float(acc_full),
            'reduced_accuracy': float(acc_reduced),
            'full_f1': float(f1_full),
            'reduced_f1': float(f1_reduced),
        }

        return self.results


# ═════════════════════════════════════════════════════════════════════════════
# PART 2: NEURAL NETWORK ROOF VERTEX PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

class RoofVertexPredictor:
    """
    Predicts 3D positions of roof vertices given LOD1.3 building geometry.

    APPROACH (supervisor's suggestion):
    - Treat the building as a solid (LOD1.3 box)
    - The top face of the box = eave level
    - Predict where to place ridge vertices ON TOP of this solid
    - Use the footprint shape and building dimensions as input

    INPUT REPRESENTATION:
    We normalize the building to a local coordinate system:
    - Origin at the footprint centroid
    - Coordinates scaled by building dimensions
    - This makes the network footprint-size invariant

    The input vector contains:
    - Normalized footprint vertices (x, y for each vertex, padded to max_vertices)
    - Building height, eave height
    - Footprint area, aspect ratio, compactness
    - Number of footprint vertices

    OUTPUT:
    - For gabled: 2 ridge vertices (6 values: x1,y1,z1, x2,y2,z2)
    - For hipped: 2 ridge endpoints (6 values: x1,y1,z1, x2,y2,z2)
    - Coordinates are in the same normalized local system

    After prediction, we denormalize back to real coordinates.

    NOTE: Uses sklearn MLPRegressor. For your thesis, you should
    upgrade to PyTorch for more flexibility (custom architectures,
    batch normalization, dropout, etc.)
    """

    MAX_FOOTPRINT_VERTICES = 4  # Fixed: rectangular footprints only

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}       # {roof_type: model}
        self.scalers_X = {}    # {roof_type: StandardScaler for input}
        self.scalers_y = {}    # {roof_type: StandardScaler for output}
        self._best_needs_scaling = {}  # {roof_type: bool}
        self.results = {}

    def prepare_vertex_data(self, building_data: list, parser_map: dict = None) -> dict:
        """
        Prepare training data for roof vertex prediction.

        For each building:
        1. Extract LOD1.3 top face vertices (normalized)
        2. Extract LOD2.2 ridge vertices (normalized) as targets

        Args:
            building_data: list of building data dicts from feature_extraction
            parser_map: dict mapping building_id to CityJSONParser
                        (needed for vertex access)

        Returns:
            dict with {roof_type: {'X': input_array, 'y': target_array,
                                    'meta': normalization_metadata}}
        """
        datasets = {'gabled': {'X': [], 'y': [], 'meta': []},
                    'hipped': {'X': [], 'y': [], 'meta': []}}

        for bd in building_data:
            roof_type = bd['roof_type']
            if roof_type not in ('gabled', 'hipped'):
                continue

            features = bd['features']
            roof_params = bd.get('roof_params', {})

            if not roof_params:
                continue

            # Get footprint 2D coordinates (stored during feature extraction)
            footprint_2d = features.get('_footprint_2d', None)
            if footprint_2d is None or len(footprint_2d) < 3:
                continue

            # ─── Build normalized input ───
            input_vec = self._build_input_vector(features, footprint_2d)
            if input_vec is None:
                continue

            # ─── Build normalized target (ridge vertex positions) ───
            target_vec = self._build_target_vector(features, roof_params, roof_type, footprint_2d)
            if target_vec is None:
                continue

            # Store normalization metadata for denormalization later
            centroid_x = features.get('centroid_x', 0)
            centroid_y = features.get('centroid_y', 0)
            meta = {
                'building_id': bd['building_id'],
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'mbr_length': features.get('mbr_length', 1),
                'mbr_width': features.get('mbr_width', 1),
                'ground_z': features.get('ground_z', 0),
                'top_z': features.get('top_z', 0),
                'building_height': features.get('building_height', 1),
            }

            datasets[roof_type]['X'].append(input_vec)
            datasets[roof_type]['y'].append(target_vec)
            datasets[roof_type]['meta'].append(meta)

        # Convert to numpy arrays
        for rt in datasets:
            if datasets[rt]['X']:
                datasets[rt]['X'] = np.array(datasets[rt]['X'])
                datasets[rt]['y'] = np.array(datasets[rt]['y'])
                print(f"  {rt}: {len(datasets[rt]['X'])} samples, "
                      f"input_dim={datasets[rt]['X'].shape[1]}, "
                      f"output_dim={datasets[rt]['y'].shape[1]}")
            else:
                datasets[rt]['X'] = np.array([]).reshape(0, 0)
                datasets[rt]['y'] = np.array([]).reshape(0, 0)

        return datasets

    def _build_input_vector(self, features: dict,
                            footprint_2d: list) -> Optional[np.ndarray]:
        """
        Build the normalized input vector for a building.

        For rectangular (4-vertex) footprints, the input contains:
        - 8 values: normalized footprint vertices (4 × XY)
        - 5 values: normalized_area, aspect_ratio, normalized_height,
                     mbr_length/scale, mbr_width/scale

        Total: 13 inputs. Each input varies across buildings — constants
        and redundant features (n_vertices, compactness, rectangularity,
        edge_length_ratio, orientation) have been removed.
        """
        centroid_x = features.get('centroid_x', 0)
        centroid_y = features.get('centroid_y', 0)
        mbr_length = features.get('mbr_length', 1)
        mbr_width = features.get('mbr_width', 1)
        building_height = features.get('building_height', 1)

        if mbr_length <= 0 or mbr_width <= 0 or building_height <= 0:
            return None

        # Normalize footprint: center at origin, scale by MBR dimensions
        scale_xy = max(mbr_length, mbr_width)
        norm_footprint = []
        for px, py in footprint_2d[:self.MAX_FOOTPRINT_VERTICES]:
            nx = (px - centroid_x) / scale_xy
            ny = (py - centroid_y) / scale_xy
            norm_footprint.extend([nx, ny])

        # Geometric features (only those that vary across rectangular buildings)
        extra_features = [
            features.get('footprint_area', 0) / (scale_xy ** 2),  # normalized area
            features.get('aspect_ratio', 1),
            building_height / scale_xy,           # normalized height
            features.get('mbr_length', 0) / scale_xy,
            features.get('mbr_width', 0) / scale_xy,
        ]

        input_vec = np.array(norm_footprint + extra_features, dtype=np.float64)
        return input_vec

    def _build_target_vector(self, features: dict, roof_params: dict,
                             roof_type: str,
                             footprint_2d: list) -> Optional[np.ndarray]:
        """
        Build the normalized target vector.

        GABLED: Only ridge_z (1 value) — ridge XY is determined by footprint
        HIPPED: Full prediction — ridge_cx, ridge_cy, ridge_z, ridge_half_len (4 values)

        All coordinates normalized relative to the building.
        """
        building_height = features.get('building_height', 1)
        ground_z = features.get('ground_z', 0)

        ridge_height = roof_params.get('ridge_height', None)
        if ridge_height is None:
            return None

        # Normalized ridge z (height above ground, scaled by building height)
        norm_ridge_z = (ridge_height - ground_z) / building_height if building_height > 0 else 0

        if roof_type == 'gabled':
            # Gabled: only predict ridge height
            # Ridge XY will be computed from footprint geometry during construction
            target = np.array([norm_ridge_z], dtype=np.float64)

        elif roof_type == 'hipped':
            # Hipped: predict full ridge position and length
            centroid_x = features.get('centroid_x', 0)
            centroid_y = features.get('centroid_y', 0)
            mbr_length = features.get('mbr_length', 1)
            mbr_width = features.get('mbr_width', 1)
            scale_xy = max(mbr_length, mbr_width)

            ridge_x = roof_params.get('ridge_x', None)
            ridge_y = roof_params.get('ridge_y', None)
            ridge_length = roof_params.get('ridge_length', None)

            if ridge_x is None or ridge_y is None:
                return None

            norm_ridge_x = (ridge_x - centroid_x) / scale_xy
            norm_ridge_y = (ridge_y - centroid_y) / scale_xy

            if ridge_length is not None:
                norm_ridge_half_len = (ridge_length / 2) / scale_xy
            else:
                norm_ridge_half_len = mbr_length / (4 * scale_xy)

            target = np.array([
                norm_ridge_x, norm_ridge_y, norm_ridge_z, norm_ridge_half_len
            ], dtype=np.float64)

        else:
            return None

        return target

    def train(self, datasets: dict, feature_names: list = None):
        """
        Train regression models for each roof type with cross-validated
        model selection across multiple model families.

        Models compared:
        - Linear Regression (baseline)
        - Decision Tree Regressor
        - Random Forest Regressor
        - Gradient Boosted Regressor
        - MLP Regressor (two sizes)
        """
        for roof_type in ['gabled', 'hipped']:
            X = datasets[roof_type]['X']
            y = datasets[roof_type]['y']

            if len(X) < 5:
                print(f"\n  Skipping {roof_type}: only {len(X)} samples")
                continue

            print(f"\n{'='*60}")
            print(f"TRAINING ROOF VERTEX PREDICTOR: {roof_type.upper()}")
            print(f"{'='*60}")
            print(f"  Samples: {len(X)}")
            print(f"  Input dim: {X.shape[1]}")
            print(f"  Output dim: {y.shape[1]}")

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )

            # ─── Define models ───
            # Each entry: (model, needs_scaling, param_grid_for_cv)
            # Models with param grids get cross-validated tuning.
            # Models without just train directly.
            model_configs = {
                'Linear Regression': {
                    'model': LinearRegression(),
                    'needs_scaling': True,
                    'params': None,  # No hyperparameters to tune
                },
                'Decision Tree': {
                    'model': DecisionTreeRegressor(random_state=self.random_state),
                    'needs_scaling': False,
                    'params': {
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                    },
                },
                'Random Forest': {
                    'model': RandomForestRegressor(
                        random_state=self.random_state, n_jobs=-1
                    ),
                    'needs_scaling': False,
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                    },
                },
                'Gradient Boosted Trees': {
                    'model': MultiOutputRegressor(
                        GradientBoostingRegressor(
                            random_state=self.random_state
                        )
                    ),
                    'needs_scaling': False,
                    'params': {
                        'estimator__n_estimators': [100, 200],
                        'estimator__max_depth': [3, 5, 7],
                        'estimator__learning_rate': [0.05, 0.1],
                    },
                },
                'MLP-Small (64-32)': {
                    'model': MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation='relu', solver='adam',
                        learning_rate='adaptive',
                        max_iter=1000, early_stopping=True,
                        validation_fraction=0.15,
                        random_state=self.random_state
                    ),
                    'needs_scaling': True,
                    'params': {
                        'learning_rate_init': [0.001, 0.005],
                    },
                },
                'MLP-Large (128-64-32)': {
                    'model': MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32),
                        activation='relu', solver='adam',
                        learning_rate='adaptive',
                        max_iter=1000, early_stopping=True,
                        validation_fraction=0.15,
                        random_state=self.random_state
                    ),
                    'needs_scaling': True,
                    'params': {
                        'learning_rate_init': [0.001, 0.005],
                    },
                },
            }

            best_model = None
            best_mae = float('inf')
            best_name = None
            best_needs_scaling = False
            self.results[roof_type] = {}

            cv = 3 if len(X_train) < 100 else 5

            for name, config in model_configs.items():
                needs_scaling = config['needs_scaling']
                param_grid = config['params']

                if needs_scaling:
                    # Wrap in Pipeline so GridSearchCV sees original units
                    # This ensures CV MAE is comparable across all models
                    from sklearn.compose import TransformedTargetRegressor
                    wrapped_model = TransformedTargetRegressor(
                        regressor=Pipeline([
                            ('scaler_X', StandardScaler()),
                            ('model', config['model'])
                        ]),
                        transformer=StandardScaler()
                    )
                    # Prefix param names to reach through the wrappers
                    if param_grid:
                        prefixed_params = {
                            f'regressor__model__{k}': v for k, v in param_grid.items()
                        }
                    else:
                        prefixed_params = None

                    X_tr = X_train  # raw, unscaled — Pipeline handles scaling
                    X_te = X_test
                    y_tr_fit = y_train
                    use_model = wrapped_model
                    use_params = prefixed_params
                else:
                    X_tr = X_train
                    X_te = X_test
                    y_tr_fit = y_train
                    use_model = config['model']
                    use_params = param_grid

                # Train with or without GridSearchCV
                if use_params and len(X_train) >= 10:
                    grid = GridSearchCV(
                        use_model, use_params,
                        cv=cv, scoring='neg_mean_absolute_error',
                        n_jobs=-1, refit=True
                    )
                    grid.fit(X_tr, y_tr_fit)
                    model = grid.best_estimator_
                    cv_score = -grid.best_score_
                    # Extract original param names for display
                    if needs_scaling:
                        best_params = {
                            k.replace('regressor__model__', ''): v
                            for k, v in grid.best_params_.items()
                        }
                    else:
                        best_params = grid.best_params_
                else:
                    model = use_model
                    model.fit(X_tr, y_tr_fit)
                    cv_score = None
                    best_params = {}

                # Predict on test set (model handles scaling internally if wrapped)
                y_pred_real = model.predict(X_te)

                y_pred_real = y_pred_real if y_pred_real.ndim == 2 else y_pred_real.reshape(-1, 1)
                y_test_real = y_test if y_test.ndim == 2 else y_test.reshape(-1, 1)

                # Store the unwrapped model and scalers for later prediction
                if needs_scaling and hasattr(model, 'regressor_'):
                    # Extract the fitted scalers and inner model for predict()
                    inner_pipeline = model.regressor_
                    self.scalers_X[roof_type] = inner_pipeline.named_steps['scaler_X']
                    self.scalers_y[roof_type] = model.transformer_
                    actual_model = inner_pipeline.named_steps['model']
                else:
                    actual_model = model

                # Evaluate each output dimension
                mae_per_dim = []
                for dim in range(y_test_real.shape[1]):
                    mae_dim = mean_absolute_error(y_test_real[:, dim], y_pred_real[:, dim])
                    mae_per_dim.append(mae_dim)

                total_mae = np.mean(mae_per_dim)
                total_rmse = np.sqrt(mean_squared_error(y_test_real.flatten(), y_pred_real.flatten()))

                self.results[roof_type][name] = {
                    'mae': float(total_mae),
                    'rmse': float(total_rmse),
                    'mae_per_dim': [float(m) for m in mae_per_dim],
                    'n_train': len(X_train),
                    'n_test': len(X_test),
                    'best_params': {k: str(v) for k, v in best_params.items()},
                }
                if cv_score is not None:
                    self.results[roof_type][name]['cv_mae'] = float(cv_score)

                dim_labels_map = {
                    'gabled': ['ridge_z'],
                    'hipped': ['ridge_cx', 'ridge_cy', 'ridge_z', 'ridge_half_len'],
                }
                dim_labels = dim_labels_map.get(roof_type, [f'dim{i}' for i in range(len(mae_per_dim))])
                print(f"\n  {name}:")
                if best_params:
                    print(f"    Best params: {best_params}")
                if cv_score is not None:
                    print(f"    CV MAE: {cv_score:.4f}")
                print(f"    Test MAE: {total_mae:.4f} (normalized)")
                print(f"    Per-dimension MAE:")
                for i, (label, mae_d) in enumerate(zip(dim_labels, mae_per_dim)):
                    print(f"      {label}: {mae_d:.4f}")

                if total_mae < best_mae:
                    best_mae = total_mae
                    best_model = actual_model
                    best_name = name
                    best_needs_scaling = needs_scaling

            self.models[roof_type] = best_model
            self._best_needs_scaling[roof_type] = best_needs_scaling
            print(f"\n  ★ Best model for {roof_type}: {best_name} (MAE={best_mae:.4f})")

            # ─── Detailed analysis of best model predictions ───
            print(f"\n  --- Sample Predictions ({roof_type}) ---")
            if best_needs_scaling and roof_type in self.scalers_X:
                X_test_scaled = self.scalers_X[roof_type].transform(X_test)
                y_pred_scaled = best_model.predict(X_test_scaled)
                if y_pred_scaled.ndim == 1:
                    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
                y_pred_final = self.scalers_y[roof_type].inverse_transform(y_pred_scaled)
            else:
                y_pred_final = best_model.predict(X_test)

            n_show = min(5, len(X_test))
            for i in range(n_show):
                true_vals = np.atleast_1d(y_test[i])
                pred_vals = np.atleast_1d(y_pred_final[i])
                errors = np.abs(true_vals - pred_vals)
                print(f"    Sample {i+1}:")
                print(f"      True:      [{', '.join(f'{v:.3f}' for v in true_vals)}]")
                print(f"      Predicted: [{', '.join(f'{v:.3f}' for v in pred_vals)}]")
                print(f"      Abs Error: [{', '.join(f'{v:.3f}' for v in errors)}]")

    def predict(self, features: dict, footprint_2d: list,
                roof_type: str) -> Optional[dict]:
        """
        Predict roof parameters for a new building.

        GABLED: Returns only ridge_z — ridge XY computed by construction module
        HIPPED: Returns full ridge vertex positions

        Args:
            features: dict of LOD1.3 features
            footprint_2d: list of (x, y) footprint coordinates
            roof_type: predicted roof type

        Returns:
            dict with 'ridge_height' and optionally 'ridge_vertices'
        """
        if roof_type not in self.models or self.models[roof_type] is None:
            return None

        input_vec = self._build_input_vector(features, footprint_2d)
        if input_vec is None:
            return None

        input_vec = input_vec.reshape(1, -1)
        model = self.models[roof_type]

        # Predict (handle scaling based on what the best model needed)
        needs_scaling = getattr(self, '_best_needs_scaling', {}).get(roof_type, False)

        if needs_scaling and roof_type in self.scalers_X:
            input_scaled = self.scalers_X[roof_type].transform(input_vec)
            pred_scaled = model.predict(input_scaled)
            if pred_scaled.ndim == 1:
                pred_scaled = pred_scaled.reshape(-1, 1)
            prediction = self.scalers_y[roof_type].inverse_transform(pred_scaled)[0]
        else:
            prediction = model.predict(input_vec)[0]

        # Ensure prediction is always an array
        prediction = np.atleast_1d(prediction)

        # Denormalize
        ground_z = features.get('ground_z', 0)
        building_height = features.get('building_height', 1)

        if roof_type == 'gabled':
            # Gabled: only ridge_z predicted
            ridge_z = float(prediction[0]) * building_height + ground_z

            # Clamp ridge height
            eave_z = ground_z + building_height
            ridge_z = max(eave_z, min(eave_z + building_height, ridge_z))

            return {
                'ridge_height': ridge_z,
                'normalized_prediction': prediction.tolist()
            }

        elif roof_type == 'hipped':
            # Hipped: full prediction — ridge_cx, ridge_cy, ridge_z, ridge_half_len
            centroid_x = features.get('centroid_x', 0)
            centroid_y = features.get('centroid_y', 0)
            mbr_length = features.get('mbr_length', 1)
            mbr_width = features.get('mbr_width', 1)
            scale_xy = max(mbr_length, mbr_width)

            ridge_cx = prediction[0] * scale_xy + centroid_x
            ridge_cy = prediction[1] * scale_xy + centroid_y
            ridge_z = prediction[2] * building_height + ground_z
            ridge_half_len = abs(prediction[3]) * scale_xy

            # Compute ridge direction from longest footprint edge
            best_dx, best_dy, best_len = 1.0, 0.0, 0.0
            for k in range(len(footprint_2d)):
                next_k = (k + 1) % len(footprint_2d)
                ex = footprint_2d[next_k][0] - footprint_2d[k][0]
                ey = footprint_2d[next_k][1] - footprint_2d[k][1]
                elen = math.sqrt(ex * ex + ey * ey)
                if elen > best_len:
                    best_len = elen
                    best_dx = ex / elen
                    best_dy = ey / elen

            dx = best_dx * ridge_half_len
            dy = best_dy * ridge_half_len

            # Clamp ridge center
            max_offset = scale_xy * 0.3
            ridge_cx = max(centroid_x - max_offset, min(centroid_x + max_offset, ridge_cx))
            ridge_cy = max(centroid_y - max_offset, min(centroid_y + max_offset, ridge_cy))

            # Clamp ridge half length
            max_half_len = mbr_length * 0.6
            ridge_half_len = min(ridge_half_len, max_half_len)
            dx = best_dx * ridge_half_len
            dy = best_dy * ridge_half_len

            # Clamp ridge height
            eave_z = ground_z + building_height
            ridge_z = max(eave_z, min(eave_z + building_height, ridge_z))

            ridge_v1 = (ridge_cx - dx, ridge_cy - dy, ridge_z)
            ridge_v2 = (ridge_cx + dx, ridge_cy + dy, ridge_z)

            return {
                'ridge_vertices': [ridge_v1, ridge_v2],
                'ridge_center': (ridge_cx, ridge_cy, ridge_z),
                'ridge_length': ridge_half_len * 2,
                'ridge_height': ridge_z,
                'normalized_prediction': prediction.tolist()
            }

        return None


# ═════════════════════════════════════════════════════════════════════════════
# COMPLETE REVISED PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_revised_pipeline(building_data: list, output_dir: str = "output"):
    """
    Run the complete revised ML pipeline:
    1. Feature selection for classification
    2. Train classifier with reduced features
    3. Train neural network for roof vertex prediction
    """
    os.makedirs(output_dir, exist_ok=True)

    # ─── Prepare classification data ───
    X_rows = []
    y_class = []
    valid_buildings = []

    for bd in building_data:
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
            y_class.append(bd['roof_type'])
            valid_buildings.append(bd)

    X = np.array(X_rows)
    y = np.array(y_class)

    print(f"\nDataset: {len(X)} buildings")
    print(f"Classes: { {c: int(np.sum(y == c)) for c in np.unique(y)} }")

    # ═══ STEP 1: Feature Selection ═══
    selector = FeatureSelector()
    selection_results = selector.analyze(X, y, ALL_FEATURES)

    # ═══ STEP 2: Train and Compare Classifiers ═══
    selected_features = selection_results['selected_features']
    selected_indices = selection_results['selected_indices']
    X_reduced = X[:, selected_indices]

    print(f"\n{'='*70}")
    print(f"CLASSIFIER COMPARISON WITH CROSS-VALIDATION ({len(selected_features)} features)")
    print(f"{'='*70}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y_enc, test_size=0.15, stratify=y_enc, random_state=42
    )

    # Scale data for models that need it (KNN, MLP)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ─── Define models with hyperparameter grids for GridSearchCV ───
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifier_configs = {
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced'],
            },
            'needs_scaling': False,
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'class_weight': ['balanced'],
            },
            'needs_scaling': False,
        },
        'Gradient Boosted Trees': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1],
            },
            'needs_scaling': False,
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
            },
            'needs_scaling': True,
        },
        'MLP': {
            'model': MLPClassifier(
                solver='adam', learning_rate='adaptive',
                early_stopping=True, max_iter=500, random_state=42
            ),
            'params': {
                'hidden_layer_sizes': [(64, 32), (128, 64, 32)],
                'learning_rate_init': [0.001, 0.005],
            },
            'needs_scaling': True,
        },
    }

    print(f"\n  Models: {', '.join(classifier_configs.keys())}")
    print(f"  Cross-validation: {cv.n_splits}-fold stratified")
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

    best_clf = None
    best_clf_name = None
    best_clf_f1 = -1
    all_classifier_results = {}

    for name, config in classifier_configs.items():
        print(f"\n  --- {name} ---")

        X_tr = X_train_s if config['needs_scaling'] else X_train
        X_te = X_test_s if config['needs_scaling'] else X_test

        grid = GridSearchCV(
            config['model'], config['params'],
            cv=cv, scoring='f1_weighted', n_jobs=-1, refit=True
        )
        grid.fit(X_tr, y_train)

        # Evaluate best model on held-out test set
        y_pred = grid.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Cross-validation score of best model
        cv_mean = grid.best_score_
        cv_std = grid.cv_results_['std_test_score'][grid.best_index_]

        print(f"    Best params: {grid.best_params_}")
        print(f"    CV F1: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"    Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}")

        all_classifier_results[name] = {
            'accuracy': float(acc),
            'f1': float(f1),
            'cv_f1_mean': float(cv_mean),
            'cv_f1_std': float(cv_std),
            'best_params': {k: str(v) for k, v in grid.best_params_.items()},
        }

        if f1 > best_clf_f1:
            best_clf_f1 = f1
            best_clf_name = name
            best_clf = grid.best_estimator_
            best_clf_needs_scaling = config['needs_scaling']

    print(f"\n  ★ Best classifier: {best_clf_name} (Test F1={best_clf_f1:.4f})")

    # ─── Final evaluation of best classifier ───
    X_te_final = X_test_s if best_clf_needs_scaling else X_test
    y_pred_final = best_clf.predict(X_te_final)
    acc_final = accuracy_score(y_test, y_pred_final)
    f1_final = f1_score(y_test, y_pred_final, average='weighted')

    print(f"\n{classification_report(y_test, y_pred_final, target_names=le.classes_, zero_division=0)}")

    cm = confusion_matrix(y_test, y_pred_final)
    print(f"Confusion Matrix:")
    print(f"  Predicted →  {' '.join(f'{c:>8}' for c in le.classes_)}")
    for i, row in enumerate(cm):
        print(f"  {le.classes_[i]:<10}  {' '.join(f'{v:>8}' for v in row)}")

    # If best model needs scaling, wrap it in a pipeline for clean predict() later
    if best_clf_needs_scaling:
        clf = Pipeline([('scaler', scaler), ('clf', best_clf)])
    else:
        clf = best_clf

    # ═══ STEP 3: Train Roof Vertex Predictor ═══
    print(f"\n{'='*70}")
    print("TRAINING NEURAL NETWORK ROOF VERTEX PREDICTOR")
    print(f"{'='*70}")

    vertex_predictor = RoofVertexPredictor()
    datasets = vertex_predictor.prepare_vertex_data(valid_buildings)
    vertex_predictor.train(datasets)

    # ═══ Save Results ═══
    results = {
        'feature_selection': {
            'selected_features': selected_features,
            'dropped_features': selection_results['dropped_features'],
            'full_accuracy': selection_results['full_accuracy'],
            'reduced_accuracy': selection_results['reduced_accuracy'],
        },
        'classifier_comparison': all_classifier_results,
        'best_classifier': {
            'model': best_clf_name,
            'n_features': len(selected_features),
            'features_used': selected_features,
            'accuracy': float(acc_final),
            'f1': float(f1_final),
            'confusion_matrix': cm.tolist(),
        },
        'vertex_prediction': vertex_predictor.results,
    }

    results_path = os.path.join(output_dir, "ml_results_revised.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return {
        'classifier': clf,
        'label_encoder': le,
        'feature_selector': selector,
        'vertex_predictor': vertex_predictor,
        'selected_features': selected_features,
        'results': results
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main: Run with synthetic data for testing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from cityjson_parser import CityJSONParser
    from feature_extraction import process_tile, filter_buildings

    # Load your tile(s) — update the path(s) to your actual files
    filepaths = [
        "data/9-564-628.city.json",
        "data/9-564-632.city.json",
        "data/9-564-636.city.json",
        "data/9-568-628.city.json",
        "data/9-568-632.city.json",
        "data/9-568-636.city.json",
        "data/9-572-628.city.json",
        "data/9-572-632.city.json",
        "data/9-572-636.city.json"
    ]

    all_building_data = []
    for fp in filepaths:
        parser = CityJSONParser(fp)
        parser.summary()
        data = process_tile(parser)
        for d in data:
            d['_parser'] = parser
        all_building_data.extend(data)

    # Filter to simple buildings
    building_data = filter_buildings(all_building_data)

    # Run the revised pipeline
    results = run_revised_pipeline(building_data)