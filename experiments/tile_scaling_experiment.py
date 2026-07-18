"""
Tile Scaling Experiment

Tests how the number of training tiles affects classification accuracy and
reconstruction quality, evaluated against a fixed set of 13 held-out tiles
that are never used for training. Training set grows in four steps: the
original 3x3 grid (9 tiles), +6 adjacent tiles (15), +7 more (22), then all
28 available training tiles.

Usage:
    Run from the repository root (so the relative ``data/`` paths resolve):
        python experiments/tile_scaling_experiment.py
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
from ml_models_revised import run_revised_pipeline
from roof_construction import run_construction_pipeline


# Fixed evaluation tiles (13) — never used in training
EVAL_TILES = [
    "data/8-576-624.city.json",
    "data/9-560-628.city.json",
    "data/9-560-644.city.json",
    "data/9-568-640.city.json",
    "data/9-572-624.city.json",
    "data/9-576-640.city.json",
    "data/9-580-636.city.json",
    "data/9-580-644.city.json",
    "data/10-560-642.city.json",
    "data/10-562-640.city.json",
    "data/10-566-626.city.json",
    "data/10-572-642.city.json",
    "data/10-574-640.city.json",
]

# Training tile configurations (nested — each includes the previous)

# Config 9: Original compact 3x3 grid
TRAIN_9 = [
    "data/9-564-628.city.json", "data/9-564-632.city.json", "data/9-564-636.city.json",
    "data/9-568-628.city.json", "data/9-568-632.city.json", "data/9-568-636.city.json",
    "data/9-572-628.city.json", "data/9-572-632.city.json", "data/9-572-636.city.json",
]

# Config 15: 9 + 6 adjacent tiles (expanding the grid)
TRAIN_15 = TRAIN_9 + [
    "data/9-564-640.city.json", "data/9-564-644.city.json",
    "data/9-568-624.city.json", "data/9-568-644.city.json",
    "data/9-560-624.city.json", "data/9-560-632.city.json",
]

# Config 22: 15 + 7 more tiles (wider geographic coverage)
TRAIN_22 = TRAIN_15 + [
    "data/9-560-636.city.json", "data/9-572-644.city.json",
    "data/9-576-632.city.json", "data/9-576-636.city.json",
    "data/9-576-644.city.json", "data/9-580-632.city.json",
    "data/9-580-640.city.json",
]

# Config 28: All available training tiles (maximum data)
# Note: 9-580-644 excluded — it's in the eval set
TRAIN_28 = TRAIN_22 + [
    "data/10-560-640.city.json", "data/10-562-642.city.json",
    "data/10-564-624.city.json", "data/10-564-626.city.json",
    "data/10-566-624.city.json", "data/10-574-642.city.json",
]

CONFIGS = [
    ("9 tiles (compact grid)", TRAIN_9),
    ("15 tiles (adjacent)", TRAIN_15),
    ("22 tiles (wider)", TRAIN_22),
    ("28 tiles (all available)", TRAIN_28),
]


# --- load and prepare training data ---

def load_training_data(tile_paths):
    """Load tiles, extract features, filter, compute neighbor features."""
    all_data = []
    parsers = []
    existing = [f for f in tile_paths if os.path.exists(f)]
    print(f"  Loading {len(existing)} tiles...")

    for fp in existing:
        parser = CityJSONParser(fp)
        data = process_tile(parser)
        for d in data:
            d['_parser'] = parser
        all_data.extend(data)
        parsers.append(parser)

    filtered = filter_buildings(all_data)
    filtered = compute_neighbor_features(filtered, radius=50.0, use_ground_truth=True)

    # Count roof types
    type_counts = {}
    for bd in filtered:
        t = bd['roof_type']
        type_counts[t] = type_counts.get(t, 0) + 1

    return filtered, parsers, type_counts


# --- evaluate on held-out tiles using evaluation.py ---

def evaluate_on_tiles(eval_tile_paths, classifier, vertex_predictor,
                      label_encoder, selected_features):
    """
    Run construction + evaluation on each eval tile using the proper
    evaluation module. Returns combined per-building results.
    """
    from evaluation import evaluate_tile, compute_aggregate_metrics

    all_per_building = []
    tile_summaries = []

    existing_eval = [f for f in eval_tile_paths if os.path.exists(f)]

    for fp in existing_eval:
        parser = CityJSONParser(fp)
        tile_name = os.path.basename(fp).replace('.city.json', '')

        # Run construction pipeline (suppress output for cleaner logs)
        result = run_construction_pipeline(
            parser=parser,
            classifier=classifier,
            vertex_predictor=vertex_predictor,
            label_encoder=label_encoder,
            selected_features=selected_features,
            output_path=None
        )

        buildings = result.get('buildings', [])
        if not buildings:
            print(f"    {tile_name}: no buildings constructed")
            continue

        # Evaluate against ground truth
        eval_result = evaluate_tile(parser, buildings)
        n_eval = eval_result['n_evaluated']
        metrics = eval_result['metrics']

        acc = metrics['classification']['accuracy']
        print(f"    {tile_name}: {n_eval} buildings, accuracy={acc:.3f}")

        all_per_building.extend(eval_result['per_building'])
        tile_summaries.append({
            'name': tile_name,
            'n_evaluated': n_eval,
            'accuracy': acc,
        })

    # Compute combined metrics across all tiles
    combined_metrics = compute_aggregate_metrics(all_per_building)

    return {
        'per_building': all_per_building,
        'metrics': combined_metrics,
        'tile_summaries': tile_summaries,
    }


def summarize_results(eval_output):
    """Extract key numbers from evaluation output for the comparison table."""
    metrics = eval_output['metrics']

    summary = {
        'overall_accuracy': metrics['classification']['accuracy'],
        'total_buildings': metrics['classification']['n_total'],
        'per_type': {}
    }

    for rtype in ['flat', 'gabled', 'hipped']:
        rt = metrics.get(rtype, {})
        rh = rt.get('ridge_height', {})
        vol = rt.get('volume', {})

        summary['per_type'][rtype] = {
            'count': rt.get('n_buildings', 0),
            'ridge_mae': rh.get('mae'),
            'ridge_median': rh.get('median_error'),
            'ridge_rmse': rh.get('rmse'),
            'ridge_bias': rh.get('mean_bias'),
            'vol_mae': vol.get('mae'),
            'vol_relative': vol.get('mean_relative_error'),
        }

    return summary


# --- main experiment ---

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    all_summaries = []

    for config_name, train_tiles in CONFIGS:
        print(f"\n{'█'*70}")
        print(f"  TRAINING CONFIG: {config_name}")
        print(f"{'█'*70}")

        start_time = time.time()

        # Load and prepare training data
        filtered, parsers, type_counts = load_training_data(train_tiles)
        print(f"  Training buildings: {len(filtered)}")
        print(f"  Roof types: {type_counts}")

        # Train models
        print(f"\n  Training models...")
        pipeline_result = run_revised_pipeline(filtered, output_dir="output")

        classifier = pipeline_result['classifier']
        label_encoder = pipeline_result['label_encoder']
        vertex_predictor = pipeline_result['vertex_predictor']
        selected_features = pipeline_result['selected_features']

        # Evaluate on fixed eval tiles
        print(f"\n  Evaluating on {len(EVAL_TILES)} held-out tiles...")
        eval_results = evaluate_on_tiles(
            EVAL_TILES, classifier, vertex_predictor,
            label_encoder, selected_features
        )

        summary = summarize_results(eval_results)
        summary['config'] = config_name
        summary['n_train_tiles'] = len([f for f in train_tiles if os.path.exists(f)])
        summary['n_train_buildings'] = len(filtered)
        summary['train_type_counts'] = type_counts
        summary['time_seconds'] = time.time() - start_time

        all_summaries.append(summary)

        # Print intermediate results
        print(f"\n  ─── Results for {config_name} ───")
        print(f"  Overall accuracy: {summary['overall_accuracy']:.4f}")
        for rtype in ['flat', 'gabled', 'hipped']:
            s = summary['per_type'][rtype]
            ridge = f"{s['ridge_mae']:.3f}m" if s['ridge_mae'] is not None else "N/A"
            print(f"  {rtype}: {s['count']} buildings, ridge MAE={ridge}")
        print(f"  Time: {summary['time_seconds']:.0f}s")

    # Final comparison table
    print(f"\n\n{'█'*75}")
    print(f"  TILE SCALING EXPERIMENT — FINAL COMPARISON")
    print(f"{'█'*75}")
    print(f"\n  Fixed evaluation set: {len(EVAL_TILES)} tiles")

    # Header
    print(f"\n  {'Config':<28} {'Train':>6} {'Eval':>6} {'Class':>7} "
          f"{'Flat':>7} {'Gabled':>7} {'Hipped':>7}")
    print(f"  {'':28} {'bldgs':>6} {'bldgs':>6} {'Acc':>7} "
          f"{'MAE':>7} {'MAE':>7} {'MAE':>7}")
    print(f"  {'─'*28} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for s in all_summaries:
        flat_mae = f"{s['per_type']['flat']['ridge_mae']:.3f}" if s['per_type']['flat']['ridge_mae'] else "N/A"
        gab_mae = f"{s['per_type']['gabled']['ridge_mae']:.3f}" if s['per_type']['gabled']['ridge_mae'] else "N/A"
        hip_mae = f"{s['per_type']['hipped']['ridge_mae']:.3f}" if s['per_type']['hipped']['ridge_mae'] else "N/A"

        print(f"  {s['config']:<28} {s['n_train_buildings']:>6} "
              f"{s['total_buildings']:>6} {s['overall_accuracy']:>6.1%} "
              f"{flat_mae:>7} {gab_mae:>7} {hip_mae:>7}")

    # Detailed per-type breakdown
    print(f"\n  {'─'*75}")
    print(f"  Detailed per-type metrics:")
    print(f"  {'─'*75}")

    for s in all_summaries:
        print(f"\n  {s['config']}:")
        for rtype in ['flat', 'gabled', 'hipped']:
            d = s['per_type'][rtype]
            if d['ridge_mae'] is not None:
                ridge_str = f"ridge MAE={d['ridge_mae']:.3f}m, median={d['ridge_median']:.3f}m"
                vol_str = f"vol MAE={d['vol_mae']:.1f}m³ ({d['vol_relative']*100:.1f}%)" if d['vol_mae'] else ""
                bias_str = f"bias={d['ridge_bias']:+.3f}m" if d.get('ridge_bias') is not None else ""
                print(f"    {rtype:<8}: n={d['count']:>4}, {ridge_str}, {bias_str}, {vol_str}")
            else:
                print(f"    {rtype:<8}: n={d['count']:>4}, no data")

    # Training data composition
    print(f"\n  {'─'*75}")
    print(f"  Training data composition:")
    print(f"  {'─'*75}")
    for s in all_summaries:
        tc = s.get('train_type_counts', {})
        total = sum(tc.values())
        hipped_pct = tc.get('hipped', 0) / total * 100 if total > 0 else 0
        print(f"  {s['config']:<28}: {total} total, "
              f"flat={tc.get('flat',0)}, gabled={tc.get('gabled',0)}, "
              f"hipped={tc.get('hipped',0)} ({hipped_pct:.1f}%)")

    # Save results
    results_path = "output/tile_scaling_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            """json.dump `default` hook: cast numpy scalars/arrays to plain Python."""
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(all_summaries, f, indent=2, default=convert)
    print(f"\n  Results saved to {results_path}")