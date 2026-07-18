"""
Evaluation: Geometric Accuracy Metrics

Compares reconstructed LOD2.2 buildings against 3DBAG ground truth on three
fronts — ridge-height error (MAE/RMSE/per-building, in meters), volume
difference (via the divergence theorem on closed meshes), and classification
accuracy (predicted vs. ground-truth roof type). All three are reported both
overall and broken down by roof type.

Usage:
    python evaluation.py output/9-564-628_lod22.city.json data/9-564-628.city.json
    python evaluation.py --all-tiles output/ data/

Or import directly:
    from evaluation import evaluate_tile, print_evaluation_report
"""

import math
import json
import os
import sys
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --- volume computation ---

def _signed_volume_of_triangle(v1, v2, v3):
    """
    Compute the signed volume of a tetrahedron formed by the origin
    and a triangle (v1, v2, v3). Used for computing volume of a closed
    mesh via the divergence theorem.

    volume = (1/6) * v1 · (v2 × v3)
    """
    cross_x = v2[1] * v3[2] - v2[2] * v3[1]
    cross_y = v2[2] * v3[0] - v2[0] * v3[2]
    cross_z = v2[0] * v3[1] - v2[1] * v3[0]

    return (v1[0] * cross_x + v1[1] * cross_y + v1[2] * cross_z) / 6.0


def compute_solid_volume(vertices: List[Tuple], faces: List[List[int]]) -> float:
    """
    Compute the volume of a closed 3D solid using the divergence theorem.

    Triangulates each face (fan triangulation from first vertex) and
    sums signed tetrahedron volumes. The solid must be closed (watertight)
    with consistent face winding for an accurate result.

    Args:
        vertices: list of (x, y, z) vertex coordinates
        faces: list of faces, each face is a list of vertex indices

    Returns:
        Absolute volume in cubic meters
    """
    total = 0.0
    for face in faces:
        if len(face) < 3:
            continue
        # Fan triangulation: v0 → v1 → v2, v0 → v2 → v3, etc.
        v0 = vertices[face[0]]
        for i in range(1, len(face) - 1):
            v1 = vertices[face[i]]
            v2 = vertices[face[i + 1]]
            total += _signed_volume_of_triangle(v0, v1, v2)

    return abs(total)


# --- ground-truth extraction from LOD2.2 ---

def extract_gt_ridge_height(parser, lod22_geometry: dict) -> Optional[float]:
    """
    Extract the ridge height (maximum roof Z) from ground-truth LOD2.2.

    Returns the absolute Z coordinate of the highest roof vertex,
    or None if no roof surfaces found.
    """
    labels = parser.get_semantic_labels(lod22_geometry)
    faces = parser.get_shell_faces(lod22_geometry)
    vertices = parser.vertices

    if not labels or not faces:
        return None

    roof_z_values = []
    for face, label in zip(faces, labels):
        if label == 'RoofSurface':
            for idx in face:
                roof_z_values.append(vertices[idx][2])

    if not roof_z_values:
        return None

    return max(roof_z_values)


def extract_gt_eave_height(parser, lod22_geometry: dict) -> Optional[float]:
    """
    Extract the eave height (minimum roof Z) from ground-truth LOD2.2.
    """
    labels = parser.get_semantic_labels(lod22_geometry)
    faces = parser.get_shell_faces(lod22_geometry)
    vertices = parser.vertices

    if not labels or not faces:
        return None

    roof_z_values = []
    for face, label in zip(faces, labels):
        if label == 'RoofSurface':
            for idx in face:
                roof_z_values.append(vertices[idx][2])

    if not roof_z_values:
        return None

    return min(roof_z_values)


def _translate_to_local(vertices: List[Tuple]) -> List[Tuple]:
    """
    Translate vertices to a local coordinate system centered at their centroid.
    This is essential for accurate volume computation via the divergence theorem,
    which computes signed tetrahedra from the origin — if vertices are in
    absolute coordinates (e.g., Dutch RD with x~85000), the origin is far away
    and numerical errors dominate.
    """
    if not vertices:
        return vertices
    n = len(vertices)
    cx = sum(v[0] for v in vertices) / n
    cy = sum(v[1] for v in vertices) / n
    cz = sum(v[2] for v in vertices) / n
    return [(v[0] - cx, v[1] - cy, v[2] - cz) for v in vertices]


def extract_gt_volume(parser, lod22_geometry: dict) -> Optional[float]:
    """
    Compute the volume of the ground-truth LOD2.2 solid.
    Translates to local coordinates before volume computation.
    """
    faces = parser.get_shell_faces(lod22_geometry)
    vertices = parser.vertices

    if not faces:
        return None

    # Deduplicate vertices while preserving index mapping
    vert_map = {}
    unique_verts = []
    mapped_faces = []

    for face in faces:
        mapped_face = []
        for idx in face:
            v = tuple(vertices[idx])
            if v not in vert_map:
                vert_map[v] = len(unique_verts)
                unique_verts.append(v)
            mapped_face.append(vert_map[v])
        mapped_faces.append(mapped_face)

    # Translate to local coordinates for accurate volume computation
    local_verts = _translate_to_local(unique_verts)

    return compute_solid_volume(local_verts, mapped_faces)


def compute_predicted_volume(building: dict) -> Optional[float]:
    """
    Compute the volume of a predicted LOD2.2 building.
    Translates to local coordinates before volume computation.

    Args:
        building: dict from buildings_for_output with 'ground_verts',
                  'eave_verts', 'ridge_v1', 'ridge_v2', 'roof_type'
    """
    from roof_construction import (
        construct_flat_roof, construct_gabled_roof, construct_hipped_roof
    )

    roof_type = building['roof_type']
    ground_verts = building['ground_verts']
    eave_verts = building['eave_verts']
    ridge_v1 = building.get('ridge_v1')
    ridge_v2 = building.get('ridge_v2')

    if roof_type == 'flat':
        result = construct_flat_roof(ground_verts, eave_verts)
    elif roof_type == 'gabled' and ridge_v1 and ridge_v2:
        result = construct_gabled_roof(ground_verts, eave_verts, ridge_v1, ridge_v2)
    elif roof_type == 'hipped' and ridge_v1 and ridge_v2:
        result = construct_hipped_roof(ground_verts, eave_verts, ridge_v1, ridge_v2)
    else:
        result = construct_flat_roof(ground_verts, eave_verts)

    # Translate to local coordinates for accurate volume computation
    local_verts = _translate_to_local(result['vertices'])

    return compute_solid_volume(local_verts, result['faces'])


# --- per-tile evaluation ---

def evaluate_tile(parser, buildings: List[dict]) -> dict:
    """
    Evaluate predicted LOD2.2 buildings against ground-truth LOD2.2.

    For each building in `buildings` (from the construction pipeline output),
    finds the matching ground-truth LOD2.2 geometry from the parser and
    computes error metrics.

    Args:
        parser: CityJSONParser with the original tile
        buildings: list of dicts from run_construction_pipeline()['buildings']

    Returns:
        dict with per-building results and aggregate metrics
    """
    from feature_extraction import derive_roof_type

    paired = parser.pair_lod13_lod22()
    # Build lookup: part_id → paired entry
    gt_lookup = {p['part_id']: p for p in paired}

    results = []

    for bld in buildings:
        part_id = bld['part_id']
        predicted_type = bld['roof_type']

        gt = gt_lookup.get(part_id)
        if gt is None:
            continue

        lod22 = gt['lod22']

        # Ground-truth roof type
        gt_roof_type = derive_roof_type(parser, lod22)
        if gt_roof_type is None or gt_roof_type == 'complex':
            continue

        entry = {
            'building_id': bld['building_id'],
            'part_id': part_id,
            'predicted_type': predicted_type,
            'pass1_type': bld.get('pass1_roof_type'),
            'gt_type': gt_roof_type,
            'type_correct': predicted_type == gt_roof_type,
        }

        # Ridge height error
        gt_ridge_z = extract_gt_ridge_height(parser, lod22)
        gt_eave_z = extract_gt_eave_height(parser, lod22)

        if predicted_type in ('gabled', 'hipped') and bld.get('ridge_v1'):
            predicted_ridge_z = bld['ridge_v1'][2]  # Both ridge verts have same Z
        elif predicted_type == 'flat':
            predicted_ridge_z = bld['eave_verts'][0][2]  # Top of flat roof
        else:
            predicted_ridge_z = None

        if gt_ridge_z is not None and predicted_ridge_z is not None:
            entry['ridge_height_error'] = predicted_ridge_z - gt_ridge_z
            entry['ridge_height_abs_error'] = abs(predicted_ridge_z - gt_ridge_z)
            entry['gt_ridge_z'] = gt_ridge_z
            entry['predicted_ridge_z'] = predicted_ridge_z

        if gt_eave_z is not None:
            entry['gt_eave_z'] = gt_eave_z

        # Volume difference
        gt_vol = extract_gt_volume(parser, lod22)
        pred_vol = compute_predicted_volume(bld)

        if gt_vol is not None and pred_vol is not None and gt_vol > 0:
            entry['gt_volume'] = gt_vol
            entry['predicted_volume'] = pred_vol
            entry['volume_difference'] = pred_vol - gt_vol
            entry['volume_abs_difference'] = abs(pred_vol - gt_vol)
            entry['volume_relative_error'] = abs(pred_vol - gt_vol) / gt_vol

        results.append(entry)

    # Aggregate metrics
    metrics = compute_aggregate_metrics(results)

    return {
        'per_building': results,
        'metrics': metrics,
        'n_evaluated': len(results),
    }


def compute_aggregate_metrics(results: List[dict]) -> dict:
    """
    Compute aggregate metrics from per-building evaluation results.
    Reports overall and per roof type, including error tolerance bands.
    """
    metrics = {}

    # Overall classification
    n_total = len(results)
    n_correct = sum(1 for r in results if r['type_correct'])
    metrics['classification'] = {
        'n_total': n_total,
        'n_correct': n_correct,
        'accuracy': n_correct / n_total if n_total > 0 else 0,
    }

    # Per roof-type metrics
    for roof_type in ['flat', 'gabled', 'hipped', 'all']:
        if roof_type == 'all':
            subset = results
        else:
            # Use ground-truth type for grouping
            subset = [r for r in results if r['gt_type'] == roof_type]

        if not subset:
            continue

        rt_metrics = {'n_buildings': len(subset)}

        # Ridge height error
        ridge_errors = [r['ridge_height_abs_error'] for r in subset
                        if 'ridge_height_abs_error' in r]
        if ridge_errors:
            rt_metrics['ridge_height'] = {
                'n_evaluated': len(ridge_errors),
                'mae': sum(ridge_errors) / len(ridge_errors),
                'rmse': math.sqrt(sum(e**2 for e in ridge_errors) / len(ridge_errors)),
                'max_error': max(ridge_errors),
                'median_error': sorted(ridge_errors)[len(ridge_errors) // 2],
            }
            # Signed errors for bias analysis
            signed_errors = [r['ridge_height_error'] for r in subset
                             if 'ridge_height_error' in r]
            if signed_errors:
                rt_metrics['ridge_height']['mean_bias'] = sum(signed_errors) / len(signed_errors)

            # Error tolerance bands
            n_ridge = len(ridge_errors)
            rt_metrics['ridge_height']['tolerance'] = {
                'within_0.25m': sum(1 for e in ridge_errors if e <= 0.25) / n_ridge,
                'within_0.50m': sum(1 for e in ridge_errors if e <= 0.50) / n_ridge,
                'within_1.00m': sum(1 for e in ridge_errors if e <= 1.00) / n_ridge,
                'within_2.00m': sum(1 for e in ridge_errors if e <= 2.00) / n_ridge,
            }

        # Volume difference
        vol_errors = [r['volume_abs_difference'] for r in subset
                      if 'volume_abs_difference' in r]
        vol_rel_errors = [r['volume_relative_error'] for r in subset
                          if 'volume_relative_error' in r]
        if vol_errors:
            rt_metrics['volume'] = {
                'n_evaluated': len(vol_errors),
                'mae': sum(vol_errors) / len(vol_errors),
                'rmse': math.sqrt(sum(e**2 for e in vol_errors) / len(vol_errors)),
                'max_error': max(vol_errors),
                'mean_relative_error': sum(vol_rel_errors) / len(vol_rel_errors) if vol_rel_errors else 0,
            }

            # Volume tolerance bands
            if vol_rel_errors:
                n_vol = len(vol_rel_errors)
                rt_metrics['volume']['tolerance'] = {
                    'within_5%': sum(1 for e in vol_rel_errors if e <= 0.05) / n_vol,
                    'within_10%': sum(1 for e in vol_rel_errors if e <= 0.10) / n_vol,
                    'within_20%': sum(1 for e in vol_rel_errors if e <= 0.20) / n_vol,
                    'within_50%': sum(1 for e in vol_rel_errors if e <= 0.50) / n_vol,
                }

        # Classification error severity
        type_correct = sum(1 for r in subset if r['type_correct'])
        rt_metrics['classification'] = {
            'correct': type_correct,
            'total': len(subset),
            'accuracy': type_correct / len(subset) if subset else 0,
        }

        metrics[roof_type] = rt_metrics

    return metrics


# --- reporting ---

def print_evaluation_report(eval_result: dict, tile_name: str = ""):
    """Pretty-print evaluation results including error tolerance."""
    metrics = eval_result['metrics']
    n = eval_result['n_evaluated']

    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT{': ' + tile_name if tile_name else ''}")
    print(f"{'='*70}")
    print(f"Buildings evaluated: {n}")

    # Classification
    cls = metrics.get('classification', {})
    print(f"\nClassification Accuracy: {cls.get('accuracy', 0):.4f} "
          f"({cls.get('n_correct', 0)}/{cls.get('n_total', 0)})")

    # Per roof-type metrics
    for roof_type in ['all', 'flat', 'gabled', 'hipped']:
        rt = metrics.get(roof_type)
        if not rt:
            continue

        label = roof_type.upper() if roof_type != 'all' else 'ALL TYPES'
        print(f"\n--- {label} ({rt['n_buildings']} buildings) ---")

        if 'ridge_height' in rt:
            rh = rt['ridge_height']
            print(f"  Ridge Height Error:")
            print(f"    MAE:    {rh['mae']:.3f} m")
            print(f"    RMSE:   {rh['rmse']:.3f} m")
            print(f"    Median: {rh['median_error']:.3f} m")
            print(f"    Max:    {rh['max_error']:.3f} m")
            if 'mean_bias' in rh:
                bias = rh['mean_bias']
                direction = "over-predicts" if bias > 0 else "under-predicts"
                print(f"    Bias:   {bias:+.3f} m ({direction})")
            if 'tolerance' in rh:
                tol = rh['tolerance']
                print(f"    Error Tolerance:")
                print(f"      Within 0.25m: {tol['within_0.25m']:.1%}")
                print(f"      Within 0.50m: {tol['within_0.50m']:.1%}")
                print(f"      Within 1.00m: {tol['within_1.00m']:.1%}")
                print(f"      Within 2.00m: {tol['within_2.00m']:.1%}")

        if 'volume' in rt:
            vol = rt['volume']
            print(f"  Volume Difference:")
            print(f"    MAE:    {vol['mae']:.2f} m³")
            print(f"    RMSE:   {vol['rmse']:.2f} m³")
            print(f"    Max:    {vol['max_error']:.2f} m³")
            print(f"    Mean relative error: {vol['mean_relative_error']:.1%}")
            if 'tolerance' in vol:
                tol = vol['tolerance']
                print(f"    Volume Tolerance:")
                print(f"      Within  5%: {tol['within_5%']:.1%}")
                print(f"      Within 10%: {tol['within_10%']:.1%}")
                print(f"      Within 20%: {tol['within_20%']:.1%}")
                print(f"      Within 50%: {tol['within_50%']:.1%}")


def analyze_pass_flips(eval_result: dict):
    """
    Analyze which buildings changed classification between Pass 1 and Pass 2,
    and whether the change was a correction (moved toward ground truth) or
    corruption (moved away from ground truth).

    Requires buildings to have 'pass1_roof_type' stored (from roof_construction.py).
    """
    results = eval_result.get('per_building', [])

    flips = []
    for r in results:
        pass1 = r.get('pass1_type')
        pass2 = r.get('predicted_type')
        gt = r.get('gt_type')

        if pass1 is None or pass2 is None or gt is None:
            continue

        if pass1 != pass2:
            was_correct = (pass1 == gt)
            now_correct = (pass2 == gt)

            if was_correct and not now_correct:
                outcome = 'corrupted'  # was right, now wrong
            elif not was_correct and now_correct:
                outcome = 'corrected'  # was wrong, now right
            elif not was_correct and not now_correct:
                outcome = 'changed_still_wrong'  # was wrong, still wrong (different wrong)
            else:
                outcome = 'unknown'

            flips.append({
                'building_id': r.get('building_id', ''),
                'pass1': pass1,
                'pass2': pass2,
                'gt': gt,
                'outcome': outcome,
            })

    if not flips:
        print("\n  No Pass 1 → Pass 2 flip data available.")
        print("  (Run with updated roof_construction.py that stores pass1_roof_type)")
        return None

    # Summarize
    n_flips = len(flips)
    n_total = len([r for r in results if r.get('pass1_type') is not None])
    corrected = sum(1 for f in flips if f['outcome'] == 'corrected')
    corrupted = sum(1 for f in flips if f['outcome'] == 'corrupted')
    changed_wrong = sum(1 for f in flips if f['outcome'] == 'changed_still_wrong')

    print(f"\n{'='*70}")
    print(f"PASS 1 → PASS 2 FLIP ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total buildings with flip data: {n_total}")
    print(f"  Buildings that changed: {n_flips} ({n_flips/n_total*100:.1f}%)")
    print(f"")
    print(f"  Corrected (wrong → right): {corrected} ({corrected/n_flips*100:.1f}% of flips)")
    print(f"  Corrupted (right → wrong): {corrupted} ({corrupted/n_flips*100:.1f}% of flips)")
    print(f"  Changed but still wrong:   {changed_wrong} ({changed_wrong/n_flips*100:.1f}% of flips)")

    net_benefit = corrected - corrupted
    print(f"\n  Net benefit: {net_benefit:+d} buildings ({'+' if net_benefit > 0 else ''}{net_benefit/n_total*100:.1f}% of total)")

    # Breakdown by transition type
    transitions = {}
    for f in flips:
        key = f"{f['pass1']} → {f['pass2']}"
        if key not in transitions:
            transitions[key] = {'total': 0, 'corrected': 0, 'corrupted': 0, 'still_wrong': 0}
        transitions[key]['total'] += 1
        if f['outcome'] == 'corrected':
            transitions[key]['corrected'] += 1
        elif f['outcome'] == 'corrupted':
            transitions[key]['corrupted'] += 1
        elif f['outcome'] == 'changed_still_wrong':
            transitions[key]['still_wrong'] += 1

    print(f"\n  Transition breakdown:")
    print(f"  {'Transition':<25} {'Total':>6} {'Corrected':>10} {'Corrupted':>10} {'Still Wrong':>12}")
    print(f"  {'─'*25} {'─'*6} {'─'*10} {'─'*10} {'─'*12}")
    for key in sorted(transitions.keys()):
        t = transitions[key]
        print(f"  {key:<25} {t['total']:>6} {t['corrected']:>10} {t['corrupted']:>10} {t['still_wrong']:>12}")

    return {
        'n_total': n_total,
        'n_flips': n_flips,
        'corrected': corrected,
        'corrupted': corrupted,
        'changed_still_wrong': changed_wrong,
        'net_benefit': net_benefit,
        'transitions': transitions,
    }


def save_evaluation_results(eval_result: dict, output_path: str):
    """Save evaluation results to JSON file."""
    # Convert for JSON serialization
    serializable = {
        'n_evaluated': eval_result['n_evaluated'],
        'metrics': eval_result['metrics'],
        'per_building': eval_result['per_building'],
    }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nEvaluation results saved to {output_path}")


# --- standalone runner ---

if __name__ == "__main__":
    from cityjson_parser import CityJSONParser
    from feature_extraction import (
        process_tile, filter_buildings, compute_neighbor_features
    )
    from ml_models_revised import run_revised_pipeline
    from roof_construction import run_construction_pipeline

    if len(sys.argv) < 3:
        print("Usage: python evaluation.py <train_tiles...> -- <eval_tiles...>")
        print("\nExample:")
        print("  python evaluation.py data/9-564-628.city.json data/9-564-632.city.json -- data/9-572-636.city.json")
        print("\nTiles before '--' are used for training.")
        print("Tiles after '--' are used for evaluation.")
        sys.exit(1)

    # Split args into train and eval tiles
    args = sys.argv[1:]
    if '--' in args:
        split_idx = args.index('--')
        train_files = args[:split_idx]
        eval_files = args[split_idx + 1:]
    else:
        print("Error: Use '--' to separate training tiles from evaluation tiles.")
        print("Example: python evaluation.py train1.json train2.json -- eval1.json")
        sys.exit(1)

    if not train_files or not eval_files:
        print("Error: Need at least one training tile and one evaluation tile.")
        sys.exit(1)

    print(f"Training tiles: {len(train_files)}")
    print(f"Evaluation tiles: {len(eval_files)}")

    # Step 1: load and prepare training data
    print(f"\n{'█'*70}")
    print(f"  LOADING TRAINING DATA")
    print(f"{'█'*70}")

    train_parsers = []
    all_building_data = []
    for fp in train_files:
        parser = CityJSONParser(fp)
        parser.summary()
        data = process_tile(parser)
        for d in data:
            d['_parser'] = parser
        all_building_data.extend(data)
        train_parsers.append(parser)

    filtered = filter_buildings(all_building_data)
    filtered = compute_neighbor_features(filtered, radius=50.0, use_ground_truth=True)

    # Step 2: train models
    print(f"\n{'█'*70}")
    print(f"  TRAINING MODELS")
    print(f"{'█'*70}")

    pipeline_result = run_revised_pipeline(filtered, output_dir="output")

    classifier = pipeline_result['classifier']
    label_encoder = pipeline_result['label_encoder']
    vertex_predictor = pipeline_result['vertex_predictor']
    selected_features = pipeline_result['selected_features']

    # Step 3: run construction + evaluation on eval tiles
    print(f"\n{'█'*70}")
    print(f"  EVALUATING ON HELD-OUT TILES")
    print(f"{'█'*70}")

    os.makedirs("output", exist_ok=True)

    all_eval_results = []
    for fp in eval_files:
        eval_parser = CityJSONParser(fp)
        eval_parser.summary()
        tile_name = eval_parser.filename.replace('.city.json', '')
        output_path = f"output/{tile_name}_lod22.city.json"

        print(f"\n--- Constructing: {eval_parser.filename} ---")
        construction_result = run_construction_pipeline(
            parser=eval_parser,
            classifier=classifier,
            vertex_predictor=vertex_predictor,
            label_encoder=label_encoder,
            selected_features=selected_features,
            output_path=output_path,
        )

        print(f"\n--- Evaluating: {eval_parser.filename} ---")
        eval_result = evaluate_tile(eval_parser, construction_result['buildings'])
        print_evaluation_report(eval_result, tile_name)

        eval_path = f"output/{tile_name}_evaluation.json"
        save_evaluation_results(eval_result, eval_path)

        all_eval_results.extend(eval_result['per_building'])

    # Overall evaluation across all eval tiles
    if len(eval_files) > 1:
        print(f"\n{'█'*70}")
        print(f"  COMBINED EVALUATION ({len(eval_files)} tiles)")
        print(f"{'█'*70}")
        combined_metrics = compute_aggregate_metrics(all_eval_results)
        combined = {
            'n_evaluated': len(all_eval_results),
            'metrics': combined_metrics,
            'per_building': all_eval_results,
        }
        print_evaluation_report(combined, "COMBINED")

        # Pass 1 -> Pass 2 flip analysis
        flip_result = analyze_pass_flips(combined)

        save_evaluation_results(combined, "output/evaluation_combined.json")

    print(f"\n{'█'*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'█'*70}")