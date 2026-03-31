"""
=============================================================================
Run Full Pipeline: Train + Construct LOD2.2
=============================================================================
Usage:
    python run_full_pipeline.py tile1.city.json tile2.city.json ...

This script:
  1. Loads CityJSON tiles
  2. Extracts features and ground truth
  3. Trains classifier (Stage 1) and vertex predictor (Stage 2)
  4. Runs geometric construction on all buildings
  5. Exports LOD2.2 CityJSON output
=============================================================================
"""

import sys
import os
import json
import warnings
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cityjson_parser import CityJSONParser
from feature_extraction import process_tile, filter_buildings
from ml_models_revised import (
    FeatureSelector, RoofVertexPredictor,
    run_revised_pipeline
)
from roof_construction import run_construction_pipeline


def main(filepaths: list):
    # ─── Step 1: Load tiles and prepare data ───
    print(f"\n{'█'*70}")
    print(f"  STEP 1: LOADING DATA")
    print(f"{'█'*70}")

    parsers = []
    all_building_data = []

    for fp in filepaths:
        parser = CityJSONParser(fp)
        parser.summary()
        data = process_tile(parser)
        for d in data:
            d['_parser'] = parser
        all_building_data.extend(data)
        parsers.append(parser)

    filtered = filter_buildings(all_building_data)
    print(f"\nTotal buildings for training: {len(filtered)}")

    # ─── Step 2: Train models ───
    print(f"\n{'█'*70}")
    print(f"  STEP 2: TRAINING MODELS")
    print(f"{'█'*70}")

    pipeline_result = run_revised_pipeline(filtered, output_dir="output")

    classifier = pipeline_result['classifier']
    label_encoder = pipeline_result['label_encoder']
    vertex_predictor = pipeline_result['vertex_predictor']
    selected_features = pipeline_result['selected_features']

    # ─── Step 3: Run construction on each tile ───
    print(f"\n{'█'*70}")
    print(f"  STEP 3: CONSTRUCTING LOD2.2 BUILDINGS")
    print(f"{'█'*70}")

    os.makedirs("output", exist_ok=True)

    for i, parser in enumerate(parsers):
        tile_name = parser.filename.replace('.city.json', '').replace('.json', '')
        output_path = f"output/{tile_name}_lod22.city.json"

        print(f"\n--- Processing tile: {parser.filename} ---")

        result = run_construction_pipeline(
            parser=parser,
            classifier=classifier,
            vertex_predictor=vertex_predictor,
            label_encoder=label_encoder,
            selected_features=selected_features,
            output_path=output_path
        )

    print(f"\n{'█'*70}")
    print(f"  DONE — Check output/ for LOD2.2 CityJSON files")
    print(f"{'█'*70}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full_pipeline.py <tile1.city.json> [tile2.city.json ...]")
        print("\nExample:")
        print("  python run_full_pipeline.py data/9-564-628.city.json")
        print("  python run_full_pipeline.py data/*.city.json")
        sys.exit(1)

    main(sys.argv[1:])