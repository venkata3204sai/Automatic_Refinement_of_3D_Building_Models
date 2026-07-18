"""
Quick sanity check: load a single tile, run feature extraction + filtering,
and print the building counts and roof-type class distribution.

Handy for confirming the parser and feature pipeline work on a new tile
before running the full pipeline.

Usage:
    Run from the repository root (so the relative ``data/`` path resolves):
        python scripts/testfile.py
"""

import os
import sys
from collections import Counter

# This script lives in scripts/; add the repository root (its parent) to the
# import path so the pipeline modules can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cityjson_parser import CityJSONParser
from feature_extraction import process_tile, filter_buildings

parser = CityJSONParser('data/10-572-640.city.json')
buildings = process_tile(parser)
filtered = filter_buildings(buildings)

print(f'Total buildings: {len(buildings)}')
print(f'After filtering: {len(filtered)}')

distribution = Counter(b.get('roof_type', 'Unknown') for b in filtered)
print(f'Class distribution: {dict(distribution)}')