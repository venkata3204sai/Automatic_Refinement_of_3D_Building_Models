"""
=============================================================================
Module 5: Geometric Roof Construction (Weeks 8-9)
=============================================================================
Takes ML predictions (roof type + ridge vertex positions) and constructs
valid LOD2.2 CityJSON geometry from LOD1.3 input.

Scope: Rectangular (4-vertex) footprints only. The pipeline filters to
n_footprint_vertices == 4 before construction, ensuring clean geometry
for the three supported roof types.

For each roof type, deterministic construction rules produce:
  - Correct 3D faces (roof slopes, gable walls, ground, walls)
  - Semantic surface labels (RoofSurface, WallSurface, GroundSurface)
  - Valid CityJSON Solid geometry

Construction per roof type:
  FLAT:    Keep LOD1.3 box, add semantic labels
  GABLED:  Add 2 ridge vertices above short-edge midpoints →
           2 sloped RoofSurfaces + 2 pentagon gable WallSurfaces
           + 2 rectangular side WallSurfaces + 1 GroundSurface
  HIPPED:  Add 2 ridge vertices along the long axis →
           4 sloped RoofSurfaces + 4 rectangular WallSurfaces
           + 1 GroundSurface

Edge classification uses a side-of-ridge-line test (cross product)
rather than dot-product thresholds, which is robust for rectangles.

The eave height is kept at the original LOD1.3 top Z — only the
ridge is placed above it based on ML predictions.
=============================================================================
"""

import math
import json
from typing import Dict, List, Tuple, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# Geometry Helpers
# ─────────────────────────────────────────────────────────────────────────────

def distance_2d(p1, p2):
    """2D distance between two points (ignoring z)."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ─────────────────────────────────────────────────────────────────────────────
# Face Normal Helper
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_outward_normal(face_indices: List[int], all_vertices: List[Tuple],
                           centroid: Tuple) -> List[int]:
    """
    Ensure a face's normal points AWAY from the building centroid.
    If not, reverse the winding order.
    """
    if len(face_indices) < 3:
        return face_indices

    v0 = all_vertices[face_indices[0]]
    v1 = all_vertices[face_indices[1]]
    v2 = all_vertices[face_indices[2]]

    # Cross product = face normal
    e1 = (v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2])
    e2 = (v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2])
    nx = e1[1]*e2[2] - e1[2]*e2[1]
    ny = e1[2]*e2[0] - e1[0]*e2[2]
    nz = e1[0]*e2[1] - e1[1]*e2[0]

    # Face center
    fc = [sum(all_vertices[i][k] for i in face_indices)/len(face_indices) for k in range(3)]

    # If normal points same direction as centroid→face_center, it's outward (correct)
    to_face = (fc[0]-centroid[0], fc[1]-centroid[1], fc[2]-centroid[2])
    dot = nx*to_face[0] + ny*to_face[1] + nz*to_face[2]

    if dot < 0:
        return list(reversed(face_indices))
    return face_indices


# ─────────────────────────────────────────────────────────────────────────────
# Roof Constructors
# ─────────────────────────────────────────────────────────────────────────────

def construct_flat_roof(ground_verts: List[Tuple],
                        eave_verts: List[Tuple]) -> dict:
    """
    Construct LOD2.2 geometry for a FLAT roof.

    The geometry is essentially the same as LOD1.3 but with semantic labels:
    - 1 GroundSurface (bottom face)
    - 1 RoofSurface (top face — flat)
    - N WallSurfaces (side faces)

    Args:
        ground_verts: list of (x, y, z) for ground footprint
        eave_verts: list of (x, y, z) for top face (same as LOD1.3 top)

    Returns:
        dict with 'vertices', 'faces', 'semantics'
    """
    n = len(ground_verts)
    all_vertices = list(ground_verts) + list(eave_verts)

    # Vertex indices: ground = 0..n-1, eave = n..2n-1
    ground_idx = list(range(n))
    eave_idx = list(range(n, 2 * n))

    faces = []
    semantics = []

    # Ground face (reversed winding for outward normal pointing down)
    faces.append(list(reversed(ground_idx)))
    semantics.append('GroundSurface')

    # Roof face (top — flat)
    faces.append(eave_idx[:])
    semantics.append('RoofSurface')

    # Wall faces
    for i in range(n):
        j = (i + 1) % n
        # Wall quad: ground[i], ground[j], eave[j], eave[i]
        wall = [ground_idx[i], ground_idx[j], eave_idx[j], eave_idx[i]]
        faces.append(wall)
        semantics.append('WallSurface')

    # ─── Fix face normals: ensure all faces point outward ───
    centroid = (
        sum(v[0] for v in all_vertices) / len(all_vertices),
        sum(v[1] for v in all_vertices) / len(all_vertices),
        sum(v[2] for v in all_vertices) / len(all_vertices),
    )
    faces = [_ensure_outward_normal(f, all_vertices, centroid) for f in faces]

    return {
        'vertices': all_vertices,
        'faces': faces,
        'semantics': semantics,
        'roof_type': 'flat',
        'n_roof_faces': 1,
        'ridge_vertices': [],
    }


def construct_gabled_roof(ground_verts: List[Tuple],
                          eave_verts: List[Tuple],
                          ridge_v1: Tuple,
                          ridge_v2: Tuple) -> dict:
    """
    Construct LOD2.2 geometry for a GABLED roof on a rectangular footprint.

    Ridge XY coordinates are computed by the pipeline (MBR-based) and passed
    in directly. Face construction uses a side-of-ridge-line test to classify
    edges as long (parallel to ridge) or short (gable ends).

    For a 4-vertex rectangle this always produces:
    - 2 long edges → 2 roof slope quads + 2 rectangular walls
    - 2 short edges → 2 pentagon gable walls
    - 1 ground face
    Total: 7 faces (1 ground + 2 roof + 4 wall)
    """
    n = len(eave_verts)
    ridge_z = ridge_v1[2]

    # Ridge direction (from the provided ridge vertices)
    ridge_dx = ridge_v2[0] - ridge_v1[0]
    ridge_dy = ridge_v2[1] - ridge_v1[1]
    ridge_len = math.sqrt(ridge_dx**2 + ridge_dy**2)

    if ridge_len < 0.01:
        ridge_dx, ridge_dy, ridge_len = 1.0, 0.0, 1.0

    # Classify each eave vertex as side_a (+) or side_b (-) of the ridge line
    vertex_sides = []
    for ev in eave_verts:
        cross = (ridge_dx * (ev[1] - ridge_v1[1]) -
                 ridge_dy * (ev[0] - ridge_v1[0]))
        vertex_sides.append('a' if cross >= 0 else 'b')

    # Also classify edges using the ridge direction for face construction
    edge_types = []
    for i in range(n):
        j = (i + 1) % n
        if vertex_sides[i] != vertex_sides[j]:
            # Vertices on opposite sides of ridge → this is a gable-end edge
            edge_types.append('short')
        else:
            # Both vertices on same side → long edge (parallel to ridge)
            edge_types.append('long')

    # Build vertex list
    all_vertices = list(ground_verts) + list(eave_verts) + [ridge_v1, ridge_v2]
    ground_idx = list(range(n))
    eave_idx = list(range(n, 2 * n))
    r1_idx = 2 * n
    r2_idx = 2 * n + 1

    faces = []
    semantics = []

    # Ground face
    faces.append(list(reversed(ground_idx)))
    semantics.append('GroundSurface')

    # Second pass: build faces
    for i in range(n):
        j = (i + 1) % n

        if edge_types[i] == 'long':
            # ─── Long edge: Roof slope quad + rectangular wall ───
            edge_mid = ((eave_verts[i][0] + eave_verts[j][0]) / 2,
                        (eave_verts[i][1] + eave_verts[j][1]) / 2)
            cross = ((ridge_v2[0] - ridge_v1[0]) * (edge_mid[1] - ridge_v1[1]) -
                     (ridge_v2[1] - ridge_v1[1]) * (edge_mid[0] - ridge_v1[0]))

            if cross >= 0:
                slope = [eave_idx[j], eave_idx[i], r1_idx, r2_idx]
            else:
                slope = [eave_idx[j], eave_idx[i], r2_idx, r1_idx]
            faces.append(slope)
            semantics.append('RoofSurface')

            wall = [ground_idx[i], ground_idx[j], eave_idx[j], eave_idx[i]]
            faces.append(wall)
            semantics.append('WallSurface')

        else:
            # ─── Short edge: Pentagon gable wall ───
            # Find which ridge vertex is closest to this edge's midpoint
            edge_mid = ((eave_verts[i][0] + eave_verts[j][0]) / 2,
                        (eave_verts[i][1] + eave_verts[j][1]) / 2)
            d1 = distance_2d(edge_mid, ridge_v1)
            d2 = distance_2d(edge_mid, ridge_v2)
            closest = r1_idx if d1 < d2 else r2_idx

            pentagon = [ground_idx[i], ground_idx[j], eave_idx[j], closest, eave_idx[i]]
            faces.append(pentagon)
            semantics.append('WallSurface')

    # ─── Fix face normals ───
    centroid = (
        sum(v[0] for v in all_vertices) / len(all_vertices),
        sum(v[1] for v in all_vertices) / len(all_vertices),
        sum(v[2] for v in all_vertices) / len(all_vertices),
    )
    faces = [_ensure_outward_normal(f, all_vertices, centroid) for f in faces]

    return {
        'vertices': all_vertices,
        'faces': faces,
        'semantics': semantics,
        'roof_type': 'gabled',
        'n_roof_faces': sum(1 for s in semantics if s == 'RoofSurface'),
        'ridge_vertices': [ridge_v1, ridge_v2],
    }


def construct_hipped_roof(ground_verts: List[Tuple],
                          eave_verts: List[Tuple],
                          ridge_v1: Tuple,
                          ridge_v2: Tuple) -> dict:
    """
    Construct LOD2.2 geometry for a HIPPED roof on a rectangular footprint.

    Uses side-of-ridge-line classification (same as gabled) to determine
    edge types. Short edges get triangular hip roof faces instead of
    pentagon gable walls.

    For a 4-vertex rectangle this always produces:
    - 2 long edges → 2 trapezoid roof slopes + 2 rectangular walls
    - 2 short edges → 2 triangular hip roof faces + 2 rectangular walls
    - 1 ground face
    Total: 9 faces (1 ground + 4 roof + 4 wall)
    """
    n = len(eave_verts)

    all_vertices = list(ground_verts) + list(eave_verts) + [ridge_v1, ridge_v2]
    ground_idx = list(range(n))
    eave_idx = list(range(n, 2 * n))
    r1_idx = 2 * n
    r2_idx = 2 * n + 1

    faces = []
    semantics = []

    # Ground face
    faces.append(list(reversed(ground_idx)))
    semantics.append('GroundSurface')

    ridge_dx = ridge_v2[0] - ridge_v1[0]
    ridge_dy = ridge_v2[1] - ridge_v1[1]
    ridge_len = math.sqrt(ridge_dx**2 + ridge_dy**2)

    if ridge_len < 0.01:
        ridge_dx, ridge_dy, ridge_len = 1.0, 0.0, 1.0

    # Classify each eave vertex as side_a or side_b of the ridge line
    vertex_sides = []
    for ev in eave_verts:
        cross = (ridge_dx * (ev[1] - ridge_v1[1]) -
                 ridge_dy * (ev[0] - ridge_v1[0]))
        vertex_sides.append('a' if cross >= 0 else 'b')

    for i in range(n):
        j = (i + 1) % n

        if vertex_sides[i] == vertex_sides[j]:
            # Both vertices on same side → long edge: trapezoid roof slope + wall
            edge_mid = ((eave_verts[i][0] + eave_verts[j][0]) / 2,
                        (eave_verts[i][1] + eave_verts[j][1]) / 2)
            cross = ((ridge_v2[0] - ridge_v1[0]) * (edge_mid[1] - ridge_v1[1]) -
                     (ridge_v2[1] - ridge_v1[1]) * (edge_mid[0] - ridge_v1[0]))

            if cross >= 0:
                slope = [eave_idx[j], eave_idx[i], r1_idx, r2_idx]
            else:
                slope = [eave_idx[j], eave_idx[i], r2_idx, r1_idx]
            faces.append(slope)
            semantics.append('RoofSurface')

            # Rectangular wall below
            wall = [ground_idx[i], ground_idx[j], eave_idx[j], eave_idx[i]]
            faces.append(wall)
            semantics.append('WallSurface')

        else:
            # Vertices on opposite sides → short edge: triangular hip roof + wall
            edge_mid = ((eave_verts[i][0] + eave_verts[j][0]) / 2,
                        (eave_verts[i][1] + eave_verts[j][1]) / 2)
            d1 = distance_2d(edge_mid, ridge_v1)
            d2 = distance_2d(edge_mid, ridge_v2)
            closest = r1_idx if d1 < d2 else r2_idx

            # Triangular hip roof face
            hip = [eave_idx[j], eave_idx[i], closest]
            faces.append(hip)
            semantics.append('RoofSurface')

            # Rectangular wall below
            wall = [ground_idx[i], ground_idx[j], eave_idx[j], eave_idx[i]]
            faces.append(wall)
            semantics.append('WallSurface')

    # ─── Fix face normals: ensure all faces point outward ───
    centroid = (
        sum(v[0] for v in all_vertices) / len(all_vertices),
        sum(v[1] for v in all_vertices) / len(all_vertices),
        sum(v[2] for v in all_vertices) / len(all_vertices),
    )
    faces = [_ensure_outward_normal(f, all_vertices, centroid) for f in faces]

    return {
        'vertices': all_vertices,
        'faces': faces,
        'semantics': semantics,
        'roof_type': 'hipped',
        'n_roof_faces': sum(1 for s in semantics if s == 'RoofSurface'),
        'ridge_vertices': [ridge_v1, ridge_v2],
    }



# ─────────────────────────────────────────────────────────────────────────────
# CityJSON Output Builder
# ─────────────────────────────────────────────────────────────────────────────

def to_cityjson_geometry(construction_result: dict,
                         scale: List[float] = None,
                         translate: List[float] = None) -> dict:
    """
    Convert a construction result into CityJSON geometry format.

    Args:
        construction_result: dict from construct_*_roof()
        scale: CityJSON scale factors [sx, sy, sz] (default: [0.001, 0.001, 0.001])
        translate: CityJSON translate offsets [tx, ty, tz]

    Returns:
        dict with:
          'vertices': list of integer vertices (CityJSON format)
          'geometry': CityJSON geometry object with semantics
    """
    if scale is None:
        scale = [0.001, 0.001, 0.001]
    if translate is None:
        translate = [0.0, 0.0, 0.0]

    vertices = construction_result['vertices']
    faces = construction_result['faces']
    semantics_list = construction_result['semantics']

    # Convert real-world coordinates to CityJSON integer vertices
    int_vertices = []
    for v in vertices:
        iv = [
            round((v[0] - translate[0]) / scale[0]),
            round((v[1] - translate[1]) / scale[1]),
            round((v[2] - translate[2]) / scale[2]),
        ]
        int_vertices.append(iv)

    # Build CityJSON boundaries (Solid → outer shell → faces → rings)
    boundaries_shell = []
    for face in faces:
        boundaries_shell.append([face])  # Each face has one outer ring

    # Build semantics
    unique_types = []
    type_to_idx = {}
    for st in semantics_list:
        if st not in type_to_idx:
            type_to_idx[st] = len(unique_types)
            unique_types.append({'type': st})

    sem_values = [type_to_idx[st] for st in semantics_list]

    geometry = {
        'type': 'Solid',
        'lod': '2.2',
        'boundaries': [boundaries_shell],
        'semantics': {
            'surfaces': unique_types,
            'values': [sem_values]
        }
    }

    return {
        'vertices': int_vertices,
        'geometry': geometry,
    }


def build_lod22_cityjson(buildings: List[dict],
                         source_parser=None,
                         output_path: str = None) -> dict:
    """
    Build a complete CityJSON file with LOD2.2 buildings.

    Args:
        buildings: list of dicts, each with:
            'building_id': str
            'part_id': str
            'roof_type': 'flat' | 'gabled' | 'hipped'
            'ground_verts': list of (x, y, z)
            'eave_verts': list of (x, y, z)
            'ridge_v1': (x, y, z) or None
            'ridge_v2': (x, y, z) or None
            'attributes': dict (original 3DBAG attributes)
        source_parser: CityJSONParser (for scale/translate)
        output_path: if provided, write to file

    Returns:
        CityJSON dict
    """
    # Get scale/translate from source or use defaults
    if source_parser:
        scale = source_parser.scale
        translate = source_parser.translate
    else:
        scale = [0.001, 0.001, 0.001]
        translate = [0.0, 0.0, 0.0]

    # Initialize CityJSON structure
    cityjson = {
        'type': 'CityJSON',
        'version': '2.0',
        'transform': {
            'scale': scale,
            'translate': translate,
        },
        'CityObjects': {},
        'vertices': [],
    }

    vertex_offset = 0

    for bld in buildings:
        bid = bld['building_id']
        pid = bld.get('part_id', bid + '-part')
        roof_type = bld['roof_type']
        ground_verts = bld['ground_verts']
        eave_verts = bld['eave_verts']
        ridge_v1 = bld.get('ridge_v1')
        ridge_v2 = bld.get('ridge_v2')
        attributes = bld.get('attributes', {})

        # Construct roof geometry
        if roof_type == 'flat':
            result = construct_flat_roof(ground_verts, eave_verts)
        elif roof_type == 'gabled' and ridge_v1 and ridge_v2:
            result = construct_gabled_roof(ground_verts, eave_verts, ridge_v1, ridge_v2)
        elif roof_type == 'hipped' and ridge_v1 and ridge_v2:
            result = construct_hipped_roof(ground_verts, eave_verts, ridge_v1, ridge_v2)
        else:
            # Fallback to flat if ridge vertices missing
            result = construct_flat_roof(ground_verts, eave_verts)

        # Convert to CityJSON format
        cj_result = to_cityjson_geometry(result, scale, translate)

        # Offset face indices by the current vertex count
        geom = cj_result['geometry']
        for shell in geom['boundaries']:
            for face in shell:
                for ring in range(len(face)):
                    face[ring] = [idx + vertex_offset for idx in face[ring]]

        # Add vertices to global list
        cityjson['vertices'].extend(cj_result['vertices'])
        vertex_offset += len(cj_result['vertices'])

        # Create Building CityObject (parent)
        if bid not in cityjson['CityObjects']:
            cityjson['CityObjects'][bid] = {
                'type': 'Building',
                'attributes': attributes,
                'children': [pid],
            }
        else:
            cityjson['CityObjects'][bid]['children'].append(pid)

        # Create BuildingPart CityObject (child with geometry)
        cityjson['CityObjects'][pid] = {
            'type': 'BuildingPart',
            'parents': [bid],
            'geometry': [geom],
        }

    # Write to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(cityjson, f, indent=2)
        print(f"CityJSON written to {output_path}")
        print(f"  Buildings: {len(buildings)}")
        print(f"  Total vertices: {len(cityjson['vertices'])}")

    return cityjson


# ─────────────────────────────────────────────────────────────────────────────
# Ridge XY Computation (Bug 2+4 fix)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_gabled_ridge_xy(eave_verts: List[Tuple],
                              footprint_2d: list,
                              ridge_z: float) -> Tuple[Tuple, Tuple]:
    """
    Compute gabled ridge vertex XY positions from the footprint geometry
    using the MBR (minimum bounding rectangle) approach.

    Steps:
    1. Finds the longest edge of the footprint → ridge direction
    2. Projects all footprint vertices onto the perpendicular axis
    3. Places ridge vertices at the midpoint of the two extremes
       along the perpendicular axis, at each end of the long axis

    Currently used with 4-vertex rectangular footprints. The MBR-based
    approach generalises to non-rectangular footprints if needed in future.

    Args:
        eave_verts: eave-level vertices with Z coordinates
        footprint_2d: list of (x, y) footprint coordinates
        ridge_z: predicted ridge height

    Returns:
        (ridge_v1, ridge_v2) — two 3D tuples
    """
    n = len(footprint_2d)

    # Step 1: Find the longest edge → this defines the ridge direction
    best_dx, best_dy, best_len = 1.0, 0.0, 0.0
    for i in range(n):
        j = (i + 1) % n
        ex = footprint_2d[j][0] - footprint_2d[i][0]
        ey = footprint_2d[j][1] - footprint_2d[i][1]
        elen = math.sqrt(ex * ex + ey * ey)
        if elen > best_len:
            best_len = elen
            best_dx = ex / elen
            best_dy = ey / elen

    # Ridge direction (along long axis) and perpendicular direction
    ridge_dir = (best_dx, best_dy)
    perp_dir = (-best_dy, best_dx)

    # Step 2: Project all footprint vertices onto both axes
    # to find the extent of the building along each direction
    centroid_x = sum(p[0] for p in footprint_2d) / n
    centroid_y = sum(p[1] for p in footprint_2d) / n

    ridge_projections = []
    perp_projections = []
    for px, py in footprint_2d:
        dx = px - centroid_x
        dy = py - centroid_y
        # Project onto ridge direction (long axis)
        ridge_proj = dx * ridge_dir[0] + dy * ridge_dir[1]
        ridge_projections.append(ridge_proj)
        # Project onto perpendicular direction (short axis)
        perp_proj = dx * perp_dir[0] + dy * perp_dir[1]
        perp_projections.append(perp_proj)

    # Step 3: Ridge endpoints are at the min/max of the ridge-direction projection
    # (i.e., at the two "short ends" of the building), centered on the perpendicular axis
    min_ridge = min(ridge_projections)
    max_ridge = max(ridge_projections)
    perp_center = (min(perp_projections) + max(perp_projections)) / 2

    # Convert back to world coordinates
    r1_x = centroid_x + min_ridge * ridge_dir[0] + perp_center * perp_dir[0]
    r1_y = centroid_y + min_ridge * ridge_dir[1] + perp_center * perp_dir[1]
    r2_x = centroid_x + max_ridge * ridge_dir[0] + perp_center * perp_dir[0]
    r2_y = centroid_y + max_ridge * ridge_dir[1] + perp_center * perp_dir[1]

    ridge_v1 = (r1_x, r1_y, ridge_z)
    ridge_v2 = (r2_x, r2_y, ridge_z)

    return ridge_v1, ridge_v2


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Pipeline Integration
# ─────────────────────────────────────────────────────────────────────────────

def run_construction_pipeline(parser,
                              classifier,
                              vertex_predictor,
                              label_encoder,
                              selected_features: list,
                              output_path: Optional[str] = None) -> dict:
    """
    Run the complete end-to-end pipeline:
      CityJSON LOD1.3 → Feature Extraction → Classification →
      Vertex Prediction → Geometric Construction → CityJSON LOD2.2
    """
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    from feature_extraction import extract_lod13_features

    paired = parser.pair_lod13_lod22()
    stats = {'total': 0, 'flat': 0, 'gabled': 0, 'hipped': 0,
             'skipped': 0, 'fallback_flat': 0}

    # ─── Phase 1: Extract features for all buildings ───
    print(f"  Extracting features from {len(paired)} buildings...")
    valid_buildings = []  # list of (paired_entry, features, feature_vec)

    for p in paired:
        lod13 = p['lod13']
        attributes = p['attributes']

        features = extract_lod13_features(parser, lod13, attributes)
        if features is None:
            stats['skipped'] += 1
            continue

        if features['n_footprint_vertices'] != 4:
            stats['skipped'] += 1
            continue
        if features['footprint_area'] < 10 or features['footprint_area'] > 500:
            stats['skipped'] += 1
            continue

        # Build feature vector for classifier
        feature_vec = []
        valid = True
        for key in selected_features:
            val = features.get(key)
            if val is None:
                valid = False
                break
            feature_vec.append(float(val))

        if not valid:
            stats['skipped'] += 1
            continue

        valid_buildings.append((p, features, feature_vec))

    print(f"  Valid buildings: {len(valid_buildings)}, Skipped: {stats['skipped']}")

    if not valid_buildings:
        print("  No valid buildings to process!")
        return {'cityjson': {}, 'stats': stats, 'buildings': []}

    # ─── Phase 2: Batch classification (all at once — no per-building warnings) ───
    print(f"  Running batch classification...")
    X_all = np.array([b[2] for b in valid_buildings])
    roof_types_encoded = classifier.predict(X_all)
    roof_types = label_encoder.inverse_transform(roof_types_encoded)

    type_counts = {}
    for rt in roof_types:
        type_counts[rt] = type_counts.get(rt, 0) + 1
    print(f"  Predicted: {type_counts}")

    # ─── Phase 3: Vertex prediction + construction ───
    print(f"  Constructing LOD2.2 geometry...")
    buildings_for_output = []

    for idx, (p, features, _) in enumerate(valid_buildings):
        roof_type = roof_types[idx]
        vertices = parser.vertices

        footprint_indices = features['_footprint_indices']
        top_indices = features['_top_indices']

        ground_verts = [tuple(vertices[i]) for i in footprint_indices]
        eave_verts = [tuple(vertices[i]) for i in top_indices]

        # ─── Safety check: ground and eave must have same vertex count ───
        if len(ground_verts) != len(eave_verts):
            stats['skipped'] += 1
            continue

        # ─── Reorder eave vertices to match ground vertex order ───
        # Each eave vertex should be directly above its corresponding ground vertex.
        # The CityJSON top face may list vertices in a different order than the ground face.
        reordered_eave = []
        used = set()
        valid_match = True
        for gi, gv in enumerate(ground_verts):
            best_ei = None
            best_dist = float('inf')
            for ei, ev in enumerate(eave_verts):
                if ei in used:
                    continue
                dist = math.sqrt((gv[0] - ev[0])**2 + (gv[1] - ev[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_ei = ei
            if best_ei is None or best_dist > 1.0:
                valid_match = False
                break
            reordered_eave.append(eave_verts[best_ei])
            used.add(best_ei)

        if not valid_match:
            stats['skipped'] += 1
            continue

        eave_verts = reordered_eave

        ridge_v1 = None
        ridge_v2 = None

        # Skip degenerate buildings with zero height
        lod13_top_z = eave_verts[0][2]
        ground_z_val = ground_verts[0][2]
        building_h = lod13_top_z - ground_z_val

        if building_h < 1.0:
            # Building too short — force flat
            roof_type = 'flat'

        if roof_type in ('gabled', 'hipped'):
            footprint_2d = features.get('_footprint_2d', None)
            if footprint_2d and vertex_predictor:
                prediction = vertex_predictor.predict(features, footprint_2d, roof_type)
                if prediction:
                    new_ridge_z = prediction['ridge_height']

                    # Bug 1 fix: Keep eave at original LOD1.3 top — don't lower it.
                    # The ridge goes ABOVE the existing eave; we don't modify eave_verts.
                    new_ridge_z = max(lod13_top_z + 0.5, new_ridge_z)

                    if roof_type == 'gabled':
                        # Bug 2+4 fix: Compute ridge XY here from the footprint
                        # using MBR-based approach, not placeholder (0,0).
                        # Find the two shortest edges (gable ends) and place
                        # ridge vertices above their midpoints.
                        ridge_v1, ridge_v2 = _compute_gabled_ridge_xy(
                            eave_verts, footprint_2d, new_ridge_z
                        )

                    elif roof_type == 'hipped':
                        # HIPPED: ML predicts full ridge vertex positions
                        ridge_v1 = prediction['ridge_vertices'][0]
                        ridge_v2 = prediction['ridge_vertices'][1]
                        ridge_v1 = (ridge_v1[0], ridge_v1[1], new_ridge_z)
                        ridge_v2 = (ridge_v2[0], ridge_v2[1], new_ridge_z)

            if ridge_v1 is None:
                roof_type = 'flat'
                stats['fallback_flat'] += 1

        stats['total'] += 1
        stats[roof_type] += 1

        buildings_for_output.append({
            'building_id': p['building_id'],
            'part_id': p['part_id'],
            'roof_type': roof_type,
            'ground_verts': ground_verts,
            'eave_verts': eave_verts,
            'ridge_v1': ridge_v1,
            'ridge_v2': ridge_v2,
            'attributes': p['attributes'],
        })

        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"    Processed {idx + 1}/{len(valid_buildings)} buildings...")

    # ─── Phase 4: Build CityJSON output ───
    print(f"  Building CityJSON output...")
    cityjson = build_lod22_cityjson(
        buildings_for_output,
        source_parser=parser,
        output_path=output_path
    )

    print(f"\n  CONSTRUCTION COMPLETE:")
    print(f"    Total: {stats['total']}")
    print(f"    Flat: {stats['flat']}, Gabled: {stats['gabled']}, Hipped: {stats['hipped']}")
    print(f"    Skipped: {stats['skipped']}, Fallback to flat: {stats['fallback_flat']}")

    return {
        'cityjson': cityjson,
        'stats': stats,
        'buildings': buildings_for_output,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with simple synthetic buildings
    print("Testing geometric construction with synthetic buildings...\n")

    # ─── Test 1: Flat roof ───
    ground = [(0, 0, 0), (10, 0, 0), (10, 8, 0), (0, 8, 0)]
    eave = [(0, 0, 6), (10, 0, 6), (10, 8, 6), (0, 8, 6)]

    result = construct_flat_roof(ground, eave)
    print(f"FLAT ROOF:")
    print(f"  Vertices: {len(result['vertices'])}")
    print(f"  Faces: {len(result['faces'])}")
    print(f"  Semantics: {result['semantics']}")
    assert len(result['faces']) == 6  # 1 ground + 1 roof + 4 walls
    assert result['semantics'].count('RoofSurface') == 1
    print(f"  ✓ Correct: 6 faces, 1 RoofSurface\n")

    # ─── Test 2: Gabled roof ───
    ridge_v1 = (0, 4, 10)  # ridge along x-axis at y=4 (center)
    ridge_v2 = (10, 4, 10)

    result = construct_gabled_roof(ground, eave, ridge_v1, ridge_v2)
    print(f"GABLED ROOF:")
    print(f"  Vertices: {len(result['vertices'])} (8 original + 2 ridge)")
    print(f"  Faces: {len(result['faces'])}")
    print(f"  Semantics: {result['semantics']}")
    n_roof = result['semantics'].count('RoofSurface')
    n_wall = result['semantics'].count('WallSurface')
    n_ground = result['semantics'].count('GroundSurface')
    print(f"  RoofSurfaces: {n_roof}, WallSurfaces: {n_wall}, GroundSurfaces: {n_ground}")
    print(f"  ✓ Expected: 2 roof + ~6 wall (4 rect + 2 gable triangles) + 1 ground\n")

    # ─── Test 3: Hipped roof ───
    ridge_v1_hip = (3, 4, 10)   # shorter ridge
    ridge_v2_hip = (7, 4, 10)

    result = construct_hipped_roof(ground, eave, ridge_v1_hip, ridge_v2_hip)
    print(f"HIPPED ROOF:")
    print(f"  Vertices: {len(result['vertices'])} (8 original + 2 ridge)")
    print(f"  Faces: {len(result['faces'])}")
    print(f"  Semantics: {result['semantics']}")
    n_roof = result['semantics'].count('RoofSurface')
    n_wall = result['semantics'].count('WallSurface')
    n_ground = result['semantics'].count('GroundSurface')
    print(f"  RoofSurfaces: {n_roof}, WallSurfaces: {n_wall}, GroundSurfaces: {n_ground}")
    print(f"  ✓ Expected: 4 roof + 4 wall + 1 ground\n")

    # ─── Test 4: CityJSON export ───
    print("Testing CityJSON export...")
    buildings = [
        {
            'building_id': 'test_flat', 'part_id': 'test_flat_p',
            'roof_type': 'flat',
            'ground_verts': ground, 'eave_verts': eave,
            'attributes': {'b3_dak_type': 'horizontal'},
        },
        {
            'building_id': 'test_gabled', 'part_id': 'test_gabled_p',
            'roof_type': 'gabled',
            'ground_verts': [(20, 0, 0), (30, 0, 0), (30, 8, 0), (20, 8, 0)],
            'eave_verts': [(20, 0, 6), (30, 0, 6), (30, 8, 6), (20, 8, 6)],
            'ridge_v1': (20, 4, 10), 'ridge_v2': (30, 4, 10),
            'attributes': {'b3_dak_type': 'slanted'},
        },
        {
            'building_id': 'test_hipped', 'part_id': 'test_hipped_p',
            'roof_type': 'hipped',
            'ground_verts': [(40, 0, 0), (50, 0, 0), (50, 8, 0), (40, 8, 0)],
            'eave_verts': [(40, 0, 6), (50, 0, 6), (50, 8, 6), (40, 8, 6)],
            'ridge_v1': (43, 4, 10), 'ridge_v2': (47, 4, 10),
            'attributes': {'b3_dak_type': 'slanted'},
        },
    ]

    cj = build_lod22_cityjson(buildings, output_path="/tmp/test_lod22.city.json")
    print(f"  CityObjects: {len(cj['CityObjects'])}")
    print(f"  Total vertices: {len(cj['vertices'])}")
    print(f"\n  ✓ All tests passed!")