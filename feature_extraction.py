"""
Feature Extraction & Ground Truth Derivation

Turns LOD1.3 geometry into the feature vectors used for training, and
derives ground-truth roof labels/parameters from the paired LOD2.2
geometry. Scoped to rectangular (4-vertex) footprints with flat, gabled,
or hipped roofs.

LOD1.3 features (footprint area/perimeter/vertex count, aspect ratio,
compactness, rectangularity, MBR dimensions, edge ratio, height,
orientation, volume) are the only things the model sees as input. LOD2.2
is used strictly to derive labels — roof type, ridge height/position/
length, slope — and never leaks into the feature vector.
"""

import math
from typing import List, Tuple, Optional


# --- 2D geometry helpers (footprint analysis) ---

def polygon_area_2d(points: List[Tuple[float, float]]) -> float:
    """Compute area of a 2D polygon using the Shoelace formula."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def polygon_perimeter_2d(points: List[Tuple[float, float]]) -> float:
    """Compute perimeter of a 2D polygon."""
    n = len(points)
    perimeter = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = points[j][0] - points[i][0]
        dy = points[j][1] - points[i][1]
        perimeter += math.sqrt(dx * dx + dy * dy)
    return perimeter


def polygon_edges_2d(points: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
    """
    Compute edge properties for a 2D polygon.
    Returns list of (length, orientation_degrees, edge_index).
    """
    n = len(points)
    edges = []
    for i in range(n):
        j = (i + 1) % n
        dx = points[j][0] - points[i][0]
        dy = points[j][1] - points[i][1]
        length = math.sqrt(dx * dx + dy * dy)
        orientation = math.degrees(math.atan2(dy, dx)) % 180  # 0-180 range
        edges.append((length, orientation, i))
    return edges


def polygon_centroid_2d(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Compute centroid of a 2D polygon."""
    n = len(points)
    if n == 0:
        return (0.0, 0.0)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    return (cx, cy)


def minimum_bounding_rectangle(points: List[Tuple[float, float]]) -> dict:
    """
    Compute the minimum area bounding rectangle using rotating calipers
    (simplified approach using edge orientations).

    Returns dict with: area, width, length, aspect_ratio, orientation
    """
    if len(points) < 3:
        return {'area': 0, 'width': 0, 'length': 0, 'aspect_ratio': 1.0, 'orientation': 0}

    edges = polygon_edges_2d(points)
    best = None
    best_area = float('inf')

    # Try the orientation of each edge as a candidate rotation
    for _, edge_angle, _ in edges:
        angle_rad = math.radians(edge_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        # Rotate all points
        rotated = []
        for px, py in points:
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            rotated.append((rx, ry))

        min_x = min(r[0] for r in rotated)
        max_x = max(r[0] for r in rotated)
        min_y = min(r[1] for r in rotated)
        max_y = max(r[1] for r in rotated)

        w = max_x - min_x
        h = max_y - min_y
        area = w * h

        if area < best_area and area > 0:
            best_area = area
            length = max(w, h)
            width = min(w, h)
            best = {
                'area': area,
                'width': width,
                'length': length,
                'aspect_ratio': length / width if width > 0 else 1.0,
                'orientation': edge_angle
            }

    return best or {'area': 0, 'width': 0, 'length': 0, 'aspect_ratio': 1.0, 'orientation': 0}


def compute_compactness(area: float, perimeter: float) -> float:
    """
    Compute isoperimetric compactness: 4π × area / perimeter².
    Value of 1.0 = perfect circle, lower = more elongated/complex.
    """
    if perimeter <= 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter * perimeter)


def compute_rectangularity(area: float, mbr_area: float) -> float:
    """
    Compute rectangularity: footprint_area / min_bounding_rect_area.
    Value of 1.0 = perfect rectangle.
    """
    if mbr_area <= 0:
        return 0.0
    return area / mbr_area


# --- 3D geometry helpers ---

def distance_3d(p1: List[float], p2: List[float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))


# --- LOD1.3 feature extraction ---

def extract_footprint(faces: List[List[int]], vertices: List[List[float]]) -> List[int]:
    """
    Identify the ground/footprint face from LOD1.3 geometry.
    The footprint is the face with the lowest average z-value.
    """
    best_face = None
    min_avg_z = float('inf')

    for face in faces:
        avg_z = sum(vertices[idx][2] for idx in face) / len(face) if face else float('inf')
        if avg_z < min_avg_z:
            min_avg_z = avg_z
            best_face = face

    return best_face or []


def extract_top_face(faces: List[List[int]], vertices: List[List[float]]) -> List[int]:
    """
    Identify the top (roof) face from LOD1.3 geometry.

    The top face must:
    1. Have the same number of vertices as the ground face
    2. All vertices at approximately the same Z (horizontal face)
    3. Have the highest average Z among such faces

    This avoids picking wall faces that happen to have high avg Z.
    """
    # First find the ground face to know expected vertex count
    ground_face = extract_footprint(faces, vertices)
    n_ground = len(ground_face)

    best_face = None
    max_avg_z = float('-inf')

    for face in faces:
        if not face:
            continue

        # Check if this face is horizontal (all vertices at ~same Z)
        z_vals = [vertices[idx][2] for idx in face]
        z_range = max(z_vals) - min(z_vals)

        if z_range > 0.5:  # Not horizontal — skip (it's a wall)
            continue

        avg_z = sum(z_vals) / len(z_vals)

        # Must have same vertex count as ground and be above ground
        ground_avg_z = sum(vertices[idx][2] for idx in ground_face) / n_ground if n_ground > 0 else 0
        if avg_z <= ground_avg_z + 0.1:  # Must be above ground
            continue

        if len(face) == n_ground and avg_z > max_avg_z:
            max_avg_z = avg_z
            best_face = face

    # Fallback: if no matching horizontal face found, use highest avg Z
    # but only among faces with same vertex count as ground
    if best_face is None:
        for face in faces:
            if not face or len(face) != n_ground:
                continue
            avg_z = sum(vertices[idx][2] for idx in face) / len(face)
            if avg_z > max_avg_z:
                max_avg_z = avg_z
                best_face = face

    # Last fallback: just highest avg Z
    if best_face is None:
        for face in faces:
            if not face:
                continue
            avg_z = sum(vertices[idx][2] for idx in face) / len(face)
            if avg_z > max_avg_z:
                max_avg_z = avg_z
                best_face = face

    return best_face or []


def extract_lod13_features(parser, geometry: dict, attributes: dict) -> Optional[dict]:
    """
    Extract all geometric features from a LOD1.3 building model.

    Args:
        parser: CityJSONParser instance (for vertex access)
        geometry: LOD1.3 geometry dict
        attributes: Building attributes dict

    Returns:
        dict of feature name → value, or None if extraction fails
    """
    faces = parser.get_shell_faces(geometry)
    if not faces:
        return None

    vertices = parser.vertices

    # Find footprint (ground face)
    footprint_indices = extract_footprint(faces, vertices)
    if not footprint_indices or len(footprint_indices) < 3:
        return None

    # Extract 2D footprint coordinates
    footprint_2d = [(vertices[idx][0], vertices[idx][1]) for idx in footprint_indices]

    # Find top face
    top_indices = extract_top_face(faces, vertices)

    # Collect all vertex indices used in this building
    all_indices = set()
    for face in faces:
        all_indices.update(face)

    z_values = [vertices[idx][2] for idx in all_indices]
    ground_z = min(z_values)
    top_z = max(z_values)

    # Footprint geometry
    area = polygon_area_2d(footprint_2d)
    perimeter = polygon_perimeter_2d(footprint_2d)
    n_vertices = len(footprint_2d)

    if area < 1.0:  # Skip degenerate buildings (< 1 m²)
        return None

    # Shape metrics
    mbr = minimum_bounding_rectangle(footprint_2d)
    compactness = compute_compactness(area, perimeter)
    rectangularity = compute_rectangularity(area, mbr['area'])
    aspect_ratio = mbr['aspect_ratio']

    # Edge analysis
    edges = polygon_edges_2d(footprint_2d)
    longest_edge = max(edges, key=lambda e: e[0])
    shortest_edge = min(edges, key=lambda e: e[0])
    edge_length_ratio = longest_edge[0] / shortest_edge[0] if shortest_edge[0] > 0 else 1.0

    building_height = top_z - ground_z

    # vol_lod22 is deliberately excluded here — it would leak LOD2.2 data into a feature
    vol_lod13 = attributes.get('b3_volume_lod13', None)

    orientation = longest_edge[1]  # degrees, 0-180
    centroid = polygon_centroid_2d(footprint_2d)

    # Everything below is LOD1.3-derivable and safe to use as an ML input;
    # LOD2.2 only ever supplies ground-truth labels, never features.
    features = {
        # Footprint geometry
        'footprint_area': area,
        'footprint_perimeter': perimeter,
        'n_footprint_vertices': n_vertices,

        # Shape descriptors
        'aspect_ratio': aspect_ratio,
        'compactness': compactness,
        'rectangularity': rectangularity,
        'mbr_length': mbr['length'],
        'mbr_width': mbr['width'],
        'edge_length_ratio': edge_length_ratio,

        # Height features (LOD1.3 only)
        'building_height': building_height,
        'ground_z': ground_z,
        'top_z': top_z,

        # Volume (LOD1.3 only)
        'vol_lod13': vol_lod13,

        # Orientation and position
        'orientation': orientation,
        'longest_edge_length': longest_edge[0],
        'centroid_x': centroid[0],
        'centroid_y': centroid[1],

        # Raw footprint data (for construction and graph representation)
        '_footprint_indices': footprint_indices,
        '_footprint_2d': footprint_2d,
        '_top_indices': top_indices,
    }

    return features


# --- LOD2.2 ground-truth derivation ---

def derive_roof_type(parser, geometry: dict) -> Optional[str]:
    """
    Derive the actual roof type from LOD2.2 semantic surfaces.

    Rules:
        1 RoofSurface  → 'flat'
        2 RoofSurfaces → 'gabled'
        4 RoofSurfaces → 'hipped'
        Other          → 'complex' (out of scope)
    """
    labels = parser.get_semantic_labels(geometry)
    n_roof = sum(1 for l in labels if l == 'RoofSurface')

    if n_roof == 1:
        return 'flat'
    elif n_roof == 2:
        return 'gabled'
    elif n_roof == 4:
        return 'hipped'
    else:
        return 'complex'


def extract_roof_parameters(parser, geometry: dict, roof_type: str) -> Optional[dict]:
    """
    Extract ground-truth roof parameters from LOD2.2 geometry.

    For gabled/hipped roofs:
        - ridge_height: absolute z of ridge vertices
        - eave_height: z of eave vertices (where roof meets walls)
        - ridge_relative_height: ridge_height - eave_height
        - ridge_x, ridge_y: position of ridge line
        - ridge_length: length of ridge line (for hipped roofs)
        - roof_slope: slope angle in degrees

    For flat roofs:
        - roof_height: z of the single roof surface
    """
    if roof_type == 'complex':
        return None

    faces = parser.get_shell_faces(geometry)
    labels = parser.get_semantic_labels(geometry)
    vertices = parser.vertices

    if not faces or not labels or len(faces) != len(labels):
        return None

    # Separate faces by semantic type
    roof_faces = []
    wall_faces = []
    ground_faces = []

    for face, label in zip(faces, labels):
        if label == 'RoofSurface':
            roof_faces.append(face)
        elif label == 'WallSurface':
            wall_faces.append(face)
        elif label == 'GroundSurface':
            ground_faces.append(face)

    # Collect all roof vertex indices
    roof_vertex_indices = set()
    for face in roof_faces:
        roof_vertex_indices.update(face)

    if not roof_vertex_indices:
        return None

    roof_z_values = [vertices[idx][2] for idx in roof_vertex_indices]

    params = {
        'roof_type': roof_type,
        'n_roof_faces': len(roof_faces),
    }

    if roof_type == 'flat':
        params['roof_height'] = sum(roof_z_values) / len(roof_z_values)
        params['ridge_relative_height'] = 0.0
        params['roof_slope'] = 0.0

    elif roof_type in ('gabled', 'hipped'):
        eave_height = min(roof_z_values)
        ridge_height = max(roof_z_values)

        params['eave_height'] = eave_height
        params['ridge_height'] = ridge_height
        params['ridge_relative_height'] = ridge_height - eave_height

        # Find ridge vertices (those at maximum z)
        ridge_tolerance = 0.05  # meters
        ridge_vertices = [idx for idx in roof_vertex_indices
                         if abs(vertices[idx][2] - ridge_height) < ridge_tolerance]

        if len(ridge_vertices) >= 2:
            # Ridge line: compute position and length
            ridge_coords = [vertices[idx] for idx in ridge_vertices]

            # Ridge midpoint (position)
            ridge_x = sum(v[0] for v in ridge_coords) / len(ridge_coords)
            ridge_y = sum(v[1] for v in ridge_coords) / len(ridge_coords)
            params['ridge_x'] = ridge_x
            params['ridge_y'] = ridge_y

            # Ridge length (distance between extreme ridge points)
            if len(ridge_vertices) == 2:
                params['ridge_length'] = distance_3d(
                    vertices[ridge_vertices[0]],
                    vertices[ridge_vertices[1]]
                )
            else:
                # For more complex cases, compute max pairwise distance
                max_dist = 0
                for i in range(len(ridge_coords)):
                    for j in range(i+1, len(ridge_coords)):
                        d = distance_3d(ridge_coords[i], ridge_coords[j])
                        if d > max_dist:
                            max_dist = d
                params['ridge_length'] = max_dist

            params['ridge_vertices'] = ridge_vertices
        else:
            # Single ridge point (unusual but possible)
            if ridge_vertices:
                v = vertices[ridge_vertices[0]]
                params['ridge_x'] = v[0]
                params['ridge_y'] = v[1]
            params['ridge_length'] = 0.0
            params['ridge_vertices'] = ridge_vertices

        # Roof slope from semantics or compute from geometry
        slope_angles = parser.get_roof_slope_angles(geometry)
        if slope_angles:
            params['roof_slope'] = sum(slope_angles) / len(slope_angles)
        else:
            # Compute from geometry: slope = atan(rise / run)
            if params.get('ridge_relative_height', 0) > 0:
                # Estimate run from footprint width / 2
                # This is approximate; real computation needs footprint analysis
                params['roof_slope'] = None  # Will be computed later with footprint info

        # For hipped roofs: compute ridge position relative to footprint
        if roof_type == 'hipped' and ground_faces:
            ground_verts = set()
            for face in ground_faces:
                ground_verts.update(face)
            ground_coords = [vertices[idx] for idx in ground_verts]
            footprint_centroid_x = sum(v[0] for v in ground_coords) / len(ground_coords)
            footprint_centroid_y = sum(v[1] for v in ground_coords) / len(ground_coords)

            params['ridge_offset_x'] = params.get('ridge_x', 0) - footprint_centroid_x
            params['ridge_offset_y'] = params.get('ridge_y', 0) - footprint_centroid_y

    return params


# --- putting it together: features + labels for one building / one tile ---

def extract_building_data(parser, paired_building: dict) -> Optional[dict]:
    """
    Extract all features and ground truth for a single paired building.

    Args:
        parser: CityJSONParser instance
        paired_building: dict from parser.pair_lod13_lod22()

    Returns:
        dict with:
            - 'building_id': str
            - 'features': dict of LOD1.3 features
            - 'roof_type': str ('flat', 'gabled', 'hipped', 'complex')
            - 'roof_params': dict of ground-truth roof parameters
            - 'attributes': original 3DBAG attributes
    """
    lod13 = paired_building['lod13']
    lod22 = paired_building['lod22']
    attributes = paired_building['attributes']

    # Extract LOD1.3 features
    features = extract_lod13_features(parser, lod13, attributes)
    if features is None:
        return None

    # Derive roof type from LOD2.2
    roof_type = derive_roof_type(parser, lod22)
    if roof_type is None:
        return None

    # Extract roof parameters from LOD2.2
    roof_params = extract_roof_parameters(parser, lod22, roof_type)

    return {
        'building_id': paired_building['building_id'],
        'part_id': paired_building['part_id'],
        'features': features,
        'roof_type': roof_type,
        'roof_params': roof_params,
        'attributes': attributes
    }


def process_tile(parser) -> List[dict]:
    """
    Process all buildings in a tile: extract features and ground truth.

    Args:
        parser: CityJSONParser instance

    Returns:
        List of building data dicts
    """
    paired = parser.pair_lod13_lod22()
    results = []
    skipped = {'no_features': 0, 'complex': 0, 'no_lod_pair': 0}

    for p in paired:
        data = extract_building_data(parser, p)
        if data is None:
            skipped['no_features'] += 1
            continue
        if data['roof_type'] == 'complex':
            skipped['complex'] += 1
            continue
        results.append(data)

    print(f"\nProcessed {len(paired)} paired buildings:")
    print(f"  Extracted: {len(results)}")
    print(f"  Skipped: {skipped}")

    # Distribution of roof types
    type_counts = {}
    for r in results:
        t = r['roof_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Roof type distribution: {type_counts}")

    return results


# --- dataset filtering ---

def filter_buildings(building_data: List[dict],
                     max_vertices: int = 4,
                     min_area: float = 10.0,
                     max_area: float = 500.0,
                     allowed_types: List[str] = None) -> List[dict]:
    """
    Filter buildings based on complexity and size constraints.

    Args:
        building_data: List of extracted building data dicts
        max_vertices: Exact footprint vertex count required (default: 4 = rectangles)
        min_area: Minimum footprint area in m²
        max_area: Maximum footprint area in m²
        allowed_types: Allowed roof types (default: flat, gabled, hipped)

    Returns:
        Filtered list
    """
    if allowed_types is None:
        allowed_types = ['flat', 'gabled', 'hipped']

    filtered = []
    for bd in building_data:
        f = bd['features']
        rt = bd['roof_type']

        # Check roof type
        if rt not in allowed_types:
            continue

        # Check vertex count
        if f['n_footprint_vertices'] != max_vertices:
            continue

        # Check area
        if f['footprint_area'] < min_area or f['footprint_area'] > max_area:
            continue

        filtered.append(bd)

    print(f"\nFiltering: {len(building_data)} → {len(filtered)} buildings")
    print(f"  Criteria: max_vertices={max_vertices}, area=[{min_area}, {max_area}], types={allowed_types}")

    # Show distribution after filtering
    type_counts = {}
    for bd in filtered:
        t = bd['roof_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Roof type distribution: {type_counts}")

    return filtered


# --- neighboring-building features ---

def compute_neighbor_features(building_data: List[dict],
                               radius: float = 50.0,
                               use_ground_truth: bool = True) -> List[dict]:
    """
    Enrich each building's feature dict with spatial neighbor context.

    For each building, finds all other buildings within `radius` meters
    (by centroid distance) and computes:
      - n_neighbors: count of nearby buildings
      - neighbor_mean_height: average building_height of neighbors
      - neighbor_mean_area: average footprint_area of neighbors
      - neighbor_frac_flat: fraction of neighbors with flat roof
      - neighbor_frac_gabled: fraction of neighbors with gabled roof
      - neighbor_frac_hipped: fraction of neighbors with hipped roof

    Args:
        building_data: list of building data dicts (must have 'features'
                       with 'centroid_x', 'centroid_y', and 'roof_type')
        radius: neighbor search radius in meters (default: 50m)
        use_ground_truth: if True, use ground-truth roof_type for
                          neighbor fractions (training). If False,
                          use 'predicted_roof_type' key (inference).

    Returns:
        Same list with neighbor features added to each building's
        feature dict. Buildings with no neighbors get neutral values.
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

    # Compute pairwise distances and find neighbors
    # For efficiency, only compute when needed (O(n²) is fine for <20k buildings)
    print(f"\n  Computing neighbor features (radius={radius}m, n={n} buildings)...")

    neighbor_counts = 0
    for i in range(n):
        cx_i, cy_i = centroids[i]
        neighbors_idx = []

        for j in range(n):
            if i == j:
                continue
            cx_j, cy_j = centroids[j]
            dist = math.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            if dist <= radius and dist > 0.1:
                neighbors_idx.append((j, dist))

        n_neighbors = len(neighbors_idx)
        neighbor_counts += n_neighbors

        if n_neighbors > 0:
            # Distance-weighted: closer neighbors count more
            weights = [1.0 / d for _, d in neighbors_idx]
            total_weight = sum(weights)

            # Weighted geometric neighbor features
            neighbor_heights = [building_data[j]['features'].get('building_height', 0)
                                for j, _ in neighbors_idx]
            neighbor_areas = [building_data[j]['features'].get('footprint_area', 0)
                              for j, _ in neighbors_idx]

            w_mean_height = sum(h * w for h, w in zip(neighbor_heights, weights)) / total_weight
            w_mean_area = sum(a * w for a, w in zip(neighbor_areas, weights)) / total_weight

            # Weighted roof type fractions
            neighbor_rt = [roof_types[j] for j, _ in neighbors_idx]
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
            # No neighbors found — use neutral values
            building_data[i]['features']['n_neighbors'] = 0
            building_data[i]['features']['neighbor_mean_height'] = building_data[i]['features'].get('building_height', 0)
            building_data[i]['features']['neighbor_mean_area'] = building_data[i]['features'].get('footprint_area', 0)
            building_data[i]['features']['neighbor_frac_flat'] = 0.33
            building_data[i]['features']['neighbor_frac_gabled'] = 0.33
            building_data[i]['features']['neighbor_frac_hipped'] = 0.33

    avg_neighbors = neighbor_counts / n if n > 0 else 0
    no_neighbor_count = sum(1 for bd in building_data if bd['features']['n_neighbors'] == 0)
    print(f"    Average neighbors per building: {avg_neighbors:.1f}")
    print(f"    Buildings with no neighbors: {no_neighbor_count}")

    return building_data


# --- export to ML-ready format ---

def to_feature_matrix(building_data: List[dict]) -> Tuple[List[List[float]], List[str], List[str]]:
    """
    Convert building data to a feature matrix for ML.

    Uses only LOD1.3-derivable features (no LiDAR/LOD2.2 data).

    Returns:
        X: List of feature vectors (numeric features only)
        y: List of roof type labels
        feature_names: List of feature column names
    """
    # LOD1.3-only features (matches ALL_FEATURES in ml_models_revised.py)
    feature_keys = [
        'footprint_area', 'footprint_perimeter', 'n_footprint_vertices',
        'aspect_ratio', 'compactness', 'rectangularity',
        'mbr_length', 'mbr_width', 'edge_length_ratio',
        'building_height',
        'orientation', 'longest_edge_length',
        'vol_lod13'
    ]

    X = []
    y = []
    valid_buildings = []

    for bd in building_data:
        features = bd['features']
        row = []
        valid = True

        for key in feature_keys:
            val = features.get(key)
            if val is None:
                valid = False
                break
            row.append(float(val))

        if valid:
            X.append(row)
            y.append(bd['roof_type'])
            valid_buildings.append(bd)

    print(f"\nFeature matrix: {len(X)} samples × {len(feature_keys)} features")
    print(f"  (Dropped {len(building_data) - len(X)} buildings with missing values)")

    return X, y, feature_keys


# --- CSV export (no pandas dependency) ---

def export_to_csv(building_data: List[dict], filepath: str):
    """Export building features and labels to CSV file."""
    if not building_data:
        print("No data to export.")
        return

    X, y, feature_names = to_feature_matrix(building_data)

    # Add building_id, roof_type, and roof parameter columns
    header = ['building_id'] + feature_names + [
        'roof_type',
        'ridge_relative_height', 'ridge_length', 'roof_slope',
        'eave_height', 'ridge_height'
    ]

    with open(filepath, 'w') as f:
        f.write(','.join(header) + '\n')

        idx = 0
        for bd in building_data:
            features = bd['features']
            # Check if this building made it through the feature matrix filter
            row = []
            valid = True
            for key in feature_names:
                val = features.get(key)
                if val is None:
                    valid = False
                    break
                row.append(str(val))

            if not valid:
                continue

            rp = bd.get('roof_params') or {}
            line = [bd['building_id']] + row + [
                bd['roof_type'],
                str(rp.get('ridge_relative_height', '')),
                str(rp.get('ridge_length', '')),
                str(rp.get('roof_slope', '')),
                str(rp.get('eave_height', '')),
                str(rp.get('ridge_height', ''))
            ]
            f.write(','.join(line) + '\n')
            idx += 1

    print(f"\nExported {idx} buildings to {filepath}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from cityjson_parser import CityJSONParser

    filepath = sys.argv[1] if len(sys.argv) > 1 else "../sample_3dbag_tile.city.json"
    parser = CityJSONParser(filepath)

    # Process all buildings
    all_data = process_tile(parser)

    # Filter for simple buildings
    filtered = filter_buildings(all_data)

    # Show detailed results
    for bd in filtered:
        print(f"\n{'─'*50}")
        print(f"Building: {bd['building_id']}")
        print(f"Roof type: {bd['roof_type']}")
        print(f"Features:")
        for k, v in bd['features'].items():
            if not k.startswith('_'):
                print(f"  {k}: {v}")
        if bd['roof_params']:
            print(f"Roof params (ground truth):")
            for k, v in bd['roof_params'].items():
                if not k.startswith('_'):
                    print(f"  {k}: {v}")

    # Export
    export_to_csv(filtered, "../building_features.csv")