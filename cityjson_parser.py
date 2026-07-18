"""
CityJSON Parser

Handles loading 3DBAG CityJSON tiles, coordinate transformation,
and pairing LOD1.3/LOD2.2 building geometries.
"""

import json
import os
from typing import Dict, List, Optional


class CityJSONParser:
    """
    Parses 3DBAG CityJSON files and provides structured access
    to building geometries at different LODs.
    """

    def __init__(self, filepath: str):
        """Load and parse a CityJSON file."""
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

        with open(filepath, 'r') as f:
            self.data = json.load(f)

        # Extract transform parameters
        transform = self.data.get('transform', {})
        self.scale = transform.get('scale', [1.0, 1.0, 1.0])
        self.translate = transform.get('translate', [0.0, 0.0, 0.0])

        # Convert all vertices to real-world coordinates once
        self.raw_vertices = self.data.get('vertices', [])
        self.vertices = self._transform_vertices()

        # Index CityObjects
        self.city_objects = self.data.get('CityObjects', {})

    def _transform_vertices(self) -> List[List[float]]:
        """Apply scale + translate to convert integer vertices to real coordinates."""
        result = []
        for v in self.raw_vertices:
            result.append([
                v[0] * self.scale[0] + self.translate[0],
                v[1] * self.scale[1] + self.translate[1],
                v[2] * self.scale[2] + self.translate[2]
            ])
        return result

    def get_vertex(self, index: int) -> List[float]:
        """Get real-world coordinates for a vertex index."""
        return self.vertices[index]

    def get_buildings(self) -> Dict[str, dict]:
        """Return all Building-type CityObjects."""
        return {k: v for k, v in self.city_objects.items()
                if v.get('type') == 'Building'}

    def get_building_parts(self) -> Dict[str, dict]:
        """Return all BuildingPart-type CityObjects."""
        return {k: v for k, v in self.city_objects.items()
                if v.get('type') == 'BuildingPart'}

    def get_building_with_parts(self, building_id: str) -> Optional[dict]:
        """Look up a building and its BuildingParts. None if the id isn't a Building."""
        building = self.city_objects.get(building_id)
        if not building or building.get('type') != 'Building':
            return None

        parts = {}
        for child_id in building.get('children', []):
            child = self.city_objects.get(child_id)
            if child:
                parts[child_id] = child

        return {
            'building_id': building_id,
            'building': building,
            'parts': parts,
            'attributes': building.get('attributes', {})
        }

    def get_geometry_by_lod(self, city_object: dict, target_lod: str) -> Optional[dict]:
        """Return the geometry entry matching target_lod (e.g. '1.3', '2.2'), or None."""
        for geom in city_object.get('geometry', []):
            if geom.get('lod') == target_lod:
                return geom
        return None

    def get_shell_faces(self, geometry: dict) -> List[List[int]]:
        """
        Extract the outer shell faces from a geometry object.
        Each face is a list of vertex indices (outer ring only, ignoring holes).

        For Solid: boundaries[0] = outer shell
        For MultiSurface: boundaries directly contains faces
        """
        boundaries = geometry.get('boundaries', [])
        geom_type = geometry.get('type', '')

        if geom_type == 'Solid':
            if not boundaries:
                return []
            shell = boundaries[0]  # outer shell
        elif geom_type == 'MultiSurface':
            shell = boundaries
        else:
            shell = boundaries[0] if boundaries else []

        # Extract outer ring of each face (index 0, ignoring holes)
        faces = []
        for face in shell:
            if face and len(face) > 0:
                faces.append(face[0])  # outer ring only

        return faces

    def get_semantic_labels(self, geometry: dict) -> List[Optional[str]]:
        """
        Get semantic surface type for each face in the geometry.

        Returns:
            List of surface type strings (e.g., 'RoofSurface', 'WallSurface')
            aligned with the faces from get_shell_faces().
            Returns None for faces without semantic info.
        """
        semantics = geometry.get('semantics', {})
        if not semantics:
            return []

        surfaces = semantics.get('surfaces', [])
        values = semantics.get('values', [[]])

        # For Solid, values[0] corresponds to outer shell
        if geometry.get('type') == 'Solid':
            face_indices = values[0] if values else []
        else:
            face_indices = values[0] if values else []

        labels = []
        for idx in face_indices:
            if idx is not None and idx < len(surfaces):
                labels.append(surfaces[idx].get('type'))
            else:
                labels.append(None)

        return labels

    def get_roof_slope_angles(self, geometry: dict) -> List[float]:
        """Extract roof slope angles (b3_hellingshoek) from LOD2.2 semantics."""
        semantics = geometry.get('semantics', {})
        surfaces = semantics.get('surfaces', [])
        angles = []
        for surf in surfaces:
            if surf.get('type') == 'RoofSurface':
                angle = surf.get('b3_hellingshoek', None)
                if angle is not None:
                    angles.append(angle)
        return angles

    def pair_lod13_lod22(self) -> List[dict]:
        """
        Find every BuildingPart that has both a LOD1.3 and a LOD2.2 geometry
        and pair them up. This is the link between model input (LOD1.3) and
        ground truth (LOD2.2) that everything downstream depends on.

        Each returned dict holds building_id, part_id, the parent building's
        attributes, and the two geometry dicts (lod13, lod22).
        """
        paired = []

        for part_id, part in self.get_building_parts().items():
            lod13 = self.get_geometry_by_lod(part, '1.3')
            lod22 = self.get_geometry_by_lod(part, '2.2')

            if lod13 is None or lod22 is None:
                continue

            # Get parent building attributes
            parent_ids = part.get('parents', [])
            parent_id = parent_ids[0] if parent_ids else None
            parent = self.city_objects.get(parent_id, {})
            attributes = parent.get('attributes', {})

            paired.append({
                'building_id': parent_id or part_id,
                'part_id': part_id,
                'attributes': attributes,
                'lod13': lod13,
                'lod22': lod22
            })

        return paired

    def summary(self) -> dict:
        """Print and return a summary of the tile."""
        buildings = self.get_buildings()
        parts = self.get_building_parts()
        paired = self.pair_lod13_lod22()

        # Count dak types
        dak_types = {}
        for b in buildings.values():
            dt = b.get('attributes', {}).get('b3_dak_type', 'unknown')
            dak_types[dt] = dak_types.get(dt, 0) + 1

        summary = {
            'file': self.filename,
            'version': self.data.get('version', 'unknown'),
            'total_vertices': len(self.vertices),
            'total_city_objects': len(self.city_objects),
            'buildings': len(buildings),
            'building_parts': len(parts),
            'paired_lod13_lod22': len(paired),
            'dak_type_distribution': dak_types,
            'coordinate_system': self.data.get('metadata', {}).get('referenceSystem', 'unknown')
        }

        print(f"\n{'='*60}")
        print(f"Tile Summary: {self.filename}")
        print(f"{'='*60}")
        for key, val in summary.items():
            print(f"  {key}: {val}")

        return summary


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "../sample_3dbag_tile.city.json"
    parser = CityJSONParser(filepath)
    parser.summary()
    paired = parser.pair_lod13_lod22()
    print(f"\nPaired buildings: {len(paired)}")
    for p in paired[:3]:
        print(f"  {p['building_id']}: dak_type={p['attributes'].get('b3_dak_type')}")