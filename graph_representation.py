"""
Graph Representation of Buildings

Represents a building as a planar graph: nodes are vertices (footprint
corners, eave points, ridge points) with x/y/z and a type; edges are the
structural connections between them (footprint, wall, roof, ridge), each
carrying length and orientation.

This gives an alternative framing of the LOD1.3 -> LOD2.2 upgrade as graph
augmentation — start from the LOD1.3 footprint graph and add the roof nodes
and edges. It's a simple dict-based graph rather than a NetworkX graph, so
it has no extra dependencies; to_networkx()/from_networkx() below convert
to/from NetworkX if you have it installed and want its algorithms.
"""

import math
from typing import Dict, List, Tuple


class BuildingGraph:
    """
    Graph representation of a building's geometry.

    Nodes are identified by integer IDs.
    Each node has attributes: x, y, z, node_type
    Each edge has attributes: length, edge_type, orientation
    """

    def __init__(self, building_id: str = ""):
        self.building_id = building_id
        self.nodes: Dict[int, dict] = {}
        self.edges: Dict[Tuple[int, int], dict] = {}
        self._next_node_id = 0

    def add_node(self, node_id: int = None, x: float = 0.0, y: float = 0.0,
                 z: float = 0.0, node_type: str = "unknown", **kwargs) -> int:
        """Add a node with position and type attributes."""
        if node_id is None:
            node_id = self._next_node_id
            self._next_node_id += 1
        else:
            self._next_node_id = max(self._next_node_id, node_id + 1)

        self.nodes[node_id] = {
            'x': x, 'y': y, 'z': z,
            'node_type': node_type,
            **kwargs
        }
        return node_id

    def add_edge(self, n1: int, n2: int, edge_type: str = "unknown", **kwargs):
        """Add an undirected edge between two nodes."""
        if n1 not in self.nodes or n2 not in self.nodes:
            raise ValueError(f"Both nodes must exist: {n1}, {n2}")

        # Compute length
        p1 = self.nodes[n1]
        p2 = self.nodes[n2]
        length = math.sqrt(
            (p1['x'] - p2['x'])**2 +
            (p1['y'] - p2['y'])**2 +
            (p1['z'] - p2['z'])**2
        )

        # Compute 2D orientation (for horizontal edges)
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        orientation = math.degrees(math.atan2(dy, dx)) % 180 if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else 0

        # Store as sorted tuple for undirected edge
        key = (min(n1, n2), max(n1, n2))
        self.edges[key] = {
            'length': length,
            'orientation': orientation,
            'edge_type': edge_type,
            **kwargs
        }

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get all neighbors of a node."""
        neighbors = []
        for (n1, n2) in self.edges:
            if n1 == node_id:
                neighbors.append(n2)
            elif n2 == node_id:
                neighbors.append(n1)
        return neighbors

    def get_nodes_by_type(self, node_type: str) -> List[int]:
        """Get all nodes of a given type."""
        return [nid for nid, attrs in self.nodes.items()
                if attrs['node_type'] == node_type]

    def get_edges_by_type(self, edge_type: str) -> List[Tuple[int, int]]:
        """Get all edges of a given type."""
        return [(n1, n2) for (n1, n2), attrs in self.edges.items()
                if attrs['edge_type'] == edge_type]

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return len(self.edges)

    def summary(self) -> str:
        """Return a text summary of the graph."""
        node_types = {}
        for n in self.nodes.values():
            t = n['node_type']
            node_types[t] = node_types.get(t, 0) + 1

        edge_types = {}
        for e in self.edges.values():
            t = e['edge_type']
            edge_types[t] = edge_types.get(t, 0) + 1

        lines = [
            f"BuildingGraph: {self.building_id}",
            f"  Nodes: {self.node_count()} {node_types}",
            f"  Edges: {self.edge_count()} {edge_types}",
        ]
        return '\n'.join(lines)

    def __repr__(self):
        return self.summary()


# --- build graphs from LOD1.3 and LOD2.2 geometry ---

def build_lod13_graph(parser, geometry: dict, building_id: str = "") -> BuildingGraph:
    """
    Build a graph from LOD1.3 geometry.

    The LOD1.3 graph represents the building as:
    - Ground nodes (footprint vertices at ground level)
    - Eave nodes (footprint vertices extruded to building height — flat top in LOD1.3)
    - Footprint edges (connecting ground nodes)
    - Top edges (connecting eave nodes)
    - Wall edges (vertical, connecting ground to eave)
    """
    graph = BuildingGraph(building_id)
    vertices = parser.vertices
    faces = parser.get_shell_faces(geometry)

    if not faces:
        return graph

    # Identify ground and top faces
    face_avg_z = []
    for face in faces:
        avg_z = sum(vertices[idx][2] for idx in face) / len(face)
        face_avg_z.append((avg_z, face))

    face_avg_z.sort(key=lambda x: x[0])
    ground_face = face_avg_z[0][1]
    top_face = face_avg_z[-1][1]

    # Map original vertex indices to graph node IDs
    vertex_to_node = {}

    # Add ground nodes
    for i, vidx in enumerate(ground_face):
        v = vertices[vidx]
        nid = graph.add_node(x=v[0], y=v[1], z=v[2], node_type='ground',
                             original_vertex=vidx)
        vertex_to_node[vidx] = nid

    # Add eave/top nodes
    for i, vidx in enumerate(top_face):
        v = vertices[vidx]
        nid = graph.add_node(x=v[0], y=v[1], z=v[2], node_type='eave',
                             original_vertex=vidx)
        vertex_to_node[vidx] = nid

    # Add footprint edges (ground ring)
    n_ground = len(ground_face)
    for i in range(n_ground):
        j = (i + 1) % n_ground
        nid_i = vertex_to_node[ground_face[i]]
        nid_j = vertex_to_node[ground_face[j]]
        graph.add_edge(nid_i, nid_j, edge_type='footprint')

    # Add top edges (eave ring)
    n_top = len(top_face)
    for i in range(n_top):
        j = (i + 1) % n_top
        nid_i = vertex_to_node[top_face[i]]
        nid_j = vertex_to_node[top_face[j]]
        graph.add_edge(nid_i, nid_j, edge_type='top')

    # Add wall edges (vertical connections)
    # Match ground and eave nodes by x,y proximity
    ground_nodes = graph.get_nodes_by_type('ground')
    eave_nodes = graph.get_nodes_by_type('eave')

    for gid in ground_nodes:
        g = graph.nodes[gid]
        # Find closest eave node in 2D
        best_eid = None
        best_dist = float('inf')
        for eid in eave_nodes:
            e = graph.nodes[eid]
            dist = math.sqrt((g['x'] - e['x'])**2 + (g['y'] - e['y'])**2)
            if dist < best_dist:
                best_dist = dist
                best_eid = eid
        if best_eid is not None and best_dist < 0.1:  # should be directly above
            graph.add_edge(gid, best_eid, edge_type='wall')

    return graph


def build_lod22_graph(parser, geometry: dict, building_id: str = "") -> BuildingGraph:
    """
    Build a graph from LOD2.2 geometry (ground truth).

    The LOD2.2 graph includes:
    - Ground nodes
    - Eave nodes (at roof base)
    - Ridge nodes (at roof peak)
    - All structural edges with semantic labels
    """
    graph = BuildingGraph(building_id)
    vertices = parser.vertices
    faces = parser.get_shell_faces(geometry)
    labels = parser.get_semantic_labels(geometry)

    if not faces or not labels:
        return graph

    # Collect vertices by semantic context
    vertex_to_node = {}

    # First pass: categorize all vertices by which faces they belong to
    vertex_faces = {}  # vertex_index -> set of face semantic types
    for face, label in zip(faces, labels):
        for vidx in face:
            if vidx not in vertex_faces:
                vertex_faces[vidx] = set()
            if label:
                vertex_faces[vidx].add(label)

    # Determine node type based on which faces a vertex belongs to
    for vidx, face_types in vertex_faces.items():
        v = vertices[vidx]

        if face_types == {'GroundSurface'} or (
            'GroundSurface' in face_types and 'WallSurface' in face_types
            and 'RoofSurface' not in face_types):
            node_type = 'ground'
        elif 'RoofSurface' in face_types and 'WallSurface' not in face_types:
            # Only roof faces → likely a ridge point
            node_type = 'ridge'
        elif 'RoofSurface' in face_types and 'WallSurface' in face_types:
            # Both roof and wall → eave point
            node_type = 'eave'
        elif 'WallSurface' in face_types and 'RoofSurface' not in face_types:
            if 'GroundSurface' in face_types:
                node_type = 'ground'
            else:
                node_type = 'eave'  # top of wall = eave
        else:
            node_type = 'unknown'

        nid = graph.add_node(x=v[0], y=v[1], z=v[2], node_type=node_type,
                             original_vertex=vidx)
        vertex_to_node[vidx] = nid

    # Add edges from face connectivity
    added_edges = set()
    for face, label in zip(faces, labels):
        n = len(face)
        for i in range(n):
            j = (i + 1) % n
            vidx_i, vidx_j = face[i], face[j]

            if vidx_i not in vertex_to_node or vidx_j not in vertex_to_node:
                continue

            nid_i = vertex_to_node[vidx_i]
            nid_j = vertex_to_node[vidx_j]
            edge_key = (min(nid_i, nid_j), max(nid_i, nid_j))

            if edge_key not in added_edges:
                # Determine edge type from the face it belongs to
                node_i_type = graph.nodes[nid_i]['node_type']
                node_j_type = graph.nodes[nid_j]['node_type']

                if label == 'GroundSurface':
                    edge_type = 'footprint'
                elif label == 'WallSurface':
                    if node_i_type == 'ground' or node_j_type == 'ground':
                        edge_type = 'wall'
                    else:
                        edge_type = 'gable'  # triangular gable end wall
                elif label == 'RoofSurface':
                    if 'ridge' in (node_i_type, node_j_type):
                        if node_i_type == 'ridge' and node_j_type == 'ridge':
                            edge_type = 'ridge'
                        else:
                            edge_type = 'slope'
                    else:
                        edge_type = 'roof_eave'
                else:
                    edge_type = 'unknown'

                graph.add_edge(nid_i, nid_j, edge_type=edge_type)
                added_edges.add(edge_key)

    return graph


# --- comparing LOD1.3 vs LOD2.2 graphs ---

def compare_graphs(lod13_graph: BuildingGraph, lod22_graph: BuildingGraph) -> dict:
    """
    Compare LOD1.3 and LOD2.2 graphs to understand what augmentation is needed.

    Returns summary of differences.
    """
    # Node analysis
    lod13_types = {}
    for n in lod13_graph.nodes.values():
        t = n['node_type']
        lod13_types[t] = lod13_types.get(t, 0) + 1

    lod22_types = {}
    for n in lod22_graph.nodes.values():
        t = n['node_type']
        lod22_types[t] = lod22_types.get(t, 0) + 1

    # Edge analysis
    lod13_edge_types = {}
    for e in lod13_graph.edges.values():
        t = e['edge_type']
        lod13_edge_types[t] = lod13_edge_types.get(t, 0) + 1

    lod22_edge_types = {}
    for e in lod22_graph.edges.values():
        t = e['edge_type']
        lod22_edge_types[t] = lod22_edge_types.get(t, 0) + 1

    # What needs to be added
    added_node_types = {}
    for t, count in lod22_types.items():
        diff = count - lod13_types.get(t, 0)
        if diff > 0:
            added_node_types[t] = diff

    return {
        'lod13_nodes': lod13_graph.node_count(),
        'lod22_nodes': lod22_graph.node_count(),
        'lod13_edges': lod13_graph.edge_count(),
        'lod22_edges': lod22_graph.edge_count(),
        'lod13_node_types': lod13_types,
        'lod22_node_types': lod22_types,
        'lod13_edge_types': lod13_edge_types,
        'lod22_edge_types': lod22_edge_types,
        'nodes_to_add': added_node_types,
        'total_nodes_to_add': lod22_graph.node_count() - lod13_graph.node_count(),
        'total_edges_to_add': lod22_graph.edge_count() - lod13_graph.edge_count(),
    }


# --- NetworkX conversion (optional dependency) ---

def to_networkx(graph: BuildingGraph):
    """
    Convert BuildingGraph to a NetworkX graph.
    Requires: import networkx as nx
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("NetworkX is required. Install with: pip install networkx")

    G = nx.Graph()
    G.graph['building_id'] = graph.building_id

    for nid, attrs in graph.nodes.items():
        G.add_node(nid, **attrs)

    for (n1, n2), attrs in graph.edges.items():
        G.add_edge(n1, n2, **attrs)

    return G


def from_networkx(G, building_id: str = "") -> BuildingGraph:
    """Convert a NetworkX graph back to BuildingGraph."""
    graph = BuildingGraph(building_id or G.graph.get('building_id', ''))

    for nid, attrs in G.nodes(data=True):
        graph.add_node(node_id=nid, **attrs)

    for n1, n2, attrs in G.edges(data=True):
        graph.add_edge(n1, n2, **attrs)

    return graph


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from cityjson_parser import CityJSONParser

    filepath = sys.argv[1] if len(sys.argv) > 1 else "../sample_3dbag_tile.city.json"
    parser = CityJSONParser(filepath)

    paired = parser.pair_lod13_lod22()
    for p in paired:
        bid = p['building_id']
        dak = p['attributes'].get('b3_dak_type', '?')

        print(f"\n{'='*60}")
        print(f"Building: {bid} (dak_type: {dak})")

        g13 = build_lod13_graph(parser, p['lod13'], bid)
        g22 = build_lod22_graph(parser, p['lod22'], bid)

        print(f"\nLOD1.3 graph:")
        print(f"  {g13.summary()}")
        print(f"\nLOD2.2 graph:")
        print(f"  {g22.summary()}")

        comp = compare_graphs(g13, g22)
        print(f"\nGraph augmentation needed:")
        print(f"  Add {comp['total_nodes_to_add']} nodes: {comp['nodes_to_add']}")
        print(f"  Add {comp['total_edges_to_add']} edges")
        print(f"  LOD2.2 edge types: {comp['lod22_edge_types']}")