"""
Extract real geometry + ML results from the thesis pipeline data and bake
them into a compact JS data file for the 3D pipeline animation.

Reads (read-only):
  data/<TILE>.city.json            LOD1.3 input + LOD2.2 ground truth
  output/<TILE>_lod22.city.json    reconstructed LOD2.2 (pipeline output)
  output/<TILE>_evaluation.json    per-building predictions + errors
  output/ml_results_revised.json   model comparison / selected features

Writes: data.js in this scratchpad directory (const ANIM = {...};)
"""
import sys, json, math, os

# repo root = two levels above this script (docs/animation/ -> repo)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(OUT_DIR, "..", ".."))
TILE = "9-572-624"
HERO = "NL.IMBAG.Pand.0307100000587119-0"   # user-chosen gabled building
CONTEXT_RADIUS = 420.0   # m around hero for the city context
NEIGHBOR_RADIUS = 50.0   # matches pipeline's compute_neighbor_features

sys.path.insert(0, REPO)
from cityjson_parser import CityJSONParser
import feature_extraction as fe

# ---------------------------------------------------------------- helpers

def r2(x):
    return round(float(x), 2)

def polygon_area_signed(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a / 2.0

def ensure_ccw(pts):
    return pts if polygon_area_signed(pts) > 0 else pts[::-1]

def earclip(pts):
    """Ear-clipping triangulation of a simple 2D polygon.
    Returns list of index triples into pts. Falls back to fan."""
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]
    idx = list(range(n))
    if polygon_area_signed(pts) < 0:
        idx = idx[::-1]
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        ear_found = False
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            ax, ay = pts[i0]; bx, by = pts[i1]; cx, cy = pts[i2]
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if cross <= 1e-12:
                continue  # reflex
            # any other vertex inside?
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                px, py = pts[j]
                d1 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                d2 = (cx - bx) * (py - by) - (cy - by) * (px - bx)
                d3 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
                if d1 >= -1e-12 and d2 >= -1e-12 and d3 >= -1e-12:
                    inside = True
                    break
            if inside:
                continue
            tris.append((i0, i1, i2))
            del idx[k]
            ear_found = True
            break
        if not ear_found:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    if not tris:  # degenerate fallback: fan
        tris = [(0, i, i + 1) for i in range(1, n - 1)]
    return tris

def newell_normal(coords):
    nx = ny = nz = 0.0
    n = len(coords)
    for i in range(n):
        x1, y1, z1 = coords[i]
        x2, y2, z2 = coords[(i + 1) % n]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)
    return (nx, ny, nz)

def triangulate_face3d(coords):
    """Triangulate a planar 3D polygon: project on dominant normal axis."""
    nx, ny, nz = newell_normal(coords)
    ax_, ay_, az_ = abs(nx), abs(ny), abs(nz)
    if az_ >= ax_ and az_ >= ay_:
        proj = [(c[0], c[1]) for c in coords]
        flip = nz < 0
    elif ax_ >= ay_:
        proj = [(c[1], c[2]) for c in coords]
        flip = nx < 0
    else:
        proj = [(c[2], c[0]) for c in coords]
        flip = ny < 0
    tris = earclip(proj)
    if flip:
        tris = [(a, c, b) for (a, b, c) in tris]
    return tris

# ---------------------------------------------------------------- load

print("Loading tiles (read-only)...")
parser = CityJSONParser(os.path.join(REPO, "data", f"{TILE}.city.json"))
out_parser = CityJSONParser(os.path.join(REPO, "output", f"{TILE}_lod22.city.json"))
with open(os.path.join(REPO, "output", f"{TILE}_evaluation.json")) as f:
    ev = json.load(f)
with open(os.path.join(REPO, "output", "ml_results_revised.json")) as f:
    mlres = json.load(f)

evmap = {e["part_id"]: e for e in ev["per_building"]}
paired = parser.pair_lod13_lod22()
paired_map = {p["part_id"]: p for p in paired}
print(f"data tile: {len(parser.get_buildings())} buildings, "
      f"{len(paired)} LOD1.3/LOD2.2 paired parts, {len(evmap)} evaluated")

# ---------------------------------------------------------------- hero pick

def features_for(part_id):
    p = paired_map.get(part_id)
    if not p:
        return None
    return fe.extract_lod13_features(parser, p["lod13"], p["attributes"])

# centroids of all evaluated buildings (for neighbor counting)
ev_centroids = {}
for pid in evmap:
    f = features_for(pid)
    if f:
        ev_centroids[pid] = (f["centroid_x"], f["centroid_y"], f)

def n_neighbors_of(pid, radius=NEIGHBOR_RADIUS):
    cx, cy, _ = ev_centroids[pid]
    cnt = 0
    for q, (qx, qy, _) in ev_centroids.items():
        if q == pid:
            continue
        if math.hypot(cx - qx, cy - qy) <= radius:
            cnt += 1
    return cnt

if HERO not in evmap:
    sys.exit(f"hero {HERO} not found in evaluation data")
if HERO not in ev_centroids:
    sys.exit(f"hero {HERO} feature extraction failed")
hero_ev = evmap[HERO]
hero_f = ev_centroids[HERO][2]
hero_nn = n_neighbors_of(HERO)
hero_rise = hero_ev["gt_ridge_z"] - hero_ev["gt_eave_z"]
print(f"\nHERO = {HERO} (user-chosen)")
print(f"  neighbors={hero_nn} rise={hero_rise:.2f}m "
      f"ridge_err={hero_ev['ridge_height_abs_error']:.3f} "
      f"vol_err={hero_ev['volume_relative_error']:.3f} "
      f"area={hero_f['footprint_area']:.1f} ar={hero_f['aspect_ratio']:.2f}")

HX, HY = hero_f["centroid_x"], hero_f["centroid_y"]
HZ = hero_f["ground_z"]

def loc(v):
    """world -> local coords (meters, z up, origin at hero centroid/ground)"""
    return [r2(v[0] - HX), r2(v[1] - HY), r2(v[2] - HZ)]

# ---------------------------------------------------------------- context city

out_parts = out_parser.get_building_parts()

def extract_lod13_prism(part_id, p):
    faces = parser.get_shell_faces(p["lod13"])
    fp_idx = fe.extract_footprint(faces, parser.vertices)
    if not fp_idx or len(fp_idx) < 3:
        return None
    fp = [(parser.vertices[i][0], parser.vertices[i][1]) for i in fp_idx]
    all_idx = set()
    for face in faces:
        all_idx.update(face)
    zs = [parser.vertices[i][2] for i in all_idx]
    gz, tz = min(zs), max(zs)
    cx = sum(x for x, _ in fp) / len(fp)
    cy = sum(y for _, y in fp) / len(fp)
    return fp, gz, tz, cx, cy

context = []
recon = []            # reconstructed LOD2.2 meshes for the finale morph
skipped = 0
for pid, p in paired_map.items():
    prism = extract_lod13_prism(pid, p)
    if prism is None:
        skipped += 1
        continue
    fp, gz, tz, cx, cy = prism
    d = math.hypot(cx - HX, cy - HY)
    if d > CONTEXT_RADIUS:
        continue
    fp_ccw = ensure_ccw(fp)
    entry = {
        "fp": [[r2(x - HX), r2(y - HY)] for x, y in fp_ccw],
        "g": r2(gz - HZ),
        "t": r2(tz - HZ),
        "d": r2(d),
    }
    if pid == HERO:
        entry["hero"] = 1
    e = evmap.get(pid)
    if e:
        entry["pt"] = {"flat": 0, "gabled": 1, "hipped": 2}[e["predicted_type"]]
        entry["ok"] = 1 if e["type_correct"] else 0
    context.append(entry)

    # reconstructed mesh (predicted LOD2.2) for finale morph
    op = out_parts.get(pid)
    if op is not None:
        g22 = out_parser.get_geometry_by_lod(op, "2.2") or (
            op["geometry"][0] if op.get("geometry") else None)
        if g22:
            faces22 = out_parser.get_shell_faces(g22)
            labels22 = out_parser.get_semantic_labels(g22)
            vidx = []
            vmap = {}
            verts = []
            tris = []
            roof_flags = []
            for fi, face in enumerate(faces22):
                coords = [out_parser.vertices[i] for i in face]
                local_ids = []
                for i in face:
                    if i not in vmap:
                        vmap[i] = len(verts)
                        verts.append(loc(out_parser.vertices[i]))
                    local_ids.append(vmap[i])
                is_roof = 1 if (fi < len(labels22) and labels22[fi] == "RoofSurface") else 0
                for (a, b, c) in triangulate_face3d(coords):
                    tris.append([local_ids[a], local_ids[b], local_ids[c]])
                    roof_flags.append(is_roof)
            recon.append({
                "i": len(context) - 1,     # index into context list
                "v": verts,
                "t": tris,
                "r": roof_flags,
                "top13": r2(tz - HZ),      # morph start z for above-ground verts
            })

print(f"\ncontext buildings within {CONTEXT_RADIUS:.0f} m: {len(context)} "
      f"({sum(1 for c in context if 'pt' in c)} evaluated, "
      f"{len(recon)} with reconstructed LOD2.2), {skipped} skipped tile-wide")

# ---------------------------------------------------------------- hero detail

def mesh_from_geometry(pp, geometry):
    faces = pp.get_shell_faces(geometry)
    labels = pp.get_semantic_labels(geometry)
    vmap, verts, tris, roof_flags = {}, [], [], []
    for fi, face in enumerate(faces):
        coords = [pp.vertices[i] for i in face]
        local_ids = []
        for i in face:
            if i not in vmap:
                vmap[i] = len(verts)
                verts.append(loc(pp.vertices[i]))
            local_ids.append(vmap[i])
        is_roof = 1 if (fi < len(labels) and labels[fi] == "RoofSurface") else 0
        for (a, b, c) in triangulate_face3d(coords):
            tris.append([local_ids[a], local_ids[b], local_ids[c]])
            roof_flags.append(is_roof)
    return {"v": verts, "t": tris, "r": roof_flags}

hero_pair = paired_map[HERO]
hero_gt_mesh = mesh_from_geometry(parser, hero_pair["lod22"])
hero_op = out_parts[HERO]
hero_pred_geom = out_parser.get_geometry_by_lod(hero_op, "2.2") or hero_op["geometry"][0]
hero_pred_mesh = mesh_from_geometry(out_parser, hero_pred_geom)

# ridge line of the predicted roof (verts at max z)
maxz = max(v[2] for v in hero_pred_mesh["v"])
ridge_pts = [v for v in hero_pred_mesh["v"] if abs(v[2] - maxz) < 0.05]
# eave z of predicted roof: min z among roof-face vertices
roof_vids = set()
for tri, rf in zip(hero_pred_mesh["t"], hero_pred_mesh["r"]):
    if rf:
        roof_vids.update(tri)
eave_z_pred = min(hero_pred_mesh["v"][i][2] for i in roof_vids) if roof_vids else None
top13_local = r2(hero_f["top_z"] - HZ)
print(f"\nhero predicted eave z (local) = {eave_z_pred}, LOD1.3 top = {top13_local}"
      f"  -> eave==top13: {abs((eave_z_pred or 0) - top13_local) < 0.06}")
print(f"hero ridge pts: {ridge_pts}")

# neighbor context, mirroring compute_neighbor_features (1/d weighting,
# predicted types = inference-time pass-1 view)
neigh = []
wsum = 0.0
acc = {"flat": 0.0, "gabled": 0.0, "hipped": 0.0}
h_acc = a_acc = 0.0
for q, (qx, qy, qf) in ev_centroids.items():
    if q == HERO:
        continue
    d = math.hypot(HX - qx, HY - qy)
    if d <= NEIGHBOR_RADIUS and d > 0.1:
        e = evmap[q]
        w = 1.0 / d
        wsum += w
        acc[e["pass1_type"]] += w
        h_acc += w * qf["building_height"]
        a_acc += w * qf["footprint_area"]
        neigh.append({"x": r2(qx - HX), "y": r2(qy - HY), "d": r2(d),
                      "t": {"flat": 0, "gabled": 1, "hipped": 2}[e["pass1_type"]]})
neighbor_feats = {
    "n_neighbors": len(neigh),
    "neighbor_mean_height": r2(h_acc / wsum) if wsum else 0,
    "neighbor_mean_area": r2(a_acc / wsum) if wsum else 0,
    "neighbor_frac_flat": round(acc["flat"] / wsum, 3) if wsum else 0,
    "neighbor_frac_gabled": round(acc["gabled"] / wsum, 3) if wsum else 0,
    "neighbor_frac_hipped": round(acc["hipped"] / wsum, 3) if wsum else 0,
}
print(f"hero neighbors within {NEIGHBOR_RADIUS:.0f} m: {len(neigh)}  {neighbor_feats}")

bh = hero_f["building_height"]
hero_out = {
    "id": HERO.replace("NL.IMBAG.Pand.", ""),
    "full_id": HERO,
    "fp": [[r2(x - HX), r2(y - HY)] for x, y in
           ensure_ccw(list(hero_f["_footprint_2d"]))],
    "ground": 0.0,
    "top13": top13_local,
    "height13": r2(bh),
    "features": {k: round(float(hero_f[k]), 3) for k in [
        "footprint_area", "footprint_perimeter", "n_footprint_vertices",
        "aspect_ratio", "compactness", "rectangularity",
        "mbr_length", "mbr_width", "edge_length_ratio",
        "building_height", "orientation", "longest_edge_length"]
        if hero_f.get(k) is not None},
    "vol_lod13": r2(hero_f["vol_lod13"]) if hero_f.get("vol_lod13") else None,
    "neighbor_features": neighbor_feats,
    "neighbors": neigh,
    "eval": {
        "predicted_type": hero_ev["predicted_type"],
        "pass1_type": hero_ev["pass1_type"],
        "gt_type": hero_ev["gt_type"],
        "ridge_err": round(hero_ev["ridge_height_abs_error"], 3),
        "vol_rel_err": round(hero_ev["volume_relative_error"], 4),
        "pred_ridge_z": r2(hero_ev["predicted_ridge_z"] - HZ),
        "gt_ridge_z": r2(hero_ev["gt_ridge_z"] - HZ),
        "gt_eave_z": r2(hero_ev["gt_eave_z"] - HZ),
    },
    "ridge_ratio_pred": round((hero_ev["predicted_ridge_z"] - HZ) / bh, 3),
    "ridge_ratio_gt": round((hero_ev["gt_ridge_z"] - HZ) / bh, 3),
    "ridge_pts": ridge_pts,
    "pred_mesh": hero_pred_mesh,
    "gt_mesh": hero_gt_mesh,
}

# ---------------------------------------------------------------- tile stats

tile_stats = {
    "tile": TILE,
    "n_buildings_tile": len(parser.get_buildings()),
    "n_paired": len(paired),
    "n_evaluated": ev["n_evaluated"],
    "tile_accuracy": round(ev["metrics"]["classification"]["accuracy"], 4),
    "n_context": len(context),
    "n_recon_context": len(recon),
}

ml_out = {
    "selected_features": mlres["feature_selection"]["selected_features"],
    "best_model": mlres["best_classifier"]["model"],
    "best_f1": round(mlres["best_classifier"]["f1"], 3),
    "classifiers": {k: round(v["f1"], 3)
                    for k, v in mlres["classifier_comparison"].items()},
    "gabled_regressor_mae": round(
        mlres["vertex_prediction"]["gabled"]["Random Forest"]["mae"], 4),
    "regressors_gabled": {k: round(v["mae"], 4)
                          for k, v in mlres["vertex_prediction"]["gabled"].items()},
}

ANIM = {
    "stats": tile_stats,
    "ml": ml_out,
    "hero": hero_out,
    "context": context,
    "recon": recon,
}

out_path = os.path.join(OUT_DIR, "data.js")
with open(out_path, "w") as f:
    f.write("const ANIM = ")
    json.dump(ANIM, f, separators=(",", ":"))
    f.write(";\n")
size = os.path.getsize(out_path)
print(f"\nwrote {out_path}  ({size/1e6:.2f} MB)")
