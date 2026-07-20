# Interactive 3D pipeline animation

A self-contained, browser-based 3D walkthrough of the LOD1.3 → LOD2.2
pipeline, told through one real building: tile **9-572-624**, hero building
**NL.IMBAG.Pand.0307100000587119** (gabled, correctly classified, ridge
height error 1.6 cm).

Eight stages follow the thesis pipeline: the LOD1.3 tile → the in-scope
building found in the map → the building isolated on its own → feature
extraction (measurements drawn on the block) → roof-type classification
(three candidate roofs hover over the building) → ridge-point prediction →
deterministic construction (the roof structure flies in and docks) →
comparison against the LiDAR ground truth → the reconstructed building
returns to the map as the whole neighborhood is reconstructed at once.

Every number and every mesh shown is a real pipeline result — extracted
from `data/`, `output/`, and `output/ml_results_revised.json`, never
hand-typed.

## Viewing

Open [index.html](index.html) in any browser (double-click works — no build,
no dependencies, no network requests), or serve the repo with GitHub Pages
and share the URL: enable Pages for the `main` branch with the `/docs`
folder, and the animation lives at `/animation/`.

Controls: auto-plays through the stages; click a stage chip to jump,
space to pause. Drag to orbit, right-drag (or shift-drag) to pan, scroll to
zoom — the scene is fully 3D at all times. The **Building layers** card
toggles the hero building's footprint, walls, and roof independently.

## Files

- `index.html` — the whole application (inline WebGL renderer, no libraries)
- `data.js` — baked geometry + results for the featured neighborhood
  (486 real LOD1.3 buildings, 285 reconstructed LOD2.2 roofs)
- `extract_animation_data.py` — regenerates `data.js` from the repo's
  `data/` and `output/` folders (read-only; edit `TILE` / `HERO` at the top
  to feature a different building)
