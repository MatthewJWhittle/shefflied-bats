# 0002: Layer pipeline must scale UK-wide, so it works tile by tile

Date: 24 Oct 2026. Status: Accepted.

## Context
Matt set UK-wide scale as a core design principle. The study area is 14,369 km² and the UK is about 243,600 km², roughly 17 times larger. Some layer steps hold the whole area in memory. Buildings rasterising (#95) peaks at about 5.5 GB for the study area, which suggests something like 90 GB UK-wide, more than one ordinary machine has. Measured baselines and projections are in `docs/scaling-assessment.md` (#56).

## Options
1. Keep whole-area in-memory steps and use a bigger machine. Rejected: memory grows with area and every run rebuilds everything.
2. Process each layer tile by tile on a fixed national tile layout. Chosen.
3. Move to a distributed raster engine now. Deferred: more moving parts than this stage needs.

## Decision
- Every layer step runs tile by tile on the fixed national 10,240 m tile layout anchored at 0,0 (EPSG:27700), the same layout the vegetation height download uses (#54).
- Each tile is read with an overlap at least as wide as the largest neighbourhood (landscape-scale) radius, then cropped back, so tile edges leave no seams.
- Tiles are independent, so they can run in parallel, and a change rebuilds only the tiles and layers it touches (#41).
- Existing whole-area steps, such as #95, can stay for now. New or reworked steps follow this record, and the old ones move over when the tiled store (#41) lands.

## Why
With this design, memory depends on the tile size rather than the area, so going UK-wide just means more tiles. It also fits two earlier rules: rebuild only what a change affects, and closeness rather than exactness (#38).

## How it's judged
1. Peak memory for any layer step stays roughly flat as the number of tiles grows. Run on 1 tile and on 16 tiles: the peak should differ by no more than 20%.
2. A tiled build of the study area matches the existing layers under the #38 tolerances (95% of cells within 1% of the layer's range for continuous layers, 95% match for category layers), with no visible seams where tiles meet.
3. Changing one input rebuilds only the tiles and layers that depend on it, and the build log shows this.

## Revisit when
- A source dataset can't be split into tiles (for example, a national statistic that needs the whole area at once).
- Neighbourhood radii grow so large that the overlap stops being small next to the tile size.
- Cloud or distributed compute is added.
