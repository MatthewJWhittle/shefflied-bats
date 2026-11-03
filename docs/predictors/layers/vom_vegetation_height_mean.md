# Vegetation canopy height — mean (`vom_vegetation_height_mean`)

Mean top-of-canopy height from the Environment Agency **Vegetation Object
Model (VOM)**, summarised to 100 m cells.

## What it shows

Average canopy-top height above ground (metres) for vegetation objects that
the VOM classifies in each model cell.

## What it does not show

- Plants or hedges **below the 2.5 m VOM threshold**
- Vertical structure inside the canopy (only the **top** of canopy)
- Leaf-on detail for features that disappear or shrink in **leaf-off** surveys

## Source (verified)

Public product page
[LIDAR Vegetation Object Model (VOM)](https://www.data.gov.uk/dataset/227ab487-e8f2-4cbb-b26a-9e6d3b662265/lidar-vegetation-object-model-vom)
(Environment Agency, OGL):

- Each pixel is “the height of **top of canopy** above ground, for all
  classified vegetation objects **above a threshold of 2.5 metres**.”
- Production is automated; EA notes possible mis-classification of some
  non-vegetation objects.

National LiDAR Programme surveys feeding these products were captured in
**winter months** (approximately November–April)
([National LIDAR Programme](https://environment.data.gov.uk/dataset/2e8d0733-4f43-48b4-9e51-631c25d1b0a9)).
Leaf-off capture plus the 2.5 m threshold means trimmed hedges and understorey
are largely absent from this predictor.

## Pipeline

`vom` rebuild step → optional 40 m height cap → mean/min/max/std on the model
grid → merge as `vom_vegetation_height_*`.

## Caveats

See YAML `caveats` (threshold, canopy-top only, leaf-off, mis-class risk, 40 m
cap). Neighbourhood versions (`*_500m` etc.) inherit these limits via lineage.

## Role

**Data / ecology** — source limits checked against the public EA pages above.
