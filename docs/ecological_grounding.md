# Ecological grounding for the South Yorkshire bat habitat models

## Ecology review

**What this is for.** A reference to check the habitat suitability models against. It sets out what UK natural history and the published literature say we *should* see for each species and record type (roost and in flight), so that choices about features, cut-offs and tuning can be judged on whether they are ecologically sensible, and not only on test scores.

**How to use it.** For each model, check four things: (1) do the top layers match the expected drivers (section 4); (2) do the response curves run in the expected direction; (3) does the suitable share of the study area sit in the expected band and order (section 5); (4) are any red flags present (sections 4 and 6). Where a model disagrees, decide whether it's real local ecology, recording bias (section 6) or a modelling artefact.

**Scope and sources.** Public sources only (listed in section 9). No figures are invented: where the literature gives no number, the expectation is written as a direction, an order or a band. The layer-by-layer expected directions are in `literature_expectations.md` / `.csv`. This doc is the reasoning behind them.

---

## 1. The study area

The repo's boundary (`sdm/data/boundaries.py`) is the four South Yorkshire districts (Barnsley, Doncaster, Rotherham and Sheffield) plus a 7 km buffer (`config.yml`: `study_area_buffer: 7000`, 100 m grid, British National Grid). The buffer takes in edges of the Dark Peak, north-east Derbyshire, West Yorkshire and north Nottinghamshire. It crosses five Natural England National Character Areas (NCAs):

| NCA | Where | Landscape |
|---|---|---|
| 51 Dark Peak | West Sheffield and Barnsley | High gritstone moorland, blanket bog, heather, gritstone edges, reservoirs, conifer blocks |
| 37 Yorkshire Southern Pennine Fringe | Sheffield's western suburbs, Penistone, Stocksbridge | Steep wooded valleys (cloughs), pasture, gritstone villages, the Don, Loxley and Rivelin valleys |
| 38 Notts, Derbyshire and Yorkshire Coalfield | Sheffield and Rotherham core, Barnsley, the Dearne | Urban and industrial, former collieries, restored spoil, river corridors, mixed farming |
| 30 Southern Magnesian Limestone | A narrow north–south band east of Rotherham and through Doncaster's west side | Arable plateau, limestone gorges (for example Anston Stones, Roche Abbey), estate woodland, caves |
| 39 Humberhead Levels | East Doncaster | Flat, low-lying drained farmland, Thorne and Hatfield Moors (lowland raised bog), drains, the Don and Went, gravel pits |

### What each landscape means for bats

| Landscape | Roosting | Foraging and commuting | Limits |
|---|---|---|---|
| **Moorland, blanket bog and gritstone edges** (Dark Peak) | Almost none: no trees, few buildings. Some crevices in rock and walls | Poor: exposed and windy, cold, few insects | Altitude and exposure. Bat activity falls with height and wind in Pennine studies (Bellamy et al. 2013, Yorkshire Dales). Expect low suitability for every species above the moorland edge |
| **Upland cloughs and wooded valleys** (Pennine Fringe) | Old oak and birch woods with tree holes; stone farm buildings and barns; old mills | Sheltered, insect-rich valley bottoms with streams; woodland edges | The best bat habitat in the west. A narrow band between moor and town. Woodland Myotis and brown long-eared should pick it up |
| **River corridors** (Don, Rivelin, Loxley, Sheaf, Porter, Dearne, Rother) | Bridges, culverts, riverside trees and mills | Daubenton's over calm water; soprano pipistrelle along tree-lined banks; commuting routes into and through the city | Lighting and weirs break up continuity. Through central Sheffield and Rotherham the corridor is lit and canalised |
| **Reservoirs** (Damflask, Agden, Strines, Ladybower edge, Langsett, Ulley, Thrybergh) | Rarely roosts themselves. Nearby woodland and buildings | Large open water: Daubenton's, soprano, noctule; Nathusius' on passage | Upland reservoirs are cold and exposed. Water there is less productive than lowland lakes |
| **Urban core** (central Sheffield, Rotherham, Barnsley, Doncaster) | Buildings (pipistrelles), but dense cores have few suitable gaps | Low. Lit streets, little vegetation. Pipistrelles and noctules feed at some lights | Light-averse species (Myotis, long-eared) avoid it (Stone et al. 2009; Russo & Ancillotto 2015) |
| **Suburbs and parks** (the "green" Sheffield west and south-west) | Houses, churches, mature park and garden trees | Gardens, parks and tree-lined streets | Good for common pipistrelle; mixed for soprano; patchy for the rest |
| **Arable and pasture to the east** (Magnesian Limestone, coalfield farmland, Humberhead Levels) | Farm buildings, churches, hedgerow trees | Hedgerows, woodland edges and watercourses carry the bats; open fields away from edges are little used (Boughey et al. 2011a) | Intensive arable is insect-poor. Bats depend on linear features |
| **Magnesian limestone gorges and caves** | Caves and mines are used for hibernation (mainly Myotis and brown long-eared) | Gorge woodlands | Hibernation sites matter for population but aren't summer roosts. The models here are summer-season records, mostly |
| **Post-industrial land** (spoil heaps, restored collieries, the Dearne valley wetlands like Old Moor) | Few roosts on the land itself. Nearby buildings | Wetlands and restored lakes in the Dearne are good foraging; bare spoil is poor | Restored land changes fast. Land cover maps may lag |
| **Lowland raised bog and wetland** (Thorne and Hatfield Moors) | Few: scrub and birch on the edges | Open, wet and insect-rich; noctule and pipistrelle feed over it | Remote, so few records. Expect under-recording |

---

## 2. What the study area actually looks like (from the model inputs)

Figures from the analysis stats in `ev_stats/` (77 layers at 100 m). Land cover is the share of each cell in a class. The mean share of each class across the study area:

| Class | Mean share | Note |
|---|---|---|
| Improved grassland | about 30% | The biggest single class |
| Arable | about 28% | Mostly east (Magnesian Limestone, Humberhead Levels) |
| Upland heathland | about 7% | Dark Peak |
| Grassland (semi-natural) | about 7% | |
| Suburban | about 6% | |
| Broadleaved woodland | about 5% | Only in about 18% of cells, but present within 1 km of 88% of cells. Small woods spread widely |
| Wetland | about 5% | Bog and fen, mainly Dark Peak and Humberhead Levels |
| Urban | about 2.5% | |
| Coniferous woodland | about 2% | |

Elevation: median about 89 m, upper 5% above about 440 m. Most of the area is lowland. The high ground is a western wedge.

What this means: a model that marks most of the lowland as suitable for a woodland specialist is wrong, because broadleaf woodland is only about 5% of land. Common pipistrelle is the only species where "much of the lowland" is a plausible answer.

**Data checks to raise with the implementation side** (they are not ecology, but they affect it):
- `terrain_dtm` reaches about 724 m. The highest ground in the Dark Peak is Kinder Scout at 636 m and is outside the buffer. Worth a check for spikes or a projection error.
- `bgs_coast_distance_to_coast` is capped at 7000 m, but the study area is 60 km or more from the sea. Most cells sit at the cap, and a few read 0. Inland, this layer can only carry noise or artefacts. It should come out of the feature set (the Nathusius' roost model uses it).
- `os_cover_water` has its 95th percentile at 100%, with a mean of about 9.5%. That seems too high for South Yorkshire, where reservoirs and rivers cover far less. Check whether no-data or the edge of the buffer is being counted as water.
- The distance layers (buildings, roads) are capped at 7000 m, so "far from everything" is a flat plateau.

---

## 3. Species: natural history in brief

UK status and distribution draw on the BCT species pages, the Mammal Society review (Mathews et al. 2018), the Mammal Society atlas (Crawley et al. 2020) and NBN Atlas maps. Record counts are those in the repo's training data (`presence_vs_background.csv`).

**Common pipistrelle (*Pipistrellus pipistrellus*).** The commonest and most widespread UK bat, found across Yorkshire. Roosts almost entirely in buildings (houses, including modern ones). Forages along edges, gardens, parks, woodland and hedgerows. It tolerates the urban edge more than any other UK bat, and will feed around some street lights. Avoids high exposed ground. Detection is easy and records are plentiful (630 in flight, 1215 roost). *Expect: widespread; the broadest map of all.*

**Soprano pipistrelle (*P. pygmaeus*).** Also widespread and common. Split from common pipistrelle only in the 1990s (Jones & van Parijs 1993), so older records are mixed. Roosts in buildings, often in larger colonies, usually near water. More tied to rivers, lakes and riparian woodland than common pipistrelle (Davidson-Watts & Jones 2006), and a little less urban-tolerant. *Expect: widespread but more patchy, following valleys and water.*

**Nathusius' pipistrelle (*P. nathusii*).** Rare, and a migrant. Many UK records are in autumn and winter, from animals coming over from the continent, but there are a few breeding sites. Strongly linked to large waterbodies, reservoirs and wetland. Roosts in buildings and trees near water. Records are few and often come from targeted surveys at lakes, so they cluster. *Expect: very restricted; the larger lowland waters only. Model only roost (no stats in `ev_stats`).*

**Noctule (*Nyctalus noctula*).** Widespread in England, though thinly. One of the largest UK bats. Roosts in trees: woodpecker holes and rot holes in mature trees, and rarely in buildings. Flies high and fast over open ground, water, woodland canopy and parkland, and travels several kilometres from roosts. Loud and easily detected, so in-flight records are spread widely. Roost records are scarcer. *Expect: in flight broad; roost restricted to mature trees and parkland.*

**Leisler's bat (*N. leisleri*).** Scarce in Britain and patchy. Yorkshire is not at its range edge, but records are few. Roosts mainly in tree holes, sometimes in buildings. Forages high in the open, like noctule. Calls overlap with noctule and serotine, so acoustic records are uncertain. *Expect: restricted, uncertain. Treat 32 roost records with caution.*

**Daubenton's bat (*Myotis daubentonii*).** Widespread wherever there's calm water. Roosts in trees near water, bridges, tunnels and culverts. Feeds low over calm water, trawling insects from the surface. Avoids light: lit stretches of water are used less. *Expect: narrow bands along rivers, canals, lakes and reservoirs. The clearest water signal of all.*

**Whiskered bat (*M. mystacinus*) and Brandt's bat (*M. brandtii*).** Both widespread but thinly recorded in northern England. The two are hard to tell apart in the hand and almost impossible from calls, so "whiskered" in-flight records are best read as whiskered/Brandt's. Both roost in buildings and trees near woodland, and forage along woodland edges, hedgerows and riparian vegetation. Brandt's is more tied to woodland and water. Light-averse. Brandt's has only 27 roost records here. *Expect: restricted to wooded valleys and well-treed farmland. Brandt's more restricted than whiskered.*

**Natterer's bat (*M. nattereri*).** Widespread but local. Roosts in old buildings (barns, churches), trees and bat boxes. Gleans insects in cluttered woodland and along tree-lined features. Strongly light-averse. Hibernates in caves and mines (possibly in the limestone gorges). *Expect: woodland and parkland; restricted.*

**Brown long-eared bat (*Plecotus auritus*).** Widespread and common in the UK, but often under-recorded by detectors because its calls are very quiet. Roosts in roof voids of old buildings (barns, churches, older houses) and in trees. Forages close to the roost, within 1–2 km, gleaning in woodland. Light-averse and a slow flier. Roost records (604) come largely from building surveys. *Expect: widespread in wooded farmland and valleys, absent from the urban core and moorland.*

---

## 4. Testable expectations by model

Direction: ↑ more suitable with more of the layer, ↓ less suitable, – no strong expectation. "Plausible suitable share" is an order and a broad band, not a target number.

| Model | Should matter, and which way | Plausible suitable share | Red flags |
|---|---|---|---|
| Common pipistrelle, roost | Distance to buildings ↓ (strong), suburban ↑, altitude ↓, open moorland ↓, broadleaf weak ↑ | Broadest roost map: most settled lowland | Woodland or water as a top driver; moorland suitable |
| Common pipistrelle, in flight | Suburban ↑, edge and tree height ↑, altitude ↓, arable ↓ (away from edges) | Broadest of all: most of the lowland | A narrow map. Urban core rated top (the core itself is poor; the edge is good) |
| Soprano pipistrelle, roost | Distance to buildings ↓, water ↑ nearby, tree cover ↑, altitude ↓ | Wide but narrower than common pipistrelle | No water signal at all |
| Soprano pipistrelle, in flight | Water ↑ (stronger than common pipistrelle), riparian trees ↑, urban ↓ | Wide but patchy, following valleys | Buildings beating water as the top layer (likely survey bias) |
| Nathusius' pipistrelle, roost | Large water ↑ (strong), wetland ↑, trees ↑, altitude ↓ | Very small: large lowland waters | Distance to coast as a driver (an artefact inland); suitability away from big water; upland reservoirs rated as highly as lowland lakes |
| Noctule, roost | Tree height and mature trees ↑, broadleaf ↑, distance to buildings – or ↑ (tree roosts), altitude ↓ | Restricted: woodland and parkland | Strong "closer to buildings" signal |
| Noctule, in flight | Broad. Open water and wetland ↑, woodland edge ↑, altitude ↓; buildings – | Broad, second only to the pipistrelles | A tight map (a high flyer that ranges widely) |
| Leisler's, roost and in flight | Mature broadleaf ↑, water and pasture ↑ for feeding, moorland ↓ | Restricted and uncertain | Strong urban or suburban signal; stable sharp maps from 32 records |
| Daubenton's, in flight | Water ↑ (top driver), riparian broadleaf ↑, urban and lighting ↓, arable ↓ | Narrow: water corridors and waterbodies | Water not in the top layers; wide spread away from water |
| Daubenton's, roost | Broadleaf near water ↑, water ↑, altitude ↓, moorland ↓ | Narrow, near water | As above |
| Whiskered, roost and in flight | Broadleaf and edges ↑, tree height ↑, urban ↓, altitude ↓, moorland ↓ | Restricted: wooded valleys and treed farmland | Strong urban signal; in flight driven by buildings |
| Brandt's, roost | Broadleaf ↑ (stronger), water ↑, urban ↓, altitude ↓ | Smaller than whiskered | Map bigger than whiskered's; a sharp map from 27 records |
| Natterer's, roost and in flight | Broadleaf and tree height ↑ (strong), urban and suburban ↓ (strong), arable ↓ | Restricted: woodland | Urban or suburban ↑ |
| Brown long-eared, roost | Broadleaf within 1–2 km ↑, tree cover ↑, old buildings (rural) ↑, urban ↓, major roads ↓ | Moderate: wooded farmland and valleys | Dense urban rated suitable; strong suburban ↑ |
| Brown long-eared, in flight | Broadleaf and tree cover ↑ (strong), urban ↓ | Moderate, close to woodland | Driven by buildings (emergence or survey bias) |

**Expected ordering of suitable share, broadest first** (a judgement from the natural history; ties are possible):
common pipistrelle (in flight ≥ roost) > noctule in flight > soprano pipistrelle > brown long-eared ≈ whiskered > Natterer's ≈ Daubenton's ≈ noctule roost > Brandt's ≈ Leisler's > Nathusius'.

**Rules that hold for every species** (the strongest and safest expectations):
- Suitability falls with altitude and exposure. Open moorland and blanket bog should be low for every species.
- Dense urban core should be low for every species except, at most, the pipistrelles and noctule in flight.
- Intensive arable away from hedges and woodland should be low.
- Calm water and the riparian woodland next to it should be good for Daubenton's, soprano, Nathusius' and noctule.

---

## 5. How the training data compare (presence vs background)

From `ev_stats/presence_vs_background.csv`. The figure is the standardised mean difference (SMD) between each model's records and its background points: positive means records sit at *higher* values than background. **Caveat:** the background points come from the bat records themselves, so a difference shows how this species' records differ from *where people record bats*, not from the landscape. Training weights are ignored here. About ±0.2 is small, ±0.5 moderate, and ±0.8 or more large.

| Model | n | Altitude | Distance to buildings | Water (500 m) | Broadleaf (500 m) | Suburban (500 m) | Urban (500 m) | Heath (500 m) | Max tree height (500 m) |
|---|---|---|---|---|---|---|---|---|---|
| Brandt's roost | 27 | -0.14 | -0.82 | +0.32 | +0.56 | -0.03 | -0.31 | -0.33 | +1.23 |
| Daubenton's in flight | 116 | -0.23 | -0.52 | +0.64 | +1.04 | +0.47 | +0.35 | -0.25 | +1.05 |
| Daubenton's roost | 124 | -0.42 | -0.62 | +0.46 | +0.79 | +0.19 | +0.23 | -0.38 | +1.16 |
| Whiskered in flight | 32 | -0.11 | -0.52 | +0.41 | +0.85 | +0.25 | -0.12 | -0.16 | +0.87 |
| Whiskered roost | 124 | -0.25 | -0.83 | +0.05 | +0.52 | -0.01 | -0.08 | -0.25 | +0.81 |
| Natterer's in flight | 43 | -0.03 | -0.57 | +0.15 | +0.68 | +0.02 | -0.16 | -0.22 | +0.69 |
| Natterer's roost | 191 | -0.36 | -0.77 | +0.07 | +0.53 | -0.10 | -0.20 | -0.35 | +0.85 |
| Leisler's in flight | 82 | -0.05 | -0.59 | +0.23 | +0.65 | +0.69 | +0.47 | -0.10 | +0.75 |
| Leisler's roost | 32 | -0.27 | -0.83 | -0.02 | +0.68 | +1.28 | +0.55 | -0.48 | +0.95 |
| Noctule in flight | 312 | -0.26 | -0.75 | +0.22 | +0.47 | +0.65 | +0.33 | -0.33 | +0.75 |
| Noctule roost | 74 | -0.54 | -0.65 | +0.24 | +0.82 | +0.53 | +0.32 | -0.47 | +1.20 |
| Common pipistrelle in flight | 630 | -0.17 | -0.74 | +0.07 | +0.27 | +0.70 | +0.41 | -0.29 | +0.58 |
| Common pipistrelle roost | 1215 | -0.43 | -0.91 | -0.02 | +0.15 | +0.52 | +0.19 | -0.38 | +0.65 |
| Soprano in flight | 265 | -0.26 | -0.70 | +0.27 | +0.54 | +0.52 | +0.29 | -0.34 | +0.87 |
| Soprano roost | 325 | -0.52 | -0.84 | +0.19 | +0.48 | +0.22 | -0.14 | -0.43 | +1.00 |
| Brown long-eared in flight | 141 | -0.30 | -0.76 | +0.13 | +0.52 | +0.46 | +0.13 | -0.40 | +0.78 |
| Brown long-eared roost | 604 | -0.35 | -0.82 | +0.05 | +0.37 | +0.04 | -0.16 | -0.32 | +0.79 |

(Nathusius' roost isn't in the stats file.)

### Where the data agree with the literature
- **Altitude and moorland** fall for every model, as expected.
- **Tree height and broadleaf** rise for every model, and most strongly for the woodland species (Brandt's, Daubenton's, noctule roost, whiskered). Common pipistrelle has the weakest broadleaf signal, as expected for a generalist.
- **Water** is strongest for Daubenton's in flight (+0.64) and stronger for soprano than for common pipistrelle (+0.27 against +0.07). Both are as expected.
- **Suburban** rises most for common pipistrelle in flight and is near zero for Natterer's, whiskered, Brandt's and brown long-eared roosts. As expected.

### Where they disagree (flags)
1. **Noctule roost, closer to buildings (-0.65).** Noctule roosts in trees. The literature expects no effect or the opposite. Most likely survey bias (records near where people look), or building-based records mislabelled. Same finding as in the earlier evidence comment.
2. **Leisler's roost, strongly suburban (+1.28) and urban (+0.55).** Leisler's mainly roosts in trees. With 32 records, a few clustered sites could cause this. Check whether the records come from one or two survey sites (for example a park or a cemetery with mature trees). Treat the model as unreliable until checked.
3. **Leisler's and noctule in flight, suburban and urban up (+0.5 to +0.7).** Partly plausible: both feed high over parks and around some lights. But it is as strong as common pipistrelle's, which is not expected. Likely observer effort in towns.
4. **Daubenton's in flight, suburban (+0.47) and urban (+0.35) up.** Unexpected for a light-averse species. A likely explanation: the Don, Sheaf and Porter run through Sheffield and Rotherham, so river records are also urban. Check that the model's urban response is not positive once water is accounted for.
5. **Brown long-eared in flight, suburban up (+0.46).** Not expected. Probably emergence records from house roosts and garden detectors.
6. **"Closer to buildings" in every model, -0.5 to -0.9.** Plausible for pipistrelles and for building-roosting colonies of long-eared, whiskered and Brandt's, but not for noctule, Leisler's, Daubenton's in flight or Natterer's in flight. Because the background is itself drawn from bat records, a near-universal effect points to a shared bias in where records come from.

---

## 6. Recording bias in UK bat records, and what it means for the models

- **Observer effort follows people.** Records cluster near towns, roads, nature reserves and the homes of active recorders. The South Yorkshire Bat Group and local bat groups record where members live and go. Remote areas (Thorne and Hatfield Moors, high moors, private estates) are under-recorded.
- **Roost records come from buildings.** Many roost records come from building inspections, development surveys and the Bat Conservation Trust's helpline/roost visits, so building roosts are hugely over-represented compared with tree roosts. That inflates a "buildings" effect for every roost model, and tree-roosting species (noctule, Leisler's) suffer most.
- **Detectors favour loud bats.** Noctule, Leisler's and pipistrelles are picked up at long range. Brown long-eared and Natterer's are quiet, and whiskered/Brandt's are hard to identify. So quiet species are under-recorded in flight, and Myotis records are often "Myotis sp." and dropped.
- **Mixed and old identifications.** Pipistrelle records before the late 1990s may not split common and soprano. Whiskered/Brandt's are confused. Leisler's and noctule calls overlap.
- **Timing.** Autumn swarming, migration (Nathusius') and hibernation records mix with summer data.
- **What it means.** Presence-only models learn the bias as habitat unless it's handled (Phillips et al. 2009; Fourcade et al. 2014). Using other bat records as background (target-group background) helps, but only when every species was recorded the same way. Standardised detector schemes reduce the bias (Newson et al. 2015). In practice: distrust buildings, roads and suburban layers as top drivers for tree-roosting and light-averse species; judge rare species (Brandt's, Leisler's, Nathusius') on how stable the map is across folds more than on mean score (Lobo et al. 2008).

---

## 7. How this should guide choices

- **Features.** Prefer layers with a clear ecological reason (woodland, tree height, water, altitude, exposure, urban and suburban, arable). Remove ones that can only act as a proxy inland (distance to coast). Treat the climate layers as gradients of altitude: they shouldn't carry a model on their own at this scale.
- **Cut-offs.** A cut-off that leaves a woodland specialist suitable across most of the lowland, or a pipistrelle restricted to a few valleys, fails the ordering in section 4. Compare the suitable shares with the order, not with a fixed number.
- **Bias.** For tree-roosting and light-averse species, test a bias-corrected background (or a check without the buildings and roads layers) and see whether the map still makes sense.
- **Sign-off.** A model is ecologically credible when its top drivers and directions match section 4, its share sits in the right place in the order, and none of its red flags show.

---

## 8. Open questions

- Are there summer-only records, or are hibernation and swarming records mixed in? (This changes the limestone caves and mines.)
- Is a Nathusius' model worth keeping with so few records, given that most are probably migrants?
- Should the in-flight whiskered model be relabelled "whiskered/Brandt's"?
- Would a local check from the South Yorkshire Bat Group's known sites be acceptable as a test (public summaries only)?

---

## 9. References

- Bat Conservation Trust. UK bat species pages. https://www.bats.org.uk/about-bats/what-are-bats/uk-bats
- Collins, J. (ed.) (2023) *Bat Surveys for Professional Ecologists: Good Practice Guidelines*, 4th edn. Bat Conservation Trust. https://www.bats.org.uk/resources/guidance-for-professionals/bat-surveys-for-professional-ecologists-good-practice-guidelines-4th-edition
- Mathews, F. et al. (2018) *A Review of the Population and Conservation Status of British Mammals*. Natural England / Mammal Society. https://publications.naturalengland.org.uk/publication/5636785878597632
- Crawley, D. et al. (2020) *Atlas of the Mammals of Great Britain and Ireland*. Mammal Society / Pelagic. https://www.mammal.org.uk/
- NBN Atlas (species records and maps). https://nbnatlas.org/
- Natural England. National Character Area profiles (NCA 30, 37, 38, 39, 51). https://www.gov.uk/government/publications/national-character-area-profiles-data-for-local-decision-making
- Sheffield and Rotherham Wildlife Trust. https://www.wildsheffield.com/
- Bellamy, C., Scott, C. & Altringham, J. (2013) Multiscale, presence-only habitat suitability models: fine-resolution maps for eight bat species. *J. Appl. Ecol.* https://doi.org/10.1111/1365-2664.12117
- Boughey, K.L., Lake, I.R., Haysom, K.A. & Dolman, P.M. (2011a) Improving the biodiversity benefits of hedgerows: how physical characteristics and the proximity of foraging habitat affect the use of linear features by bats. *Biol. Conserv.* https://doi.org/10.1016/j.biocon.2011.02.017
- Boughey, K.L., Lake, I.R., Haysom, K.A. & Dolman, P.M. (2011b) Effects of landscape-scale broadleaved woodland configuration and extent on roost location for six bat species across the UK. *Biol. Conserv.* https://doi.org/10.1016/j.biocon.2011.06.008
- Davidson-Watts, I. & Jones, G. (2006) Differences in foraging behaviour between *Pipistrellus pipistrellus* and *P. pygmaeus*. *J. Zool.* https://doi.org/10.1111/j.1469-7998.2005.00016.x
- Fourcade, Y. et al. (2014) Mapping species distributions with MAXENT using a geographically biased sample of presence data. *PLoS ONE.* https://doi.org/10.1371/journal.pone.0097122
- Jones, G. & van Parijs, S.M. (1993) Bimodal echolocation in pipistrelle bats: are cryptic species present? *Proc. R. Soc. B.* https://doi.org/10.1098/rspb.1993.0017
- Lintott, P.R. et al. (2015) Differential responses to woodland character and landscape context by cryptic bats in urban environments. *PLoS ONE.* https://doi.org/10.1371/journal.pone.0126850
- Lobo, J.M., Jiménez-Valverde, A. & Real, R. (2008) AUC: a misleading measure of the performance of predictive distribution models. *Glob. Ecol. Biogeogr.* https://doi.org/10.1111/j.1466-8238.2007.00358.x
- Mathews, F. et al. (2015) Barriers and benefits: implications of artificial night-lighting for the distribution of common bats in Britain and Ireland. *Phil. Trans. R. Soc. B.* https://doi.org/10.1098/rstb.2014.0124
- Newson, S.E., Evans, H.E. & Gillings, S. (2015) A novel citizen science approach for large-scale standardised monitoring of bat activity and distribution, evaluated in eastern England. *Biol. Conserv.* https://doi.org/10.1016/j.biocon.2015.06.009
- Phillips, S.J. et al. (2009) Sample selection bias and presence-only distribution models: implications for background and pseudo-absence data. *Ecol. Appl.* https://doi.org/10.1890/07-2153.1
- Razgour, O., Hanmer, J. & Jones, G. (2011) Using multi-scale modelling to predict habitat suitability for species of conservation concern: the grey long-eared bat as a case study. *Biol. Conserv.* https://doi.org/10.1016/j.biocon.2011.08.010
- Russ, J.M. & Montgomery, W.I. (2002) Habitat associations of bats in Northern Ireland: implications for conservation. *Biol. Conserv.* https://doi.org/10.1016/S0006-3207(02)00089-7
- Russo, D. & Ancillotto, L. (2015) Sensitivity of bats to urbanization: a review. *Mamm. Biol.* https://doi.org/10.1016/j.mambio.2014.10.003
- Stone, E.L., Jones, G. & Harris, S. (2009) Street lighting disturbs commuting bats. *Curr. Biol.* https://doi.org/10.1016/j.cub.2009.05.058
- Stone, E.L. et al. (2015) The impacts of new street light technologies: experimentally testing the effects on bats of changing from low-pressure sodium to white metal halide. *Phil. Trans. R. Soc. B.* https://doi.org/10.1098/rstb.2014.0127

All DOIs were checked against Crossref and all web links returned a working page on 20 Oct 2026. Two sources from the brief aren't linked: the South Yorkshire Bat Group site didn't respond, and the Peak District National Park landscape strategy page blocked the check. Their content isn't relied on here.
