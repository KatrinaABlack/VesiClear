# membrane_refinement_robust_gaps_v9

Parent: v6 (`membrane_refinement_robust_edges_v6`). v6 remains production; this branch
explores NOT drawing a leaflet through genuine gaps / low-quality regions of the membrane.

## Change vs v6
`leaflets_from_points` (`refinement_pipeline/fit_centerline.py`) gains **gap clipping**
(clip-output approach):

- The global robust fit is UNCHANGED (supported arcs are identical to v6).
- After fitting, spans that bridge a **genuine gap** are removed from the OUTPUT, so the
  leaflet is emitted as open arc(s) instead of a closed loop drawn through no-evidence.
- A **genuine gap** = a contiguous run of LOW-CONFIDENCE samples (`weight < gap_confidence`,
  which includes dropped weight==0 picks AND kept-but-weak ones) whose bridged arc length
  exceeds `gap_min_arc`. Shorter dropouts are still bridged (as v6).

Key finding that shaped this: the regions where v6 wanders off the true membrane are NOT
weight==0 gaps - they are runs of picks that pass the 0.1 drop floor but are weak (~0.1-0.2)
and wrong. So the gap criterion is **confidence-based**, not dropped-based.

A gapped vesicle now emits multiple (inner, center, outer) triples (one per supported arc).

## New params (base_params)
- `gap_clip = True`        (off => exact v6 behaviour)
- `gap_min_arc = 175.0`    (A; bridged run longer than this = genuine gap)
- `gap_confidence = 0.2`   (weight below this = unsupported for clipping; drop floor stays 0.1)

## Status / not yet done
- Calibrated only by eye on mic 11466188286478496947 (see test_data/gaps_v9/). Thresholds
  (`gap_confidence`, `gap_min_arc`) likely need a sweep.
- NOT wired into the production CLI (`pick_membrane_robust.py` / `picking_parser.py`) yet -
  would need the three params added there for a full 41-mic run.

## Diagnostics (test_data/gaps_v9/)
- `gap_diagnostic.py`  -> gap_diagnostic.png  (where/how long the gaps are, weight==0 based)
- `compare_gap.py <uid>` -> compare_gap_<uid>.png (v6 bridge red vs v9 clip cyan, grid)
- `/tmp/zoom_gap.py <uid> <ci...>` -> zoom with picks coloured by confidence
