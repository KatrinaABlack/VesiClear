# vesiclear

`vesiclear` refines vesicle membrane positions from SAM segmentation masks on cryo-EM
micrographs and subtracts the membrane density to reveal membrane-proximal particles.
Developed for EGFR–EGF virus-like particles.

## Layout

```
vesiclear/                 the package
  pick_membrane_robust.py  leaflet-refinement entry point
  refinement_pipeline/     baseline, evidence/picking, robust spline fit + gap clipping
  helpers/                 IO, logging, parsing
  subtraction/             membrane subtraction
```

## 1. Leaflet refinement

1. **Baseline** — smooth the SAM contour and resample at even arc length (clean reference
   curve + normals).
2. **Box & pick** — perpendicular intensity profile per sample, cross-correlated with a bilayer
   template; the correlation peak gives the membrane displacement and a confidence.
3. **Robust centerline fit** — penalized B-spline (P-spline) of the displacement field with IRLS
   robust reweighting; picks are re-selected within a shrinking band around the fitted prior
   (×3) and profiles are re-boxed perpendicular to the refined line (×2).
4. **Gap clipping** — spans where the membrane signal is genuinely absent (a run of
   low-confidence picks longer than `gap_min_arc`) are not drawn; the leaflet is emitted as
   open arc(s) only where there is real evidence.
5. **Output** — `inner / intermembrane / outer` coordinate arrays (one set per arc) + overlays.

Entry point (run in place of `generate_picks.py` in the vesicle-picker workflow):

```bash
python vesiclear/pick_membrane_robust.py \
  --membrane_template /path/to/intensity_template.npz \
  --picks_dir first_picks/ --spline_dir splines/ --logs_dir logs/ \
  ./config.ini
```

Gap-clip options: `--gap_confidence` (default 0.2), `--gap_min_arc` (default 175 Å),
`--no_gap_clip` to disable.

## 2. Membrane subtraction

`vesiclear/subtraction/membrane_subtraction_v4_clipped.py` removes the masked membrane region by
inpainting: local-mean replacement plus clipped, high-pass spectral noise matched to the
surrounding background, blended over a distance ramp. CLI-driven (`--help`).

## Install / dependencies

```bash
pip install -e .
```

Requires `numpy`, `scipy`, `opencv-python`, `mrcfile`, `matplotlib`; the refinement entry point
also needs the `vesicle_picker` package (SAM mask import + cryoSPARC access).

## License

MIT — see [LICENSE](LICENSE).
