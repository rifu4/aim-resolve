# Plan: reuse of unchanged components (B, C)

Improvement to the iterative pipeline (`detect → model → pre-fit → reconstruct`): components that did
not change are taken over from the previous reconstruction (or, at a zoom step, fitted to their own
previous map) instead of being pre-fitted again through masks. This saves most of the pre-fit time,
keeps faint structure, and starts the reconstruction close to the previous posterior.

- **B. Model step:** stable boxes between iterations (hysteresis), which C needs for exact copies.
- **C. Pre-fit:** copy, zoom-fit or refit per overlap group.

The detection with the Claude agent (history, embedded sources) is planned separately in
`PLAN_agent_detection.md` (parts A and A2). It helps the agent keep its decisions stable, which
makes B and C apply more often, but B and C work without it.

## Context: current state (Nov 2026)

- **Detection:** `steering/scripts/detect.py` with `base_detect.method: unet | agent | manual`. The
  agent tools are in `src/aim_resolve/agent_detect.py`, and headless runs use
  `steering/run_with_claude.sh`. Detections within `base_detect.edge_dist` of the field edge are
  dropped. The agent's pixel settings and `edge_dist` are scaled by the zoom factor of the
  reconstruction.
- **Model step:** `steering/scripts/model.py` → `masks_from_maps` (`src/aim_resolve/mask.py`), then
  `model_objects`, `model_tiles` and `model_points` (`src/aim_resolve/modeling.py`). It writes
  `files/{it}_mdl.yml` and `files/{it}_msk.npz`.
  - Boxes are the detection mask plus a margin (`margin_fac`, `margin_min`, both scaled with the zoom).
  - A box that fits into `tile_size` becomes a tile, otherwise an object (up to `max_objects`).
  - Offsets are recomputed every iteration (`get_offset`).
  - **Zoom per iteration:** `base_model.zoom` is one factor or a list with one factor per iteration
    (whole factors only, e.g. `[1, 1, 1, 2]`); `base_model.zoom_bg` refines the background as well.
  - Objects and tiles can use their own prior sections (`params_sf_comp`, `params_mf_comp`).
- **Pre-fit:** `steering/scripts/pre_fit.py` → `transition_addt` (`src/aim_resolve/transition.py`).
  1. The background is fitted on the old reconstruction R outside all component masks (cold start).
  2. Each component is fitted in its own mask on R minus the background, with 15 KL steps
     (`base_trans`) and a **fixed noise std** (`max_std · max`). The per-pixel noise model had
     dropped faint structure.
  3. The noise model starts at its prior (`base_prefit.reset_noise: true`); otherwise it is copied
     only if its shape is unchanged (it lives on the Fourier grid, so a zoom changes it).
- **Zoom transition** (`transition_zoom`, used by `nifty/extend_rec.py`): fits each new component
  to its own previous map (`fit: own_kl`, tiles per copy), no masks and no separation needed.
- **Lessons since the first version of this plan:**
  - The pre-fit needs freedom in the zero mode: with 0.02 or 0.1 instead of 0.3, components stayed
    at their start offset (`o1` of `agent4`). Narrow values only match the final posterior.
  - At a zoom step, the `addt` pre-fit has to re-separate unchanged components through masks; the
    zoom transition does this better, because each component keeps its own target.
- **Measured cost:**
  - Pre-fit at 1024² (run `fast_vi_1f_1024_1z_agent2`, iteration 1): about 4 min for the background,
    8–10 min each for the two large objects, about 3 min each for small objects and the tiles.
  - Reconstruction at 1024²: about 3 h; at 3072²: about 1 h per KL step, about 20 h in total.

---

## B. Model step: stable boxes (hysteresis)

Without this, boxes rarely repeat exactly. A mask that is slightly different gives a box that is
1 px larger, and no component can be copied.

### Same-source test
For each new detection, compare it with the old components. Use the detection masks (before
margins) from `files/{it-1}_det.npz` and `files/{it-1}_msk.npz`, not the boxes.

| criterion | measure | start value |
|---|---|---|
| one-to-one | the new mask overlaps exactly one old component, and that one overlaps no other new mask | required (otherwise split or merge) |
| overlap | IoU(new mask, old mask) | ≥ 0.7 |
| still fits | new mask plus a small margin (e.g. `margin_min / 2`) lies inside the old box | required (otherwise the box must grow) |
| not too loose | area(box the new mask would get) / area(old box) | ≥ 0.7 (otherwise the box should shrink) |
| same type | object → object, tile → tile, point → point | required |

**If all criteria hold,** the model step writes the old component unchanged: same grid (space,
center, factor), same prior, same taper and the **old offset**. Otherwise the component gets a new
box and offset as now.

- **Tiles:** the same test per tile, with the tile box in place of the object box. The tile keeps
  its old center.
- **Points:** the same pixel within the snap radius (3 px).
- **Zoom:** the pixel thresholds (`fit_margin`, `point_radius`) are given for the base grid and
  scaled by the zoom factor of the iteration, like the detection settings. At a zoom step the old
  box is compared in sky coordinates, so an unchanged source keeps its box at the finer resolution.

### Implementation
- **Implemented** (`src/aim_resolve/reuse.py`, `steering/scripts/model.py`, `base_model.reuse`):
  - Sources are compared by their cores (the detections without margins), one-to-one within their
    kind; nested or touching components do not block a match (they are the overlap groups of C).
  - A kept object gets the old section (grid with the current zoom factor, prior, taper, offset); a
    kept tile the old center and offset; both get the old mask.
  - The model file gets a section `reuse.{it}` (never built as a model, kept by `clean_dict`):
    `objects: {new prefix: old prefix}` and `tiles: [[new copy index, old copy index], ...]`.
  - All decisions with their values (IoU, area ratio, fits, reason) go to
    `{odir}/reuse/{it}_reuse.json`.
  - Points are not handled yet (reported as "kind not handled").
  - Test on `agent3`, iteration 2 → 3: 26 of 27 components kept (both objects, 24 of 25 tiles; one
    new tile covers two old ones), with and without a 2× zoom at that step.

---

## C. Pre-fit: copy, zoom-fit or refit

### Copy rule (transition)
A component is copied only if **all** of these hold:
1. **Identical model:** same grid, prior parameters and taper. After B this is an exact equality
   check, with the old offset.
2. **Unchanged overlap group** (see below).
3. ~~**The old component already holds the emission** (flux share ≥ `flux_min`)~~ — dropped: the
   old reconstruction is exactly the sum of the old components and the old background, so the share
   is always 1 for an isolated component (it cannot see emission left in the background), and a
   nested component always fails because its neighbours count in the denominator. Changes are caught
   by the same-source test (B) and the group rule.

### Overlap groups (nested and touching components)
- **The problem:**
  - A new tile inside a copied object (e.g. a compact source in the jet box) would see nothing to
    fit, because the copied object already contains its flux.
  - A removed tile inside a copied object would lose its flux at the start: its region lies inside
    the object mask, so the background does not take it.
- **Groups:** components whose boxes overlap, chained transitively. A tile in an object that touches
  another object puts all three into one group. Isolated tiles and points form groups of one.
- **Copy a group** only if every member matches an old component (criteria 1 and 3) and the old
  model had exactly this group: no member added, removed or changed.
- **Otherwise refit the whole group.**

### Three outcomes per group
| outcome | when | what |
|---|---|---|
| **copy** | group unchanged, same grid | take over the latent parameters, with the old offsets |
| **zoom-fit** | group unchanged, but the grids are refined by the zoom of this iteration | fit each component to its own previous map, as `transition_zoom` with `fit: own_kl` (tiles per copy); offsets from `get_offset` on the finer grid; no masks |
| **refit** | group new or changed | as now: on R − copied − background, inside the masks |

- **Why zoom-fit:** with the zoom per iteration, the zoom step is where reuse helps most. The
  components usually do not change there, and an `addt` refit through masks loses faint structure.
- The copy rule's criteria 1 (identical model) and 3 (flux share) apply to zoom-fit as well, with
  "identical model" meaning: same box in sky coordinates, same prior and taper, finer factor.

### Targets
R is the previous total reconstruction (all old components plus the old background).

1. **Copied components** C₁…Cₖ: take their latent parameters from the previous samples, renamed to
   the new prefixes (e.g. `o0.1 …` → `o1.2 …`), with their old offsets.
   Then compute R' = R − ΣCᵢ.
2. **Background:** a **cold-start refit** on R'.
   - Masked only at the refitted groups. The regions of the copied components stay in, because there
     R' is exactly the old background.
   - **Why no warm start:** the excitations are Fourier modes. The old background's flux at a new
     source sits in modes that the masked fit does not constrain, so it would survive and the new
     component would get nothing.
   - **Possible middle ground (later):** copy only the hyperparameters (power spectrum, zero mode)
     and reset the excitations to zero.
3. **Refitted groups:** each member is fitted in its own mask on (R' − B).clip(0), as in the
   current pre-fit, with fixed noise.
4. **Zoom-fitted groups** are fitted to their own previous maps and then subtracted like copied
   ones (R' = R − ΣC_copied − ΣC_zoom-fitted) before the background fit.
5. **Background with `zoom_bg: true`:** refitted on the finer grid; R' is mapped onto it.
6. **Noise:** follows `base_prefit.reset_noise`. In the skip path the noise model is copied only if
   the reset is off and its shape is unchanged.

### Cases
| case | handling |
|---|---|
| newly detected source (was in the old background) | the background is fitted outside its mask, so it only interpolates there; the flux stays in R' − B and goes to the new component |
| component changed a lot (box grew or shrank clearly, split, merged, type changed) | treated as new; the old version is not subtracted, so its full flux stays in R'; latents cannot be carried over to a different grid |
| component changed slightly | the hysteresis (B) keeps the old box, so the component is copied (if its group is unchanged) |
| component removed | not copied; its flux stays in R' and goes to the background (or to the group that contains its region, which is then refitted) |
| new component overlaps a copied one | the two are in the same group, so the whole group is refitted |
| unchanged components at a zoom step | zoom-fit, no masks |
| nothing new, changed or removed | copy the background too (and the noise model, unless reset); **skip the pre-fit entirely** |

### Tiles: copy per tile
- The tile component is one model with many copies, and the set of tiles changes between
  iterations. Per tile with an identical center, copy its slice of the excitations. Only new tiles
  are fitted.
- The shared hyperparameters (power spectrum) are copied as long as the tile prior is unchanged.
- **Open question:** fitting only some copies of a VModel needs a mask or a stop-gradient on the
  copied slices. Simpler first version: if any tile of a group changes, refit that tile only, with
  the others fixed. If that is too complicated, refit the whole tile component and copy only
  objects and points in the first version.

### Further speed-up: stop on convergence
- The refits that remain always run 10 KL steps. Stop them when the relative energy change per step
  falls below a threshold.
- This is more reliable now that the pre-fit uses fixed noise.

### Implementation
- `transition_addt`:
  - the kept components from the section `reuse.{it}` of the model file;
  - group building, the copy decision, the R' / background / group fits;
  - the skip path when nothing changed.
- Helpers:
  - `match_components` (grid equality, or equality up to the zoom factor, + groups);
  - zoom-fit through the existing `own_kl_fit` / `own_to_new` of `transition_zoom`;
  - `copy_latents(samples_old, old_prefix, new_prefix)`, which renames the keys, including the
    per-tile slices;
  - `component_flux_fraction` for criterion 3.
- `pre_fit.py` passes the reuse information and logs every decision:
  - "o1.2 = o0.1: copied (flux 0.97)"
  - "o2.2: refitted (IoU 0.55)"
  - "group {o0.2, t0.2[3]}: refitted (new member)"
- The final joint plot `{it}_sky.png` stays; the per-component plots are made for refitted and
  zoom-fitted components and carry the outcome in their step label (copied components are listed
  in the log only).

### Config
A new section, merged like the others:

```yaml
base_model:
  reuse:
    enabled: true
    iou_min: 0.7
    area_min: 0.7
    fit_margin: 0.5      # fraction of margin_min the new mask must keep inside the old box
    point_radius: 3      # px
    flux_min: 0.9        # transition: old component must hold this share of R - background
base_trans:
  stop_rel_energy: 1.0e-3   # optional early stopping of the remaining fits
```

`enabled: false` gives the current behaviour.

---

## Tests
- **Unit tests**
  - Same-source test: shifted, grown, shrunk, split and merged masks; tiles and points; at a zoom
    step (same source on a 2× finer grid).
  - Overlap groups: nested tile in an object, a chain of three, isolated tiles.
  - `copy_latents`: renamed keys, tile slices; the copied component evaluates identically to the
    old one.
  - Flux criterion on a synthetic case where the old component misses an extension.
  - Skip path: unchanged model → pre-fit skipped, samples equal to the old ones.
- **Pipeline tests** offline on `fast_vi_1f_1024_1z_agent3`:
  - iterations 2 → 3 (no zoom): the reuse decisions, pre-fit time with and without reuse, and the
    start point of the reconstruction (residuals and energy of the first KL steps);
  - iteration 3 → 4 at 2× (the zoomed iteration that the pipeline runs now with the `addt`
    pre-fit): zoom-fit against that pre-fit, per component and for faint structure.

## Order of work
1. **B:** hysteresis and the `reuse.{it}` section in the model step (incl. the zoom scaling). Done.
2. **C:** copy for objects and tiles (per tile), overlap groups, background refit, logging.
   Implemented (`reuse.py`: `box_regions`, `overlap_groups`, `copy_plan`, `renamed_latents`,
   `tile_slices`; `transition_addt(reuse=...)`, decisions in `{odir}/reuse/{it}_prefit.json`);
   decisions checked on `agent3` 2 → 3 (24 of 27 copied), full pre-fit run still to do.
3. **C:** zoom-fit for unchanged groups at zoom steps.
4. **C:** skip path when nothing changed.
5. **C:** per-tile copying.
6. Early stopping of the remaining fits.
7. **Later, separate plan:** A and A2 in `PLAN_agent_detection.md`. A shares the matching code
   with B.

## Open questions
- Should a copied component keep its old offset in the YAML for good, or be re-anchored after the
  reconstruction?
- Per-tile copying inside one VModel: masking or stop-gradient, or split the tiles into "copied" and
  "new" tile components?
- Thresholds: check the logged values from the first runs before tightening or loosening them.
- Zoom-fit with `zoom_bg: false`: the background stays coarse; is that enough for the faint sources
  that only show up at the finer resolution, or should the zoom step switch to `zoom_bg: true`?
