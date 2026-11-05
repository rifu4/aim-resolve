# Plan: detection with the Claude agent (A, A2)

Improvements to the detection step of the iterative pipeline (`detect → model → pre-fit →
reconstruct`) with the Claude agent:

- **A. Blind decision, then comparison with the history:** the agent first decides on the current
  reconstruction only, then compares its decisions with all previous iterations and settles the
  disagreements.
- **A2. Compact sources embedded in extended objects:** separated as point sources instead of
  being dropped.

These come after the reuse of unchanged components (`PLAN_reuse.md`, parts B and C). A uses the
same-source test of B for its matching; A2 relies on the overlap groups of C in the pre-fit.

## Context: current state (Nov 2026)

- **Detection:** `steering/scripts/detect.py` with `base_detect.method: unet | agent | manual`. The
  agent tools are in `src/aim_resolve/agent_detect.py`, and headless runs use
  `steering/run_with_claude.sh` (snakemake with `--nolock --rerun-triggers mtime`).
- **Task and submission:** the agent gets `agent/{it}_task/` (`task.md`, `overview.png`, candidates)
  and the reconstruction `files/{it-1}_rec.pkl`. It writes `agent/{it}_submission.json` (objects,
  groups, compact_ids, extra_compact).
- **Settings:** the agent's pixel settings (`merge_dist`, `pair_dist`, `compact_box`, `snap_radius`)
  and `base_detect.edge_dist` are scaled by the zoom factor of the reconstruction; detections within
  `edge_dist` of the field edge are dropped.
- **Cost:** a detection takes 3–4 min, about 40k new and 8k output tokens (plus cached input).

---

## A. Detection: blind decision, then comparison with the history

### Goals
- **Consistency:** the same source keeps the same treatment across iterations unless the new
  reconstruction gives a reason to change it. No flip-flopping on borderline candidates.
- **Focus on changes:**
  - sources that appear once bright neighbours are modelled;
  - rejected candidates that persist or grow;
  - emission that moved between the background and a component.
- **Check the model's result:**
  - a tile that captured two sources and should be split;
  - a box that is too tight;
  - a component that absorbed artefacts or background;
  - a lobe group that is really the continuation of a jet.

### Why two phases
- **Anchoring:** an agent that sees its previous answer first tends to copy it. In the blind phase
  it cannot, so its detection is an independent judgement of the current reconstruction.
- **Agreement means something:** where the blind detection matches the previous iteration, that is
  real confirmation, not copying.
- **Disagreements are the signal:** the second phase only deals with the places where the blind and
  previous decisions differ, which are exactly the cases worth a closer look.
- **Measurable:** comparing the blind and final submissions per iteration shows how often the history
  changed the agent's mind, and in which direction.
- **The opposite risk:** in phase 2 the agent may fall back to the old decision by default.
  - Neither side wins by default: every disagreement is decided from the images, with a reason.
  - If the evidence is unclear, the **blind decision stands**, because it is based on the newer
    reconstruction.

### Workflow
1. **Phase 1, blind** (as now).
   - The agent gets only `agent/{it}_task/` (`task.md`, overview, candidates) and the current
     reconstruction.
   - It writes `agent/{it}_submission_blind.json`, with the same schema as today.
2. **Comparison, in code** (not by the agent).
   - The blind decisions are matched by position to **all previous iterations**: their submissions
     (`agent/{k}_submission.json`), detection maps (`files/{k}_det.npz`) and final models
     (`files/{k}_mdl.yml`, i.e. what the model step made of the decisions), for k < it.
   - **Source tracks:** matching iteration by iteration links the decisions into one track per
     source, e.g. "accepted in it 1, rejected in it 2, back as blind candidate in it 3". A
     disagreement is judged against the last iteration, but the agent sees the full track. A source
     that flip-flops is flagged as such.
   - Matching uses the same-source test from `PLAN_reuse.md`, part B (one-to-one, IoU, same type), applied to the
     decisions instead of the model boxes.
   - It produces a short list of **disagreements**:
     - **new:** in the blind decision, not in the previous one (e.g. a source that appeared after its
       bright neighbour was modelled);
     - **missing:** a previous component that the blind decision does not contain;
     - **grouping:** a split or merge (two blind objects for one previous object, a different lobe
       group);
     - **type:** object ↔ tile ↔ point, or extended ↔ compact;
     - **candidate status:** a compact candidate accepted in one and rejected in the other, with the
       brightness change.
   - Pure box-size differences are **left out**, because the hysteresis in the model step (`PLAN_reuse.md`, part B)
     handles those anyway.
   - Output: `agent/{it}_task/diff.md` and one image per disagreement in `agent/{it}_task/diff/`.
     Each image has a crop of the current reconstruction with the blind decision drawn in, and the
     previous fitted component(s) in the same region.
   - `diff.md` also has a compact **history table**, with one row per source track (decision per
     iteration, type, flux). The agent thus sees the full history, not only the last iteration.
     Tracks without any change are summarised in one line per type.
3. **Phase 2, reconciliation.**
   - **Skipped if there are no disagreements:** the blind submission becomes the final one.
   - The agent reads `diff.md` and the images and decides each disagreement: keep the blind
     decision, take the previous one, or something new. It can still use the tools
     (`view`/`sheet`/`peaks`/`stats`, and the new `components` view below).
   - It writes the final `agent/{it}_submission.json` (same schema) with a new field `changes`: one
     entry per disagreement with the decision and a short reason.
   - Best run in the **same Claude session** (`claude -p --resume <session>` in
     `run_with_claude.sh`), so that the agent still has the images it looked at in phase 1. Phase 2
     is then short and cheap.
4. **Iteration 1:** there is no previous iteration, so only phase 1 runs.

### Implementation
1. **Save the component maps after each reconstruction.**
   - `reconstruct.py` writes `files/{it}_rec_components.npz`: the posterior mean of every component
     on its own grid, plus the background.
   - Tiles are saved unmapped, `(n_tiles, 32, 32)`.
   - The file is small and needs no extra evaluation beyond `samples.mean`.
2. **Comparison step** (`agent_detect.py`, a new function `compare_with_previous`).
   - Inputs: the blind submission, the current candidates, and the previous submission, detection
     maps, model and component maps.
   - Output: the disagreement list as data, `diff.md` and the images.
   - The same matching is reused for the candidate status ("candidate 37: rejected in it 2, now 3×
     brighter").
3. **New tool:** `python -m aim_resolve.agent_detect components <it> [--prefix o0.2 ...]` shows the
   previous iteration's fitted components (sub-grids for tiles and points, via `plot_components` /
   `plot_panels`), so that the agent can look at more than the diff images.
4. **Detect step** (`detect.py`, methods `agent` and `manual`).
   - Manual: the task stops twice. First it waits for `{it}_submission_blind.json`, then, if there
     are disagreements, for `{it}_submission.json`.
   - Agent (API): the two phases are two turns of the same conversation.
   - The final maps are built from the final submission, as now.
5. **Wrapper** (`run_with_claude.sh`).
   - Phase 1 runs with `--output-format json`, so the wrapper gets the session id.
   - Snakemake then runs once more: the detect step writes `diff.md` and stops again.
   - Phase 2 runs with `--resume <session id>` and a short prompt that points to `diff.md`. The write
     permission is `Edit(//…/agent/{it}_submission.json)`.
   - If there is no diff, the detect step copies the blind submission and continues.
6. **Prompts** (`SYSTEM_PROMPT`, `task.md`, wrapper).
   - Phase 1 contains no word about history.
   - Phase 2 instructions:
     - decide every disagreement from the evidence;
     - if unclear, keep the blind decision;
     - explicitly check whether a previous component absorbed artefacts or background, a tile holds
       two sources, a box is too tight, or a lobe group is the continuation of a jet;
     - give a reason for each decision in `changes`.
7. **Log:** `agent/{it}_agent.json` stores both phases. In addition, one line per iteration goes to
   `agent/history.json`: number of disagreements, how many kept blind / took previous / new, so
   that the stability can be followed over the run.

### Priority
1. Component maps, comparison step and `diff.md` with images; phase 2 in manual mode.
2. Two-phase wrapper with `--resume`.
3. The `components` tool and the history log.
4. Optional, later: a difference image (the last minus the second-to-last reconstruction) in the
   diff, if the agent misses newly emerging sources.

---

## A2. Detection: compact sources embedded in extended objects

### Problem
Compact sources that sit on top of an extended object are dropped.
- **Example:** run `fast_vi_1f_1024_1z_agent2`, iteration 1, candidate 13 at (438, 480). It is a
  point-like source at the lower-left edge of the central galaxy's southern lobe.
  - Its contrast is 36. That is low only because it is measured against the lobe emission around
    it; isolated sources in the same image reach several thousand.
  - The agent rejected it as "the southern tip of the central object".
- **Cause 1, prompt:** `SYSTEM_PROMPT` says "reject candidates that are part of an extended object".
- **Cause 2, code:** `boxes_to_maps` skips every compact source whose position is already covered by
  an object mask ("compact sources: skip those inside other sources"). Such a source is lost even if
  the agent accepts it.
- **What already works:** the model can superpose components. `ComponentModel` adds points and tiles
  on top of objects, and runs already have tiles inside object boxes. What is missing is the
  detection rule and the separation in the pre-fit.

Separating such sources should improve the reconstruction. The object's correlated field otherwise
has to produce an unresolved peak, which its smoothness prior works against, and the peak's flux
leaks into its surroundings.

### Detection rule
- A compact candidate inside an object is accepted as **embedded** if it is point-like and clearly
  above the local emission of the object.
- **Local contrast** as a hint for the agent: the peak against the median of a ring a few pixels
  around it (e.g. radius 3–6 px), instead of against the global background. It is computed in
  `find_candidates` and shown in the candidate list and on the contact sheets.
- **What to separate:**
  - unrelated sources on top of a lobe (like #13);
  - unresolved cores;
  - point-like sources close to hotspots or knots, if they are clearly separate from them.
- **What stays with the object:** hotspots and knots that are resolved parts of the jet or lobe
  structure. The agent decides case by case.
- `task.md` and `SYSTEM_PROMPT` get a section with this rule and an example image of an embedded
  source.

### Submission and conversion
- New fields:
  - `embedded_ids`: candidates accepted as embedded;
  - `extra_embedded`: positions that the candidate list misses.
- `boxes_to_maps`:
  - embedded sources become **points** (`ps_map`) at the snapped peak;
  - the object keeps its full mask, and embedded sources are no longer dropped.
- **Points rather than small tiles:** a 32 px tile on top of a lobe would compete with the object's
  field over its whole area. A point only takes the unresolved part.
- `masks_from_maps` / `model_points`: points inside object masks are kept (the config already has
  `points: true`). They get an offset from the local peak minus the ring median, not from the total
  flux in a box.

### Pre-fit separation
- **Problem:** today all components are fitted on the same target (R minus the background). An
  object would also fit the flux of an embedded point, so that flux is counted twice.
- **New order inside an object's overlap group:**
  1. Fit the embedded points first, on peak minus local level, i.e. on R minus the background minus
     the ring median around each point.
  2. Subtract the fitted point images from the target of the object.
  3. Then fit the object.
- **Reuse (`PLAN_reuse.md`, part C):** a point inside an object belongs to the object's overlap group, so the two are
  copied or refitted together.

### Tests
- `boxes_to_maps`: an embedded source inside an object mask ends up in `ps_map`, and the object mask
  stays complete.
- Local contrast: a point on a smooth gradient has a high local contrast; a resolved knot within a
  filament has a low one.
- Pre-fit: on a synthetic object with a point on top, the point flux is recovered and the object
  does not contain it (flux sum check).
- On run `fast_vi_1f_1024_1z_agent2`, iteration 1: with the new rule, candidate 13 should be
  accepted as embedded.

### Tests (A)
- **Comparison step:** unit tests with synthetic submissions (new, missing, split, merge, type
  change, candidate status). Box-only differences must not appear in the diff.
- **Two-phase agent:** on iteration 2 of an existing run:
  - run phase 1 headless and check the diff and its images;
  - run phase 2 with `--resume`;
  - compare the blind and final submissions and read the reasons in `changes`;
  - also check the skip path (no disagreements → the blind submission becomes final).

### Open questions (A, A2)
- In phase 2, should the agent also be allowed to request "keep box of component X" explicitly, so
  that it can drive the hysteresis instead of leaving it to the mask comparison alone?
- ~~Should phase 2 see only the last iteration or the full history?~~ Decided: the full history,
  as source tracks (see the comparison step in A).
- Embedded sources (A2):
  - Which local-contrast threshold should be suggested to the agent?
  - Should point-like peaks at hotspots or cores be separated by default, or only when they are
    clearly distinct from the surrounding structure?
