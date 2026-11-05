"""Reuse of unchanged components between pipeline iterations (stable boxes, part B of PLAN_reuse.md).

The model step compares the masks of the new detection with the masks of the previous iteration
(``files/{it-1}_msk.npz``). A new component that is the same source as an old one keeps the old box:
same grid (space and center, the factor of the current zoom), prior, taper and offset, plus a
``reuse`` marker with the old prefix. The pre-fit (part C) can then copy or zoom-fit it.

Masks are on the grid of the current iteration. A mask is the detection (value 1) plus a margin
that falls off below 1 (``add_margin``): the core (>= 1) is the detected source, the support (> 0)
the region of its box.
"""

import numpy as np

CORE = 1.0 - 1e-6


def upsample_mask(mask, factor):
    """Mask on a grid refined by the whole factor `factor` (each pixel becomes factor x factor pixels)."""
    factor = int(factor)
    if factor == 1:
        return np.asarray(mask)
    mask = np.asarray(mask)
    return np.repeat(np.repeat(mask, factor, axis=-2), factor, axis=-1)


def bbox(mask):
    """Bounding box (i0, i1, j0, j1), inclusive, of the non-zero pixels of a 2D mask (None if empty)."""
    idx = np.argwhere(np.asarray(mask) > 0)
    if idx.size == 0:
        return None
    (i0, j0), (i1, j1) = idx.min(axis=0), idx.max(axis=0)
    return int(i0), int(i1), int(j0), int(j1)


def _area(box):
    return (box[1] - box[0] + 1) * (box[3] - box[2] + 1)


def _inside(box, outer):
    return box[0] >= outer[0] and box[1] <= outer[1] and box[2] >= outer[2] and box[3] <= outer[3]


def same_source(new, old, margin, iou_min=0.7, area_min=0.7):
    """Compare a new and an old component mask (same grid).

    Parameters
    ----------
    new, old : np.ndarray
        2D masks with margins (core = 1, margin < 1).
    margin : float
        Margin in pixels that the core of the new mask has to keep inside the box of the old mask.
    iou_min, area_min : float
        Minimum intersection over union of the supports and minimum ratio of the box areas
        (new box / old box).

    Returns
    -------
    ok : bool
        True if the new mask is the same source and fits the old box.
    stats : dict
        The measured values (iou, area ratio, fits) and, if not ok, the reason.
    """
    new_sup, old_sup = np.asarray(new) > 0, np.asarray(old) > 0
    union = np.logical_or(new_sup, old_sup).sum()
    iou = float(np.logical_and(new_sup, old_sup).sum() / union) if union else 0.0
    new_box, old_box, core_box = bbox(new_sup), bbox(old_sup), bbox(np.asarray(new) >= CORE)
    if new_box is None or old_box is None or core_box is None:
        return False, dict(iou=iou, reason="empty mask")
    m = int(np.ceil(margin))
    grown = (core_box[0] - m, core_box[1] + m, core_box[2] - m, core_box[3] + m)
    fits = _inside(grown, old_box)
    area = _area(new_box) / _area(old_box)
    stats = dict(iou=round(iou, 3), area=round(area, 3), fits=bool(fits))
    if iou < iou_min:
        return False, stats | dict(reason=f"iou {iou:.2f} < {iou_min}")
    if not fits:
        return False, stats | dict(reason="does not fit the old box")
    if area < area_min:
        return False, stats | dict(reason=f"box area ratio {area:.2f} < {area_min}")
    return True, stats


def match_components(new, old, margin, iou_min=0.7, area_min=0.7):
    """Match new components to old ones.

    Parameters
    ----------
    new, old : dict
        ``{name: (kind, mask)}`` with kind 'object', 'tile' or 'point' and 2D masks on the same grid
        (tiles and points one entry per copy, e.g. ``'t0.3[4]'``).
    margin, iou_min, area_min : float
        See ``same_source``.

    Returns
    -------
    dict
        ``{new name: {'old': old name or None, ...stats, 'reason': ...}}``. Sources are compared by
        their cores (the detections, without margins), so that nested or touching components (a
        compact source in a jet box, neighbouring tiles) do not count; those are the overlap groups
        of the pre-fit. A new component is matched one-to-one within its kind: its core overlaps the
        core of exactly one old component of the same kind (otherwise a merge), that one overlaps no
        other new core of the kind (otherwise a split), and ``same_source`` holds. A core that only
        overlaps old components of another kind is a type change.
    """
    new_core = {k: np.asarray(m) >= CORE for k, (_, m) in new.items()}
    old_core = {k: np.asarray(m) >= CORE for k, (_, m) in old.items()}
    hits = {n: [o for o in old_core if np.logical_and(new_core[n], old_core[o]).any()] for n in new_core}
    out = {}
    for n, (kind, mask) in new.items():
        same = [o for o in hits[n] if old[o][0] == kind]
        if not same:
            other = [o for o in hits[n] if old[o][0] != kind]
            reason = f"type changed ({old[other[0]][0]} -> {kind})" if other else "new source"
            out[n] = dict(old=None, reason=reason)
            continue
        if len(same) > 1:
            out[n] = dict(old=None, reason=f"covers {len(same)} old {kind}s (merge)")
            continue
        o = same[0]
        split = [m for m in hits if new[m][0] == kind and o in hits[m]]
        if len(split) > 1:
            out[n] = dict(old=None, reason=f"old {o} covered by {len(split)} new {kind}s (split)")
            continue
        ok, stats = same_source(mask, old[o][1], margin, iou_min, area_min)
        out[n] = dict(old=o if ok else None, **stats) | ({} if ok else {"candidate": o})
    return out


def component_masks(mask_dct, it):
    """``{name: (kind, mask)}`` of the objects, tiles and points of iteration `it` in a mask dictionary."""
    out = {}
    for k, v in mask_dct.items():
        if not k.endswith(f".{it}") or k.startswith("bg"):
            continue
        v = np.asarray(v)
        if k.startswith("o"):
            out[k] = ("object", v)
        elif k.startswith(("t", "p")):
            kind = "tile" if k.startswith("t") else "point"
            for i, m in enumerate(v if v.ndim == 3 else v[None]):
                out[f"{k}[{i}]"] = (kind, m)
    return out
