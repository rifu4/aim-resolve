"""Source detection in reconstructed images with a Claude vision agent.

Alternative to the U-Net + DBSCAN detection. The agent inspects the
reconstruction through a few image tools and returns bounding boxes of
extended objects plus the compact sources it accepts from a candidate list.
Every source is turned into a mask: masks of a single pixel become point
sources, all others (small or large) extended objects. The result has the same
``ps_map`` / ``cl_map`` format as the U-Net pipeline produces.

All coordinates exchanged with the agent are array indices ``(i, j)`` of the
image, with ``i`` along axis 0 (vertical in the plots) and ``j`` along axis 1
(horizontal in the plots).
"""

import base64
import io
import json

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from scipy.ndimage import label, maximum_filter, median_filter

SYSTEM_PROMPT = """\
You are an expert in radio astronomy helping to build a sky model for an \
iterative Bayesian (NIFTy) image reconstruction.

You are given the posterior mean of a reconstructed radio sky image. Your job \
is to identify all sources in it so that each one can get its own model \
component in the next reconstruction round. There are two kinds:

- Extended objects: resolved emission (radio galaxies with lobes and jets, \
diffuse blobs, extended halos). Report one axis-aligned bounding box per \
physically distinct object. Physically connected structures (e.g. a core with \
its two lobes) belong to one object; separate galaxies get separate boxes. \
Make the boxes generous: include the faint outskirts, because emission outside \
a box is lost to the background model.
- Compact sources: small sources of one to a few pixels. You are given a list \
of compact candidates (local maxima that stand out from their surroundings by \
at least the given contrast). Review them and accept every candidate that is \
a real source; reject candidates that are part of an extended object you \
boxed, sit on the edge of an extended object, or are peaks of reconstruction \
artefacts. Being complete matters: also faint compact sources should be \
included. If you see a compact source that is not in the list, add its \
position as an extra compact source. You do not have to decide whether a \
compact source is a point source or slightly extended; this is determined \
automatically from its pixels.
- Double-lobed radio galaxies: radio galaxies often appear as two lobes on \
opposite sides of a (sometimes faint or invisible) core. Two or more compact \
sources close to each other, of similar brightness, elongated towards or \
pointing at a common centre, or connected by a thin bridge or jet, are the \
lobes (and core) of one galaxy. Submit them as one group of candidate ids \
instead of as separate compact sources. A list of close candidate pairs is \
provided as a hint, with two measurements: bridge (> 0 means the emission \
between the peaks stays above the local background, i.e. they are connected) \
and align (how well each lobe is elongated towards the other, 1 = pointing at \
each other, None = too small to tell). Pairs passing both are marked LIKELY \
LOBES. Zoom in to decide: most close pairs are unrelated neighbours, a marked \
pair can still be one source with two peaks or two unrelated sources, and \
lobes without a visible bridge (e.g. two round blobs on opposite sides of a \
faint core) are not marked but can still be one galaxy.

Guidelines:
- The images are shown on a logarithmic colour scale by default. Faint, \
smooth large-scale emission at the floor of the dynamic range is the \
background and is modelled separately; do not box it.
- Reconstruction artefacts such as ripples, stripes or concentric ring \
patterns around bright sources are not sources.
- Use the tools to zoom into regions before you decide. Zoomed views mark the \
compact candidates with circles and their ids. Review the compact candidates \
in batches of up to 40 with view_candidates (one small zoom per candidate) \
rather than one region at a time. Coordinates are array indices: \
i is the vertical axis, j the horizontal axis, as labelled in the images. Box \
limits are inclusive-exclusive: [i0, i1) x [j0, j1).
- When you are done, call submit_detection exactly once with your final \
answer.
"""

TOOLS = [
    {
        "name": "view_region",
        "description": (
            "Render the region [i0, i1) x [j0, j1) of the reconstruction as an image "
            "with labelled index axes (i vertical, j horizontal). Compact candidates "
            "are marked with circles, and with their ids if the region is at most "
            "256 pixels wide."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "i0": {"type": "integer"},
                "i1": {"type": "integer"},
                "j0": {"type": "integer"},
                "j1": {"type": "integer"},
                "scale": {"type": "string", "enum": ["log", "linear"]},
            },
            "required": ["i0", "i1", "j0", "j1", "scale"],
            "additionalProperties": False,
        },
    },
    {
        "name": "view_candidates",
        "description": (
            "Render a contact sheet of 48x48 pixel zooms (log scale) around up to 40 compact "
            "candidates, one panel per id, titled with id, position and contrast. The reviewed "
            "candidate is circled in cyan, neighbouring candidates in white with their ids. "
            "Use this to review the candidate list in batches."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_peaks",
        "description": (
            "List local maxima in the region [i0, i1) x [j0, j1) whose value is at "
            "least min_rel_flux times the global image maximum, brightest first, with "
            "their contrast to the local surroundings."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "i0": {"type": "integer"},
                "i1": {"type": "integer"},
                "j0": {"type": "integer"},
                "j1": {"type": "integer"},
                "min_rel_flux": {"type": "number"},
            },
            "required": ["i0", "i1", "j0", "j1", "min_rel_flux"],
            "additionalProperties": False,
        },
    },
    {
        "name": "region_stats",
        "description": (
            "Statistics of the region [i0, i1) x [j0, j1): flux sum and fraction of "
            "the total flux, maximum and its position, median, and number of pixels "
            "above the background level."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "i0": {"type": "integer"},
                "i1": {"type": "integer"},
                "j0": {"type": "integer"},
                "j1": {"type": "integer"},
            },
            "required": ["i0", "i1", "j0", "j1"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_detection",
        "description": (
            "Submit the final detection: one bounding box per extended object, groups of "
            "candidate ids forming one object (lobes and core of a radio galaxy), the ids "
            "of the remaining accepted compact candidates, and positions of extra compact "
            "sources missing from the candidate list. Call this exactly once at the end."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "i0": {"type": "integer"},
                            "i1": {"type": "integer"},
                            "j0": {"type": "integer"},
                            "j1": {"type": "integer"},
                            "notes": {"type": "string"},
                        },
                        "required": ["name", "i0", "i1", "j0", "j1", "notes"],
                        "additionalProperties": False,
                    },
                },
                "groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "ids": {"type": "array", "items": {"type": "integer"}},
                            "notes": {"type": "string"},
                        },
                        "required": ["name", "ids", "notes"],
                        "additionalProperties": False,
                    },
                },
                "compact_ids": {"type": "array", "items": {"type": "integer"}},
                "extra_compact": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "i": {"type": "integer"},
                            "j": {"type": "integer"},
                        },
                        "required": ["i", "j"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["objects", "groups", "compact_ids", "extra_compact"],
            "additionalProperties": False,
        },
    },
]


class _Image:
    """Reconstructed image with the helpers behind the agent tools."""

    def __init__(self, val, n_orders=5, view_px=768, contrast_size=15):
        self.val = val
        self.shp = val.shape
        self.max = float(val.max())
        self.floor = self.max / 10**n_orders
        self.bg = max(float(np.median(val)), self.floor)
        self.view_px = view_px
        self.contrast_size = contrast_size
        self._contrast = None
        self.candidates = []

    @property
    def contrast(self):
        """Ratio of each pixel to the median of its surroundings."""
        if self._contrast is None:
            local = median_filter(self.val, size=self.contrast_size)
            self._contrast = self.val / np.maximum(local, self.floor)
        return self._contrast

    def clip_box(self, i0, i1, j0, j1):
        """Clip a box to the image and make sure it is not empty."""
        i0, i1 = sorted((int(np.clip(i0, 0, self.shp[0])), int(np.clip(i1, 0, self.shp[0]))))
        j0, j1 = sorted((int(np.clip(j0, 0, self.shp[1])), int(np.clip(j1, 0, self.shp[1]))))
        if i1 == i0:
            i0, i1 = max(i0 - 1, 0), min(i1 + 1, self.shp[0])
        if j1 == j0:
            j0, j1 = max(j0 - 1, 0), min(j1 + 1, self.shp[1])
        return i0, i1, j0, j1

    def render(self, i0, i1, j0, j1, scale="log"):
        """Render a region as a PNG with index axes, returned base64-encoded."""
        i0, i1, j0, j1 = self.clip_box(i0, i1, j0, j1)
        sub = self.val[i0:i1, j0:j1]
        if scale == "log":
            vmax = max(float(sub.max()), self.floor * 10)
            norm = LogNorm(vmin=self.floor, vmax=vmax, clip=True)
            sub = np.clip(sub, self.floor, None)
        else:
            norm = Normalize(vmin=float(sub.min()), vmax=float(sub.max()))

        size = self.view_px / 100
        fig = Figure(figsize=(size, size), dpi=100)
        ax = fig.subplots()
        im = ax.imshow(
            sub,
            origin="lower",
            cmap="inferno",
            norm=norm,
            extent=(j0 - 0.5, j1 - 0.5, i0 - 0.5, i1 - 0.5),
            interpolation="nearest",
        )

        # mark the compact candidates, with ids in zoomed views
        cands = [c for c in self.candidates if i0 <= c["i"] < i1 and j0 <= c["j"] < j1]
        if cands:
            ii, jj = np.array([[c["i"], c["j"]] for c in cands]).T
            ax.scatter(jj, ii, s=60, facecolors="none", edgecolors="cyan", linewidths=0.8)
            if max(i1 - i0, j1 - j0) <= 256:
                for c in cands:
                    ax.annotate(
                        str(c["id"]), (c["j"], c["i"]), xytext=(4, 4),
                        textcoords="offset points", color="cyan", fontsize=8,
                    )

        ax.set_xlim(j0 - 0.5, j1 - 0.5)
        ax.set_ylim(i0 - 0.5, i1 - 0.5)
        ax.set_xlabel("j (axis 1)")
        ax.set_ylabel("i (axis 0)")
        ax.set_title(f"i [{i0}, {i1}), j [{j0}, {j1}), {scale} scale")
        ax.grid(color="white", alpha=0.25, linewidth=0.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    def render_sheet(self, ids, half=24, cols=8):
        """Contact sheet of small zooms around the given candidates, returned base64-encoded."""
        by_id = {c["id"]: c for c in self.candidates}
        cands = [by_id[k] for k in ids if k in by_id]
        if not cands:
            raise ValueError(f"no known candidate ids in {list(ids)}")
        rows = -(-len(cands) // cols)
        fig = Figure(figsize=(2.0 * cols, 2.1 * rows), dpi=90)
        axs = np.atleast_1d(fig.subplots(rows, cols)).ravel()
        for ax in axs:
            ax.set_axis_off()
        for ax, c in zip(axs, cands, strict=False):  # the last row may have empty panels
            i0, i1, j0, j1 = self.clip_box(c["i"] - half, c["i"] + half, c["j"] - half, c["j"] + half)
            sub = np.clip(self.val[i0:i1, j0:j1], self.floor, None)
            ax.imshow(
                sub,
                origin="lower",
                cmap="inferno",
                norm=LogNorm(vmin=self.floor, vmax=max(float(sub.max()), self.floor * 10)),
                extent=(j0 - 0.5, j1 - 0.5, i0 - 0.5, i1 - 0.5),
                interpolation="nearest",
            )
            # mark all candidates in the zoom, the reviewed one in cyan, neighbours in white
            for d in self.candidates:
                if i0 <= d["i"] < i1 and j0 <= d["j"] < j1:
                    col = "cyan" if d["id"] == c["id"] else "white"
                    ax.scatter(d["j"], d["i"], s=160, facecolors="none", edgecolors=col, linewidths=0.6)
                    if d["id"] != c["id"]:
                        ax.annotate(str(d["id"]), (d["j"], d["i"]), xytext=(3, 3), textcoords="offset points", color="white", fontsize=6)
            ax.set_xlim(j0 - 0.5, j1 - 0.5)
            ax.set_ylim(i0 - 0.5, i1 - 0.5)
            ax.set_title(f"#{c['id']} ({c['i']},{c['j']}) con {c['contrast']:.0f}", fontsize=7)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    def find_peaks(self, i0, i1, j0, j1, min_rel_flux=1e-3, min_contrast=0.0, size=5, n_max=50):
        """Local maxima in a region, brightest first."""
        i0, i1, j0, j1 = self.clip_box(i0, i1, j0, j1)
        sub = self.val[i0:i1, j0:j1]
        con = self.contrast[i0:i1, j0:j1]
        is_peak = (
            (sub == maximum_filter(sub, size=size))
            & (sub >= min_rel_flux * self.max)
            & (con >= min_contrast)
        )
        loc = np.argwhere(is_peak)
        flux, con = sub[is_peak], con[is_peak]
        order = np.argsort(-flux)[:n_max]
        return [
            dict(
                i=int(loc[k, 0] + i0),
                j=int(loc[k, 1] + j0),
                rel_flux=float(f"{flux[k] / self.max:.3g}"),
                contrast=float(f"{con[k]:.3g}"),
            )
            for k in order
        ]

    def find_candidates(self, min_contrast=20.0, merge_dist=5.0, n_max=500):
        """Compact source candidates: local maxima standing out from their surroundings.

        Chains of peaks closer than *merge_dist* are merged into one candidate at
        the brightest peak, with all peak positions kept as ``members``.
        """
        peaks = self.find_peaks(
            0, self.shp[0], 0, self.shp[1],
            min_rel_flux=10 * self.floor / self.max,
            min_contrast=min_contrast, size=7, n_max=n_max,
        )

        # union-find over close peaks, the root is always the brightest (first) peak
        root = list(range(len(peaks)))

        def find(a):
            while root[a] != a:
                a = root[a]
            return a

        for a in range(len(peaks)):
            for b in range(a + 1, len(peaks)):
                dist = np.hypot(peaks[a]["i"] - peaks[b]["i"], peaks[a]["j"] - peaks[b]["j"])
                if dist <= merge_dist:
                    ra, rb = find(a), find(b)
                    root[max(ra, rb)] = min(ra, rb)

        self.candidates = []
        for a, p in enumerate(peaks):
            if find(a) != a:
                continue
            members = [dict(i=q["i"], j=q["j"]) for b, q in enumerate(peaks) if find(b) == a]
            self.candidates.append(dict(id=len(self.candidates), **p, members=members))
        return self.candidates

    def find_pairs(self, max_dist=40.0, max_ratio=15.0, min_bridge=0.0, min_align=0.8):
        """Close candidate pairs of similar flux, possible lobes of one radio galaxy.

        Each pair gets two morphological features: ``bridge``, the lowest point of
        the profile between the peaks above the local background, as fraction of the
        fainter peak (> 0: the peaks are connected), and ``align``, the |cos| between
        each lobe's major axis and the line to the other lobe (1: pointing at each
        other, None: too small to measure). Pairs with ``bridge > min_bridge`` and
        both ``align >= min_align`` are flagged as ``likely`` lobes.
        """
        pairs = []
        for a, ca in enumerate(self.candidates):
            for cb in self.candidates[a + 1 :]:
                dist = float(np.hypot(ca["i"] - cb["i"], ca["j"] - cb["j"]))
                ratio = max(ca["rel_flux"], cb["rel_flux"]) / min(ca["rel_flux"], cb["rel_flux"])
                if dist > max_dist or ratio > max_ratio:
                    continue
                i0, i1, j0, j1 = self.clip_box(
                    min(ca["i"], cb["i"]) - 10, max(ca["i"], cb["i"]) + 11,
                    min(ca["j"], cb["j"]) - 10, max(ca["j"], cb["j"]) + 11,
                )
                bg = max(self.bg, _border_median(self.val[i0:i1, j0:j1]))
                bridge = self.bridge(ca, cb, bg)
                align = [self.alignment(ca, cb, bg), self.alignment(cb, ca, bg)]
                likely = bridge > min_bridge and all(x is not None and x >= min_align for x in align)
                pairs.append(
                    dict(
                        ids=[ca["id"], cb["id"]],
                        dist=round(dist, 1),
                        flux_ratio=round(ratio, 1),
                        bridge=round(bridge, 3),
                        align=[None if x is None else round(x, 2) for x in align],
                        likely=likely,
                    )
                )
        return pairs

    def bridge(self, ca, cb, bg):
        """Lowest point of the profile between two peaks above bg, as fraction of the fainter peak."""
        pa, pb = np.array([ca["i"], ca["j"]], float), np.array([cb["i"], cb["j"]], float)
        perp = np.array([pa[1] - pb[1], pb[0] - pa[0]]) / np.linalg.norm(pb - pa)
        low = np.inf
        for t in np.linspace(0, 1, int(np.ceil(np.linalg.norm(pb - pa) * 2)) + 1):
            p = pa + t * (pb - pa)
            # maximum across the line (+-1 px), so slightly curved bridges are followed
            across = [np.clip(np.round(p + s * perp).astype(int), 0, np.array(self.shp) - 1) for s in (-1, 0, 1)]
            low = min(low, max(self.val[tuple(q)] for q in across))
        faint = min(self.val[ca["i"], ca["j"]], self.val[cb["i"], cb["j"]])
        return float((low - bg) / max(faint - bg, 1e-30))

    def alignment(self, c, other, bg, frac=0.05, half=8):
        """|cos| between a lobe's major axis and the direction to the other lobe, None if too small."""
        i, j = c["i"], c["j"]
        i0, i1, j0, j1 = self.clip_box(i - half, i + half + 1, j - half, j + half + 1)
        sub = self.val[i0:i1, j0:j1]
        lbl, _ = label(sub > bg + frac * (self.val[i, j] - bg))
        pix = np.argwhere(lbl == lbl[i - i0, j - j0]).astype(float)
        if len(pix) < 3:
            return None
        w = sub[tuple(pix.astype(int).T)]
        cov = np.cov((pix - (pix * w[:, None]).sum(0) / w.sum()).T, aweights=w)
        major = np.linalg.eigh(cov)[1][:, -1]
        d = np.array([other["i"] - i, other["j"] - j], float)
        return float(abs(major @ d) / np.linalg.norm(d))

    def region_stats(self, i0, i1, j0, j1):
        """Summary statistics of a region."""
        i0, i1, j0, j1 = self.clip_box(i0, i1, j0, j1)
        sub = self.val[i0:i1, j0:j1]
        k = np.unravel_index(np.argmax(sub), sub.shape)
        return dict(
            box=[i0, i1, j0, j1],
            flux_sum=float(sub.sum()),
            flux_fraction=float(sub.sum() / self.val.sum()),
            rel_max=float(sub.max() / self.max),
            argmax=[int(k[0] + i0), int(k[1] + j0)],
            rel_median=float(np.median(sub) / self.max),
            n_above_bg=int((sub > self.bg).sum()),
        )

    def summary(self):
        """Text summary of the full image for the first message."""
        return dict(
            shape=list(self.shp),
            max=self.max,
            dynamic_range_floor=self.floor,
            background_level=self.bg,
            n_compact_candidates=len(self.candidates),
        )


def _border_median(sub):
    """Median of the pixels on the border of a 2D array."""
    return float(np.median(np.concatenate([sub[0], sub[-1], sub[1:-1, 0], sub[1:-1, -1]])))


def boxes_to_maps(
    val,
    objects,
    compact=(),
    groups=(),
    n_orders=5,
    obj_threshold=1e-3,
    obj_min_frac=0.1,
    compact_threshold=0.05,
    compact_box=8,
    snap_radius=3,
    min_pix=4,
):
    """Convert object boxes and compact source positions into detection maps.

    Every source is turned into a mask. Single-pixel masks become point sources,
    all others extended objects.

    Parameters
    ----------
    val : np.ndarray
        2D reconstructed image.
    objects : list of dict
        Boxes of extended objects with keys ``i0``, ``i1``, ``j0``, ``j1``.
    compact : list of dict, optional
        Positions of compact sources with keys ``i`` and ``j``.
    groups : list of list of dict, optional
        Groups of compact source positions forming one object, e.g. the lobes
        of a radio galaxy. Each group becomes one object made of the masks of
        its members and any bridge between them above the threshold.
    n_orders : int, optional
        Dynamic range below the image maximum used as the background floor.
        Default is 5.
    obj_threshold : float, optional
        Mask threshold of the extended objects as fraction of (box peak -
        background) above the background. Default is 1e-3.
    compact_threshold : float, optional
        Mask threshold of the compact sources as fraction of (peak - local
        background) above the local background. Default is 0.05.
    compact_box : int, optional
        Half-size of the box around a compact source. Default is 8.
    snap_radius : int, optional
        Compact sources are moved to the brightest pixel within this radius.
        Default is 3.
    min_pix : int, optional
        Connected mask components smaller than this are discarded, except for
        the one containing the box peak. Default is 4.
    obj_min_frac : float, optional
        Besides their largest component (main body), extended objects keep only
        components of at least this fraction of the main body's size, e.g. lobes
        separated at the threshold, but not stray patches of the background.
        Default is 0.1.

    Returns
    -------
    ps_map : np.ndarray
        Binary point-source map of the same shape as *val*.
    cl_map : np.ndarray
        Array of shape ``(n_objects, *val.shape)`` with one binary map per
        object, sorted by descending pixel count.
    """
    img = _Image(val, n_orders=n_orders)
    taken = np.zeros(val.shape, dtype=bool)
    ps_map = np.zeros(val.shape)
    maps = []

    def snap(src):
        """Move a position to the brightest pixel nearby."""
        i0, i1, j0, j1 = img.clip_box(
            src["i"] - snap_radius, src["i"] + snap_radius + 1,
            src["j"] - snap_radius, src["j"] + snap_radius + 1,
        )
        k = np.unravel_index(np.argmax(val[i0:i1, j0:j1]), (i1 - i0, j1 - j0))
        return k[0] + i0, k[1] + j0

    # snap the compact sources and group members, extended objects must not claim them
    peaks = [snap(src) for src in compact]
    group_peaks = [[snap(src) for src in grp] for grp in groups]
    peak_map = np.zeros(val.shape, dtype=bool)
    for p in peaks + [p for grp in group_peaks for p in grp]:
        peak_map[p] = True

    def add_mask(box, frac, keep_all):
        i0, i1, j0, j1 = box
        sub = val[i0:i1, j0:j1]
        free = ~taken[i0:i1, j0:j1]
        if not free.any():
            return
        peak = np.unravel_index(np.argmax(np.where(free, sub, -np.inf)), sub.shape)
        bg = max(img.bg, _border_median(sub)) if min(sub.shape) > 2 else img.bg
        mask = (sub > bg + frac * max(float(sub[peak]) - bg, 0.0)) & free

        # compact sources keep the component of their peak; extended objects keep their
        # largest component (main body) and other components of at least obj_min_frac of
        # its size, except those containing a compact source (sources of their own)
        lbl, n_lbl = label(mask)
        keep = [lbl[peak]] if lbl[peak] else []
        if keep_all and n_lbl:
            sizes = np.bincount(lbl.ravel(), minlength=n_lbl + 1)
            compact_lbl = set(np.unique(lbl[peak_map[i0:i1, j0:j1]]))
            keep = [int(np.argmax(sizes[1:])) + 1]
            min_size = max(min_pix, obj_min_frac * sizes[keep[0]])
            keep += [k for k in np.flatnonzero(sizes >= min_size) if k not in compact_lbl]
        mask = np.isin(lbl, [k for k in keep if k > 0])

        # fall back to the peak (compact) or the free part of the box (extended)
        if not mask.any():
            mask = np.zeros_like(free)
            mask[peak] = True
            if keep_all:
                mask = free

        src = np.zeros(val.shape, dtype=bool)
        src[i0:i1, j0:j1] = mask
        taken[:] |= src
        if src.sum() == 1:
            ps_map[src] = 1
        else:
            maps.append(src.astype("float64"))

    # extended objects first, they claim their pixels
    for obj in objects:
        add_mask(img.clip_box(obj["i0"], obj["i1"], obj["j0"], obj["j1"]), obj_threshold, True)

    # groups (e.g. the lobes of a radio galaxy): one object from the members and the bridges between them
    for grp in group_peaks:
        grp = [p for p in grp if not taken[p]]
        if not grp:
            continue
        ii, jj = np.array(grp).T
        i0, i1, j0, j1 = img.clip_box(
            ii.min() - compact_box, ii.max() + compact_box + 1,
            jj.min() - compact_box, jj.max() + compact_box + 1,
        )
        sub = val[i0:i1, j0:j1]
        free = ~taken[i0:i1, j0:j1]
        bg = max(img.bg, _border_median(sub))
        low = min(float(val[p]) for p in grp)
        lbl, _ = label((sub > bg + compact_threshold * max(low - bg, 0.0)) & free)
        mask = np.isin(lbl, [lbl[i - i0, j - j0] for i, j in grp if lbl[i - i0, j - j0]])
        for i, j in grp:
            mask[i - i0, j - j0] = True
        src = np.zeros(val.shape, dtype=bool)
        src[i0:i1, j0:j1] = mask
        taken |= src
        maps.append(src.astype("float64"))

    # compact sources: skip those inside other sources
    for i, j in peaks:
        if taken[i, j]:
            continue
        box = img.clip_box(i - compact_box, i + compact_box + 1, j - compact_box, j + compact_box + 1)
        add_mask(box, compact_threshold, False)

    # sort the object maps by size in descending order
    cl_map = np.array(maps).reshape((len(maps),) + val.shape)
    cl_map = cl_map[np.argsort(-cl_map.sum(axis=(1, 2)), kind="stable")]
    return ps_map, cl_map


def _prepare(reconstruction, n_orders, view_px, min_contrast, merge_dist, pair_dist, pair_ratio):
    """Image, compact candidates, pair hints and the introductory text of a detection task."""
    val = np.squeeze(np.asarray(reconstruction.val, dtype="float64"))
    if val.ndim != 2:
        raise ValueError(f"agent detection needs a 2D image, got shape {val.shape}")
    img = _Image(val, n_orders=n_orders, view_px=view_px)
    cands = img.find_candidates(min_contrast=min_contrast, merge_dist=merge_dist)
    pairs = img.find_pairs(max_dist=pair_dist, max_ratio=pair_ratio)

    cand_txt = "\n".join(
        f"{c['id']}: i={c['i']} j={c['j']} rel_flux={c['rel_flux']} contrast={c['contrast']}"
        + (f" (merged with {len(c['members']) - 1} adjacent peaks)" if len(c["members"]) > 1 else "")
        for c in cands
    )
    pair_txt = "\n".join(
        f"{p['ids'][0]} + {p['ids'][1]}: dist={p['dist']} flux_ratio={p['flux_ratio']} "
        f"bridge={p['bridge']} align={p['align']}" + (" LIKELY LOBES" if p["likely"] else "")
        for p in pairs
    )
    intro = (
        "Full reconstruction (log scale, compact candidates circled). Image summary:\n"
        + json.dumps(img.summary(), indent=1)
        + f"\n\nCompact candidates (contrast >= {min_contrast}, brightest first):\n"
        + cand_txt
        + f"\n\nClose candidate pairs (dist <= {pair_dist} px, flux ratio <= {pair_ratio}), "
        "possible lobes of one radio galaxy:\n"
        + (pair_txt or "none")
        + "\n\nIdentify the extended objects, review the compact candidates and the pairs, "
        "and submit your detection."
    )
    return val, img, cands, pairs, intro


def check_submission(submission, cands):
    """Check a detection submission for missing fields, wrong types and unknown candidate ids."""
    errors = []
    for key in ("objects", "groups", "compact_ids", "extra_compact"):
        if not isinstance(submission.get(key), list):
            errors.append(f"'{key}' must be a list")
    if errors:
        raise ValueError("invalid submission: " + "; ".join(errors))

    def is_int(x):
        return isinstance(x, int) and not isinstance(x, bool)

    for k, obj in enumerate(submission["objects"]):
        if not all(is_int(obj.get(b)) for b in ("i0", "i1", "j0", "j1")):
            errors.append(f"objects[{k}] needs integer i0, i1, j0, j1")
    for k, src in enumerate(submission["extra_compact"]):
        if not (is_int(src.get("i")) and is_int(src.get("j"))):
            errors.append(f"extra_compact[{k}] needs integer i, j")
    known = {c["id"] for c in cands}
    ids = list(submission["compact_ids"])
    for k, grp in enumerate(submission["groups"]):
        if not isinstance(grp.get("ids"), list) or len(grp["ids"]) < 2:
            errors.append(f"groups[{k}] needs a list 'ids' with at least two candidate ids")
        else:
            ids += grp["ids"]
    unknown = sorted({k for k in ids if k not in known}, key=str)
    if unknown:
        errors.append(f"unknown candidate ids {unknown} (valid: 0..{len(cands) - 1})")
    if errors:
        raise ValueError("invalid submission: " + "; ".join(errors))


def submission_to_maps(val, submission, cands, **mask_kwargs):
    """Detection maps from a submission, see ``boxes_to_maps`` for the mask settings.

    Ids in a group are not used as single compact sources; merged candidates
    (several adjacent peaks) are built like groups of their peaks.
    """
    by_id = {c["id"]: c for c in cands}
    groups = [
        [m for k in grp["ids"] if k in by_id for m in by_id[k]["members"]]
        for grp in submission["groups"]
    ]
    grouped = {k for grp in submission["groups"] for k in grp["ids"]}
    compact = list(submission["extra_compact"])
    for k in submission["compact_ids"]:
        if k not in by_id or k in grouped:
            continue
        if len(by_id[k]["members"]) > 1:
            groups.append(by_id[k]["members"])
        else:
            compact.append(by_id[k])
    return boxes_to_maps(val, submission["objects"], compact, [grp for grp in groups if grp], **mask_kwargs)


def agent_detect(
    reconstruction,
    model="claude-opus-5-5",
    effort="high",
    max_turns=25,
    n_orders=5,
    view_px=768,
    min_contrast=20.0,
    merge_dist=5.0,
    pair_dist=40.0,
    pair_ratio=15.0,
    obj_threshold=1e-3,
    obj_min_frac=0.1,
    compact_threshold=0.05,
    compact_box=8,
    snap_radius=3,
    max_tokens=16000,
    client=None,
    print_det=True,
):
    """Detect point sources and extended objects with a Claude vision agent.

    Parameters
    ----------
    reconstruction : ImageData
        Reconstructed image (posterior mean) with a 2D ``val`` array.
    model : str, optional
        Claude model id. Default is 'claude-opus-5-5'.
    effort : str, optional
        Effort level of the model. Default is 'high'.
    max_turns : int, optional
        Maximum number of model calls. Default is 25.
    n_orders : int, optional
        Dynamic range of the log rendering and background floor. Default is 5.
    view_px : int, optional
        Size of the rendered images in pixels. Default is 768.
    min_contrast : float, optional
        Minimum contrast to the local surroundings for compact candidates.
        Default is 20.
    merge_dist : float, optional
        Candidates closer than this (pixels) are merged into one. Default is 5.
    pair_dist, pair_ratio : float, optional
        Maximum distance (pixels) and flux ratio of candidate pairs listed to the
        agent as possible lobes of one radio galaxy. Default is 40 and 15.
    obj_threshold, obj_min_frac, compact_threshold, compact_box, snap_radius : optional
        Mask settings, see ``boxes_to_maps``.
    max_tokens : int, optional
        Maximum number of output tokens per model call. Default is 16000.
    client : anthropic.Anthropic, optional
        Client to use. Default creates one from the environment.
    print_det : bool, optional
        Whether to print the number of detected sources. Default is True.

    Returns
    -------
    ps_map : np.ndarray
        Binary point-source map.
    cl_map : np.ndarray
        One binary map per extended object, sorted by descending size.
    log : dict
        Transcript of the agent run (text, tool calls, submission, usage).
    """
    if client is None:
        try:
            import anthropic
        except ImportError as err:
            raise ImportError(
                "agent_detect needs the 'anthropic' package, which is only installed in the "
                "'agent' environment. Run with `pixi run -e agent ...` (or "
                "`pip install anthropic`), or use base_detect.method: unet."
            ) from err
        client = anthropic.Anthropic()

    val, img, cands, pairs, intro = _prepare(
        reconstruction, n_orders, view_px, min_contrast, merge_dist, pair_dist, pair_ratio
    )

    def image_block(data):
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}

    messages = [
        {
            "role": "user",
            "content": [
                image_block(img.render(0, img.shp[0], 0, img.shp[1], "log")),
                {"type": "text", "text": intro},
            ],
        }
    ]
    log = dict(
        model=model, effort=effort, candidates=cands, pairs=pairs, events=[],
        usage=dict(input_tokens=0, output_tokens=0),
    )
    submission = None

    for turn in range(max_turns):
        response = client.beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        log["usage"]["input_tokens"] += response.usage.input_tokens
        log["usage"]["output_tokens"] += response.usage.output_tokens
        log["served_by"] = response.model

        if response.stop_reason == "refusal":
            raise RuntimeError(f"agent_detect: request refused ({response.stop_details})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("agent_detect: response hit max_tokens")

        messages.append({"role": "assistant", "content": response.content})

        # run the requested tools and collect the results in one user message
        results = []
        for block in response.content:
            if block.type == "text":
                log["events"].append(dict(turn=turn, text=block.text))
            if block.type != "tool_use":
                continue
            args = block.input
            log["events"].append(dict(turn=turn, tool=block.name, input=args))

            if block.name == "submit_detection":
                submission = args
                content = "Detection received."
            elif block.name == "view_region":
                box = img.clip_box(args["i0"], args["i1"], args["j0"], args["j1"])
                content = [image_block(img.render(*box, args["scale"]))]
            elif block.name == "view_candidates":
                ids = args["ids"][:40]
                try:
                    content = [image_block(img.render_sheet(ids))]
                except ValueError as err:
                    content = str(err)
            elif block.name == "find_peaks":
                box = (args["i0"], args["i1"], args["j0"], args["j1"])
                content = json.dumps(img.find_peaks(*box, min_rel_flux=args["min_rel_flux"]))
            elif block.name == "region_stats":
                content = json.dumps(img.region_stats(args["i0"], args["i1"], args["j0"], args["j1"]))
            else:
                content = f"Unknown tool {block.name}."
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})

        if submission is not None:
            break

        if results:
            if turn == max_turns - 2:
                results.append(
                    {"type": "text", "text": "This is your last turn: call submit_detection now."}
                )
            messages.append({"role": "user", "content": results})
        else:
            messages.append(
                {"role": "user", "content": "Please call submit_detection with your final answer."}
            )

    if submission is None:
        raise RuntimeError(f"agent_detect: no detection submitted after {max_turns} turns")
    log["submission"] = submission

    ps_map, cl_map = submission_to_maps(
        val, submission, cands,
        n_orders=n_orders,
        obj_threshold=obj_threshold,
        obj_min_frac=obj_min_frac,
        compact_threshold=compact_threshold,
        compact_box=compact_box,
        snap_radius=snap_radius,
    )
    if print_det:
        print(f"n point sources: {int(ps_map.sum())}")
        print(f"n objects: {cl_map.shape[0]}")
        print(f"tokens in/out: {log['usage']['input_tokens']}/{log['usage']['output_tokens']}")
    return ps_map, cl_map, log


# ---------------------------------------------------------------------------
# manual mode: the agent's role is taken by a person or an interactive Claude
# session, which reads an exported task and writes the submission as JSON
# ---------------------------------------------------------------------------


def _task_settings(n_orders, view_px, min_contrast, merge_dist, pair_dist, pair_ratio):
    return dict(
        n_orders=n_orders, view_px=view_px, min_contrast=min_contrast,
        merge_dist=merge_dist, pair_dist=pair_dist, pair_ratio=pair_ratio,
    )


def export_task(reconstruction, task_dir, rec_path=None, submission_file=None, **settings):
    """Write a detection task for manual mode.

    The task contains the same information the API agent gets: the overview
    image (``overview.png``), the instructions and candidate/pair lists
    (``task.md``), the candidates as JSON (``candidates.json``), and an empty
    submission (``submission_template.json``). Zooms, peaks and statistics are
    available through ``python -m aim_resolve.agent_detect``.
    """
    import os

    val, img, cands, pairs, intro = _prepare(reconstruction, **settings)
    os.makedirs(task_dir, exist_ok=True)

    with open(os.path.join(task_dir, "overview.png"), "wb") as f:
        f.write(base64.standard_b64decode(img.render(0, img.shp[0], 0, img.shp[1], "log")))
    with open(os.path.join(task_dir, "candidates.json"), "w") as f:
        json.dump(dict(settings=settings, candidates=cands, pairs=pairs), f, indent=1)
    with open(os.path.join(task_dir, "submission_template.json"), "w") as f:
        json.dump(dict(objects=[], groups=[], compact_ids=[], extra_compact=[]), f, indent=1)

    opts = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in settings.items() if k != "view_px")
    rec = rec_path or "<rec.pkl>"
    tool = f"python -m aim_resolve.agent_detect {{}} {rec} {opts}"
    schema = json.dumps(TOOLS[-1]["input_schema"], indent=1)
    task = f"""# Source detection task

## Instructions

{SYSTEM_PROMPT}

## Tools

Run from any directory with the aim-resolve environment (all coordinates are array indices):

- zoom:  `{tool.format("view")} --box I0 I1 J0 J1 [--scale linear] --out FILE.png`
- candidate sheet (up to 40 ids): `{tool.format("sheet")} --ids ID ID ... --out FILE.png`
- peaks: `{tool.format("peaks")} --box I0 I1 J0 J1 --min-rel-flux 0.001`
- stats: `{tool.format("stats")} --box I0 I1 J0 J1`

## Submission

Write the final detection as JSON to:

    {submission_file or "<submission.json>"}

(start from `submission_template.json` in this directory). Schema of `submit_detection`:

```json
{schema}
```

Then run the detect step again; it builds the detection maps from the submission.

## Image (overview.png)

{intro}
"""
    with open(os.path.join(task_dir, "task.md"), "w") as f:
        f.write(task)
    return cands, pairs


def manual_detect(
    reconstruction,
    task_dir,
    submission_file,
    rec_path=None,
    n_orders=5,
    view_px=768,
    min_contrast=20.0,
    merge_dist=5.0,
    pair_dist=40.0,
    pair_ratio=15.0,
    obj_threshold=1e-3,
    obj_min_frac=0.1,
    compact_threshold=0.05,
    compact_box=8,
    snap_radius=3,
    print_det=True,
    **_api_settings,
):
    """Source detection with a manually written submission (no API calls).

    If *submission_file* does not exist yet, the task is exported to *task_dir*
    and None is returned. Otherwise the submission is checked and converted into
    detection maps exactly as in ``agent_detect``. Settings only used by the API
    agent (model, effort, ...) are ignored.

    Returns
    -------
    None or (ps_map, cl_map, log)
    """
    import os

    settings = _task_settings(n_orders, view_px, min_contrast, merge_dist, pair_dist, pair_ratio)
    if not os.path.exists(submission_file):
        export_task(reconstruction, task_dir, rec_path, submission_file, **settings)
        return None

    with open(submission_file) as f:
        submission = json.load(f)
    val, _, cands, pairs, _ = _prepare(reconstruction, **settings)
    check_submission(submission, cands)

    ps_map, cl_map = submission_to_maps(
        val, submission, cands,
        n_orders=n_orders,
        obj_threshold=obj_threshold,
        obj_min_frac=obj_min_frac,
        compact_threshold=compact_threshold,
        compact_box=compact_box,
        snap_radius=snap_radius,
    )
    log = dict(model="manual", candidates=cands, pairs=pairs, submission=submission)
    if print_det:
        print(f"n point sources: {int(ps_map.sum())}")
        print(f"n objects: {cl_map.shape[0]}")
    return ps_map, cl_map, log


def main(argv=None):
    """Command line access to the agent tools for manual mode."""
    import argparse

    from .img_data import ImageData

    parser = argparse.ArgumentParser(prog="python -m aim_resolve.agent_detect", description=main.__doc__)
    parser.add_argument("tool", choices=["view", "sheet", "peaks", "stats"])
    parser.add_argument("rec", help="reconstruction pickle ({it}_rec.pkl)")
    parser.add_argument("--box", nargs=4, type=int, metavar=("I0", "I1", "J0", "J1"))
    parser.add_argument("--ids", nargs="+", type=int, help="candidate ids for sheet (up to 40)")
    parser.add_argument("--scale", choices=["log", "linear"], default="log")
    parser.add_argument("--out", help="output PNG of view or sheet")
    parser.add_argument("--min-rel-flux", type=float, default=1e-3)
    parser.add_argument("--n-orders", type=int, default=5)
    parser.add_argument("--view-px", type=int, default=768)
    parser.add_argument("--min-contrast", type=float, default=20.0)
    parser.add_argument("--merge-dist", type=float, default=5.0)
    parser.add_argument("--pair-dist", type=float, default=40.0)
    parser.add_argument("--pair-ratio", type=float, default=15.0)
    args = parser.parse_args(argv)

    val = np.squeeze(np.asarray(ImageData.load(args.rec, dtype="float32").val, dtype="float64"))
    img = _Image(val, n_orders=args.n_orders, view_px=args.view_px)
    if args.tool == "sheet":
        if not (args.ids and args.out):
            parser.error("sheet needs --ids and --out")
        img.find_candidates(min_contrast=args.min_contrast, merge_dist=args.merge_dist)
        with open(args.out, "wb") as f:
            f.write(base64.standard_b64decode(img.render_sheet(args.ids[:40])))
        print(args.out)
        return
    if args.box is None:
        parser.error(f"{args.tool} needs --box")
    box = img.clip_box(*args.box)
    if args.tool == "view":
        if not args.out:
            parser.error("view needs --out")
        img.find_candidates(min_contrast=args.min_contrast, merge_dist=args.merge_dist)
        with open(args.out, "wb") as f:
            f.write(base64.standard_b64decode(img.render(*box, args.scale)))
        print(args.out)
    elif args.tool == "peaks":
        print(json.dumps(img.find_peaks(*box, min_rel_flux=args.min_rel_flux), indent=1))
    else:
        print(json.dumps(img.region_stats(*box), indent=1))


if __name__ == "__main__":
    main()
