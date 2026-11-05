"""Tests for aim_resolve.reuse: the same-source test between pipeline iterations."""

import numpy as np

from aim_resolve.mask import add_margin
from aim_resolve.reuse import (
    bbox,
    component_masks,
    match_components,
    same_source,
    upsample_mask,
)

N = 128


def blob(i0, i1, j0, j1, margin=6, n=N):
    """Detection rectangle with a margin, as masks_from_maps builds it."""
    m = np.zeros((n, n))
    m[i0:i1, j0:j1] = 1
    return add_margin(m, margin, round=False)


class TestSameSource:
    def test_identical(self):
        ok, stats = same_source(blob(40, 60, 40, 70), blob(40, 60, 40, 70), margin=3)
        assert ok and stats["iou"] == 1.0

    def test_slightly_shifted_still_fits(self):
        ok, _ = same_source(blob(41, 61, 41, 71), blob(40, 60, 40, 70), margin=3)
        assert ok

    def test_grown_does_not_fit(self):
        ok, stats = same_source(blob(30, 70, 40, 70), blob(40, 60, 40, 70), margin=3)
        assert not ok and ("fit" in stats["reason"] or "iou" in stats["reason"])

    def test_shrunk_too_loose(self):
        ok, stats = same_source(blob(48, 52, 50, 56), blob(30, 70, 30, 80), margin=1, iou_min=0.0)
        assert not ok and "area" in stats["reason"]


class TestMatch:
    def test_one_to_one_split_merge_type_new(self):
        old = {"o0.1": ("object", blob(20, 40, 20, 50)), "o1.1": ("object", blob(80, 100, 80, 110)),
               "t0.1[0]": ("tile", blob(60, 64, 10, 14, margin=4))}
        new = {
            "o0.2": ("object", blob(21, 41, 20, 50)),            # same source
            "o1.2": ("object", blob(80, 90, 80, 110)),            # split of old o1 ...
            "o2.2": ("object", blob(92, 100, 80, 110)),           # ... into two
            "o3.2": ("object", blob(58, 66, 8, 16, margin=4)),    # tile -> object
            "t0.2[0]": ("tile", blob(5, 9, 100, 104, margin=4)),  # new source
        }
        res = match_components(new, old, margin=3)
        assert res["o0.2"]["old"] == "o0.1"
        assert res["o1.2"]["old"] is None and "split" in res["o1.2"]["reason"]
        assert res["o3.2"]["old"] is None and "type" in res["o3.2"]["reason"]
        assert res["t0.2[0]"]["old"] is None and res["t0.2[0]"]["reason"] == "new source"

    def test_merge(self):
        old = {"o0.1": ("object", blob(20, 40, 20, 50)), "o1.1": ("object", blob(42, 60, 20, 50))}
        new = {"o0.2": ("object", blob(20, 60, 20, 50))}
        res = match_components(new, old, margin=3)
        assert res["o0.2"]["old"] is None and "merge" in res["o0.2"]["reason"]

    def test_zoom_step(self):
        # the same source on a 2x finer grid: the old mask upsampled by 2 matches the new one
        old = upsample_mask(blob(20, 40, 20, 50, n=64), 2)
        new = blob(40, 80, 40, 100, margin=12)
        assert old.shape == new.shape == (N, N)
        res = match_components({"o0.2": ("object", new)}, {"o0.1": ("object", old)}, margin=6)
        assert res["o0.2"]["old"] == "o0.1"


def test_component_masks_and_bbox():
    tiles = np.stack([blob(10, 14, 10, 14, margin=2), blob(50, 54, 60, 64, margin=2)])
    dct = {"o0.2": blob(20, 40, 20, 50), "t0.2": tiles, "bg.2": np.ones((N, N)), "sum": np.ones((N, N))}
    comps = component_masks(dct, 2)
    assert set(comps) == {"o0.2", "t0.2[0]", "t0.2[1]"}
    assert comps["t0.2[1]"][0] == "tile"
    assert bbox(comps["t0.2[1]"][1]) == (48, 55, 58, 65)


class TestCopyPlan:
    @staticmethod
    def region(i0, i1, j0, j1):
        r = np.zeros((64, 64), bool)
        r[i0:i1, j0:j1] = True
        return r

    def test_groups(self):
        from aim_resolve.reuse import overlap_groups

        regs = {"o0": self.region(0, 20, 0, 20), "t[0]": self.region(10, 14, 10, 14),
                "o1": self.region(18, 30, 18, 30), "t[1]": self.region(50, 54, 50, 54)}
        groups = sorted(sorted(g) for g in overlap_groups(regs))
        assert groups == [["o0", "o1", "t[0]"], ["t[1]"]]

    def test_copy_refit(self):
        from aim_resolve.reuse import copy_plan

        new = {"o0.3": self.region(0, 20, 0, 20), "t0.3[0]": self.region(10, 14, 10, 14),
               "t0.3[1]": self.region(50, 54, 50, 54), "t0.3[2]": self.region(30, 34, 50, 54)}
        old = {"o0.2": self.region(0, 20, 0, 20), "t0.2[0]": self.region(10, 14, 10, 14),
               "t0.2[1]": self.region(50, 54, 50, 54), "t0.2[2]": self.region(40, 44, 5, 9)}
        pairs = {"o0.3": "o0.2", "t0.3[0]": "t0.2[0]", "t0.3[1]": "t0.2[1]"}
        same = dict.fromkeys(pairs, True)
        plan = copy_plan(new, old, pairs, same)
        assert plan["o0.3"]["action"] == plan["t0.3[0]"]["action"] == "copy"  # nested, both kept
        assert plan["t0.3[1]"]["action"] == "copy"
        assert plan["t0.3[2]"]["action"] == "refit" and "new" in plan["t0.3[2]"]["reason"]
        # a new tile inside the object: the whole group is refitted
        new["t0.3[3]"] = self.region(5, 8, 5, 8)
        plan = copy_plan(new, old, pairs, same)
        assert plan["o0.3"]["action"] == "refit" and plan["t0.3[0]"]["action"] == "refit"
        # a dropped old component in the object's region, or a new grid: refit
        old["t0.2[3]"] = self.region(2, 4, 2, 4)
        del new["t0.3[3]"]
        assert copy_plan(new, old, pairs, same)["o0.3"]["action"] == "refit"
        del old["t0.2[3]"]
        assert copy_plan(new, old, pairs, same | {"o0.3": False})["o0.3"]["action"] == "refit"


def test_renamed_latents_and_tile_slices():
    from aim_resolve.reuse import renamed_latents, tile_slices

    tree = {"o1.2 i0 xi": np.ones((3, 3)), "o1.2 i0 zeromode": 0.5, "o10.2 i0 xi": np.zeros(2),
            "t0.2 i0 xi": np.arange(4.0)[:, None] * np.ones((4, 2)), "t0.2 i0 zeromode": np.arange(4.0)}
    ren = renamed_latents(tree, "o1.2", "o0.3")
    assert set(ren) == {"o0.3 i0 xi", "o0.3 i0 zeromode"}  # o10.2 is not o1.2
    new = {"t0.3 i0 xi": np.full((3, 2), -1.0), "t0.3 i0 zeromode": np.full(3, -1.0)}
    out = tile_slices(tree, "t0.2", "t0.3", new, {0: 3, 2: 1})
    assert list(out["t0.3 i0 zeromode"]) == [3.0, -1.0, 1.0]
    assert out["t0.3 i0 xi"][0, 0] == 3.0 and out["t0.3 i0 xi"][1, 0] == -1.0
