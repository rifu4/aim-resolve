"""Tests for aim_resolve.agent_detect (no API calls, the client is mocked)."""

import base64
from types import SimpleNamespace

import numpy as np
import pytest

from aim_resolve.agent_detect import _Image, agent_detect, boxes_to_maps


def gaussian(shape, i, j, sigma, amp):
    ii, jj = np.indices(shape)
    return amp * np.exp(-((ii - i) ** 2 + (jj - j) ** 2) / (2 * sigma**2))


@pytest.fixture
def sky():
    """Faint background with a large blob, a small blob, a compact source and a point source."""
    shape = (128, 128)
    val = np.full(shape, 1e-3)
    val += gaussian(shape, 40, 40, 8, 10.0)
    val += gaussian(shape, 90, 100, 3, 5.0)
    val += gaussian(shape, 20, 100, 0.8, 2.0)
    val[100, 20] += 50.0
    return val


# ---------------------------------------------------------------------------
# boxes_to_maps
# ---------------------------------------------------------------------------


class TestBoxesToMaps:
    """Tests for the conversion of agent output into detection maps."""

    def test_shapes_and_sorting(self, sky):
        """Objects are returned as binary maps sorted by descending size."""
        objects = [
            dict(i0=80, i1=101, j0=90, j1=111),  # small blob first
            dict(i0=10, i1=71, j0=10, j1=71),  # large blob second
        ]
        ps_map, cl_map = boxes_to_maps(sky, objects)

        assert ps_map.shape == sky.shape
        assert cl_map.shape == (2,) + sky.shape
        assert set(np.unique(cl_map)) <= {0.0, 1.0}
        sizes = cl_map.sum(axis=(1, 2))
        assert sizes[0] > sizes[1]
        assert cl_map[0, 40, 40] == 1
        assert cl_map[1, 90, 100] == 1

    def test_masks_stay_inside_boxes(self, sky):
        """Mask pixels never leave the object box."""
        _, cl_map = boxes_to_maps(sky, [dict(i0=20, i1=60, j0=25, j1=55)])
        idx = np.argwhere(cl_map[0])
        assert idx[:, 0].min() >= 20 and idx[:, 0].max() < 60
        assert idx[:, 1].min() >= 25 and idx[:, 1].max() < 55

    def test_threshold_shrinks_mask(self, sky):
        """A higher threshold keeps fewer pixels of the same box."""
        box = [dict(i0=10, i1=71, j0=10, j1=71)]
        _, low = boxes_to_maps(sky, box, obj_threshold=0.0)
        _, high = boxes_to_maps(sky, box, obj_threshold=0.1)
        assert high.sum() < low.sum()
        assert high[0, 40, 40] == 1

    def test_overlapping_boxes_do_not_share_pixels(self, sky):
        """Pixels covered by two boxes belong to the first object only."""
        objects = [dict(i0=10, i1=71, j0=10, j1=71), dict(i0=30, i1=90, j0=30, j1=90)]
        _, cl_map = boxes_to_maps(sky, objects)
        assert cl_map.sum(axis=0).max() <= 1

    def test_single_pixel_source_is_point(self, sky):
        """A compact source with a one-pixel mask becomes a point source."""
        ps_map, cl_map = boxes_to_maps(sky, [], [dict(i=98, j=22)], snap_radius=3)
        assert ps_map[100, 20] == 1
        assert ps_map.sum() == 1
        assert cl_map.shape[0] == 0

    def test_resolved_compact_source_is_object(self, sky):
        """A compact source spanning several pixels becomes a small object."""
        ps_map, cl_map = boxes_to_maps(sky, [], [dict(i=20, j=100)])
        assert ps_map.sum() == 0
        assert cl_map.shape[0] == 1
        assert 1 < cl_map[0].sum() < 30
        assert cl_map[0, 20, 100] == 1

    def test_compact_source_inside_object_is_skipped(self, sky):
        """Compact sources on top of an extended object are not added twice."""
        ps_map, cl_map = boxes_to_maps(sky, [dict(i0=10, i1=71, j0=10, j1=71)], [dict(i=40, j=40)])
        assert ps_map.sum() == 0
        assert cl_map.shape[0] == 1

    def test_stray_patches_are_dropped_from_objects(self, sky):
        """Small patches in an object box are dropped, comparable ones (separated lobes) kept."""
        sky = sky.copy()
        sky[60:62, 15:17] += 1.0  # stray 4 px patch, far below 10% of the blob (~1200 px)
        sky += gaussian(sky.shape, 18, 60, 3.5, 5.0)  # second lobe, separate, ~180 px
        box = [dict(i0=10, i1=71, j0=10, j1=71)]
        _, cl_map = boxes_to_maps(sky, box, obj_threshold=0.05)
        assert cl_map[0, 40, 40] == 1 and cl_map[0, 18, 60] == 1
        assert cl_map[0, 60, 15] == 0
        _, cl_map = boxes_to_maps(sky, box, obj_threshold=0.05, obj_min_frac=0.0)
        assert cl_map[0, 60, 15] == 1

    def test_compact_source_in_box_but_separate_is_kept(self, sky):
        """A compact source not connected to the object in its box becomes its own source."""
        sky = sky + gaussian(sky.shape, 15, 65, 0.8, 2.0)
        box = [dict(i0=10, i1=71, j0=10, j1=71)]

        # without the compact source, the small separate patch is not part of the object either
        # (below obj_min_frac of the main body); with obj_min_frac=0 the object claims it
        _, cl_map = boxes_to_maps(sky, box)
        assert cl_map.shape[0] == 1 and cl_map[0, 15, 65] == 0
        _, cl_map = boxes_to_maps(sky, box, obj_min_frac=0.0)
        assert cl_map[0, 15, 65] == 1

        _, cl_map = boxes_to_maps(sky, box, [dict(i=15, j=65)])
        assert cl_map.shape[0] == 2
        assert cl_map[0, 15, 65] == 0 and cl_map[0, 40, 40] == 1
        assert cl_map[1, 15, 65] == 1

    def test_bright_compact_source_does_not_replace_object_body(self, sky):
        """A compact source brighter than the object in its box does not take the object's place."""
        sky = sky + gaussian(sky.shape, 15, 65, 0.8, 50.0)
        box = [dict(i0=10, i1=71, j0=10, j1=71)]
        # compact candidates on the object body itself (e.g. its core) do not release the body
        compact = [dict(i=15, j=65), dict(i=40, j=40)]
        _, cl_map = boxes_to_maps(sky, box, compact)
        body = next(m for m in cl_map if m[40, 40] == 1)
        assert body.sum() > 100 and body[15, 65] == 0
        assert any(m[15, 65] == 1 for m in cl_map)

    def test_group_becomes_one_object(self, sky):
        """Two lobes submitted as a group form a single object with both lobes."""
        sky = sky + gaussian(sky.shape, 110, 60, 0.8, 2.0) + gaussian(sky.shape, 110, 80, 0.8, 2.0)
        lobes = [dict(i=110, j=60), dict(i=110, j=80)]

        ps_map, cl_map = boxes_to_maps(sky, [], [], [lobes])
        assert ps_map.sum() == 0
        assert cl_map.shape[0] == 1
        assert cl_map[0, 110, 60] == 1 and cl_map[0, 110, 80] == 1
        # without a bridge, the faint gap between the lobes is not part of the object
        assert cl_map[0, 110, 70] == 0

        # as single compact sources, the lobes are two objects
        _, cl_map = boxes_to_maps(sky, [], lobes)
        assert cl_map.shape[0] == 2

    def test_group_includes_bridge(self, sky):
        """A bright bridge connecting the lobes is part of the group object."""
        sky = sky + gaussian(sky.shape, 110, 60, 0.8, 2.0) + gaussian(sky.shape, 110, 80, 0.8, 2.0)
        sky[110, 61:80] += 0.5
        _, cl_map = boxes_to_maps(sky, [], [], [[dict(i=110, j=60), dict(i=110, j=80)]])
        assert cl_map.shape[0] == 1
        assert cl_map[0, 110, 60:81].all()

    def test_group_member_in_object_box_is_not_claimed(self, sky):
        """An extended object does not swallow a separate lobe of a group."""
        sky = sky + gaussian(sky.shape, 15, 65, 0.8, 2.0) + gaussian(sky.shape, 5, 90, 0.8, 2.0)
        box = [dict(i0=10, i1=71, j0=10, j1=71)]
        _, cl_map = boxes_to_maps(sky, box, [], [[dict(i=15, j=65), dict(i=5, j=90)]])
        assert cl_map.shape[0] == 2
        grp = next(m for m in cl_map if m[5, 90] == 1)
        assert grp[15, 65] == 1 and grp[40, 40] == 0

    def test_empty_box_falls_back_to_full_box(self, sky):
        """A box without emission above the background is used as a whole."""
        _, cl_map = boxes_to_maps(sky, [dict(i0=0, i1=6, j0=120, j1=128)])
        assert cl_map.shape[0] == 1
        assert cl_map[0].sum() == 6 * 8

    def test_no_detections(self, sky):
        """Empty input gives an empty point map and zero object maps."""
        ps_map, cl_map = boxes_to_maps(sky, [], [])
        assert ps_map.sum() == 0
        assert cl_map.shape == (0,) + sky.shape


# ---------------------------------------------------------------------------
# _Image tools
# ---------------------------------------------------------------------------


class TestImageTools:
    """Tests for the helpers behind the agent tools."""

    def test_render_returns_png(self, sky):
        img = _Image(sky, view_px=200)
        img.find_candidates()
        data = img.render(0, 64, 0, 64, "log")
        assert base64.standard_b64decode(data)[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_sheet(self, sky):
        """The contact sheet is a PNG; unknown ids are rejected."""
        img = _Image(sky)
        cands = img.find_candidates()
        data = img.render_sheet([c["id"] for c in cands] + [999])
        assert base64.standard_b64decode(data)[:8] == b"\x89PNG\r\n\x1a\n"
        with pytest.raises(ValueError, match="no known candidate ids"):
            img.render_sheet([999])

    def test_find_peaks(self, sky):
        peaks = _Image(sky).find_peaks(0, 128, 0, 128, min_rel_flux=0.02)
        found = {(p["i"], p["j"]) for p in peaks}
        assert {(100, 20), (40, 40), (90, 100), (20, 100)} <= found
        assert peaks[0]["i"] == 100 and peaks[0]["j"] == 20

    def test_candidates_are_compact(self, sky):
        """Compact sources are candidates, the centre of the large blob is not."""
        cands = _Image(sky).find_candidates(min_contrast=20)
        found = {(c["i"], c["j"]) for c in cands}
        assert {(100, 20), (20, 100)} <= found
        assert (40, 40) not in found
        assert [c["id"] for c in cands] == list(range(len(cands)))

    def test_find_pairs(self, sky):
        """Close candidates of similar flux are listed as pairs."""
        sky = sky + gaussian(sky.shape, 110, 60, 0.8, 2.0) + gaussian(sky.shape, 110, 80, 0.8, 2.0)
        img = _Image(sky)
        cands = img.find_candidates(min_contrast=20)
        ids = {(c["i"], c["j"]): c["id"] for c in cands}
        pairs = [tuple(sorted(p["ids"])) for p in img.find_pairs(max_dist=25)]
        assert tuple(sorted((ids[(110, 60)], ids[(110, 80)]))) in pairs
        # the bright point source is far from and much brighter than the lobes
        assert not any(ids[(100, 20)] in p for p in pairs)

    def test_pairs_flag_likely_lobes(self, sky):
        """Elongated, connected lobes pointing at each other are flagged, round separate ones not."""
        ii, jj = np.indices(sky.shape)

        def lobe(i, j, amp):
            return amp * np.exp(-((ii - i) ** 2 / (2 * 0.7**2) + (jj - j) ** 2 / (2 * 2.0**2)))

        # elongated along j, on a common line, with a faint bridge
        lobes = sky + lobe(110, 55, 2.0) + lobe(110, 75, 2.0)
        lobes[110, 58:73] += 0.05
        img = _Image(lobes)
        cands = img.find_candidates(min_contrast=20)
        ids = {(c["i"], c["j"]): c["id"] for c in cands}
        pair = next(p for p in img.find_pairs(max_dist=25) if set(p["ids"]) == {ids[(110, 55)], ids[(110, 75)]})
        assert pair["bridge"] > 0 and min(pair["align"]) > 0.9
        assert pair["likely"]

        # round, separate blobs
        blobs = sky + gaussian(sky.shape, 110, 55, 0.8, 2.0) + gaussian(sky.shape, 110, 75, 0.8, 2.0)
        img = _Image(blobs)
        cands = img.find_candidates(min_contrast=20)
        ids = {(c["i"], c["j"]): c["id"] for c in cands}
        pair = next(p for p in img.find_pairs(max_dist=25) if set(p["ids"]) == {ids[(110, 55)], ids[(110, 75)]})
        assert pair["bridge"] <= 0
        assert not pair["likely"]

    def test_close_candidates_are_merged(self, sky):
        """A chain of close peaks becomes one candidate at the brightest peak."""
        sky = sky.copy()
        sky[60, 110] += 3.0
        sky[64, 110] += 1.0
        sky[68, 110] += 0.8
        cands = _Image(sky).find_candidates(min_contrast=20, merge_dist=5)
        merged = next(c for c in cands if (c["i"], c["j"]) == (60, 110))
        assert {(m["i"], m["j"]) for m in merged["members"]} == {(60, 110), (64, 110), (68, 110)}
        assert not any((c["i"], c["j"]) in {(64, 110), (68, 110)} for c in cands)
        assert [c["id"] for c in cands] == list(range(len(cands)))

    def test_clip_box(self, sky):
        assert _Image(sky).clip_box(-10, 500, 60, 60) == (0, 128, 59, 61)


# ---------------------------------------------------------------------------
# agent_detect with a mocked client
# ---------------------------------------------------------------------------


def tool_use(name, args, id):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id)


def response(*blocks, stop_reason="tool_use"):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=None,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=10),
    )


class FakeClient:
    """Returns scripted responses and records the requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        # snapshot the message list, the loop keeps appending to it
        self.requests.append(kwargs | dict(messages=list(kwargs["messages"])))
        return self.responses.pop(0)


def empty_submission():
    return dict(objects=[], groups=[], compact_ids=[], extra_compact=[])


class TestAgentDetect:
    """Tests for the agent loop."""

    def test_scripted_run(self, sky):
        """The loop runs tools, stops at the submission and builds the maps."""
        cands = _Image(sky).find_candidates()
        point_id = next(c["id"] for c in cands if (c["i"], c["j"]) == (100, 20))
        submission = dict(
            objects=[
                dict(name="big", i0=10, i1=71, j0=10, j1=71, notes=""),
                dict(name="small", i0=80, i1=101, j0=90, j1=111, notes=""),
            ],
            groups=[],
            compact_ids=[point_id],
            extra_compact=[dict(i=20, j=100)],
        )
        client = FakeClient(
            [
                response(
                    SimpleNamespace(type="text", text="Looking closer."),
                    tool_use("view_region", dict(i0=0, i1=64, j0=0, j1=64, scale="log"), "t1"),
                    tool_use("find_peaks", dict(i0=0, i1=128, j0=0, j1=128, min_rel_flux=0.01), "t2"),
                ),
                response(tool_use("submit_detection", submission, "t3")),
            ]
        )
        rec = SimpleNamespace(val=sky)
        ps_map, cl_map, log = agent_detect(rec, client=client, view_px=200, print_det=False)

        assert ps_map[100, 20] == 1 and ps_map.sum() == 1
        assert cl_map.shape == (3,) + sky.shape
        assert log["submission"] == submission
        assert log["usage"]["input_tokens"] == 200

        # the candidates are listed in the first message
        assert "Compact candidates" in client.requests[0]["messages"][0]["content"][1]["text"]

        # both tool results are returned in one user message, the image as image block
        results = client.requests[1]["messages"][-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
        assert results[0]["content"][0]["type"] == "image"

    def test_nudge_without_tool_call(self, sky):
        """A turn without tool calls is answered with a reminder to submit."""
        client = FakeClient(
            [
                response(SimpleNamespace(type="text", text="Done."), stop_reason="end_turn"),
                response(tool_use("submit_detection", empty_submission(), "t1")),
            ]
        )
        _, cl_map, _ = agent_detect(SimpleNamespace(val=sky), client=client, print_det=False)
        assert cl_map.shape[0] == 0
        assert "submit_detection" in client.requests[1]["messages"][-1]["content"]

    def test_no_submission_raises(self, sky):
        """Running out of turns without a submission is an error."""
        client = FakeClient(
            [response(tool_use("region_stats", dict(i0=0, i1=10, j0=0, j1=10), f"t{k}")) for k in range(3)]
        )
        with pytest.raises(RuntimeError, match="no detection submitted"):
            agent_detect(SimpleNamespace(val=sky), client=client, max_turns=3, print_det=False)

    def test_refusal_raises(self, sky):
        client = FakeClient([response(stop_reason="refusal")])
        with pytest.raises(RuntimeError, match="refused"):
            agent_detect(SimpleNamespace(val=sky), client=client, print_det=False)


# ---------------------------------------------------------------------------
# manual mode
# ---------------------------------------------------------------------------


class TestManualDetect:
    """Tests for the manual mode (task export and hand-written submission)."""

    def test_exports_task_without_submission(self, sky, tmp_path):
        """Without a submission the task is exported and None is returned."""
        from aim_resolve.agent_detect import manual_detect

        task_dir, sub_file = tmp_path / "task", tmp_path / "sub.json"
        res = manual_detect(SimpleNamespace(val=sky), str(task_dir), str(sub_file), rec_path="rec.pkl")
        assert res is None
        for name in ["task.md", "overview.png", "candidates.json", "submission_template.json"]:
            assert (task_dir / name).exists()
        task = (task_dir / "task.md").read_text()
        assert "Compact candidates" in task and str(sub_file) in task
        assert "python -m aim_resolve.agent_detect view rec.pkl" in task

    def test_builds_maps_from_submission(self, sky, tmp_path):
        """A submission gives the same maps as in API mode."""
        import json

        from aim_resolve.agent_detect import manual_detect

        cands = _Image(sky).find_candidates()
        point_id = next(c["id"] for c in cands if (c["i"], c["j"]) == (100, 20))
        submission = dict(
            objects=[dict(name="big", i0=10, i1=71, j0=10, j1=71, notes="")],
            groups=[], compact_ids=[point_id], extra_compact=[dict(i=20, j=100)],
        )
        sub_file = tmp_path / "sub.json"
        sub_file.write_text(json.dumps(submission))
        ps_map, cl_map, log = manual_detect(SimpleNamespace(val=sky), str(tmp_path / "task"), str(sub_file), print_det=False)
        exp_ps, exp_cl = boxes_to_maps(sky, submission["objects"], [dict(i=100, j=20), dict(i=20, j=100)])
        assert np.array_equal(ps_map, exp_ps) and np.array_equal(cl_map, exp_cl)
        assert log["model"] == "manual" and log["submission"] == submission

    def test_invalid_submission_raises(self, sky):
        """Unknown ids and missing fields are reported, not silently dropped."""
        from aim_resolve.agent_detect import check_submission

        cands = _Image(sky).find_candidates()
        with pytest.raises(ValueError, match="unknown candidate ids"):
            check_submission(dict(objects=[], groups=[], compact_ids=[999], extra_compact=[]), cands)
        with pytest.raises(ValueError, match="'groups' must be a list"):
            check_submission(dict(objects=[], compact_ids=[], extra_compact=[]), cands)
        with pytest.raises(ValueError, match="at least two"):
            check_submission(dict(objects=[], groups=[dict(ids=[0])], compact_ids=[], extra_compact=[]), cands)
