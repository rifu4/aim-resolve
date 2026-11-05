"""Script for detecting point sources and extended objects in reconstructed images."""

import json
import os
import sys

import numpy as np

from aim_resolve import (
    ImageData,
    agent_detect,
    dbscan_clustering,
    dynamic_limits,
    manual_detect,
    model_predict,
    objects2points,
    plot_panels,
    plot_style,
    split_style,
    yaml_load,
)

# pixel sizes of the agent settings (with the defaults of agent_detect) and the size of the rendered views
AGENT_PIXEL_KEYS = {"merge_dist": 5.0, "pair_dist": 40.0, "compact_box": 8, "snap_radius": 3}
MAX_VIEW_PX = 1536


def scale_agent_settings(agent_dct, zoom):
    """Agent settings with the pixel sizes scaled by `zoom` (the zoom factor of the reconstruction).

    The rendered views grow with the zoom as well, up to MAX_VIEW_PX (about the largest image size
    the vision input uses), so that the overview still shows compact sources.
    """
    agent_dct = dict(agent_dct)
    if zoom <= 1:
        return agent_dct
    for key, default in AGENT_PIXEL_KEYS.items():
        val = agent_dct.get(key, default)
        agent_dct[key] = type(default)(round(val * zoom)) if isinstance(default, int) else float(val) * zoom
    agent_dct["view_px"] = min(MAX_VIEW_PX, int(agent_dct.get("view_px", 768)) * zoom)
    print(f"agent settings scaled by the zoom {zoom}:", {k: agent_dct[k] for k in (*AGENT_PIXEL_KEYS, "view_px")})
    return agent_dct


def main():
    """Run source detection on a reconstructed image.

    Uses either a U-Net model with DBSCAN clustering or a Claude agent,
    selected by ``base_detect.method``: ``unet``, ``agent`` (Claude API) or
    ``manual`` (agent task answered by hand, see ``aim_resolve.agent_detect.manual_detect``).
    """
    _, files = sys.argv[0], sys.argv[1:]
    opt_pkl, base_yml, it = files

    # load model and base yaml-files and extract output directory
    base_dct = yaml_load(base_yml)
    plt_dct = base_dct["base_plot"]
    odir = base_dct["base_opt"]["odir"]
    method = base_dct.get("base_detect", {}).get("method", "unet")

    # load the reconstructed image and grid
    rec = ImageData.load(opt_pkl, dtype="float32")

    # pixel sizes in the settings refer to the base grid: scale them by the zoom factor of the reconstruction
    # (a 3x zoomed reconstruction has 3x more pixels per source), so one config works at any resolution
    zoom = int(np.max(np.atleast_1d(rec.grid.factor)))
    agent_dct = scale_agent_settings(base_dct.get("base_agent", {}), zoom)

    # task, submission and log of the agent detection go to {odir}/agent
    adir = f"{odir}/agent"
    if method in ("agent", "manual"):
        os.makedirs(adir, exist_ok=True)

    if method == "agent":
        # detect point sources and objects with the Claude agent and save its transcript
        ps_map, cl_map, log = agent_detect(rec, **agent_dct)
        with open(f"{adir}/{it}_agent.json", "w") as f:
            json.dump(log, f, indent=1)

    elif method == "manual":
        # agent detection with a hand-written submission: export the task and stop if
        # the submission is missing, otherwise build the maps from it
        task_dir = f"{adir}/{it}_task"
        sub_file = f"{adir}/{it}_submission.json"
        res = manual_detect(
            rec, task_dir, sub_file, rec_path=os.path.abspath(opt_pkl), **agent_dct
        )
        if res is None:
            print(f"detection task written to {task_dir}/task.md")
            print(f"write the submission to {sub_file} and run the detect step again")
            sys.exit(1)
        ps_map, cl_map, log = res
        with open(f"{adir}/{it}_agent.json", "w") as f:
            json.dump(log, f, indent=1)

    elif method == "unet":
        # detect point sources and objects in the reconstructed image using the U-Net model
        seg_dct = base_dct["base_seg"]
        ps_map, oj_map = model_predict(rec, **seg_dct, print_ps=False)

        # load the clustering settings and cluster the detected objects
        cl_dct = base_dct["base_clu"]
        cl_alg = cl_dct.pop("alg")
        cl_map, noise_map = dbscan_clustering(oj_map, print_cl=True, **cl_dct)

        # convert one-pixel objects to point sources
        ps_map = objects2points(ps_map, noise_map, print_ps=True, **cl_dct)

    else:
        raise ValueError(f"unknown detection method '{method}', use 'unet', 'agent' or 'manual'")

    # drop detections at the border of the field: emission just outside the field wraps around (periodic
    # convolution) and shows up on opposite edges, and sources cut by the border cannot be modelled
    edge_dist = int(base_dct.get("base_detect", {}).get("edge_dist", 0) or 0) * zoom
    if edge_dist > 0:
        inner = np.zeros(ps_map.shape[-2:], dtype=bool)
        inner[edge_dist:-edge_dist, edge_dist:-edge_dist] = True
        keep = [i for i, m in enumerate(cl_map) if np.all(inner[m > 0])]
        for i in sorted(set(range(len(cl_map))) - set(keep)):
            pix = np.argwhere(cl_map[i] > 0)
            print(f"object {i} dropped: within {edge_dist} px of the field edge (centre {pix.mean(axis=0).round().astype(int)})")
        cl_map = np.asarray(cl_map)[keep] if keep else np.zeros((0,) + ps_map.shape[-2:])
        n_ps = int((ps_map > 0).sum())
        ps_map = np.where(inner, ps_map, 0)
        if int((ps_map > 0).sum()) < n_ps:
            print(f"{n_ps - int((ps_map > 0).sum())} point source(s) dropped: within {edge_dist} px of the field edge")

    # plot the reconstruction with a contour around each detected object and circles around the point sources
    theme, drange, pnl_dct = split_style(base_dct.get("base_style"))
    rec_val = np.squeeze(np.asarray(rec.val, dtype="float64"))
    vmin, vmax = dynamic_limits(rec_val, drange, plt_dct.get("vmin"), plt_dct.get("vmax"))
    radius = max(4.0, 0.006 * max(rec_val.shape))
    overlay = dict(
        contours=list(cl_map),
        contour_numbers=True,
        circles=[(x, y, radius) for x, y in np.argwhere(ps_map > 0)],
    )
    with plot_style(**theme):
        plot_panels(
            [
                dict(
                    arrays=[rec_val],
                    labels=[f"{int((ps_map > 0).sum())} points, {len(cl_map)} objects"],
                    overlays=[overlay],
                    cmap=plt_dct.get("cmap", "inferno"),
                    vmin=vmin,
                    vmax=vmax,
                )
            ],
            name=f"{it}_det.png",
            odir=f"{odir}/plots",
            step=f"detect · it {it}",
            **pnl_dct,
        )

    # save the detected point sources and clustered objects
    np.savez(f"{odir}/files/{it}_det", ps_map=ps_map, cl_map=cl_map)


if __name__ == "__main__":
    main()
