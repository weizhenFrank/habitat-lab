import json
import gzip
import glob
import tqdm
import numpy as np

import habitat_sim

num_inf = 0.0
total_eps = 0.0


for dset_file in tqdm.tqdm(
    glob.glob("data/datasets/pointnav/gibson/v1/train/**/*.json.gz", recursive=True)
):
    with gzip.open(dset_file, "rt") as f:
        dset = json.load(f)

    episodes = dset["episodes"]

    pf = None
    scene = None
    filt_eps = []
    for ep in tqdm.tqdm(episodes, leave=False):
        if pf is None or scene != ep["scene_id"]:
            pf = habitat_sim.PathFinder()
            pf.load_nav_mesh(ep["scene_id"].replace("glb", "navmesh"))
            assert pf.is_loaded

            scene = ep["scene_id"]

        path = habitat_sim.ShortestPath()
        path.requested_start = ep["start_position"]
        path.requested_end = ep["goals"][0]["position"]

        pf.find_path(path)

        total_eps += 1.0

        if not np.isfinite(path.geodesic_distance):
            num_inf += 1.0
        else:
            filt_eps.append(ep)

    dset["episodes"] = filt_eps
    with gzip.open(dset_file, "wt") as f:
        json.dump(dset, f)


print(num_inf, total_eps)
