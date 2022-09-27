#!/usr/bin/env python3

import argparse
import glob
import gzip
import json
import multiprocessing
import os

import habitat
import numpy as np
import tqdm
from habitat.datasets import make_dataset
from habitat.utils.visualizations import maps

"""
python scripts/get_topdown_maps.py \
       /coc/testnvme/jtruong33/results/outdoor_nav_results/spot_depth_context_resnet18_map_prevact_sincos32_log_rot_100_0.5_robot_scale_0.1_sd_1_TEST/ddppo_pointnav_spot.yaml \
"""
parser = argparse.ArgumentParser()
parser.add_argument("config_yaml")
args = parser.parse_args()

CONFIG_YAML = args.config_yaml
# Pass absolute path to the scene glb


def get_topdown_maps():
    save_dir = "/coc/testnvme/jtruong33/google_nav/habitat-lab/topdown_maps"
    scenes = glob.glob(
        "/coc/testnvme/nyokoyama3/fair/spot_nav/habitat-lab/data/spot_goal_headings_hm3d/train/content/*.json.gz"
    )
    print("scenes: ", scenes)

    cfg = habitat.get_config(CONFIG_YAML)
    failed_scenes = []

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    for scene in scenes:
        with gzip.open(scene, "r") as f:
            data = json.loads(f.read(), encoding="utf-8")
            scene_id = data["episodes"][1]["scene_id"]
            scene_name = os.path.basename(scene_id).split(".")[0]
            if any(
                scene_name in s
                for s in glob.glob(
                    "/coc/testnvme/jtruong33/google_nav/habitat-lab/topdown_maps/*.npy"
                )
            ):
                continue
            z_heights = list(
                set(
                    [
                        data["episodes"][i]["start_position"][1]
                        for i in range(len(data["episodes"]))
                    ]
                )
            )
            tmp_heights = list(set(np.round(z_heights)))
            z_heights_n = [find_nearest(z_heights, t) for t in tmp_heights]
            if len(z_heights_n) > 0:
                print("LOADING SCENE: ", scene_id)
                print("Z HEIGHTS: ", z_heights)
                print("FILTERED Z HEIGHTS: ", z_heights_n)
                cfg.defrost()
                if ".basis" in scene_id:
                    scene = "".join(scene_id.split(".basis"))
                cfg.SIMULATOR.SCENE = os.path.join("data/scene_datasets", scene)
                cfg.freeze()
                sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

                stacked_map_res = [0.05, 0.1, 0.2, 0.5]
                map_resolution = 100
                try:
                    for z_height in z_heights_n:
                        for mpp in stacked_map_res:
                            print("get top down map: ", z_height, map_resolution, mpp)
                            _top_down_map = maps.get_topdown_map(
                                sim.pathfinder,
                                z_height,
                                map_resolution,
                                False,
                                mpp,
                            )
                            save_name = f"{scene_name}_{np.round(z_height):.1f}_{map_resolution}_{mpp}.npy"
                            print("np.save: ", save_name)
                            np.save(
                                os.path.join(save_dir, save_name),
                                _top_down_map,
                            )
                except:
                    print("FAILED SCENE: ", scene_id)
                    failed_scenes.append(scene_id)
                sim.close()
    print("failed scenes: ", failed_scenes)

    # cfg = habitat.get_config(CONFIG_YAML)
    # cfg.defrost()
    # cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    # cfg.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    # cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    # cfg.freeze()
    #
    # _dataset = make_dataset(id_dataset=cfg.DATASET.TYPE, config=cfg.DATASET)
    # iter_option_dict = {
    #     k.lower(): v for k, v in cfg.ENVIRONMENT.ITERATOR_OPTIONS.items()
    # }
    # _episode_iterator = _dataset.get_episode_iterator(**iter_option_dict)
    # curr_uniq_id = None
    # curr_scene = None
    # ctr = 0
    #
    # print("pre iteration")
    # for episode in _episode_iterator:
    #     scene_name = os.path.basename(episode.scene_id).split(".")[0]
    #
    #     print("for loop")
    #     print("ctr: ", ctr)
    #     z_height = episode.start_position[1]
    #     print("z height: ", z_height)
    #
    #     # episode_uniq_id = f"{episode.scene_id}_{z_height}"
    #     # print("EPISODE UNIQ ID: ", episode_uniq_id)
    #     done_scenes =
    #     print("episode scene id: ", episode.scene_id, curr_scene)
    #     if episode.scene_id != curr_scene:
    #         if any("xp4FyfQ6Wr5" in aa for aa in a):
    #             continue
    #         cfg.defrost()
    #         scene = episode.scene_id
    #         if ".basis" in episode.scene_id:
    #             scene = "".join(episode.scene_id.split(".basis"))
    #         cfg.SIMULATOR.SCENE = scene
    #         cfg.freeze()
    #         print("making sim")
    #         sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    #         print("finished making sim")
    #         curr_scene = episode.scene_id
    #         # curr_uniq_id = f"{episode.scene_id}_{z_height}"
    #     stacked_map_res = [0.05, 0.1, 0.2, 0.5]
    #     map_resolution = 100
    #     for mpp in stacked_map_res:
    #         print("get top down map")
    #         _top_down_map = maps.get_topdown_map(
    #             sim.pathfinder,
    #             z_height,
    #             map_resolution,
    #             False,
    #             mpp,
    #         )
    #         print("np.save")
    #         np.save(
    #             f"{scene_name}_{z_height:.3f}_{map_resolution}_{mpp}.npy",
    #             _top_down_map,
    #         )
    #     ctr += 1


if __name__ == "__main__":
    # print('\n'.join(SCENES))
    get_topdown_maps()
