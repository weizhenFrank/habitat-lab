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
    cfg = habitat.get_config(CONFIG_YAML)
    cfg.defrost()
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 1
    cfg.freeze()

    _dataset = make_dataset(id_dataset=cfg.DATASET.TYPE, config=cfg.DATASET)
    iter_option_dict = {
        k.lower(): v for k, v in cfg.ENVIRONMENT.ITERATOR_OPTIONS.items()
    }
    _episode_iterator = _dataset.get_episode_iterator(**iter_option_dict)
    curr_uniq_id = None
    curr_scene = None
    ctr = 0

    save_dir = "/coc/testnvme/jtruong33/google_nav/habitat-lab/topdown_maps/"
    done_scenes = glob.glob(save_dir + "*.npy")

    print("pre iteration")
    failed_scenes = []
    for episode in _episode_iterator:
        scene_name = os.path.basename(episode.scene_id).split(".")[0]
        print("ctr: ", ctr, scene_name)
        if any(scene_name in s for s in done_scenes):
            continue
        z_height = episode.start_position[1]
        print("episode scene id: ", episode.scene_id, z_height, curr_scene)
        if episode.scene_id != curr_scene:
            if curr_scene is not None:
                sim.close()
            cfg.defrost()
            scene = episode.scene_id
            if ".basis" in episode.scene_id:
                scene = "".join(episode.scene_id.split(".basis"))
            cfg.SIMULATOR.SCENE = scene
            cfg.freeze()
            sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
            curr_scene = episode.scene_id
            # curr_uniq_id = f"{episode.scene_id}_{z_height}"
        try:
            stacked_map_res = [0.05, 0.1, 0.2, 0.5]
            map_resolution = 100
            for mpp in stacked_map_res:
                _top_down_map = maps.get_topdown_map(
                    sim.pathfinder,
                    z_height,
                    map_resolution,
                    False,
                    mpp,
                )
                np.save(
                    f"/coc/testnvme/jtruong33/google_nav/habitat-lab/topdown_maps/{scene_name}_{z_height:.3f}_{map_resolution}_{mpp}.npy",
                    _top_down_map,
                )
        except:
            failed_scenes.append(scene_name)
        ctr += 1
    print("failed scenes: ", failed_scenes)


if __name__ == "__main__":
    # print('\n'.join(SCENES))
    get_topdown_maps()
