#!/usr/bin/env python3

import argparse
import glob
import gzip
import multiprocessing
from os import path as osp

import tqdm

import habitat
from habitat.datasets.pointnav.pointnav_generator import (
    generate_pointnav_episode,
)

parser = argparse.ArgumentParser()
parser.add_argument("config_yaml")
parser.add_argument("out_dir")
parser.add_argument("num_episodes_per_scene", type=int)
# glb_dir on FAIR should be:
# - /datasets01/gibson/011719/491_scenes/ for Gibson 4+
# - /datasets01/hm3d/090121/ for HM3D
parser.add_argument("glb_dir")
parser.add_argument("-v", "--val", action="store_true")
# Next arg decides whether to use Gibson 4+ or HM3D
parser.add_argument("-m", "--matterport", action="store_true")
args = parser.parse_args()

# Task config yaml file. Needed for agent radius and height.
# MAKE SURE THAT RGB_SENSOR IS NOT BEING USED, ONLY DEPTH (consumes GPU)
CONFIG_YAML = args.config_yaml

# Folder where json.gz files will be saved:
TRAIN_EP_DIR = args.out_dir

NUM_EPISODES_PER_SCENE = args.num_episodes_per_scene

if args.matterport:
    split = "val" if args.val else "train"
    split_dir = osp.join(args.glb_dir, split)
    SCENES = glob.glob(osp.join(split_dir, "*/*.basis.glb"))
else:
    if args.val:
        # scenes = [
        #     "Cantwell",
        #     "Denmark",
        #     "Eastville",
        #     "Edgemere",
        #     "Elmira",
        #     "Eudora",
        #     "Greigsville",
        #     "Mosquito",
        #     "Pablo",
        #     "Ribera",
        #     "Sands",
        #     "Scioto",
        #     "Sisters",
        #     "Swormville",
        # ]
        scenes = ["ferst"]
    else:
        scenes = [
            "Adrian",
            "Albertville",
            "Anaheim",
            "Andover",
            "Angiola",
            "Annawan",
            "Applewold",
            "Arkansaw",
            "Avonia",
            "Azusa",
            "Ballou",
            "Beach",
            "Bolton",
            "Bowlus",
            "Brevort",
            "Capistrano",
            "Colebrook",
            "Convoy",
            "Cooperstown",
            "Crandon",
            "Delton",
            "Dryville",
            "Dunmor",
            "Eagerville",
            "Goffs",
            "Hainesburg",
            "Hambleton",
            "Haxtun",
            "Hillsdale",
            "Hometown",
            "Hominy",
            "Kerrtown",
            "Maryhill",
            "Mesic",
            "Micanopy",
            "Mifflintown",
            "Mobridge",
            "Monson",
            "Mosinee",
            "Nemacolin",
            "Nicut",
            "Nimmons",
            "Nuevo",
            "Oyens",
            "Parole",
            "Pettigrew",
            "Placida",
            "Pleasant",
            "Quantico",
            "Rancocas",
            "Reyno",
            "Roane",
            "Roeville",
            "Rosser",
            "Roxboro",
            "Sanctuary",
            "Sasakwa",
            "Sawpit",
            "Seward",
            "Shelbiana",
            "Silas",
            "Sodaville",
            "Soldier",
            "Spencerville",
            "Spotswood",
            "Springhill",
            "Stanleyville",
            "Stilwell",
            "Stokes",
            "Sumas",
            "Superior",
            "Woonsocket",
        ]
    SCENES = [osp.join(args.glb_dir, s + ".glb") for s in scenes]


# Pass absolute path to the scene glb
def _generate_fn(scene):
    scene_key = osp.basename(scene)
    out_file = osp.join(TRAIN_EP_DIR, scene_key.split(".")[0] + ".json.gz")

    # Skip if it already exists
    if osp.isfile(out_file):
        return

    cfg = habitat.get_config(CONFIG_YAML)
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim,
            NUM_EPISODES_PER_SCENE,
            is_gen_shortest_path=False,
            geodesic_to_euclid_min_ratio=1.05,
        )
    )
    for ep in dset.episodes:
        if args.matterport:
            ep.scene_id = scene.replace(args.glb_dir, "")
            if not ep.scene_id.startswith("/"):
                ep.scene_id = "/" + ep.scene_id
            ep.scene_id = "hm3d" + ep.scene_id
        else:
            # ep.scene_id = osp.join("gibson", scene_key)
            ep.scene_id = osp.join("ferst", scene_key)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())


def generate_gibson_large_dataset():
    scenes = SCENES

    with multiprocessing.Pool(27) as pool, tqdm.tqdm(
        total=len(scenes)
    ) as pbar:
        for _ in pool.imap_unordered(_generate_fn, scenes):
            pbar.update()


if __name__ == "__main__":
    print(len(SCENES), "scenes found")
    generate_gibson_large_dataset()
