# from skimage.draw import disk
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

print("loading maps")
context_maps = np.load(
    "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/eval_context_maps_1157_multi_wpts.npy"
)
# print("loading waypoints")
#
context_waypoint_maps = np.load(
    "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/eval_context_waypoint_maps_1157_multi_wpts.npy"
)
context_waypoints = np.load(
    "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/eval_context_waypoint_maps_1157_multi_wpts.npy"
)
# print("loading goals")
#
# context_goals = np.load(
#     "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/eval_context_goals_10k_rot.npy"
# )
#
# IMG_DIR = (
#     "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/sl_planning_imgs/train"
# )
# EVAL_IMG_DIR = (
#     "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/sl_planning_imgs/eval"
# )
# G_IMG_DIR = (
#     "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/sl_planning_imgs/1157"
# )
TMP_IMG_DIR = (
    "/coc/testnvme/jtruong33/google_nav/habitat-lab/sl/sl_planning_imgs/tmp"
)
i = 0
# print("writing imgs")
#
# for m in context_maps:
#     cv2.imwrite(os.path.join(TMP_IMG_DIR, f"cat_{i}.png"), m[:, :, 0] * 255)
#     i += 1
# print(context_maps.shape)
for m, w in zip(context_maps, context_waypoint_maps):
    overlay = m[:, :, 0].copy()
    overlay[m[:, :, 1] == 1] = 0.3
    print("overlay: ", overlay.shape)
    print("w: ", w.shape)
    o = np.ones((256, 1)) * 255
    cat_img = np.concatenate(
        [
            m[:, :, 0] * 255.0,
            o,
            w * 255,
            o,
            overlay * 255.0,
        ],
        axis=1,
    )
    cv2.imwrite(os.path.join(TMP_IMG_DIR, f"cat_{i}.png"), cat_img)
    if i == 100:
        break
    i += 1

# for m, w, g in zip(context_maps, context_waypoint_maps, context_goals):
#
#     # overlay = m.copy()
#     # overlay[g == 1.0] = 0.3
#     overlay = m[:, :, 0].copy()
#     overlay[m[:, :, 1] == 1] = 0.3
#     overlay[w == 1.0] = 0.7
#     o = np.ones((256, 1)) * 255
#     cat_img = np.concatenate(
#         [m[:, :, 0] * 255.0, o, g * 255.0, o, w * 255, o, overlay * 255.0],
#         axis=1,
#     )
#     cv2.imwrite(os.path.join(TMP_IMG_DIR, f"cat_{i}.png"), cat_img)
#     if i == 100:
#         break
#     i += 1
