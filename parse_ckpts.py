import glob
import os
import sys

import numpy as np

pth = sys.argv[1]

ckpt_idxs = []
action_losses = []
avg_spls = []
for ckpt in sorted(glob.glob(os.path.join(pth, "*.pth"))):
    ckpt_name = os.path.basename(ckpt).split(".pth")[0]
    ckpt_split = ckpt_name.split("_")
    ckpt_idxs.append(ckpt_name)
    action_losses.append(ckpt_split[1])
    avg_spls.append(ckpt_split[2])

lowest_action_loss_idx = np.argmin(np.array(action_losses))
print("lowest_action_loss: ", action_losses[lowest_action_loss_idx])
print("ckpt: ", ckpt_idxs[lowest_action_loss_idx])
print("")
highest_spl_idx = np.argmax(np.array(avg_spls))
print("highest_spl: ", avg_spls[highest_spl_idx])
print("ckpt: ", ckpt_idxs[highest_spl_idx])
