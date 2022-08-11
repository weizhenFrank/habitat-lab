import glob
import os
import sys
import shutil

import numpy as np

pth = sys.argv[1]

for ckpt in sorted(glob.glob(os.path.join(pth, "*.ckpt"))):
    new_name = ckpt.split(".ckpt")[0] + ".pth"
    shutil.move(ckpt, new_name)