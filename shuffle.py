import numpy as np
import os

path = 'train/MOSI/'
with open(os.path.join(path,"target_filtered.txt"), 'r') as f:
    lines = f.read().splitlines()
randomize = np.arange(len(lines))
np.random.shuffle(randomize)
np.save(os.path.join(path,"shuffle.npy"), randomize)