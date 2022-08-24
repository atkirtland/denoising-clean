"""

RIPS AFRL 2020
Version August 19, 2020

Here is a sample experiment file.

"""

import sys

sys.path.append(".")

from lib import write_header, save, demo

import numpy as np

# parallel
from multiprocessing import Pool, cpu_count


"""

Run a list of experiments using generated data

"""


# timestep 0.001
# a = 40.0
# b = 3.0
# c = 28.0
# state0 = [-0.1, 0.5, -0.6]

experiments = []

#for ts in np.logspace(2, 5, 20):
experiments.append(
    {
        "X": "chen_0.npy",
        "Y": "chen_2.npy",
        "tau": 10,
        "training_start": 0,
        "training_length": 8500000,
        "testing_start": 9500000,
        "testing_length": 5000,
        "error_scale": 10,
        "d": 3,
        "mode": "voronoi",
        "kmeans": False,
        #"epsilon": 3,
        #"subset_size": int(2 ** (1.72 * np.log10(ts) + 1.08)),
        "subset_size": 20000,
        "interpolation": "linear",
        "visualize_jun": True,
        "visualize_reconstruction": False,
        "save_bincounts": False,
        "visualize_error": False,
        "visualize_bad_idxs": False,
        "ylabel": "$Z(t)$",
        "ax0xlim": -20,
        "ax0ylim": 25,
        "ax1xlim": 0,
        "ax1ylim": 35,
        # "ax1xlim": -15,
        # "ax1ylim": 20,
        "timestep": 0.001,
    }
)

write_header()

# Use the commented out code below to run multiple experiments in parallel.
# This can speed up the code in many cases.
# with Pool(cpu_count()) as p:
# for experiment in experiments:
#     result = demo(experiment)

for experiment in experiments:
    result = demo(experiment)
    save(experiment, result)
    experiment["mode"] = "voronoi"
    experiment["subset_size"] = result["nonempty"]
