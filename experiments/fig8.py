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


# default lorenz settings, 0.01 timestep

experiments = []

experiments.append(
    {
        "X": "lorenz_100_0.npy",
        "Y": "lorenz_100_2.npy",
        "tau": 17,
        "training_start": 0,
        "training_length": 850000,
        "testing_start": 950000,
        "testing_length": 1000,
        "error_scale": 30,
        "d": 3,
        "mode": "voronoi",
        "kmeans": False,
        #"epsilon": 3,
        #"subset_size": int(2 ** (1.72 * np.log10(ts) + 1.08)),
        "subset_size": 1000,
        "interpolation": "linear",
        "visualize_jun": True,
        "visualize_reconstruction": False,
        "save_bincounts": False,
        "visualize_error": False,
        "visualize_bad_idxs": False,
        "ylabel": "$Z(t)$",
        "ax0xlim": -25,
        "ax0ylim": 25,
        "ax1xlim": -25,
        "ax1ylim": 75
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
