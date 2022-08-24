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


experiments = []

for ts in np.logspace(2, 5, 20):
    experiments.append(
        {
            "X": "x.npy",
            "Y": "z.npy",
            "tau": 17,
            "training_start": 0,
            "training_length": 10000,
            "testing_start": 50000,
            "testing_length": 10000,
            "error_scale": 15,
            "d": 3,
            "mode": "voronoi",
            "kmeans": False,
            "epsilon": 3,
            "subset_size": int(2 ** (1.72 * np.log10(ts) + 1.08)),
            "interpolation": "none",
            "visualize_reconstruction": False,
            "save_bincounts": False,
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
