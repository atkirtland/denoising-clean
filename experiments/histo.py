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

To get the histogram plots, use the relevant cell `visualizations.ipynb` after collecting data by running this experiment
for uniform, voronoi without kmeans, and voronoi with kmeans.
There should be 14165 (nonempty) cells for each.

"""


experiments = []


experiments.append(
    {
        "X": "lorenz_0.npy",
        "Y": "lorenz_2.npy",
        # 'Z': "lorenz_1.npy",
        "tau": 17,
        "training_start": 0,
        "training_length": 1000000,
        "testing_start": 100000,
        "testing_length": 10000,
        "error_scale": 10,
        "d": 3,
        "mode": "uniform",
        "kmeans": True,
        # "epsilon": 5.1,
        "boxes_per_dim": 100,
        "subset_size": 14165,
        "interpolation": "none",
        "visualize_reconstruction": True,
        "save_bincounts": True,
        'visualize_dots': False,
        'visualize_jun': False
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