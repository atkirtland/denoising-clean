"""

RIPS AFRL 2020
Version August 19, 2020

This experiment file can be used to generate results that can make the subset optimization plots.
Just run this file, then copy the lines it generated in `results.csv` to a new CSV file in `visualizations/`, then follow the visualization in `visualizations/visualizations.ipynb` to get the plots.


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

times = np.logspace(2.8, 5, 10)
times = [int(i) for i in times]
counter = 0
sizes = [50, 100, 200, 400]
for s in sizes:
    for t in times:
        while counter != 10:
            experiments.append(
                {
                    "X": "lorenz_0.npy",
                    "Y": "lorenz_2.npy",
                    "tau": 17,
                    "d": 2,
                    "training_start": 0,
                    "training_length": 1000000,
                    "training_size": t,
                    "testing_start": 1500000,
                    "testing_length": 100000,
                    "error_scale": 15,
                    "mode": "voronoi",
                    "kmeans": False,
                    "subset_size": s,
                    "interpolation": "none",
                    "visualize_error": True,
                }
            )
            counter += 1
        counter = 0


write_header()

# Use the commented out code below to run multiple experiments in parallel.
# This can speed up the code in many cases.
# with Pool(cpu_count()) as p:
for experiment in experiments:
    result = demo(experiment)
    save(experiment, result)
