"""

RIPS AFRL 2020
Version August 19, 2020

This file is a modified version of an old version of lib that was used to compute the number of nonempty cells quickly.
It could potentially be applied to work involving the box counting dimension in the future.
It requires modification to work with our current I/O system.
We include it here for the sake of completeness.

"""

# basic packages
import numpy as np

# seeding
import random

# timing
import time

# file io
import os
from sklearn.metrics import mean_squared_error

# lorenz
from scipy.integrate import odeint
from scipy.interpolate import griddata, Rbf
from numba import jit

# geometry
from scipy.spatial import Delaunay, ConvexHull, convex_hull_plot_2d
from scipy.spatial import cKDTree

# voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans, MiniBatchKMeans

# alpha shape 3d
from collections import defaultdict

# 2D plotting
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# parallel
from multiprocessing import Pool, cpu_count

# saving files
import json
import csv

# histogram
import scipy.stats as stats


def uniform2(data, epsilon, save_bincounts=False):
    """
     
    This function is old code that could calculate the uniform grid
    It needs to be updated to be usable with the main code in this file

    Parameters:
        None

    Returns:
        None
     
    """

    d = len(data[0])

    column_mins = data[:, 0:d].min(axis=0)
    column_maxs = data[:, 0:d].max(axis=0)

    dim_boxes = np.ceil((column_maxs - column_mins) / epsilon).astype(int)

    print("mins: ", column_mins)
    print("maxs: ", column_maxs)
    print(dim_boxes)

    point_bins = np.floor(
        (data - column_mins) / (column_maxs - column_mins) * dim_boxes
    ).astype(int)

    # subtract 1 to treat the column maxes
    for i in range(len(point_bins)):
        for j in range(len(point_bins[i])):
            if point_bins[i][j] == dim_boxes[j]:
                point_bins[i][j] -= 1

    s = set()

    for i in range(len(point_bins)):
        s.add(tuple(point_bins[i]))

    N = len(s)
    print("N(epsilon): ", N)
    print("box dim: ", np.log(N) / np.log(1/epsilon))





def demo(experiment):
    """

    Parameters:
        experiment:
            A dictionary That specificies the parameters to be used in this trial, namely
            - trainingtime
            - testingtime
            - timestep
            - tau
            - d
            - interpolation
    Returns:
        ret:
            A dictionary that specificies results from the experiment, namely
            - runtime of the trial
            - RMS error
            - Pearson Correlation Coefficient

    """

    start = time.time()

    # The second argument of the "get" function specifies what to set the returned variable
    # to in the case that the first argument is not contained in the dictionary.
    # In this case, we fill in the output with the default experiment's value.
    # The default experiment is specific in the main code at the bottom.
    trainingtime = experiment.get("trainingtime", default_experiment["trainingtime"])
    testingtime = experiment.get("testingtime", default_experiment["testingtime"])
    timestep = experiment.get("timestep", default_experiment["timestep"])
    tau = experiment.get("tau", default_experiment["tau"])
    d = experiment.get("d", default_experiment["d"])
    interpolation = experiment.get("interpolation", default_experiment["interpolation"])
    do_kmeans = experiment.get("kmeans", default_experiment["kmeans"])
    mode = experiment.get("mode", default_experiment["mode"])
    epsilon = experiment.get("epsilon", default_experiment["epsilon"])

    # We need to seed the code because if we are using Pools, then all the experiments that start
    # at the same time will share the same initial seed, which is slightly undesirable.
    random.seed()
    np.random.seed()
    totaltime = trainingtime + testingtime

    # bin_size = 10*int(subset_size ** (1.0 / d))
    # uniform_results = uniform(trainingpts, testingpts, noisy, bin_size)
    uniform2(data[:int(trainingtime/timestep)], epsilon)


def writer_header():
    """

    This function writes the list of variables under consideration to a CSV file that will be used for printing results

    Parameters:
        None

    Returns:
        None

    """

    parameters_list = [
        "trainingtime",
        "testingtime",
        "timestep",
        "tau",
        "d",
        "interpolation",
        "time",
        "error",
        "kmeans",
    ]
    with open("results.csv", "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(parameters_list)


def save(experiment, result, filetype="csv"):
    """

    This function writes the parameters and results associated with one experiment to the file.
    We write one experiment at a file to avoid losing data in the event of errors during experimentation.

    Parameters:
        experiment:
            A dictionary that specifies the parameters of the experiment
        result:
            A dictionary that specifies the results of the experiment
        filetype:
            Specifies whether to write the results to a CSV or JSON file.
            Only CSV may be working at this points

    Returns:
        None

    """

    if filetype == "csv":
        with open("results.csv", "a") as csv_file:
            writer = csv.writer(csv_file)
            experiment_list = [
                experiment.get("trainingtime", default_experiment["trainingtime"]),
                experiment.get("testingtime", default_experiment["testingtime"]),
                experiment.get("timestep", default_experiment["timestep"]),
                experiment.get("tau", default_experiment["tau"]),
                experiment.get("d", default_experiment["d"]),
                experiment.get("interpolation", default_experiment["interpolation"]),
                experiment.get("kmeans", default_experiment["kmeans"]),
                result["time"],
                result["error"],
            ]
            writer.writerow(experiment_list)


if __name__ == "__main__":
    """

    Run a list of experiments using generated data

    """

    default_experiment = {
        "trainingtime": 100.0,
        "testingtime": 100.0,
        "timestep": 0.01,
        "tau": 0.1,
        "d": 2,
        "interpolation": "lin",
        "lorenz_parameters": {"rho": 30.0, "sigma": 10.0, "beta": 8.0 / 3.0,},
        "kmeans": True,
        "sample_size": "usual",
        "mode": "uniform",
        "epsilon": 0.9,
    }

    start_time = 0.0
    end_time = 100000.0
    timestep = 0.01

    start1 = time.time()
    filename = "" + str(timestep) + "_" + str(start_time) + "_" + str(end_time)
    data = np.load(os.path.join("data", filename + ".npy"))
    end1 = time.time()
    print(end1 - start1, data.shape)

    experiments = []
    for d in range(2, 3, 1):
        for it in np.linspace(np.log10(100000), np.log10(100000), 1):
            for epsilon in [0.1]:
                experiments.append(
                    {
                        "trainingtime": 10**it,
                        "testingtime": 1000,
                        "timestep": timestep,
                        "d": d,
                        "mode": "uniform",
                        "kmeans": False,
                        "epsilon": epsilon,
                    }
                )

    # writer_header()

    # Use the commented out code below to run multiple experiments in parallel.
    # This can speed up the code in many cases.
    # with Pool(cpu_count()) as p:
    for experiment in experiments:
        result = demo(experiment)
        # # results = p.map(demo, experiments)
