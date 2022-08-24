"""

RIPS AFRL 2020
Version August 19, 2020

This file is a modified version of an old version of lib that was used to compute the number of nonempty cells quickly.
It was used to visually compare differnt reconstructions for different levels of error.
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


def reconstruct_shadow(data, max_d, tau):
    l = []
    total = len(data) - (max_d - 1) * tau
    for i in range(max_d):
        l.append(data[i * tau : total + i * tau].reshape((total, 1)))
    shadow = np.concatenate(tuple(l), axis=1)
    return shadow


def random_idx_subset_containing(length, size, idx_subset=None):
    """

    Calculates a random subset/sample of a set of points

    Parameters:
        points: the set from which you want a sample
        size: the size of the sample
        idx_subset: a subset of "points" that is always included

    Returns:
        sample: the random sample

    """

    remaining = np.arange(0, length)
    if idx_subset:
        # the np.setdiff1d function returns an np array of the indices from 0
        # len(points) (in "remaining") that are not contained in "idx_subset"
        remaining = np.setdiff1d(remaining, idx_subset)
        size = size - idx_subset.size

    sample = np.random.choice(remaining, size=size, replace=False)
    if idx_subset:
        idx_subset = np.append(idx_subset, sample)
        return idx_subset

    return sample


def uniform(
    trainingpts, testingpts, Ytrain, epsilon, save_bincounts=False, interpolation="None"
):
    """

    This function is old code that could calculate the uniform grid
    It needs to be updated to be usable with the main code in this file

    Parameters:
        None

    Returns:
        None

    """

    print("epsilon: ", epsilon)

    d = len(trainingpts[0])

    column_mins = trainingpts[:, 0:d].min(axis=0)
    column_maxs = trainingpts[:, 0:d].max(axis=0)
    dim_boxes = np.ceil((column_maxs - column_mins) / epsilon).astype(int)
    print("dim_boxes: ", dim_boxes)

    trainingbins = {}
    testingbins = {}

    # we are choosing the grid based on all the points, not only the training points, because there could be testing points outside it
    # (grid selection is just dependent on column_mins, column_maxs, and dim_boxes)

    point_bins = np.floor(
        (trainingpts - column_mins) / (column_maxs - column_mins) * dim_boxes
    ).astype(int)
    # subtract 1 to treat the column maxes
    for i in range(len(point_bins)):
        for j in range(len(point_bins[i])):
            if point_bins[i][j] == dim_boxes[j]:
                point_bins[i][j] -= 1

    for i in range(len(point_bins)):
        # if not trainingbins[tuple(point_bins[i])]:
        #     trainingbins[tuple(point_bins[i])] = []
        key = tuple(point_bins[i])
        if key in trainingbins:
            trainingbins[key].append(i)
        else:
            trainingbins[key] = [i]

    # N = sum([1 for idx, x in np.ndenumerate(trainingbins) if x])
    N = len(trainingbins)
    print("Number of nonempty boxes: ", N)

    # Average Ys for each bin
    Yrec = np.zeros(len(testingpts))

    # Yavg contains the average of the noisy points in each bin
    Yavg = {}

    for idx, x in trainingbins.items():
        mean = np.mean(Ytrain[x])
        Yavg[idx] = mean

    # testing phase
    point_bins = np.floor(
        (testingpts - column_mins) / (column_maxs - column_mins) * dim_boxes
    ).astype(int)

    # subtract 1 to treat the column maxes
    for i in range(len(point_bins)):
        for j in range(len(point_bins[i])):
            if point_bins[i][j] == dim_boxes[j]:
                point_bins[i][j] -= 1

    for i in range(len(point_bins)):
        key = tuple(point_bins[i])
        if key in testingbins:
            testingbins[key].append(i)
        else:
            testingbins[key] = [i]

    # for each of the sample points
    for i, x in testingbins.items():
        Yrec[x] = Yavg.get(i, 0)

    if interpolation == "lin":

        sample = [
            (list(column_mins + epsilon * np.array(idx) + epsilon / 2), Yavg[idx])
            for idx, x in trainingbins.items()
        ]
        sample, Yavg = zip(*sample)
        sample = list(sample)
        Yavg = list(Yavg)

        # the "linear" interpolation option here performs baryocentric interpolation on each index
        # of whatever triangulation it forms
        # Note: it seems like the "cubic" option has higher variance wrt sample, and doesn't do better
        # and the "nearest" option has lower variance
        Yrec = griddata(sample, Yavg, testingpts, method="linear")
        bad_idxs = [i for i in range(len(Yrec)) if np.isnan(Yrec[i])]
        grid = griddata(sample, Yavg, testingpts[bad_idxs], method="nearest")
        counter = 0
        for i in range(len(testingpts)):
            if np.isnan(Yrec[i]):
                Yrec[i] = grid[counter]
                counter += 1
        good_idxs = [i for i in range(len(Yrec))]
        return {"Yrec": Yrec, "good_idxs": good_idxs, "bad_idxs": bad_idxs}

    return {"Yrec": Yrec}


def voronoi(sample, trainingpts, testingpts, Ytrain, interpolation="lin"):
    """

    This function performs the binning and interpolation following the
    unstructured mesh approach.

    Parameters:
        sample: the sample from which to build the mesh
        trainingpts: the training points
        testingpts: the testing points
        noisy: the noisy Y signal
        interpolation: the type of interpolation to be used

    Returns:
        A dictionary with keys:
            Yrec: the reconstructed Y signal
            good_idxs: the list of indices of training points that are
                reconstructed. The lets us see what happens when the outliers
                are ignored.
            bad_idxs: the points that are outside the convex hull of the sample set.
                This variable is returned if it is calculated as a side product
                of the interpolation.

    """

    # The dimension of the shadow manifold
    d = len(sample[0])
    subset_size = len(sample)
    trainingbins = [[] for i in range(subset_size)]
    testingbins = [[] for i in range(subset_size)]

    # We use a cKDTree for nearest neighbor searches
    # cKDTree is much faster than KDTree because it is compiled in C
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
    tree = cKDTree(sample)
    # The first returned variable is the distance to the NN, so we can ignore it
    # "point_idxs" has the same length as "trainingpts", and specifies the index of
    # nearest point in "sample" to it
    _, point_idxs = tree.query(trainingpts)
    for i in range(len(point_idxs)):
        trainingbins[point_idxs[i]].append(i)

    Yavg = []
    for k in range(len(trainingbins)):
        Yavg.append(np.mean(Ytrain[trainingbins[k]]))

    # Yrec is the reconstructed Y signal
    Yrec = np.zeros(len(testingpts))

    if interpolation == "delaunay":
        tri = Delaunay(sample)
        simplices = tri.find_simplex(testingpts)
        X = tri.transform[simplices, :d]
        Y = testingpts - tri.transform[simplices, d]
        b = np.array([x.dot(y) for x, y in zip(X, Y)])
        # np.c_ concatenates by column
        bcoords = np.c_[b, 1 - b.sum(axis=1)]
        # the indices of points that go outside the range
        bad_idxs = [i for i in range(len(testingpts)) if simplices[i] == -1]
        for i in range(len(testingpts)):
            if simplices[i] == -1:
                bin_idx = np.argmin(np.sum((sample - testingpts[i]) ** 2, axis=1))
                Yrec[i] = Yavg[bin_idx]
            else:
                in_tri = simplices[i]
                in_tri_vertices = tri.simplices[in_tri]
                vec = [Yavg[j] for j in in_tri_vertices]
                Yrec[i] = np.dot(bcoords[i], vec)
        good_idxs = [i for i in range(len(Yrec))]
        return {"Yrec": Yrec, "good_idxs": good_idxs, "bad_idxs": bad_idxs}

    elif interpolation == "none":
        # testingphase
        for i in range(len(testingpts)):
            point_index = np.argmin(
                np.sum((sample - testingpts[i]) ** 2, axis=1)
            )  # rows of bins
            testingbins[point_index].append(i)
        for i in range(0, len(testingbins)):
            for j in testingbins[i]:
                Yrec[j] = Yavg[i]
        print(Yrec)
        return {"Yrec": Yrec}

    elif interpolation == "lin":
        # the "linear" interpolation option here performs baryocentric interpolation on each index
        # of whatever triangulation it forms
        # Note: it seems like the "cubic" option has higher variance wrt sample, and doesn't do better
        # and the "nearest" option has lower variance
        Yrec = griddata(sample, Yavg, testingpts, method="linear")
        bad_idxs = [i for i in range(len(Yrec)) if np.isnan(Yrec[i])]
        grid = griddata(sample, Yavg, testingpts[bad_idxs], method="nearest")
        counter = 0
        for i in range(len(testingpts)):
            if np.isnan(Yrec[i]):
                Yrec[i] = grid[counter]
                counter += 1
        good_idxs = [i for i in range(len(Yrec))]
        return {"Yrec": Yrec, "good_idxs": good_idxs, "bad_idxs": bad_idxs}

    elif interpolation == "lin2":
        # Note: it seems like the "cubic" option has higher variance wrt sample, and doesn't do better
        # and the "nearest" option has lower variance
        Yrec = griddata(sample, Yavg, testingpts, method="linear")
        good_idxs = [i for i in range(len(Yrec)) if not np.isnan(Yrec[i])]
        Yrec = Yrec[good_idxs]
        return {"Yrec": Yrec, "good_idxs": good_idxs}

    elif interpolation == "rbf":
        rbfi = Rbf(*zip(*sample), Yavg)
        Yrec = rbfi(*zip(*testingpts))
        print(Yrec.shape)
        return {"Yrec": Yrec}


def demo(experiment):
    """

    Parameters:
        experiment:
            A dictionary That specificies the parameters to be used in this trial, namely
            - trainingtime
            - testingtime
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

    exp = {}
    for key, val in default_experiment.items():
        exp[key] = experiment.get(key, val)

    # We need to seed the code because if we are using Pools, then all the experiments that start
    # at the same time will share the same initial seed, which is slightly undesirable.
    random.seed()
    np.random.seed()

    X = np.load(os.path.join("data", exp["X"]))
    X = X.reshape((len(X), 1))
    Y = np.load(os.path.join("data", exp["Y"]))
    Y = Y.reshape((len(Y), 1))
    s = time.time()
    points = reconstruct_shadow(X, max_d=exp["d"], tau=exp["tau"])
    print("time to shadow: ", time.time()-s)
    X = X[:len(points)]
    Y = Y[:len(points)]

    Yrecs = []

    # errs = np.linspace(0, 0.5, 5)
    # errs = [, 0.3, 0.6, 0.9, 1.2, 1.5, 2, 3, 5]
    errs = [0] + [8*2**(-n) for n in range(10)]
    # errs = [0, 0.3, 0.6]
    for scale in errs:

        noise = np.random.normal(loc=0.0, scale=scale, size=(len(Y), 1))
        # noise = np.zeros((len(Y), 1))
        noisy = Y + noise

        # "points" is the reconstructed shadow manifold, as read from a file in main
        idxs = random_idx_subset_containing(
            exp["training_end"] - exp["training_start"], exp["training_size"]
        )
        trainingpts = points[exp["training_start"] + idxs, 0 : exp["d"]]
        Ytrain = noisy[exp["training_start"] + idxs]
        noisy_test = noisy[exp["testing_start"] : exp["testing_end"]]

        testingpts = points[exp["testing_start"] : exp["testing_end"]]
        Y_test = Y[exp["testing_start"] : exp["testing_end"]]

        # for Voronoi
        if exp["mode"] == "voronoi":

            # We found that the function 7.27\sqrt{x} fit our data for one parameter set fairly well.
            # The function can adjusted as desired, and the constant in front needs to change in some cases.
            # subset_size = int(7.27 * (trainingtime/100) ** (0.5))
            subset_size = exp["subset_size"]

            if exp["kmeans"]:
                # uses lloyd's or elkan's algorithms
                centertime = time.time()
                sample = MiniBatchKMeans(subset_size).fit(trainingpts).cluster_centers_
                print("kmeans time: ", time.time() - centertime)
            else:
                idx_sample = random_idx_subset_containing(exp["training_size"], subset_size)
                sample = trainingpts[idx_sample]

            voronoi_results = voronoi(
                sample, trainingpts, testingpts, Ytrain, interpolation=exp["interpolation"]
            )
            good_idxs = voronoi_results.get("good_idxs", [i for i in range(len(Y_test))])
            Y_test = Y_test[good_idxs]
            Yrec = voronoi_results["Yrec"]

        # for Uniform
        elif exp["mode"] == "uniform":
            uniform_results = uniform(
                trainingpts,
                testingpts,
                Ytrain,
                exp["epsilon"],
                interpolation=exp["interpolation"],
            )
            Yrec = uniform_results["Yrec"]

        Y_test = Y_test.reshape((len(Y_test),))
        Yrec = Yrec.reshape((len(Yrec),))

        ms = mean_squared_error(Y_test, Yrec)
        rmse = (ms) ** (0.5)

        pcc, _ = stats.pearsonr(Y_test, Yrec)

        end = time.time()
        print("experiment finished. time: ", end - start)
        print("rmse: ", rmse)
        print("pcc: ", pcc)
        ret = {"time": end - start, "rmse": rmse, "pcc": pcc}

        Yrecs.append(Yrec)

    

    if exp["visualize_reconstruction"]:
        # visualize small segment of reconstruction
        import matplotlib.pyplot as plt

        # fig, ax = plt.subplots()
        # et = int(5000)
        # # ax.plot(noisy_test[:et])
        # for Yrec in Yrecs:
        #     ax.plot(Yrec[:et])
        # ax.plot(Y_test[:et])
        # ax.legend(errs+["original"])
        # plt.show()

        a = []
        for i in range(1, len(errs)):
            # print(errs[0], errs[i+1], stats.pearsonr(Yrecs[i], Yrecs[i+1])[0])
            a.append(stats.pearsonr(Yrecs[0], Yrecs[i])[0])
        b = []
        for i in range(1, len(errs)):
            # print("original", errs[i], stats.pearsonr(Y_test, Yrecs[i])[0])
            b.append(stats.pearsonr(Y_test, Yrecs[i])[0])
        c = []
        for i in range(1, len(errs)):
            # print("original", errs[i], stats.pearsonr(Y_test, Yrecs[i])[0])
            c.append(stats.pearsonr(Yrecs[i-1], Yrecs[i])[0])

        print(a)
        print(b)
        print(c)

        plt.hist(Ytrain, 10)
        plt.show()
        plt.hist(Y_test, 10)
        plt.show()


    return ret


def writer_header():
    """

    This function writes the list of variables under consideration to a CSV file that will be used for printing results

    Parameters:
        None

    Returns:
        None

    """

    parameters_list = [key for key in default_experiment] + [
        key for key in default_result
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
                experiment.get(key, val) for key, val in default_experiment.items()
            ] + [result.get(key, val) for key, val in default_result.items()]
            writer.writerow(experiment_list)


if __name__ == "__main__":
    """

    Run a list of experiments using generated data

    """

    default_experiment = {
        "X": "",
        "Y": "",
        "tau": 0,
        "d": 2,
        "training_start": 0,
        "training_end": 10000,
        "training_size": 1000,
        "testing_start": 12000,
        "testing_end": 13000,
        "error_scale": 1,
        "mode": "uniform",
        "interpolation": "lin",
        "kmeans": True,
        "subset_size": 200,
        "epsilon": 3.5,
        "visualize_reconstruction": False,
    }

    default_result = {
        "time": 0.0,
        "rmse": 0.0,
        "pcc": 0.0,
    }

    experiments = []
    experiments.append(
        {
            "X": "anode_cathode.npy",
            "Y": "cathode_pearson.npy",
            "tau": 20,
            "training_start": 0,
            "training_end": 100000,
            "training_size": 50000,
            "testing_start": 110000,
            "testing_end": 120000,
            "error_scale": 10,
            "d": 30,
            "mode": "voronoi",
            "kmeans": False,
            "subset_size": 200,
            "interpolation": "none",
            "visualize_reconstruction": True,
        }
    )

    writer_header()

    # Use the commented out code below to run multiple experiments in parallel.
    # This can speed up the code in many cases.
    # with Pool(cpu_count()) as p:
    for experiment in experiments:
        result = demo(experiment)
        # results = p.map(demo, experiments)
        save(experiment, result)
