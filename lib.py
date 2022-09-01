"""

RIPS AFRL 2020
Version August 19, 2020

This file contains functions that do the majority of the work. 
Some of the functions are called in the experiment files, examples of which are in the `experiments` folder.

"""

# basic packages
from cmath import exp
import numpy as np

# seeding
import random

# timing
import time

# file io
import os

# lorenz
from scipy.interpolate import griddata, Rbf

# geometry
from scipy.spatial import Delaunay, ConvexHull, convex_hull_plot_2d
from scipy.spatial import cKDTree
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans, MiniBatchKMeans

# 2D plotting
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# saving files
import json
import csv

# histogram, stats
import scipy.stats as stats
from sklearn.metrics import mean_squared_error

# a Dict specifying parameters for a single experiment run
default_experiment = {
    # these are the X and Y signal; their values should be "name.npy", where
    # `data/name.npy` is a scalar data signal saved with `np.save`, as done
    # in the data generation files.
    "X": "",
    "Y": "",
    # this variable is an positive integer. if its value is $n$, then the program will
    # automatically only use every $n$th entry in X and Y to work with; it was meant to
    # test simulating a different timestep for signals with a fixed timestep.
    "timestep_multiple": 1,
    # a positive integer specifying the time delay by number of indices in the signal
    "tau": 0,
    # an integer > 2 specifying the the dimension of the reconstruction
    "d": 2,
    # a nonnegative integer specifying the index of X and Y at which the training interval begins
    "training_start": 0,
    # a positive integer specifying how long the training interval is
    "training_length": 10000,
    # a nonnegative integer or the value -1 specifying how many points to sample from the training interval for the training set
    # if its value is -1, then the entire training interval is used for training
    "training_size": -1,
    # a nonnegative integer specifying the index of X and Y at which the testing interval begins
    "testing_start": 12000,
    # a positive integer specifying how long the testing interval is
    "testing_length": 1000,
    # a nonnegative integer or the value -1 specifying how many points to sample from the testing interval for the testing set
    # if its value is -1, then the entire testing interval is used for testing
    "testing_size": -1,
    # a nonnegative real number specifying the standard deviation of the normal noise added to the signal
    # if its value is 0, then no noise is added
    "error_scale": 1,
    # whether to use uniform or voronoi mesh, possible values are "uniform" or "voronoi"
    "mode": "uniform",
    # what type of interpolation to use.
    # possible values for voronoi are "none", "linear", "delaunay", and "rbf"
    # where "delaunay" is basically "linear" but with a delaunay traingulation
    # and "rbf" was for an old attempt to use an RBF network for extrapolation
    # possible values for uniform are "none", "linear", and "delaunay"
    "interpolation": "linear",
    # a Boolean specifying whether or not to use $k$-means if the mode is "voronoi"
    "kmeans": False,
    # a positive integer specifying how many cells to use if the mode is "voronoi"
    "subset_size": 200,
    # WARNING: Using epsilon leads to bad results for linear interpolation. that's why we use boxes_per_dim now
    # a positive real number specifying the side length of the cells to use if the mode is "uniform"
    # "epsilon": 3.5,
    "boxes_per_dim": 10,
    # a Boolean specifying whether or not to show a plot showing the reconstructed signal and the original signal
    "visualize_reconstruction": False,
    # a Boolean specifying whether or not to save a txt file specifying the number of training points that were in each cell in the reconstruction
    # should work for both voronoi and uniform
    "save_bincounts": False,
    # a Boolean specifying whether or not to show a heatmap plot showing the 2D submanifold of the shadow manifold (the first two coordinates of it) with
    # color indicating the squared error of the corresponding reconstructed points vs the original signal
    "visualize_error": False,

    # a Boolean specifying whether to visplay a visualization in the format Jun requested
    "visualize_jun": False,
    # a Boolean specifying whether to highlight the bad_idxes outside the convex hull of the training set
    "visualize_bad_idxs": False,
    # the timestep used in the generated data -- used for visualization
    "timestep": 0.01,
    # the xlabel displayed on the graph
    "xlabel": "$X(t)$",
    # the ylabel displayed on the graph
    "ylabel": "$Y(t)$",
    # the axes limits for the top plot in a Jun visualization plot
    "ax0xlim": -25,
    "ax0ylim": 25,
    # the axes limits for the middle and bottom plots in a Jun visualization plot
    "ax1xlim": -25,
    "ax1ylim": 75,
    # whether to make the true target signal dotted
    "truetargetdotted": True,
    # whether to save the noisy signal for later analysis
    "saveNoisy": False,
    # whether to save the reconstruction for later analysis
    "saveRec": False,
}

# a Dict specifying which numbers should be returned from an experiment
default_result = {
    # an integer specifying the number of nonempty cells in the reconstruction
    # works for both modes
    "nonempty": 0,
    # specifies the amount of time the experiment ran for
    "time": 0.0,
    # specifies the RMSE of the reconstruction vs the original
    "rmse": 0.0,
    # specifies the Pearson Correlation Coefficient (PCC) of the reconstruction vs the original
    "pcc": 0.0,
}


def reconstruct_shadow(data, max_d, tau):
    """

    Reconstructs a shadow manifold given a signal.

    Parameters:
        data: a 1D signal
        max_d: the dimension of the shadow manifold
        tau: the time delay

    Returns:
       shadow: the max_d dimensional shadow manifold

    """

    l = []
    # we need to chop this bit off from what we can return because of how each
    # shadow dimension is on another piece of the data.
    total = len(data) - (max_d - 1) * tau
    for i in range(max_d):
        l.append(data[i * tau : total + i * tau].reshape((total, 1)))
    shadow = np.concatenate(tuple(l), axis=1)
    return shadow


def uniform(Xtrain, Xtest, Ytrain, boxes_per_dim, save_bincounts=False, interpolation="None"):
    """

    This function can compute a reconstruction following the uniform grid approach.

    Parameters:
        Xtrain: the points of the X signal on which to train the grid
        Xtest: the points of the X signal on which to reconstruct the corresponding Y signal
        Ytrain: the points of the Y signal that correspond to Xtrain
        boxes_per_dim: an integer specifying the number of boxes to use in each dimension
        save_bincounts: whether or not to save a txt file containing the counts of the trainingbins for use in plotting histograms
        interpolation: whether or not to use linear interpolation (if yes, use the value "linear")
        Ytest: 

    Returns:
        A dictionary containing the values:
            Yrec: the reconstructed signal
            nonempty: the number of nonempty cells
            bad_idxs: the indices outside the convex hull

    """

    #embedding dimension
    d = len(Xtrain[0]) 

    # get the minimum and maximum elements of each column
    column_mins = Xtrain[:, 0:d].min(axis=0)
    column_maxs = Xtrain[:, 0:d].max(axis=0)

    # number of boxes in each dimension is just copied for each dimension as of now
    dim_boxes = np.full(d, boxes_per_dim)

    trainingbins = {}
    testingbins = {}

    # we are choosing the grid based on all the points, not only the training points, because there could be testing points outside it
    # (grid selection is just dependent on column_mins, column_maxs, and dim_boxes)

    # figure out which bin each point will be in
    point_bins = np.floor(
        (Xtrain - column_mins) / (column_maxs - column_mins) * dim_boxes
    ).astype(int)
    # subtract 1 to treat the column maxes, as otherwise they'd be at the top idx, which is not a bin
    # so we just go through every bin and substract 1 if it's too high
    for i in range(len(point_bins)):
        for j in range(len(point_bins[i])):
            if point_bins[i][j] == dim_boxes[j]:
                point_bins[i][j] -= 1

    # for each point either add its index to the corresponding bin or make a new list for that bin with its index
    for i in range(len(point_bins)):
        key = tuple(point_bins[i])
        if key in trainingbins:
            trainingbins[key].append(i)
        else:
            trainingbins[key] = [i]

    N = len(trainingbins)
    print("Number of nonempty boxes: ", N)

    # saves counts of the bins to a txt file
    if save_bincounts:
        bin_counts = [len(binn) for binn in trainingbins.values()]
        from datetime import datetime
        np.savetxt("bin_counts_"+str(datetime.now().replace(microsecond=0).isoformat()+".txt"), bin_counts)

    # this will be the recovered signal
    Yrec = np.zeros(len(Xtest))

    # Yavg contains the average of the noisy points in each bin
    Yavg = {}

    # set the average of each bin to the average of the points in Ytrain whose X part (the X value at the same time) is in it
    for idx, x in trainingbins.items():
        mean = np.mean(Ytrain[x])
        Yavg[idx] = mean

    # testing phase
   
    # again, find which bins the testing points are in
    point_bins = np.floor(
        (Xtest - column_mins) / (column_maxs - column_mins) * dim_boxes
    ).astype(int)

    # subtract 1 to treat the column maxes
    # same idea as last time
    for i in range(len(point_bins)):
        for j in range(len(point_bins[i])):
            if point_bins[i][j] == dim_boxes[j]:
                point_bins[i][j] -= 1

    # same idea as last time
    for i in range(len(point_bins)):
        key = tuple(point_bins[i])
        if key in testingbins:
            testingbins[key].append(i)
        else:
            testingbins[key] = [i]

    # for each of the testingbins, set the recovered value for all the points in it to Yavg[i] if that exists, or else 0 (for the outlier points)
    for i, x in testingbins.items():
        Yrec[x] = Yavg.get(i, 0)

    # the dictionary we will return
    ret = {}
    ret["Yrec"] = Yrec
    ret["nonempty"] = N

    # We do this to get values for the cell centers, as this is used for all the interpolation methods.
    if interpolation != "none":

        # sample is a list of 2-tuples, where each element of the list corresponds to one nonempty bin
        # and where the first element of each tuple is the coordinates of the center of that bin
        # and the second element of each tuple is the Yavg value for that bin
        # we need this for out interpolation methods
        epsilons = (column_maxs - column_mins)/dim_boxes
        sample = [
            (list(column_mins + np.multiply(epsilons, np.array(idx)) + epsilons / 2), Yavg[idx])
            for idx, x in trainingbins.items()
        ]
        sample, Yavg = zip(*sample)
        sample = list(sample)
        Yavg = list(Yavg)

    # The "linear" and "delaunay" interpolation options were directly copied from the Voronoi code.
    # That section may have more comments if it is difficult to understand.

    if interpolation == "linear":

        # the "linear" interpolation option here performs baryocentric interpolation on each index
        # of whatever triangulation it forms
        # Note: it seems like the "cubic" option has higher variance wrt sample, and doesn't do better
        # and the "nearest" option has lower variance

        Yavg = np.array(Yavg)
        Yrec = griddata(sample, Yavg, Xtest, method="linear")
        bad_idxs = [i for i in range(len(Yrec)) if np.isnan(Yrec[i])]
        grid = griddata(sample, Yavg, Xtest[bad_idxs], method="nearest")
        counter = 0
        for i in range(len(Xtest)):
            if np.isnan(Yrec[i]):
                Yrec[i] = grid[counter]
                counter += 1

        ret["Yrec"] = Yrec
        ret["bad_idxs"] = bad_idxs

    if interpolation == "delaunay":
        
        # We did not work through the details of this algorithm for computing and using the Delaunay triangulation.
        # it was found online, and a brief test made it seem like it works well
        tri = Delaunay(sample)
        simplices = tri.find_simplex(Xtest)
        X = tri.transform[simplices, :d]
        Y = Xtest - tri.transform[simplices, d]
        b = np.array([x.dot(y) for x, y in zip(X, Y)])
        # np.c_ concatenates by column
        bcoords = np.c_[b, 1 - b.sum(axis=1)]
        # the indices of points that go outside the range
        bad_idxs = [i for i in range(len(Xtest)) if simplices[i] == -1]
        for i in range(len(Xtest)):
            if simplices[i] == -1:
                bin_idx = np.argmin(np.sum((sample - Xtest[i]) ** 2, axis=1))
                Yrec[i] = Yavg[bin_idx]
            else:
                in_tri = simplices[i]
                in_tri_vertices = tri.simplices[in_tri]
                vec = [Yavg[j] for j in in_tri_vertices]
                Yrec[i] = np.dot(bcoords[i], vec)

        # bad_idxs are the idxs of points outside the convex hull
        ret["bad_idxs"] = bad_idxs
        ret["Yrec"] = Yrec

    return ret


def voronoi(
    sample, Xtrain, Xtest, Ytrain, interpolation="linear", save_bincounts=False
):
    """

    This function performs the binning and interpolation following the
    unstructured mesh approach.

    Parameters:
        sample: the sample from which to build the mesh
        Xtrain: the training points
        Xtest: the testing points
        Ytrain: the points of the Y signal that correspond to Xtrain
        interpolation: the type of interpolation to be used
        save_bincounts: whether or not to save the number of points in each bin to a txt file

    Returns:
        A dictionary with keys:
            Yrec: the reconstructed Y signal
            bad_idxs: the points that are outside the convex hull of the sample set.
                This variable is returned if it is calculated as a side product
                of the interpolation.

    """

    # The dimension of the shadow manifold
    d = len(sample[0])
    subset_size = len(sample) #number of Voronoi cells
    # Each element of trainingbins and testingbins will contain the indices of the training points or testing points residing in the 'i'-th Voronoi cell
    trainingbins = [[] for i in range(subset_size)] 
    testingbins = [[] for i in range(subset_size)] 
    
    # We use a cKDTree for nearest neighbor searches
    # cKDTree is much faster than KDTree because it is compiled in C
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
    tree = cKDTree(sample)
    # The first returned variable is the distance to the NN, so we can ignore it
    # "point_idxs" has the same length as "Xtrain", and specifies the index of
    # nearest point in "sample" to it
    _, point_idxs = tree.query(Xtrain)
    # Identify training points on the shadow manifold with their Voronoi cells
    for i in range(len(point_idxs)):
        trainingbins[point_idxs[i]].append(i) 
    # Average the values of Y corresponding to the training points that reside in the same Voronoi cell 
    Yavg = []
    for k in range(len(trainingbins)):
        Yavg.append(np.mean(Ytrain[trainingbins[k]]))

    # Yrec is the reconstructed Y signal
    Yrec = np.zeros(len(Xtest))

    ret = {}

    if save_bincounts:
        bin_counts = [len(binn) for binn in trainingbins]
        from datetime import datetime
        np.savetxt("bin_counts_"+str(datetime.now().replace(microsecond=0).isoformat()+".txt"), bin_counts)
    
    # As far as we know, this is just like the "linear" interpolation option, but with always a Delaunay triangulation instead of an unknown triangulation.
    # Because the performance was about the same with either option, we just did the linear interpolation option because its code was more understandable.
    if interpolation == "delaunay":
        tri = Delaunay(sample)
        simplices = tri.find_simplex(Xtest)
        # The next few lines below were taken from a website, and we didn't check them carefully.
        X = tri.transform[simplices, :d]
        Y = Xtest - tri.transform[simplices, d]
        b = np.array([x.dot(y) for x, y in zip(X, Y)])
        # np.c_ concatenates by column
        bcoords = np.c_[b, 1 - b.sum(axis=1)]
        # the indices of points that go outside the range
        bad_idxs = [i for i in range(len(Xtest)) if simplices[i] == -1]
        for i in range(len(Xtest)):
            if simplices[i] == -1:
                bin_idx = np.argmin(np.sum((sample - Xtest[i]) ** 2, axis=1))
                Yrec[i] = Yavg[bin_idx]
            else:
                in_tri = simplices[i]
                in_tri_vertices = tri.simplices[in_tri]
                vec = [Yavg[j] for j in in_tri_vertices]
                Yrec[i] = np.dot(bcoords[i], vec)

        ret["bad_idxs"] = bad_idxs

    elif interpolation == "none":
        Yavg = np.array(Yavg)
        Yrec = griddata(sample, Yavg, Xtest, method="nearest")

    elif interpolation == "linear":
        # the "linear" interpolation option here performs baryocentric interpolation on each index
        # of whatever triangulation it forms
        # Note: it seems like the "cubic" option has higher variance wrt sample, and doesn't do better
        # and the "nearest" option has lower variance

        Yavg = np.array(Yavg)
        Yrec = griddata(sample, Yavg, Xtest, method="linear")
        bad_idxs = [i for i in range(len(Yrec)) if np.isnan(Yrec[i])] #points outside of the convex hull of the sample points
        grid = griddata(sample, Yavg, Xtest[bad_idxs], method="nearest")

        counter = 0
        for i in range(len(Xtest)):
            if np.isnan(Yrec[i]):
                Yrec[i] = grid[counter]
                counter += 1

        ret["bad_idxs"] = bad_idxs

    # We made one attempt at using RBF networks for extrpolating values, but that didn't work well at all, so we ditched it.
    elif interpolation == "rbf":
        rbfi = Rbf(*zip(*sample), Yavg)
        Yrec = rbfi(*zip(*Xtest))

    ret["Yrec"] = Yrec
    return ret


def demo(experiment):
    """

    Parameters:
        experiment:
            A dictionary that contains the parameters to be used in this experiment, namely some subset of the variables defined in "default_experiment"

    Returns:
        ret:
            A dictionary that specificies results from the experiment, namely the same values that are in "default_result"

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

    training_end = exp["training_start"] + exp["training_length"]
    testing_end = exp["testing_start"] + exp["testing_length"]

    # We only build the shadow manifold out of as many points as we need to to save a bit of time, so
    # we calculate here how about how many points we will need.
    max_len = max(training_end, testing_end)

    X = np.load(os.path.join("data", exp["X"]))
    # below: only extracts every nth element from the signals, where n is the timestep multiple
    X = X[:: exp["timestep_multiple"]]
    X = X.reshape((len(X), 1))
    Y = np.load(os.path.join("data", exp["Y"]))
    Y = Y[:: exp["timestep_multiple"]]
    Y = Y.reshape((len(Y), 1))

    s = time.time()

    # We add some to max_len because reconstruct_shadow automatically cuts off this many because of how the time delays work.
    shadow = reconstruct_shadow(
        X[: max_len + (exp["d"] - 1) * exp["tau"]], max_d=exp["d"], tau=exp["tau"]
    )

    print("time to shadow: ", time.time() - s)

    # We need to shorten Y to the right length now (and we shorten X as well just to make sure - shouldn't do anything)
    X = X[:max_len]
    Y = Y[:max_len]

    # Here we add the noise
    noise = np.random.normal(loc=0.0, scale=exp["error_scale"], size=(max_len, 1))
    Yprime = Y + noise

    # Here we select a random sample of points to be the training points
    print(training_end - exp["training_start"], exp["training_size"])
    if exp["training_size"] != -1:
        idxs = np.random.choice(
            training_end - exp["training_start"],
            size=exp["training_size"],
            replace=False,
        )
    else:
        idxs = np.array(range(training_end - exp["training_start"]))

    # Here we take the first d columns of shadow as the training data
    Xtrain = shadow[exp["training_start"] + idxs, 0 : exp["d"]]
    # We take all of Yprime
    Ytrain = Yprime[exp["training_start"] + idxs]

    if exp["testing_size"] != -1:
        idxs = np.random.choice(
            testing_end - exp["testing_start"], size=exp["testing_size"], replace=False
        )
    else:
        idxs = np.array(range(testing_end - exp["testing_start"]))

    Xtest = shadow[exp["testing_start"] + idxs]
    Ytest = Y[exp["testing_start"] + idxs]

    sample = []

    # for Voronoi
    if exp["mode"] == "voronoi":

        if exp["kmeans"]:
            # uses lloyd's or elkan's algorithms
            centertime = time.time()
            # This line sometimes gets a warning, but after researching it a bit I think it's fine
            sample = (
                MiniBatchKMeans(n_clusters=exp["subset_size"])
                .fit(Xtrain)
                .cluster_centers_
            )
            print("kmeans time: ", time.time() - centertime)

        else:
            # otherwise pick the subset randomly
            idx_sample = np.random.choice(
                len(Xtrain), size=exp["subset_size"], replace=False
            )
            sample = Xtrain[idx_sample]

        voronoi_results = voronoi(
            sample,
            Xtrain,
            Xtest,
            Ytrain,
            interpolation=exp["interpolation"],
            save_bincounts=exp["save_bincounts"],
        )
        Yrec = voronoi_results["Yrec"]

        # All the cells are nonempty for Voronoi, so we just return the subset size.
        nonempty = exp["subset_size"]

    # for Uniform
    elif exp["mode"] == "uniform":
        uniform_results = uniform(
            Xtrain,
            Xtest,
            Ytrain,
            exp["boxes_per_dim"],
            interpolation=exp["interpolation"],
            save_bincounts=exp["save_bincounts"],
        )
        Yrec = uniform_results["Yrec"]

        # We return this for use in comparing the two models.
        nonempty = uniform_results["nonempty"]

    Ytest = Ytest.reshape((len(Ytest),))
    Yrec = Yrec.reshape((len(Yrec),))

    ms = mean_squared_error(Ytest, Yrec)
    rmse = (ms) ** (0.5)

    pcc, _ = stats.pearsonr(Ytest, Yrec)

    end = time.time()
    print("experiment finished. time: ", end - start)
    print("pcc: ", pcc)
    print("rmse: ", rmse)

    ret = {"nonempty": nonempty, "time": end - start, "rmse": rmse, "pcc": pcc}

    times = np.arange(0, exp["testing_length"]*exp["timestep"], exp["timestep"])

    if exp["visualize_bad_idxs"]:
        hull = Delaunay(Xtrain)
        bad_train_idxs = np.where(hull.find_simplex(Xtest)==-1)

    if exp["visualize_jun"]:
        # visualize small segment of reconstruction
        import matplotlib.pyplot as plt
        plt.style.use('default')

        fig, axs = plt.subplots(3, 1,figsize = (15,7))
        axs[0].tick_params(axis='y', which='major', pad=15)
        axs[1].tick_params(axis='y', which='major', pad=15)
        axs[2].tick_params(axis='y', which='major', pad=15)
        fig.legend(loc=2, prop={'size': 6})

        axs[0].plot(times, Xtest[:,0], c="black",linewidth = 1)
        axs[0].set_ylabel(exp["xlabel"],fontsize = 15)
        axs[0].legend(
            ["Reference Signal $X(t)$"], loc='upper right'
        )

        et = int(50000)

        noisy = Yprime[exp["testing_start"] + idxs]
        axs[1].set_ylabel('Data',fontsize = 15)
        if exp["truetargetdotted"]:
            axs[1].plot(times, noisy, c="black", zorder=0, alpha=.40,linewidth = 1)
            axs[1].plot(times, Ytest[:et], "--", c="red", zorder=1, alpha=1,linewidth = 1)
        else:
            axs[1].plot(times, noisy, c="black")
            axs[1].plot(times, Ytest[:et], c="red")
        axs[1].legend(
            ["Noisy Target Signal", "True Target Signal "+exp["ylabel"]], loc='upper right'
        )


        if exp["truetargetdotted"]:
            axs[2].plot(times, Yrec[:et], c="blue",linewidth = 1)
            axs[2].plot(times, Ytest[:et], "--", c="red", alpha=1,linewidth = 1)
        else:
            axs[2].plot(times, Yrec[:et], c="blue")
            axs[2].plot(times, Ytest[:et], c="red")
        axs[2].legend(
            ["Denoised Target Signal", "True Target Signal"], loc='upper right'
        )
        axs[2].set_ylabel(exp["ylabel"],fontsize = 15)

        axs[0].set_ylim(exp["ax0xlim"], exp["ax0ylim"])
        axs[1].set_ylim(exp["ax1xlim"], exp["ax1ylim"])
        axs[2].set_ylim(exp["ax1xlim"], exp["ax1ylim"])
        plt.xlabel("Time Elapsed ($t$)",fontsize = 15)
        axs[0].set_xticklabels([])
        axs[1].set_xticklabels([])
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.show()
    
    if exp["saveNoisy"]:
        with open('noisy.npy', 'wb') as f:
            np.save(f, noisy)
    if exp["saveRec"]:
        with open('rec.npy', 'wb') as f:
            np.save(f, Yrec)

    if exp["visualize_reconstruction"]:
        # visualize small segment of reconstruction
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # todo why is this here
        et = int(50000)
        ax.plot(times, Yrec[:et], zorder=0)
        ax.plot(times, Ytest[:et], zorder=1, alpha=0.85)
        if exp["visualize_bad_idxs"]:
            import copy
            tempx = np.array(bad_train_idxs, dtype=object)*exp["timestep"]
            tempx = copy.deepcopy(tempx)

            tempy = np.array(Ytest[:et][bad_train_idxs], dtype=object)
            tempy = copy.deepcopy(tempy)

            ax.scatter(tempx, tempy, c="red", s=10, zorder=2)

        ax.set(xlabel="Time Elapsed $t$")
        ax.set(ylabel=exp["ylabel"])
        ax.set(title="Target Signal Reconstruction")
        ax.legend(
            ["Denoised Target Signal", "True Target Signal", "Outlier Points"]
        )
        plt.show()

    if exp["visualize_bad_idxs"]:
        
        points = Xtest
        x1 = points[:, 0]
        y1 = points[:, 1]
        plt.plot(x1, y1, alpha=0.5, color="lightslategrey")
        badpts = points[bad_train_idxs]
        plt.scatter(badpts[:,0], badpts[:,1], c="red", s=5)
        plt.title("Points Out of Hull")
        plt.show()

    if exp["visualize_error"]:

        def plot_vor1(vor, points, errpts, title=""):
            import matplotlib.pyplot as plt

            if exp["d"] == 2 and exp["mode"] == "voronoi":
                fig = voronoi_plot_2d(vor, show_vertices=False, show_points=False)
            x1 = points[:, 0]
            y1 = points[:, 1]


            # you can uncomment this line to see some modiciations like absolute value of error
            # errpts = [i for i in errpts]
            plt.scatter(x1, y1, s=1, c=errpts)
            plt.clim(0, 1)
            plt.title("Testing Manifold Reconstruction Errors")
            cbar = plt.colorbar()
            cbar.set_label("Square of Error")
            plt.show()

        if exp["mode"] == "voronoi":
            vor = Voronoi(sample)
        else:
            vor = 0
        errpts = []

        for i in range(len(Yrec)):
            errpts.append(((Yrec[i] - Ytest[i]) ** 2).tolist())
        plot_vor1(vor, Xtest, errpts)


    return ret


def write_header():
    """

    This function writes the list of variables in each experiment and result to a CSV file that will be used for printing results

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


def save(experiment, result):
    """

    This function writes the parameters and results associated with one experiment to the file.
    We write one experiment at a file to avoid losing data in the event of errors during experimentation.

    Parameters:
        experiment:
            A dictionary that specifies the parameters of the experiment
        result:
            A dictionary that specifies the results of the experiment

    Returns:
        None

    """

    with open("results.csv", "a") as csv_file:
        writer = csv.writer(csv_file)
        experiment_list = [
            experiment.get(key, val) for key, val in default_experiment.items()
        ] + [result.get(key, val) for key, val in default_result.items()]
        writer.writerow(experiment_list)
