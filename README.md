# 2020 AFRL RIPS Team

This is the code from the 2020 AFRL RIPS team.

## Workflow

Our general workflow is as follows.
First, generate a scalar data signal in the form of a `.npy` file and place it into `data/`.
In other words, it will be a 1D numpy array that is saved through the `numpy.save` command.
We provide methods for generating this signal from the raw HET data we were provided as well as from attractors like the Lorenz system in the `data_generation` directory.

Next, write an file in `experiments/` that specifies which parameters to use in running the kinds of tests you want.
Specifically, the file should follow the format of one of the experiments already in `experiments/`.
We specify each individual experiment to run by a dictionary that contains specific keys that correspond to experimental parameters.
For example, keys include `X` that specifies the X signal in reconstruction, `training_start` that specifies where to begin the training window, and `subset_size` that specifies how many cells to include.
The list of all possible experimental parameters are those in `default_dictionary` in `lib.py`; these parameters are used in the case that some parameters are not specified in the experiment file.

Then, we run the experiment file, which should append its results to `results.csv`.
For example, an experiment file could be run as `python experiments/experiment.py` when based in the main directory.
You should be based in the main directory for each of these steps in order for files to go where they should.
Each set of experiments (corresponding to one run of an experiment file) prints a CSV header line in addition to its data lines so that it is apparent which lines correspond to which experiment.
We then copy the experiment result lines and the CSV header above them to a new CSV file in `visualizations/results/[date]/`.
Lastly, we generate visualizations using the Jupyter notebook in `visualizations/`.

Note: As mentioned above, files should be run from this main directory, so you will generally need to prepend file names with the name of the directory their are in.


## Files

- `README.md`: This file is the one you are reading.
- `raw/`: contains the CSV files with HET data provided by AFRL
- `data_generation/`: contains files related to generating data
  - `clean_het_data.py`: a script that cleans the raw HET data. 
    - We run it as, for example, `python data_generation/clean_het_data.py raw/Cxx--2017_09_29--00041_Anode + Cathode current_(5e-03s_0_pts_avg).csv data/anode_cathode`.
    - We recommend placing the cleaned data files into `data/`. Each file should then contain a scalar signal for use in data analysis.
  - `clean_all_het_data.py `: cleans all the HET data (which is not included in this public release) at once
  - `generate_attractor.py`: this script generates Lorenz data and places the results in the `data/` directory.
    - The parameters for the generated system can be modified in the file.
- `data/`: The data generation/cleaning files will generate this directory when you run them. It should contain the scalar signals used in the experiments as `.npy` files.
- `lib.py`: This file contains the functions that are used for the data analysis. We call these functions from a file in `experiments/` to generate data.
- `experiments/`: This directory contains several sets of "experiments" that use the functions in `lib.py` to generate results and appends the results to `results.csv`. The files in it can be used as templates for other experiments.
- `results.csv`: The experiments file generates this file when you run it. It contains the results of all the experiments you have run, where it inserts a CSV header line between groups of experiments to make it clear which experiments are grouped together. The parameters listed in the file are precisely those in `default_dictionary`, then `default_result`.
- `visualizations/`: This directory includes files generally relating to visualization.
  - `visualizations.ipynb`: This file contains most of the current visualizations that are used in the paper and presentations. Visualizations in this file use as data files in `results/`.
  - `old-visualizations.ipynb`: This file contains older visualizations that are not being currently used, and they may use data that was generated in an old or incorrect way.
  - `optimize_tau_d.ipynb`: This file contains code for the Average Mutual Information (AMI) method of finding tau, for generating the videos of the shadow manifold changing with different values of tau, and for Cao's method of finding the minimal embedding dimension.
  - `results/`: This directory contains subdirectories labelled by dates. Each of these dated directories has files that were generated on that date. Most of these files are CSV, though some may be other formats, like `txt` for bin counting and histograms.
- `specialized`: These files are generally standalone modifications of older versions of `lib` that were used to run particular tests or generate graphics that deviate from when the default `lib` file can provide for. Some modifications to them may be required to get their I/O working.

