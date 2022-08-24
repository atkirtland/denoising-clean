"""

RIPS AFRL 2020
Version August 19, 2020

This file reads provided HET data and prepares it for use by experiments.
Output CSV file into the raw directory by the name Cxx--DATE--00041_SIGNALNAME_(5e-03s_0_pts_avg).csv

Note that there is no extension on the outfile name because it automatically gets a .npy added to it.
It is recommended to place the file in data directory.

"""

import csv
import numpy as np
import sys
import os

infile = sys.argv[1]
outfile = sys.argv[2]

if not os.path.exists("data"):
    os.makedirs("data")

with open(infile, "r") as f:
    # Clear the variable titles in AFRL provided files (first 3 lines)
    next(f)
    next(f)
    next(f)
    reader = csv.reader(f)
    l = [float(val) for time, val in reader]
    np.save(outfile, np.array(l))
