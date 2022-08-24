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

if not os.path.exists("data"):
    os.makedirs("data")

# assign directory
directory = 'raw'
out_dir = "data"
 
# iterate over files in
# that directory
for fullfn in os.listdir(directory):
    shortfn = str.lower(fullfn[23:-23]).replace(" ", "_")
#   checking if it is a file
    if os.path.isfile(os.path.join(directory, fullfn)):
        infile = os.path.join(directory, fullfn)
        outfile = os.path.join(out_dir, shortfn)
        with open(infile, "r") as f:
            # Clear the variable titles in AFRL provided files (first 3 lines)
            next(f)
            next(f)
            next(f)
            reader = csv.reader(f)
            l = [float(val) for time, val in reader]
            np.save(outfile, np.array(l))
