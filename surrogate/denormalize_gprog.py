#!/usr/bin/python
import numpy as np
import sys

# ---- configuration ----
input_file = sys.argv[1] # gprog testing output
ranges_file = sys.argv[2] # denormalizations ranges
output_file = sys.argv[3] # denomalized result
idx=int(sys.argv[4])+1

# Load data
data = np.loadtxt(input_file, delimiter=None)
ranges = np.loadtxt(ranges_file, delimiter=None)

# Column-wise min–max normalization
col_min = ranges[0]
col_max = ranges[1]

# Avoid division by zero for constant columns
range_ = col_max - col_min
range_[range_ == 0] = 1.0

denormalized = data*range_[idx] + col_min[idx]

# Save result
np.savetxt(output_file, denormalized, fmt="%g", delimiter=" ")
