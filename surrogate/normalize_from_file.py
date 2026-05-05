#!/usr/bin/python
import numpy as np
import sys

# ---- configuration ----
input_file = sys.argv[1] # input to normalize
output_file = sys.argv[2] # result
ranges_file = sys.argv[3] # to load ranges

# Load data
ranges = np.loadtxt(ranges_file, delimiter=None)
data = np.loadtxt(input_file, delimiter=None)

# Column-wise min–max normalization
col_min = ranges[0]
col_max = ranges[1]

# Avoid division by zero for constant columns
range_ = col_max - col_min
range_[range_ == 0] = 1.0

normalized = (data - col_min) / range_

# Save result
np.savetxt(output_file, normalized, fmt="%g", delimiter=" ")
