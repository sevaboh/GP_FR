#!/usr/bin/python
import numpy as np
import sys

# ---- configuration ----
full_file = sys.argv[1] # full - to calculate ranges
input_file = sys.argv[2] # input to normalize
output_file = sys.argv[3] # result
ranges_file = sys.argv[4] # to save ranges

# Load data
full_data = np.loadtxt(full_file, delimiter=None)
data = np.loadtxt(input_file, delimiter=None)

# Column-wise min–max normalization
col_min = full_data.min(axis=0)
col_max = full_data.max(axis=0)

# Avoid division by zero for constant columns
range_ = col_max - col_min
range_[range_ == 0] = 1.0

normalized = (data - col_min) / range_

# Save result
np.savetxt(output_file, normalized, fmt="%g", delimiter=" ")
np.savetxt(ranges_file, [col_min,col_max], fmt="%g", delimiter=" ")
