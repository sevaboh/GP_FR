#!/usr/bin/python
import sys
import pandas as pd
import numpy as np

def find_nearest_outputs(file1_path, file2_path, n_outputs, output_path):
    # Load files (assuming space-separated)
    df1 = pd.read_csv(file1_path, sep='\s+', header=None)
    df2 = pd.read_csv(file2_path, sep='\s+', header=None)

    # Identify column indices
    out_cols = list(range(n_outputs))
    in_cols = list(range(n_outputs, df1.shape[1]))

    # 1. Calculate normalization ranges from File 2
    # Range = max - min for each output column
    ranges = df2[out_cols].max() - df2[out_cols].min()
    # Avoid division by zero if a column has constant values
    ranges = ranges.replace(0, 1)

    # 2. Group File 2 by the input vector for fast lookup
    # We use a dictionary or groupby for efficient access
    grouped_f2 = df2.groupby(in_cols)

    results = []

    # 3. Iterate through File 1
    for _, row in df1.iterrows():
        input_vec = tuple(row[in_cols])
        target_output = row[out_cols].values
        
        try:
            # Get all possible outputs for this input from File 2
            candidates = grouped_f2.get_group(input_vec)[out_cols].values
            
            # Calculate normalized squared differences
            # Distance = sum( ((out2 - out1) / range)^2 )
            diffs = (candidates - target_output)# / ranges.values
            dist_sq = np.sum(diffs**2, axis=1)
            
            # Find the index of the minimum distance
            nearest_idx = np.argmin(dist_sq)
            best_output = candidates[nearest_idx]
            
            # Combine nearest output with the input vector
            results.append(np.concatenate([best_output, input_vec]))
            
        except KeyError:
            # Save output from file 1
            results.append(np.concatenate([target_output, input_vec]))
            continue

    # Save to file
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, sep=' ', header=False, index=False)
    print(f"Done! Results saved to {output_path}")


if (len(sys.argv)>=5):
    find_nearest_outputs(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
