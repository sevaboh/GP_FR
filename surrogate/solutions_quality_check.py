#!/usr/bin/python
import pandas as pd
import sys

def update_outputs(file1_path, file2_path, file3_path, n_outputs, threshold, output_file):
    # 1. Load the datasets with space separation
    # sep='\s+' handles one or more spaces/tabs
    df1 = pd.read_csv(file1_path, sep='\s+', header=None)
    df2 = pd.read_csv(file2_path, sep='\s+', header=None)
    df3 = pd.read_csv(file3_path, sep='\s+', header=None)

    # 2. Identify Input and Output Columns
    # File 1 & 2: First n columns are outputs, last 9 are inputs
    output_cols = df1.columns[:n_outputs].tolist()
    input_cols_f1f2 = df1.columns[-9:].tolist()
    
    # File 3: Columns 2-10 (index 1 to 9) are inputs, last is R2
    input_cols_f3 = df3.columns[1:10].tolist()
    r2_col = df3.columns[-1]

    # 3. Create a filtering mask from File 3
    # We find the input combinations where R2 exceeds the threshold
    valid_mask = df3[df3[r2_col] > threshold][input_cols_f3]
    print("R2>"+str(threshold)+" in "+str(len(valid_mask)))
    
    # Rename columns of the mask to match File 1/2 for easy merging
    valid_mask.columns = input_cols_f1f2

    # 4. Merge to get the "Better" values from File 1
    # This aligns File 1 data with the valid input sets found in File 3
    updates = pd.merge(valid_mask, df1, on=input_cols_f1f2, how='inner')

    # 5. Perform the replacement in File 2
    # Set inputs as index for both to align the update correctly
    df2.set_index(input_cols_f1f2, inplace=True)
    updates.set_index(input_cols_f1f2, inplace=True)

    # Replace the output columns in df2 with values from df1 (via updates)
    df2.update(updates[output_cols])

    # 6. Reset index and save as space-separated
    df2.reset_index(inplace=True)
    
    # Ensure columns are in original order: [Outputs..., Inputs...]
    final_cols = output_cols + input_cols_f1f2
    df2[final_cols].to_csv(output_file, sep=' ', index=False, header=None)

# Usage
update_outputs(
    file1_path=sys.argv[1],
    file2_path=sys.argv[2],
    file3_path=sys.argv[3],
    n_outputs=int(sys.argv[4]),       # Set your value for n
    threshold=float(sys.argv[5]),     # Set your R2 threshold
    output_file=sys.argv[6]
)
