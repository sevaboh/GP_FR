#!/usr/bin/python
import numpy as np
import sys,math

def plot(da1,da2,p):
    import matplotlib.pyplot as plt
    # Create the figure and increase its size
    plt.figure(figsize=(12, 8))

    # Create the scatter plot
    if p==1:
        scatter = plt.scatter(da1, da2, cmap='viridis_r', s=50, edgecolor='k')
    if p==2:
        plt.plot(da1)
        plt.plot(da2)

    # Display the plot
    plt.show()
def compute_metrics(filename, idx, p,delimiter=None,r2a=0):
    # Load data
    try:
        data = np.genfromtxt(filename, delimiter=delimiter,filling_values=np.nan, invalid_raise=False)
    except:
        data = np.genfromtxt(filename, delimiter=delimiter,skiprows=1,filling_values=np.nan, invalid_raise=False)
    data = data[~np.isnan(data).any(axis=1)]

    if data[0,0]=="x1":
        data=data[1:,:]
    x = data[:, idx*2]
    if (2*idx+1 != data.shape[1]):
        y = data[:, idx*2+1]
    else:
        y = data[:, idx*2-1]

    # Mask zero reference values
    mask = x != 0
    if not np.any(mask):
        raise ValueError("All reference values y are zero.")

    x = x[mask]
    y = y[mask]

    # Median relative error
    median_rel_error = np.median(np.abs(x - y) / np.abs(x))

    # Relative L2 error
    rel_l2_error = np.linalg.norm(x - y) / np.linalg.norm(x)

    # R^2 coefficient
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((x - np.mean(x)) ** 2)

    if (r2a==0):
        if (ss_tot!=0.0):
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2=0
    else:
        r2=np.corrcoef(x,y)[0][1]
    if (r2<0):
        r2=0
    if math.isfinite(r2):
        
        res= {
        "median_relative_error": median_rel_error,
        "relative_L2_error": rel_l2_error,
        "R2": r2
        }
        print(i)
        for key, value in res.items():
            print(f"{key}: {value:.6e}")
        if p!=0:
            plot(x,y,p)
        return res


if __name__ == "__main__":
    filename = sys.argv[1]
    p=0
    if len(sys.argv)>=3:
        delimiter = sys.argv[2]
    else:
        delimiter = None
    if delimiter == "None":
        delimiter = None
    if len(sys.argv)>=4:
        p = int(sys.argv[3])
    try:
        data = np.genfromtxt(filename, delimiter=delimiter,filling_values=np.nan, invalid_raise=False)
    except:
        data = np.genfromtxt(filename, delimiter=delimiter,skiprows=1,filling_values=np.nan, invalid_raise=False)
    r2a=0
    if len(sys.argv)>=5:
        r2a = int(sys.argv[4])
    for i in range(int(data.shape[1]/2)+1):
        if (2*i!=data.shape[1]):
            results = compute_metrics(filename,i,p,delimiter,r2a)

