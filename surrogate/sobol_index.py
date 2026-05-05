#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load your CSV
# ---------------------------
df = pd.read_csv("res_pr.txt") # real data
#df = pd.read_csv("res_pr_generated_gprog_join.txt") # real data
X = df.iloc[:, 0:9].values # 9 input parameters
print(X)
y = df.iloc[:, 26].values # specific concentration value (index 11 - start, next - +3)
print(y)
# ---------------------------
# 2. Fit surrogate model
# ---------------------------

print(1)
model = RandomForestRegressor(
    n_estimators=800,
    n_jobs=-1,
    random_state=0
)
model.fit(X, y)
print(2)
# ---------------------------
# 3. Define problem for SALib
# ---------------------------
problem = {
    'num_vars': 9,
    'names': ['uinp', 'cdur','cinp', 'D', 'kmu','a','b','ae','tau_r'],
    # IMPORTANT: put the true ranges of your data here
    'bounds': [
        [df.x1.min(), df.x1.max()],
        [df.x2.min(), df.x2.max()],
        [df.x3.min(), df.x3.max()],
        [df.x4.min(), df.x4.max()],
        [df.x5.min(), df.x5.max()],
        [df.x6.min(), df.x6.max()],
        [df.x7.min(), df.x7.max()],
        [df.x8.min(), df.x8.max()],
        [df.x9.min(), df.x9.max()]
    ]
}
print(3)
# ---------------------------
# 4. Generate Saltelli samples
# ---------------------------
N = 4096  # base sample size (increase for accuracy)
param_values = saltelli.sample(problem, N, calc_second_order=True)
print(4)
# ---------------------------
# 5. Evaluate surrogate on Saltelli samples
# ---------------------------
y_pred = model.predict(param_values)
print(5)
# ---------------------------
# 6. Sobol sensitivity analysis
# ---------------------------
Si = sobol.analyze(problem, y_pred, calc_second_order=True, print_to_console=True)
print(6)
print("\nFirst-order indices:", Si['S1'])
print("Total-order indices:", Si['ST'])
print("Second-order indices:\n", Si['S2'])

# Extract first-order and total-order indices
S1 = Si["S1"]
ST = Si["ST"]
labels = problem["names"]

# Sort indices by total effect (largest to smallest)
sorted_idx = np.argsort(ST)[::-1]
labels_sorted = [labels[i] for i in sorted_idx]
S1_sorted = S1[sorted_idx]
ST_sorted = ST[sorted_idx]

# ----- 5. TORNADO PLOT -----
plt.figure(figsize=(8, 4))
y_pos = np.arange(len(labels))

plt.barh(y_pos, ST_sorted, edgecolor='black', alpha=0.7, label="ST (Total Effect)")
plt.barh(y_pos, S1_sorted, edgecolor='black', alpha=0.9, label="S1 (First Order)")

plt.yticks(y_pos, labels_sorted)
plt.xlabel("Sobol Index Value")
plt.legend()
plt.gca().invert_yaxis()  # Highest on top
plt.tight_layout()
plt.show()
