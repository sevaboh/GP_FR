This subdirectory contains the two pipelines for building and evaluating GP- and neural-network-based surrogate models (space-time pointwise and time-dependent) for a fPDE solver (`fractal-fract-PC`) that simulates mass transport in fractal-fractured porous media.

---

### `surrogate/` — Python Scripts

#### `nn2.py`
Trains and apply a PyTorch MLP surrogate model on the processed dataset (`res_pr.txt`). Supports direct mode (predicting fPDE outputs from parameters) and inverse mode (predicting parameters from fPDE outputs). Command-line arguments control the training fraction, noise level, output index, hidden layer size, and an optional pre-trained model file for applying the MLP to the dataset.

#### `normalize_all.py`
Computes column-wise min–max normalization ranges from a reference (full) dataset and applies them to a second input file, saving both the normalized data and the min/max ranges to disk. Used during training to ensure consistent scaling.

#### `normalize_from_file.py`
Applies pre-computed min–max ranges (loaded from a file) to normalize a new input dataset. Used to normalize parameter sets before passing them through a trained surrogate.

#### `denormalize_gprog.py`
Reverses min–max normalization on a single column of GP output predictions, converting results from normalized scale back to physical units using stored range values.

#### `gp_trees_clustering.py`
Performs hierarchical clustering of GP expression trees using tree edit distance (ZSS library). Parses the serialized tree format used by `gen_prog_test_v2.cpp`, computes a pairwise distance matrix, and optionally visualizes the result as a dendrogram to identify structurally similar discovered formulas.

#### `sobol_index.py`
Computes Sobol sensitivity indices for the direct surrogate problem (which of the 9 model input parameters most influences a given fPDE output). Fits a Random Forest surrogate on `res_pr.txt`, then uses SALib to compute first-order and total-order Sobol indices via Saltelli sampling.

#### `sobol_index_inv.py`
Same as `sobol_index.py` but for the inverse problem: computes how much variance in an input parameter is explained by different concentration values at various times and locations.

#### `stat.py`
Computes prediction quality metrics (R², and optionally RMSE) between predicted and true values from a two-column output file. Also supports scatter and line plotting for visual inspection of surrogate accuracy.

#### `poly_approx.py`
Fits a polynomial regression surrogate (via scikit-learn `PolynomialFeatures`) and converts the resulting polynomial into the Polish-notation tree format used by `gen_prog_test_v2.cpp`.

#### `pca_kern.py`
Performs Kernel PCA on the 9-dimensional input parameter space, reduces it to 2 components, and applies DBSCAN clustering. Used optionally for restricting the dataset while approximating the dependency between time-dependent GP surrogate's scalar parameters and the parameters of fPDE.

#### `solutions_nearest.py`
For each row in a solutions database, finds the closest matching row in a reference file (by Euclidean distance in output space after normalization), grouped by input configuration. Used to identify the most similar previously-computed solutions for warm-starting the surrogate.

#### `rows_nearest.py`
Given a target row index in a file, finds the nearest preceding row (by Euclidean distance in the 9-dimensional parameter space) and returns its index. Used to find parameter-space neighbors for sequential evaluation ordering.

#### `rows_sorting.py`
Sorts the rows of a dataset using a nearest-neighbor traversal (greedy + 2-opt improvement) through the 9-dimensional parameter space. Produces a path through the dataset that varies parameters smoothly.

#### `solutions_quality_check.py`
Cross-references three datasets: GP/polynomial regression solutions, previously found solutions database, and a reference R² table. For each input configuration where R² exceeds a given threshold, the prediction from the first file is used; otherwise - from the solutions database.

---

### `surrogate/` — Shell Scripts

#### `gen_prog_run`
The main shell wrapper for training and evaluating the point-wise GP surrogate for a single output column. It extracts the relevant column from `res_pr.txt`, splits data into training/testing sets, normalizes, runs `gen_prog_test_v2.cpp`, and denormalizes the predictions. Supports direct (forward) and inverse modes, as well as test-only evaluation using a previously trained model.

#### `gen_prog_run_t`
Top-level pipeline script for training time-series GP surrogates.

#### `gen_prog_run_t_stat`
Computes statistics on time-series GP surrogate predictions.

#### `gen_through_surrogate.sh`
Orchestrates generation of new dataset and their evaluation through the best available space-time pointwise surrogate (GP or NN, whichever achieves higher R² on testing data). Supports three modes: random parameter generation (results in a synthetic dataset), parameter set from training dataset used to build time-dependent surrogate (to compare the accuracy of time-dependent and point-wise surrogates), and parameters set generate through inverse surrogates (to test combined usage of inverse and direct point-wise surrogates).

#### `run_full_process.sh`
Top-level pipeline script for training space-time point-wise GP and NN surrogates. Runs five small-dataset training rounds, selects the best models, retrains on a large dataset for low-R² cases, generates a synthetic dataset, retrains again, and finally trains inverse surrogate models or restricted datasets.

#### `run_all_l0.sh`
Runs one full round of direct and inverse GP and NN training for all output columns on the "basic" dataset.

#### `run_all_l1.sh`
Same as `run_all_l0.sh` but operates on a synthetic dataset.

#### `run_big.sh`
Same as `run_all_l0.sh` but operates on a "large" dataset.

#### `run_gen_large.sh`
Generates a synthetic dataset by applying best GP or NN direct pointwise surrogate to random inputs calling `gen_through_surrogate.sh`.

#### `run_inv_test.sh`
Trains GP and NN surrogates for the inverse problem (recovering 9 physical parameters from fPDE output profiles) on restricted subsets of spatial/temporal observations, then evaluates prediction quality.

#### `generate_real_dataset.sh`
Runs the `fractal-fract-PC` solver 1000 times with randomly sampled parameters and collects solver output for multiple spatial points. Produces the raw training data for training space-time pointwise surrogates.

#### `generate_real_dataset_t.sh`
Same as `generate_real_dataset.sh` but records output at a single spatial coordinate (x = 0.3) over time, producing a dataset for traing time-dependent surrogates.

#### `apply_gen_prog_t.sh`
Applies the time-series GP surrogate (from `gp_t_series4/`) to a single parameter set. Outputs denormalized time-series predictions.

#### `compare_gen_prog.sh`
Comparison script for pointwise and time-dependent surrogates. Generates synthetic predictions through the pointwise surrogate, applies the time-dependent GP surrogate, and merges both into a joint comparison file.

#### `select_best.sh`
Across multiple training run directories, selects the GP formula and NN model achieving the highest R² for each output. Creates symlinks in a given directory, then evaluates the best ensemble.

#### `convert_txt.sh`
Converts raw training dataset generated by `generate_real_dataset.sh` into the format suitable for pointwise GP/NN surrogates training. Extracts parameter values and takes the logarithm of concentration and pressure outputs, then removes rows containing NaN or infinity.

#### `mixfiles.sh`
Merges two CSV files column-by-column (interleaves their columns), aligning rows by position. Used to combine predictions in `gen_through_surrogate.sh`.

---

### `surrogate/` — Configuration Files

#### `config_gprog.txt`
GP configuration for pointwise surrogate training.

#### `config_gprog2.txt`
GP configuration for time-dependent surrogate training.

#### `config_gprog2ga.txt`
Configuration for gradient descent+GA scalar parameters refining within time-dependent surrogate training.

#### `config_gprog3.txt`
GP configuration used for approximating the dependency between time-dependent GP surrogate's scalar parameters and the parameters of fPDE.

---

### `surrogate/gp_t_series*/` — Time-Series GP Results

Each of these four subdirectories (`gp_t_series2`, `gp_t_series3`, `gp_t_series4`, `gp_t_series5`) stores results from one GP experiment on time-series data:

- **`formulas.txt`** — The best symbolic formula found by GP, written in human-readable form with coefficient sub-expressions as a function of physical parameters.
- **`gp_t_stat.txt`** — Statistics table: parameter values and R² score for each evaluated dataset entry.
- **`best.png` / `worst.png`** — Plots comparing GP predictions vs. true values for the best and worst-performing cases.
- **`gp_t_series*.zip`** — Archive of all raw GP run outputs for that experiment.
- **`combined_gp_application.ods` / `predictions.ods`** — (in `gp_t_series4`) Spreadsheets summarizing combined surrogate application results and predictions across parameter sets.
- **`res_gp_compare_*.txt`** — (in `gp_t_series4`) Text files comparing GP surrogate predictions against solver reference values.

---

### `surrogate/const_v1/` and `surrogate/sin_v1/` — Point-wise GP and NN surrogate results for non-periodic (`const_v1`) and periodic (`sin_v1`) external pressures:

- **`best3_gp_mixed_results.ods` / `best3_nn_mixed_results.ods`** — Spreadsheets comparing the GP or the best GP or NN surrogate predictions against reference values.
- **`rest_gp_mixed_results.ods`**, **`rest_nn_mixed_results.ods`**, **`rest_t_gp_mixed_results.ods`**, **`rest_t_nn_mixed_results.ods`** — Spreadsheets comparing the GP or the best GP or NN surrogate predictions against reference values using inverse surrogate build upon the restricted dataset.
- **`r2_test.ods`** — Summary spreadsheet of R² scores on the stages of surrogates training.
- **`const_v1_1.tar.gz` / `const_v1_2.tar.gz`** (or `sin_v1_*.tar.gz`) — Compressed archives containing the full raw outputs.

---

### `surrogate/` — Dataset Text Files

#### `log3b.txt`, `log3c.txt`, `log3d.txt`, `log3e.txt`
Training datasets for time-dependent surrogates (solutions in the point x=0.3): `b` - one varying parameter, non-periodic external pressures; `c` - 3 varying parameters , non-periodic external pressures; `d` - all 9 varying parameters, non-periodic external pressures; `e` - all 9 varying parameters, periodic external pressures.

#### `res_const_0.txt`, `res_const_1.txt`
"Basic" and "large" datasets for training pointwise surrogates in the case of non-periodic external pressures.

#### `res_sin_0.txt`, `res_sin_1.txt`
"Basic" and "large" datasets for training pointwise surrogates in the case of periodic external pressures.
