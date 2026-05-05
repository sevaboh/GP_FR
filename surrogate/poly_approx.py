#!/usr/bin/python
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import re

def build_term_polish(coeff, powers, feature_names):
    """
    Builds a Polish notation string for a single term: coeff * (x_i * x_j * ...)
    """
    # Start with the constant coefficient: 4 {coeff} 1
    term_str = f"4 {coeff} 1"
    
    # Powers is an array of exponents for each input (e.g., [2, 0, 1, ...])
    for i, p in enumerate(powers):
        for _ in range(int(p)):
            # Wrap the existing expression in a multiplication: 1 1 1
            # Current expression * Input (6 {i} 1)
            term_str = f"1 1 1\n{term_str}\n6 {i} 1"
    return term_str

def denormalize_coefficients(coef, intercept, powers, scaler_y):
    coef_denorm = []
    intercept_denorm = intercept

    for i, c in enumerate(coef):
        coef_denorm.append(c)

    coef_denorm = np.array(coef_denorm)

    # Adjust for output scaling
    coef_denorm *= scaler_y.scale_[0]
    intercept_denorm = intercept_denorm * scaler_y.scale_[0] + scaler_y.mean_[0]

    return coef_denorm, intercept_denorm

def relative_error_score(y_true, y_pred, threshold=0.01):
    eps = 1e-12  # avoid division by zero
    rel_err = np.log(1.0+np.abs(y_pred - y_true))
    return np.sum(rel_err)

def objective(params, X_poly, y_true):
    intercept = params[0]
    coefs = params[1:]
    y_pred = X_poly @ coefs + intercept
    return relative_error_score(y_true, y_pred)

datafile=sys.argv[1]
degree_=int(sys.argv[2])

# 1. Load the data
# Assuming data is space-separated
df = pd.read_csv(datafile, sep='\s+', header=None)

# 2. Extract inputs (Columns 2-10 -> Indices 1 to 9)
X = df.iloc[:, 1:10].values

# 3. Identify potential output columns
# We exclude the input range and look at everything else
all_indices = set(range(df.shape[1]))
input_indices = set(range(1, 10))
output_candidates = list(all_indices - input_indices)

# 4. Set up Polynomial Features
poly = PolynomialFeatures(degree=degree_)
X_poly = poly.fit_transform(X)
power_matrix = poly.powers_

results = {}

for col in output_candidates:
    y = df.iloc[:, col].values
    
    # Check if the column is a "meaningless fixed value" (standard deviation is 0)
    if np.std(y) == 0:
        continue

    # Normalize output
    scaler_y = StandardScaler()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    # 5. Fit the model
    model = LinearRegression()
    model.fit(X_poly, y_norm)
    # Initial params from linear regression
    init_params = np.concatenate([[model.intercept_], model.coef_])

    res = minimize(
        objective,
        init_params,
        args=(X_poly, y_norm),
        method='Powell',   # derivative-free → important
        options={'maxiter': 200}
    )
    opt_params = res.x
    intercept_norm = opt_params[0]
    coef_norm = opt_params[1:]
    r1= X_poly @ model.coef_ + model.intercept_
    r2= X_poly @ coef_norm + intercept_norm

    model.coef_=coef_norm
    model.intercept_=intercept_norm
    
    coef_denorm, intercept_denorm = denormalize_coefficients(coef_norm,intercept_norm,power_matrix, scaler_y)
    # Store coefficients and the score (R^2)
    results[col] = {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "score": model.score(X_poly, y_norm)
    }
    if col!=0:
        print(f"Column {col} processed. R^2 Score: {results[col]['score']:.4f}")
#        print(results[col]['coefficients'])
        with open('best_c'+str(col+1)+'.txt', 'w') as f:
            active_terms=[]
            # Add intercept (the "4 c 1" term)
            if abs(model.intercept_) > 1e-9:
                active_terms.append(f"4 {intercept_denorm} 1")
            # Process each polynomial combination
            for i, coeff in enumerate(coef_denorm):
                if abs(coeff) > 1e-9:
                    # power_matrix[i] is an array like [1, 0, 1, 0...] 
                    # meaning this coefficient belongs to x1 * x3
                    term_polish = build_term_polish(coeff, power_matrix[i], None)
                    active_terms.append(term_polish)
            # Chain all terms with "0 1 1" (Addition)
            if active_terms:
                final_expression = active_terms[0]
                for i in range(1, len(active_terms)):
                    final_expression = f"0 1 1\n{final_expression}\n{active_terms[i]}"
            f.write(final_expression + "\n\n")
            f.close()