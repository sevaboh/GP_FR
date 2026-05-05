#!/usr/bin/python
import sys
import math

def dist(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

fname = sys.argv[1]

rows = []
vecs = []

with open(fname) as f:
    for line in f:
        parts = line.strip().split()
        rows.append(line.rstrip())
        vecs.append([float(parts[i]) for i in range(1,10)])  # columns 2–10

n = len(vecs)

# --- distance matrix ---
D = [[0]*n for _ in range(n)]
for i in range(n):
    for j in range(i+1,n):
        d = dist(vecs[i], vecs[j])
        D[i][j] = D[j][i] = d

# --- nearest neighbor path (start from row 0) ---
visited = [False]*n
visited[0] = True
path = [0]

for _ in range(n-1):
    last = path[-1]
    best = None
    best_d = float("inf")
    for i in range(n):
        if not visited[i] and D[last][i] < best_d:
            best_d = D[last][i]
            best = i
    visited[best] = True
    path.append(best)

# --- 2-opt improvement ---
improved = True
while improved:
    improved = False
    for i in range(1, n-2):
        for j in range(i+1, n-1):
            a, b = path[i-1], path[i]
            c, d = path[j], path[j+1]

            if D[a][c] + D[b][d] < D[a][b] + D[c][d]:
                path[i:j+1] = reversed(path[i:j+1])
                improved = True

# --- output reordered rows ---
for i in path:
    print(rows[i])

