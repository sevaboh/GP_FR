#!/usr/bin/python
import sys
import math

def parse_line(line):
    return list(map(float, line.strip().split()))

def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def find_closest_above_by_value(filename, target_value):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = [parse_line(line) for line in lines]

    # Find target line index
    target_index = None
    for i, row in enumerate(data):
        if row[0] == target_value:
            target_index = i
            break

    if target_index is None:
        return -1
    if target_index == 0:
        return -1

    target = data[target_index][1:10]

    min_dist = float('inf')
    best_value = None

    # Compare only with lines above
    for i in range(target_index):
        candidate = data[i][1:10]
        dist = euclidean_distance(candidate, target)

        if dist < min_dist:
            min_dist = dist
            best_value = data[i][0]

    return int(best_value)


# Example usage
filename = sys.argv[1]
target_value = int(sys.argv[2])
result = find_closest_above_by_value(filename, target_value)
print(result)