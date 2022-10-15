import numbers
from math import sqrt, pow
import numpy as np


def euclidean_distance(a: (numbers.Real, numbers.Real), b: (numbers.Real, numbers.Real)):
    return sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2))


def manhattan_distance(a: (numbers.Real, numbers.Real), b: (numbers.Real, numbers.Real)):
    return b[0] - a[0] + b[1] - a[1]

