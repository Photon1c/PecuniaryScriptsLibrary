# utils/vectors.py
import math

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def lerp(a, b, t):
    return a + (b - a) * t

def dist1d(a, b):
    return abs(a - b)

def sign(x):
    return 1 if x >= 0 else -1
