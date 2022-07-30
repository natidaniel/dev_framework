import numpy as np
import math


def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y