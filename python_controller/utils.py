import numpy as np

# Normalize phi to be between -pi and pi
def norm_angle(phi: float) -> float:
    return np.angle(np.exp(1j * phi))

def stride(i, n, k = None, p = None):
    if (k is None and p is None):
        return slice(i * n, (i + 1) * n)
    else:
        return (stride(i, n), stride(k, p))

def stridesq(i, n):
    return (stride(i, n), stride(i, n))