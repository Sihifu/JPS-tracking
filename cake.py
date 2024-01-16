import numpy as np

def cake(radius):
    # Creates a circular kernel with ones
    cake = np.arange(-np.ceil(radius),np.ceil(radius)+1)**2
    cake = cake[:,None] + cake
    cake=cake<=radius**2
    return cake.astype(np.bool_)

