import numpy as np

from numpy import polyval

# Step function in respect to flow
# Units are m^3s^-1
def step(flow, cutoff=0.1, reward=1):
    return reward if flow > cutoff else 0


def parabola(flow, coeffs=[-400, 80, -3]):
    return np.polyval(coeffs, flow)


'''
 page 18 for the explanation 
 H: height threshold
 F: flow threshold
 height and depth is used interchangably?
'''
def paper3(flow, height, H, F, c=[2, 0.25, 1.5, 10, 3]):
    Hf = H * flow
    if height < Hf <= F:
        return c[0] - c[1]*height
    
    if height >= Hf and Hf <= F:
        return c[0] - c[2]*height
    
    if height < Hf and Hf > F:
        return -c[3]*flow - c[1]*height + c[4]
    
    if height >= Hf > F:
        return -c[3]*flow - c[2]*height + c[4]
