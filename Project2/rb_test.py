#/usr/bin/python3


import random
import math
import sys

def rb_test(nums):
    rosenbrock = 0    
    for i in range(len(nums)-1):
        rosenbrock += (((1-nums[i])**2) + \
                       100*((nums[i+1] - \
                       (nums[i]**2))**2))
    return rosenbrock
