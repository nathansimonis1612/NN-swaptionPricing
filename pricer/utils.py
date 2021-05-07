import numpy as np 

def finiteDifference(self, plus, minus, h=0.0001):
    """Central-difference estimator"""
    return (plus-minus)/2*h

