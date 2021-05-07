import numpy as np
from scipy.stats import norm
import scipy.optimize as opt
from numba import njit

N = norm.cdf
n = norm.pdf

class Bachelier(object):
    def __init__(self, curve, swaption, strike):
        self.curve = curve
        self.swaption = swaption
        self.strike = strike
    
    def swaptionPrice(self, sigma):
        """Swaption price under bachelier"""  
        optSide = 1 if self.swaption.side == "payer" else -1
         
        BPV = np.sum(self.swaption.tau * self.curve.P_M(self.swaption.pmtsVec))        
        F =  self.swaption.swapRate(self.curve)

        x = (F - self.strike)
        v = sigma * np.sqrt(self.swaption.optMat)
        d = optSide * (x/v)
        
        price = BPV * v * (d*N(d) + n(d))
        return price
    
    def impliedVolatility(self, price):
        """Normal volatility given a swaption price"""
        obj = lambda x: self.swaptionPrice(x) - price
        sigma = opt.brentq(obj, -1e-10, 0.05, xtol=1e-12, rtol=1e-15, maxiter=1000)
        return sigma
        