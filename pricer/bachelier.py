import numpy as np
from scipy.stats import norm
import scipy.optimize as opt
from numba import njit

N = norm.cdf
n = norm.pdf

class Bachelier(object):
    def __init__(self, curve, swaption, strike, sigma=None):
        self.curve = curve
        self.swaption = swaption
        self.strike = strike
        self.sigma = sigma
    
    def swaptionPrice(self, sigma=None):
        """Swaption price under bachelier""" 
        if sigma!=None:
            self.sigma = sigma
        assert self.sigma != None, "Vol parameter missing"
        
        optSide = 1 if self.swaption.side == "payer" else -1
         
        BPV = np.sum(self.swaption.tau * self.curve.P_M(self.swaption.pmtsVec))        
        F =  self.swaption.swapRate(self.curve)

        x = (F - self.strike)
        v = self.sigma * np.sqrt(self.swaption.optMat)
        d = optSide * (x/v)
        
        price = BPV * v * (d*N(d) + n(d))
        return price
    
    def impliedVolatility(self, price):
        """Normal volatility given a swaption price"""
        obj = lambda x: self.swaptionPrice(x) - price
        sigma = opt.brentq(obj, -1e-10, 0.05, xtol=1e-12, rtol=1e-15, maxiter=1000)
        return sigma
        