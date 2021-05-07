import numpy as np

class Curve(object):
    def __init__(self, maturities, yields):
        self.maturities = maturities
        self.yields = yields
        
    def P_M(self, T):
        """Discount factor from the market with maturity T"""
        if hasattr(T, "__len__"):
            T = np.where(T<0, 1.0, T)
        
        rate = np.interp(T, self.maturities, self.yields)
        return np.exp(-T * rate)
    
    def instantFwd(self, T, h=0.0001):
        """Instantaneous fwd interest rate"""
        der = (np.log(self.P_M(T+h))-np.log(self.P_M(T-h)))/2*h
        return -der