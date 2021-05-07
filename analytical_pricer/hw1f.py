import numpy as np
from scipy.stats import norm
import scipy.optimize as opt

class hullWhite(object):
    def __init__(self, a, sigma, curve):
        
        self.a = np.maximum(a, 1e-12)
        self.sigma = np.maximum(sigma, 1e-12)
        self.curve = curve       
        assert (self.a>0) & (self.sigma>=0)

    def _B(self, t, T):
        """B(t, T) part from B&M book in pricing pure discount bond"""
        return (1/self.a) * (1-np.exp(-self.a*(T-t)))

    def _A(self, t, T):
        """A(t, T) part from B&M book in pricing pure discount bond"""
        P0T = self.curve.P_M(T)
        P0t = self.curve.P_M(T)
        f0t = self.curve.instantFwd(t)
        par = self._B(t, T)*f0t\
        	-((self.sigma**2)/(4*self.a)*(1-np.exp(-2*self.a*t))*(self._B(t, T)**2))
        return P0T/P0t * np.exp(par)


    def _P(self, t, T, r):
        """Price of a pure discount bond: (3.39 from B&M book"""
        return self._A(t, T) * np.exp(-self._B(t, T) * r)

    def zbo(self, optMat, bondMat, strike, side):
        """European option on a ZCB: (3.40 & 3.41 from B&G book"""
        assert side in ["call","put"], "Must be call or put"

        P0T = self.curve.P_M(optMat)
        P0S = self.curve.P_M(bondMat)

        sigma_p = self.sigma * np.sqrt((1-np.exp(-2*self.a*(optMat))) / (2*self.a))*self._B(optMat, bondMat)
        h = (1/sigma_p) * np.log(P0S/(P0T*strike)) + (sigma_p/2)

        zbc = P0S * norm.cdf(h) - strike*P0T*norm.cdf(h-sigma_p)
        zbp = strike*P0T*norm.cdf(-h+sigma_p) - P0S*norm.cdf(-h)

        if side=="call":
            return zbc
        elif side=="put":
            return zbp

    def swaptionPrice(self, swaption, strike):
        """European swaption: (3.44 & 3.45 from B&M book)"""
        if swaption.side=="payer":
            optSide = "put"
        else:
            optSide = "call"

        # Year fraction from ti−1 to ti
        tau = np.diff(np.r_[swaption.optMat, swaption.pmtsVec])
        c = strike*tau
        c[-1] = c[-1]+1.0

        # r* that makes sum(c_i * P(T, t_i, r*)) = 1
        obj = lambda x: np.sum(c*self._P(swaption.optMat, swaption.pmtsVec, x)) - 1
        rStar = opt.brentq(obj, -0.8, 0.8, xtol=1e-12, rtol=1e-15, maxiter=100)
        X = self._P(swaption.optMat, swaption.pmtsVec, rStar)

        assert len(X) == len(tau) == len(swaption.pmtsVec)
        price = np.sum(c * self.zbo(swaption.optMat, swaption.pmtsVec, X, optSide))
        return np.maximum(price, 1e-15)






