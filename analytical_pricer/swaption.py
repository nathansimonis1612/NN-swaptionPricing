import numpy as np


class Swaption(object):
    def __init__(self, optMat, swapMat, swapFreq, payerOrReceiver):
        assert payerOrReceiver in ["payer", "receiver"]
        self.optMat = optMat
        self.swapMat = swapMat
        self.swapFreq = swapFreq
        self.side = payerOrReceiver
        
        self.pmtsVec = np.arange(
            start = self.optMat + self.swapFreq,
            stop = self.optMat + self.swapMat + self.swapFreq,
            step = self.swapFreq
        )        
        self.tau = np.diff(np.r_[self.optMat, self.pmtsVec])
        
    def fwdRate(self, curve):
        """Simply compounded forward interest rate"""
        P0T = curve.P_M(self.optMat)
        P0S = curve.P_M(self.swapMat+self.optMat)
        return (1/(self.swapMat)) * ((P0T/P0S)-1)
    
    def swapRate(self, curve):
        """Forward swap rate"""
        P0T = curve.P_M(self.optMat)
        P0S = curve.P_M(self.swapMat+self.optMat)
        BPV = np.sum(self.tau * curve.P_M(self.pmtsVec))
        return (P0T-P0S)/BPV
        