from bachelier import Bachelier
from curve import Curve
from hw1f import hullWhite
from swaption import Swaption
import numpy as np

#Toy example
maturities = np.array([7/360,180/360,1,2,3,5,10,15,20])
ytm = np.array([0.02,0.025, 0.03, 0.035, 0.0375, 0.04, 0.04, 0.04, 0.039])

zeroCurve = Curve(maturities, ytm)
swpn = Swaption(optMat=5, swapMat=20, swapFreq=0.5, payerOrReceiver="payer")

hw = hullWhite(a=1e-20, sigma=0.01, curve=zeroCurve)
b = Bachelier(zeroCurve, swpn, strike=0.01, sigma=0.01)

hwPrice = hw.swaptionPrice(swpn, strike=0.01)
bPrice = b.swaptionPrice()

print(f"HW price:{hwPrice:.4f} | Bachelier price:{bPrice:.4f}")
print(f"Implied Volatility:{b.impliedVolatility(bPrice):.4f}")