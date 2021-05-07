"""Script used to generate swaption prices"""

import os
os.chdir("..")

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as st
from itertools import product
import utils
from pricer.curve import Curve
from pricer.hw1f import hullWhite
from pricer.swaption import Swaption
from pricer.bachelier import Bachelier


def SwaptionPrice2d(combinations, shuffRates):
    """Create dataset for swaption prices in 2D (strike, optMat)"""
    # Generate hw1f vol parameter and different swap maturities. 
    simSigma = st.skewnorm.rvs(a=4, loc=0.005, scale=0.01, size=budget)
    simSigma = np.maximum(1e-20, simSigma)
    
    simSwap = np.random.choice([2, 5, 10, 15, 20, 30], size=(budget,))
    
    df = []
    priceMatrix = []
    volMatrix = []

    for sigma, S, ytm in tqdm(zip(simSigma, simSwap, shuffRates.values)):
        # Curve object
        curve = Curve(maturities, ytm)
        # Init swaption prices and vector
        priceVect = np.zeros(len(combinations))
        volVect = np.zeros(len(combinations))
        
        for i, comb in enumerate(combinations):
            chgX, T = comb.values()
            swpn = Swaption(T, S, swapFreq=0.5, payerOrReceiver="payer") # Only consider payer swaptions.
            atmStrike = swpn.swapRate(curve)
            hw1f = hullWhite(0.05, sigma, curve=curve)
            bach = Bachelier(curve, swpn, atmStrike+chgX)
            
            priceVect[i] = hw1f.swaptionPrice(swpn, strike=atmStrike+chgX)
            volVect[i] = bach.impliedVolatility(priceVect[i])

        data = dict(zip(str_maturities, ytm))
        data["sigma"] = sigma
        data["swapMat"] = S
        
        priceMatrix.append(priceVect)
        volMatrix.append(volVect)
        df.append(data)

    X = pd.DataFrame(df)
    y1 = pd.DataFrame(priceMatrix, columns=[str(comb) for comb in combinations])
    y2 = pd.DataFrmae(volMatrix, columns=[str(comb) for comb in combinations])
    X.to_csv("data/2DpricingVector-input.csv", index=False)
    y1.to_csv("data/2DpricingVector-output.csv", index=False)
    y2.to_csv("data/2DvolVector-output.csv", index=False)
        
        
        
if __name__ == "__main__":
    budget = int(1e6)
    # Import historical curves

    histCurves = utils.import_data("data/ESTR_historical.xlsx", sheet="rates")
    histCurves.dropna(inplace=True)
    histCurves = histCurves.astype(np.float)
    histCurves *= 0.01
    maturities = np.array([    
        7/360, 30/360, 90/360, 180/360, 1, 3, 5, 10, 15, 20, 30, 50
    ])
    str_maturities = list(map(lambda x: str(x)+" yrs", list(np.around(maturities,4))))
    histCurves.columns = str_maturities
    
    # =============================================================================


    # Randomly sample budget curves from 
    shuffRates = histCurves.copy(deep=True).sample(budget, replace=True)

    # =============================================================================
    # Grid for 2D curve
    grid = {
    'chgStrike' : np.arange(-0.02, 0.025, step=0.005),
    'optMat' : [0.5, 1, 2, 5, 10, 15, 20, 30],
    }
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    SwaptionPrice2d(combinations, shuffRates)