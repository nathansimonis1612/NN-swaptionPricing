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


def SwaptionPrice1d(grid, shuffRates):
    """Create dataset to make point-wise predictions of the swaption price"""
    df = []

    for params, ytm in tqdm(zip(grid.values, shuffRates.values)):   
        sigma, T, S, chgX, side = params.ravel()
        curve = Curve(maturities, ytm)
        swpn = Swaption(T, S, 0.5, side) 
        atmRate = swpn.fwdRate(curve)

        model = hullWhite(0.05, sigma, curve=curve)
        price = model.swaptionPrice(swpn, strike=atmRate+chgX)

        data = dict(zip(str_maturities, ytm))
        data = {**data, **dict(zip(grid.keys(), params))}
        data["atmRate"] = atmRate
        data["strike"] = atmRate + chgX
        data["swaptionPrice"] = price    
        df.append(data)   
    df = pd.DataFrame(df)
    df.to_csv("data/training.csv", index=False)


def SwaptionPrice3d(combinations, shuffRates):
    """Create dataset for swaption prices in 3D (strike, optMat, swapMat)"""
    # Generate hw1f vol parameter. 
    simSigma = st.skewnorm.rvs(a=4, loc=0.005, scale=0.01, size=budget)
    simSigma = np.maximum(1e-20, simSigma)
    
    df = []
    priceMatrix = []

    for sigma, ytm in tqdm(zip(simSigma, shuffRates.values)):
        curve = Curve(maturities, ytm)
        priceVect = np.zeros(len(combinations))
        
        for i, comb in enumerate(combinations):
            chgX, T, S  = comb.values()
            swpn = Swaption(T, S, swapFreq=0.5, payerOrReceiver="payer") # Only consider payer swaptions.
            atmStrike = swpn.swapRate(curve)
            model = hullWhite(0.05, sigma, curve=curve)
            priceVect[i] = model.swaptionPrice(swpn, strike=atmStrike+chgX)

        data = dict(zip(str_maturities, ytm))
        data["sigma"] = sigma
        
        priceMatrix.append(priceVect)
        df.append(data)

    X = pd.DataFrame(df)
    y = pd.DataFrame(priceMatrix, columns=[str(comb) for comb in combinations])
    X.to_csv("data/pricingVector-input.csv", index=False)
    y.to_csv("data/pricingVector-output", index=False)


def SwaptionPrice2d(combinations, shuffRates):
    """Create dataset for swaption prices in 2D (strike, optMat)"""
    # Generate hw1f vol parameter and different swap maturities. 
    simSigma = st.skewnorm.rvs(a=4, loc=0.005, scale=0.01, size=budget)
    simSigma = np.maximum(1e-20, simSigma)
    simSwap = np.random.choice([2, 5, 10, 15, 20, 30], size=(budget,))
    
    # Shuffle historical rates at random so we do not have to sample inside the loop.
    shuffRates = histCurves.copy(deep=True).sample(budget, replace=True)

    df = []
    priceMatrix = []

    for sigma, S, ytm in tqdm(zip(simSigma, simSwap, shuffRates.values)):
        curve = Curve(maturities, ytm)
        priceVect = np.zeros(len(combinations))
        
        for i, comb in enumerate(combinations):
            chgX, T = comb.values()
            swpn = Swaption(T, S, swapFreq=0.5, payerOrReceiver="payer") # Only consider payer swaptions.
            atmStrike = swpn.swapRate(curve)
            model = hullWhite(0.05, sigma, curve=curve)
            priceVect[i] = model.swaptionPrice(swpn, strike=atmStrike+chgX)

        data = dict(zip(str_maturities, ytm))
        data["sigma"] = sigma
        data["swapMat"] = S
        
        priceMatrix.append(priceVect)
        df.append(data)

    X = pd.DataFrame(df)
    y = pd.DataFrame(priceMatrix, columns=[str(comb) for comb in combinations])
    X.to_csv("data/2DpricingVector-input.csv", index=False)
    y.to_csv("data/2DpricingVector-output.csv", index=False)
        
        
        
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

    # Grid for point-wise predictions
    grid = pd.DataFrame({
        "sigma":st.skewnorm.rvs(a=4, loc=0.005, scale=0.01, size=budget),
        #"sigma":np.random.uniform(0, 0.04, (budget,)),
        "optMat":np.random.uniform(90/360, 15, (budget,)),
        "swapMat":np.random.choice([2, 5, 10, 15, 20, 30], size=(budget,)),
        "chgStrike":np.random.uniform(-0.02, 0.02, (budget,)),
        "side":np.random.choice(["payer", "receiver"], (budget))                                  
    })
    

    shuffRates = histCurves.copy(deep=True).sample(budget, replace=True)

    # Call function
    #SwaptionPrice1d(grid, shuffRates)
    
    
    # =============================================================================
    grid = {
    'chgStrike' : [-0.02, -0.01, 0., 0.01, 0.02],
    'optMat' : [1,2,5,10,20],
    'swapMat' : [2, 5, 10, 20, 30]
    }
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    #SwaptionPrice3d(combinations, shuffRates)
    
    
    # =============================================================================
    
    grid = {
    'chgStrike' : np.arange(-0.02, 0.025, step=0.005),
    'optMat' : [0.5, 1, 2, 5, 10, 15, 20, 30],
    }
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    SwaptionPrice2d(combinations, shuffRates)