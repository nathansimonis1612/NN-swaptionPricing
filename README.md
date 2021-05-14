# NN-swaptionPricing
In this repo, we analyse if deep neural networks are a viable way to speed-up the computations required to price swaptions.  This could be used in the process of calibrating a model.   

To generate synthetic data, open the **data** folder and launch the `generateData.py` script.  Currently this computes a grid of swaption prices under the hw1f model for different strikes and maturities.  From these prices, the bachelier implied volatility is calculated.  The input vector is a yield curve sampled from 10 years worth of historical ESTR curves, the hw1f vol and the tenor of the swap.  The implementations of the hw1f pricing and bachelier implied vol are in the **pricer** folder.

To run the NN and see the results, please see the `NN-impliedVol-2Dgrid.ipynb`file.  Please note that the weights of the NN have been saved and do not need to trained again. 

The `Swaption_NN.pdf` is a deck of slides that goes over the methodology and the results of the analysis.
