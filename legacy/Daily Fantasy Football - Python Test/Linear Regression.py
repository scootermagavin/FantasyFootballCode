#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 20:22:52 2022

@author: scotthoran
"""
import random
import pandas as pd
from pandas import DataFrame
import statsmodels.formula.api as smf
from os import path


# change this to your data directory
DATA100 = '/Users/scotthoran/OneDrive/Documents/Python_with_Baseball/ltcwbb-files-0.6.1/data/100-game-sample'

# this (*_nb) is a version that i made; if you saved your own version feel free
# to remove and use that
dfp = pd.read_csv(path.join(DATA100, 'pitches_w_inplay_nb.csv'))
dfb = pd.read_csv(path.join(DATA100, 'atbats.csv'))

# book picks up here

# first OLS
model = smf.ols(formula='inplay ~ mph + mph2', data=dfp)
results = model.fit()
results.summary2()

def prob_inplay(mph):
    b0, b1, b2 = results.params
    return (b0 + b1*mph + b2*(mph**2))

prob_inplay(85)
prob_inplay(90)
prob_inplay(95)
prob_inplay(98)

dfp['inplay_hat'] = results.predict(dfp)
dfp[['inplay', 'inplay_hat', 'mph']].sample(5)
