# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:20:40 2021

@author: Amirhosein
"""
from chefboost import Chefboost as chef
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#5 training dataset
df = pd.read_csv('ecgoutnorm.csv')
#config = {'enableGBM': True, 'epochs': 7, 'learning_rate': 1, 'max_depth': 5}
#config = {'enableRandomForest': True, 'num_of_trees': 5}
config = {'algorithm': 'ID3', 'enableRandomForest':True, 'num_of_trees': 20} #Set algorithm to ID3, C4.5, CART, CHAID or Regression
model = chef.fit(df, config = config)

