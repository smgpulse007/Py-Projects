#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:28:19 2019

@author: shaileshdudala
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


gusto_train = pd.read_csv("/Users/shaileshdudala/Downloads/splicejn_test_data.csv")
gusto_test = pd.read_csv("/Users/shaileshdudala/Downloads/splicejn_train_data.csv")

YG_train = gusto_train.iloc[:,0]
YG_test = gusto_test.iloc[:,0]

gusto_train = gusto_train.iloc[:,1:]
gusto_test = gusto_test.iloc[:,1:]