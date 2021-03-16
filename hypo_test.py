#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:05:05 2021

@author: bprasad
"""
import numpy as np
import matplotlib as plt
from scipy.stats import norm
import statsmodels.stats.multitest as multi

def hypothesis_testing(w1, w2, w3):
# get the linear comination of weights after matrix multiplication
# get the z-score 
    w = np.matmul(np.matmul(w3, w2), w1)
    w = (w - np.mean(w))/np.std(w)

    # plot the hostogram for elements of w: a check for normality
    plt.pyplot.hist(x = w.flatten(), bins = 25, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.pyplot.xlabel('Z-score')
    plt.pyplot.ylabel('Frequency')

    # hypothesis testing
    w_p_value = np.empty((w.shape[0], w.shape[1]))

    for x in range(w.shape[0]):
        w_p_value[x, ] = 2 * (1 - norm.cdf(abs(w[x, ])))

    # multiple testing using B-H method.
    p_val_multi = multi.multipletests(w_p_value.flatten(), method = "fdr_bh", alpha = 0.05)
    p_val_corr = np.reshape(p_val_multi[1], (w.shape[0], w.shape[1]))
    
    return(w, p_val_corr)







