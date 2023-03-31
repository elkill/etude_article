#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 19:56:59 2023

@author: othman
"""

import math
import numpy as np

class Geometric:
    def __init__(self, T, degree, nbTimeSteps, size, strike):
        self.T_ = T
        self.nbTimeSteps_ = nbTimeSteps
        self.size_ = size
        self.strike_ = strike
        self.degree_ = degree
        
    def payoff(self, path, tau):
        St = path[tau, :]
        prod = np.prod(St)
        prod = math.pow(prod, 1/(float(self.size_)))
        prod -= self.strike_
        return - np.minimum(0, prod)
    
    def payoffVect(self, St):
        prod = np.prod(St)
        prod = math.pow(prod, 1/(float(self.size_)))
        prod -= self.strike_
        
        return - np.minimum(0, prod)
