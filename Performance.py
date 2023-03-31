#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 19:58:11 2023

@author: othman
"""

import math
import numpy as np

class Performance:
    def __init__(self, T, degree, nbTimeSteps, size, strike, coeffs):
        self.T_ = T
        self.nbTimeSteps_ = nbTimeSteps
        self.size_ = size
        self.strike_ = strike
        self.degree_ = degree
        self.coeffs_ = np.zeros((size,))
        self.coeffs_[:] = coeffs.data
        
    def payoff(self, path, tau):
        res = 0
        for d in range(self.size_):
            trans = self.coeffs_[d] * path[tau,d]
            res = np.maximum(res, trans)
        res -= self.strike_
        return np.maximum(0, res)
    
    def payoffVect(self, St):
        res = 0
        for d in range(self.size_):
            trans = self.coeffs_[d] * St[d]
            res = np.maximum(res, trans)
        res -= self.strike_
        return np.maximum(0, res)

def main():
    print("Performance")
    # votre code ici

if __name__ == '__main__':
    main() 
