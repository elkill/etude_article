#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:01:09 2023

@author: othman
"""

import math
import numpy as np

class BlackScholesModel:
    def __init__(self, size, rho, r, sigma, divid, spot):
        self.size_ = size
        self.rho_ = rho
        self.r_ = r
        self.sigma_ = np.zeros(size)
        self.sigma_[:] = sigma[:]
        self.divid_ = np.zeros(size)
        self.divid_[:] = divid[:]
        self.spot_ = np.zeros(size)
        self.spot_[:] = spot[:]
             
        
    def asset(self, C, path, T, step, nbTimeSteps):
        Gk = np.zeros(self.size_)
        L = np.zeros(self.size_)
        div = 0.0
        sigma = 0.0
        expon = 0.0
        value = 0.0

        path[0,:] = self.spot_
        for k in range(1, nbTimeSteps):
            Gk = np.random.normal(0, 1, self.size_)
            for d in range(self.size_):
                div = self.divid_[d]
                sigma = self.sigma_[d]
                L[:] = C[d,:]
                expon = math.exp((self.r_ - div - sigma*sigma/2.0)*step + sigma*math.sqrt(step)*np.dot(L, Gk))
                value = expon*path[k-1,d]
                path[k,d] = value

        del L, Gk

def main():
    print("BlackScholesModel")
    # votre code ici

if __name__ == '__main__':
    main()     
