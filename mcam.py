#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 02:13:35 2023

@author: othman
"""
import sys
import json
from Basket import Basket
from BlackScholesModel import BlackScholesModel
from tools import tools
import numpy as np
from scipy.optimize import minimize

#hyperparametre K
K = 3
# Lecture du fichier .txt
with open('/home/othman/blm-master/dat/basket_d5.txt', 'r') as file:
    content = file.readlines()
# Initialisation des variables
rho = degree = T = size = dates = n_samples = 0
spot = sigma = divid = coeffs = None
type = ""
strike = 0.0
opt = None
c = False
# Extraction des paramètres
for line in content:
    ine = line.strip()
    if "correlation" in line:
        rho = float(line.split()[-1])
    elif "degree for polynomial regression" in line:
        degree = int(line.split()[-1])
    elif "option type" in line:
        type = line.split()[-1].strip()
    elif "maturity" in line:
        T = float(line.split()[-1])
    elif "model size" in line:
        size = int(line.split()[-1])
    elif "dates" in line:
        dates = int(line.split()[-1])
    elif "spot" in line:
        spot =[float(line.split()[-1])]
    elif "volatility" in line:
        sigma = [float(line.split()[-1])]
    elif "interest rate" in line:
        r = float(line.split()[-1])
    elif "dividend rate" in line:
        divid = [float(line.split()[-1])]
    elif "strike" in line:
        strike = float(line.split()[-1])
    elif "MC iterations" in line:
        n_samples = int(line.split()[-1])
    elif "payoff coefficients" in line:
        coeffs = [float(line.split()[-1])]
    if type == "exchange" or type == "basket":
        opt = Basket(T, degree, dates, size, strike, coeffs)
        c = True
        """
    elif type == "bestof":
        opt = Performance(T, degree, dates, size, strike, coeffs)
        c = True
    elif "geometric_put" in line:
        opt = Geometric(T, degree, dates, size, strike)
        """
opt.typo()
bs = BlackScholesModel(size,rho,r, sigma, divid, spot);
bs.typo()
tl = tools(bs, opt, n_samples,K)


print(tl.Optim())

def fun(x):
    print("call")
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

bnds = ((0, None), (0, None))
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},

        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},

        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds)
print("res")
print(res.x)
print(res.fun)
print(res)

print("toto")