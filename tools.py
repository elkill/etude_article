#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:12:11 2023

@author: othman
"""
from math import sqrt, log, exp
from Basket import Basket
from BlackScholesModel import BlackScholesModel
import functools
from typing import List, Callable
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.integrate import quad

class tools:
    def __init__(self, bs, opt, nb_samples,K):
        self.mod_ = bs
        self.opt_ = opt
        
        
        self.nb_samples_ = nb_samples
        self.K_ = K
        
        self.step_ = opt.T_ / opt.nbTimeSteps_ 
        self.allPath_ = self.CreateAllPath()
        self.Base_ =self.gram_schmidt()
    
    
    def CreateAllPath(self):
        correl = np.zeros((1,1))
        correl[0][0] = 1
        allPath = np.empty(self.nb_samples_ , dtype=object)
        for i in range(self.nb_samples_):
            path = np.zeros((self.opt_.nbTimeSteps_, 1))
            self.mod_.asset(correl, path, self.opt_.T_,self.step_,self.opt_.nbTimeSteps_)
            allPath[i] = path
        return allPath
    
    def polynom_result(self, vector, x):
        res = 0
        for i in range(len(vector)):
            res += (x**i)*vector[i]
        return res
    
    def norm(self, vector, points):
        res = 0
        for i in range(len(points)):
            for j in range(len(points[0])):
                res += self.polynom_result(vector, log(points[i][j]))
        return sqrt(abs(res))
    
    def inner_product(self, u, v, matrice):
        #print("innerprod")
        # Use the polarization identity to compute the inner product
        u_plus_v_norm_sq = self.norm(u + v, matrice)
        u_minus_v_norm_sq = self.norm(u - v, matrice)
        result = (u_plus_v_norm_sq * u_plus_v_norm_sq - u_minus_v_norm_sq * u_minus_v_norm_sq) / 4.0
        return result
    
    def gram_schmidt(self):
        A = np.identity(self.K_ +1)
        Q = np.zeros_like(A)
        for i in range(A.shape[0]):
            # Take the i-th vector
            v = A[i]
            # Subtract the projection of v onto the span of the previous orthonormal vectors
            for j in range(i):
                v -= self.inner_product(Q[j], A[i], self.allPath_) * Q[j]
            # Normalize the resulting vector
            Q[i] = v / self.norm(v, self.allPath_)
        return Q
    
    def Yt(self, path, t):
        index = int(t//self.step_)
        return exp(-self.mod_.r_ * t) * self.opt_.payoffVect(path[index][:])

    def Lambdat(self, polynom, t, path):
        res = 0
        index = int(t//self.step_)
        rebased_polynom = np.dot(self.Base_, polynom)
        spot = path[index]
        p_x = self.polynom_result(rebased_polynom, log(spot))
        if(self.opt_.payoffVect(path[index][:]) > 0):
            res = exp(p_x)
        return res
    
    def Ut(self, polynom, t, path):
        integration = 0
        nb_step = 0
        while(nb_step*self.step_ < t):
            integration += self.Lambdat(polynom, nb_step*self.step_, path) * self.step_
            nb_step += 1
        integration += self.Lambdat(polynom, nb_step*self.step_, path) * (self.step_*nb_step - t)
        return exp(-integration)
    
    def phi(self, polynom, path):
        integration = 0
        for i in range(self.opt_.nbTimeSteps_):
            integration += quad(lambda t: self.Yt(path, t) * self.Ut(polynom, t, path) * self.Lambdat(polynom, t, path), i*self.step_, (i+1)*self.step_)[0]
        return integration + self.Yt(path, self.opt_.T_) * self.Ut(polynom, self.opt_.T_, path)
    
    def psy(self,polynom, allpath):
        self.x_appels.append(self.i)
        self.i += 1
        M = len(self.allPath_)
        res = 0
        for path in self.allPath_:
            res += self.phi(polynom, path)
        self.fx.append(res / M)
        return res / M
    
    def optimization(self):
        coefficients = [0]*(1+self.K_)
        bounds = [(None, None)] * (self.K_+1)
        result = minimize(lambda x: -self.psy(x, self.allPath_), x0=coefficients, bounds=bounds, method='L-BFGS-B')
        return result
    
    def Price(self):
        return - self.optimization().fun
    
   

def main():
    print("todo")
    # votre code ici

if __name__ == '__main__':
    main()          



        

