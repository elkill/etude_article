#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:12:11 2023

@author: othman
"""
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
        self.step_ = opt.T_ / opt.nbTimeSteps_ 
        self.K_ = K
        self.m_ = 1
        self.Mat = np.zeros((self.opt_.nbTimeSteps_ + 1,nb_samples ))
        self.Lpath_  = np.empty(nb_samples, dtype=object)#= np.zeros((nb_samples,))
        self.SetUpPaths(self.Lpath_, self.opt_.size_, self.opt_.nbTimeSteps_, self.opt_.T_, self.Mat)
        #self.Base_ = self.generateFunctions(self.K_)
        #print(self.Mat)
        self.Base_ =self.gram_schmidt()
        #the two above are imutable for all c
        self.t_values = [self.step_ * i for i in range(opt.nbTimeSteps_+1)]
        #self.Yt_values = [self.Yt(t) for t in self]
        #the two above depend on c, should be computer for every c (using compute_integrals)
        self.Ut_integral = [0 for i in range(opt.nbTimeSteps_)]
        self.LambdaUYt_integral = [0 for i in range(opt.nbTimeSteps_)]
        
    def typo(self):
        print("tools")
    
    
    def SetUpPaths(self,Lpath,size,nbTimeSteps,T,Mat):
        print("setuppaths")
        #vectSt = np.zeros(size)
        C = np.full((size, size), self.mod_.rho_)
        for i in range(size):
            C[i,i] = 1
        C = np.linalg.cholesky(C)
        for l in range(self.nb_samples_):
            path = np.zeros((nbTimeSteps + 1, size))
            self.mod_.asset(C,path, T,self.step_,nbTimeSteps)
            for i in range(nbTimeSteps + 1):
                Mat[i][l] = path[i][0]
            #vectSt = path[nbTimeSteps, :]
            self.Lpath_[l] = path
        #del vectSt
        del C
        
    def evaluate(self,vector, x):
        res = 0
        for i in range(len(vector)):
            res += (x**i)*vector[i]
        return res
    

    def norm(self, vector, N, M, Mat):
        #print("norm")
        result = 0
        for n in range(N):
            for m in range(M):
                result += self.evaluate(vector, Mat[n][m])
        return result / (N * M)
    
    def generateFunctions(self, k: int) -> List[callable]:
        #print("generatefuncts")
        functions = []
        for i in range(k+1):
            functions.append(functools.partial(lambda x,il=i: x**il))
        return functions
    
    def SetUp(self):
        self.SetUpPaths(self.Lpath_, self.opt_.size_, self.opt_.nbTimeSteps_, self.opt_.T_)
        self.Base_ = self.generateFunctions(self.K_)

    def inner_product(self, u, v, matrice):
        #print("innerprod")
        # Use the polarization identity to compute the inner product
        u_plus_v_norm_sq = self.norm(u + v,self.opt_.nbTimeSteps_ + 1,self.nb_samples_, self.Mat)
        u_minus_v_norm_sq = self.norm(u - v,self.opt_.nbTimeSteps_ + 1,self.nb_samples_ , self.Mat)
        result = (u_plus_v_norm_sq - u_minus_v_norm_sq) / 4.0
        
        return result
    
    def gram_schmidt(self):
    
        A = np.identity(self.K_+1)
        Q = np.zeros_like(A)
        for i in range(A.shape[0]):
            # Take the i-th vector
            v = A[i]
            # Subtract the projection of v onto the span of the previous orthonormal vectors
            for j in range(i):
                v -= self.inner_product(Q[j], A[i], self.Mat) * Q[j]
            # Normalize the resulting vector
            Q[i] = v / self.norm(v,self.opt_.nbTimeSteps_ + 1,self.nb_samples_, self.Mat)
        return Q
    
    def Compute_integrals(self,c):
        for i in range(self.opt_.nbTimeSteps_):
            self.Ut_integral[i] = quad((lambda x,cl=c,Lam = self.Lambdat: Lam(x, cl)),self.t_values[i],self.t_values[i+1])[0]

            
         
    def Yt(self, t):
        #print("yt")
        vectSt = np.zeros(self.opt_.size_)
        t_index = int(t / self.step_)
        vectSt = self.Lpath_[self.m_][t_index, :]
        return np.exp(-t * self.mod_.r_) * self.opt_.payoffVect(vectSt)
    
    def Lambdat(self,t,c):
        #print("lambdat")
        vectSt = np.zeros(self.opt_.size_)
        t_index = int(t/self.step_)
        vectSt = self.Lpath_[self.m_-1][t_index, :]
        if self.opt_.payoffVect(vectSt) <= 0:
            return 0
        # Calcul de l'intérieur du polynôme
        vectStLOG = np.log(vectSt)
        p = 0
        for j in range(self.opt_.size_):
            for i in range(self.K_+1):
                p += c[i]*self.evaluate(self.Base_[:][i],(vectStLOG[j]))
        return np.exp(p)
    
    def Store_Lambdat(self,c):
        lambdat_values = [self.Lambdat(myt,c) for myt in self.t_values]
        self.lambdat_values = lambdat_values
    
    def Ut(self, t, c):
        self.Store_Lambdat(c)
        #print("ut")
        # calcul de l'intégrale
        #res = quad((lambda x,cl=c,Lam = self.Lambdat: Lam(x, cl)),0,t)[0]
        t_index = int(t / self.step_)
        res =  sum(self.Ut_integral[0:t_index+1])
        return np.exp(-res)
    
    def Phi(self, c):
        print("call phi")
        integ = 0
        for i in range (self.opt_.nbTimeSteps_):
            integ += quad((lambda x,cl=c, U = self.Ut, Y = self.Yt,Lam = self.Lambdat : Lam(x, cl)*U(x,cl)*Y(x)),i*self.step_, (i+1)*self.step_)[0]
        # Ut*Yt
        result = self.Ut(self.opt_.T_, c) * self.Yt(self.opt_.T_)
        # l'exp intgrl
        """
        n = 1000 # choisir n convenable
        dx = self.opt_.T_ / float(n)
        x = 0.0
        sum = 0.0
        for i in range(n):
            mid = (x + x + dx) / 2.0
            sum += self.Ut(mid, c) * self.Yt(mid) * self.Lambdat(mid, c) * dx
            x += dx
        """

        return integ + result
    
    def TheFct(self, c):
        print("call fct")
        result = 0.0
        while self.m_ < self.nb_samples_:
            result += self.Phi(c)
            self.m_ += 1
            print(self.m_)
        return result / float(self.nb_samples_)

 
    def TheFct_min(self, c):
        self.Compute_integrals(c)
        self.m_ = 0
        print("call fctmin")
        return - self.TheFct(c)
   
    def Optim(self):
        print("optim")
        bounds = [(None, None)] * (self.K_+1)
        print(bounds)
        c0 = [0]*(self.K_+1)  # point de départ initial
        print("avant minim")
        result = minimize(self.TheFct_min, x0=c0, bounds=bounds,method='L-BFGS-B')
        print("apres minim")
        print(result.fun)
        print(result.x)
        return result
        #result.fun : val min de la fct
        #result.x : val du vect de coeffs 
        
    def Price(self):
        res = self.Optim()
        return self.TheFct(res.x)

def main():
    print("todo")
    # votre code ici

if __name__ == '__main__':
    main()          



        

