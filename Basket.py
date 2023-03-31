import math
import numpy as np

class Basket:
    def __init__(self, T, degree, nbTimeSteps, size, strike, coeffs):
        self.T_ = T
        self.nbTimeSteps_ = nbTimeSteps
        self.size_ = size
        self.strike_ = strike
        self.degree_ = degree
        self.coeffs_ = np.zeros(size)
        self.coeffs_ = coeffs

    def typo(self):
        print("basket")
        
    def payoff(self, path, tau):
        St = np.zeros(self.size_)
        St[:] = path[tau,:]
        sum = np.dot(self.coeffs_, St)
        sum -= self.strike_
        return np.maximum(0, sum)

    def payoffVect(self, St):
        sum = np.dot(self.coeffs_, St)
        sum -= self.strike_

        return np.maximum(0, sum)

def main():
    print("todo")
    # votre code ici

if __name__ == '__main__':
    main()     