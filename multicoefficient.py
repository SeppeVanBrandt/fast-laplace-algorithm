import numpy as np
import scipy.stats as st
from pyDOE import lhs
import sys
from polynomial import *

class MultiCoefficient:
    def __init__(self, k_vec):
        self.mu = 0
        self.k_vec = k_vec
        self.gamma = 0
        self.multi_polynomial = None
        self.design_vector = None
        self.s = 0
        self.q = 0
        self.S = 0
        self.Q = 0

    def calc_gamma(self, l): #calculate gamma according to (34) in [1]
        if pow(self.q, 2) - self.s <= 0:
            return 0
        elif l == 0:
            return (pow(self.q,2)-self.s)/pow(self.s, 2)
        else:
            discriminant = pow(self.s+2*l,2)-4*l*(self.s-pow(self.q,2)+l)
            if discriminant < 0:
                discriminant = 0
            return (-self.s*(self.s+2*l)+self.s*np.sqrt(discriminant))/(2*l*pow(self.s,2))

    def calc_likelihood_increase(self, l): #calculate possible marginal likelihood increase according to (43) in [1]
        old_likely = 0.5*(np.log(1/(1+self.gamma*self.s)) + (pow(self.q,2)*self.gamma)/(1+self.gamma*self.s) - l*self.gamma)
        gamma_ = self.calc_gamma(l)
        new_likely = 0.5*(np.log(1/(1+gamma_*self.s)) + (pow(self.q,2)*gamma_)/(1+gamma_*self.s) - l*gamma_)
        return new_likely - old_likely

    def __repr__(self):
        return str(self.k_vec)+" "+str(self.mu)

    def __str__(self):
        return f'{self.k_vec}\t{self.gamma}\t{self.mu}\t{self.multi_polynomial}'
