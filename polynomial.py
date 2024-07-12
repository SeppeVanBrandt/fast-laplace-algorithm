import math as m
import copy
import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt

# This file defines the Polynomial and MultiPolynomial classes
# Additionally, classes are also defined for several special families of polynomials such as Hermite, Legendre, etc.

# For more information, see Chihara, Theodore S. An introduction to orthogonal polynomials, Chapter 4


class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __add__(self, other):
        temp = []
        coe1 = copy.deepcopy(self.coefficients)
        coe2 = copy.deepcopy(other.coefficients)
        num_zeros = len(coe1) - len(coe2)
        if num_zeros < 0:
            coe1.extend([0 for _ in range(-num_zeros)])
        elif num_zeros > 0:
            coe2.extend([0 for _ in range(num_zeros)])
        for i in range(len(coe1)):
            temp.append(coe1[i] + coe2[i])
        return Polynomial(temp)

    def __mul__(self, other):
        temp = [c * other for c in self.coefficients]
        return Polynomial(temp)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __call__(self, x):
        temp = self.coefficients[0]
        for i in range(1, len(self.coefficients)):
            temp += self.coefficients[i] * pow(x, i)
        return temp

def polynomial_mul_x(polynomial):
    coe1 = copy.deepcopy(polynomial.coefficients)
    coe1.insert(0, 0)
    return Polynomial(coe1)

def polynomial_mul(poly1, poly2):
    coe1 = copy.deepcopy(poly1.coefficients)
    coe2 = copy.deepcopy(poly2.coefficients)
    lst = []
    for i in range(len(coe2)):
        temp = coe2[i] * poly1
        for _ in range(i):
            temp = polynomial_mul_x(temp)
        lst.append(temp)
    result = Polynomial([0])
    for v in lst:
        result += v
    return result

def polynomial_derivative(polynomial):
    coe1 = []
    for i in range(1, len(polynomial.coefficients)):
        coe1.append(polynomial.coefficients[i] * D(i))
    return Polynomial(coe1)

def polynomial_integral(polynomial):
    coe1 = [D(0)]
    for i in range(len(polynomial.coefficients)):
        coe1.append(polynomial.coefficients[i] / D(i + 1))
    return Polynomial(coe1)

def build_polynomials(a_list, b_list):
    temp = []
    temp.append(Polynomial([1]))                # p_0
    temp.append(Polynomial([-a_list[0], 1]))   # p_1
    for k in range(2, len(a_list) + 1):
        temp.append(polynomial_mul_x(temp[k-1]) + (-1 * a_list[k-1] * temp[k-1]) + (-1 * b_list[k-1] * temp[k-2]))    # p_k
    return temp

def monic_polynomial(k):
    coe = [0] * k
    coe.append(1)
    return Polynomial(coe)

# Multivariate

def calc_k_vec_list(dim, P):
    def extend_list(lst1, lst2):
        temp = []
        for i in lst1:
            for j in lst2:
                temp.append(i + [j])
        return temp

    all_vals = range(0, P + 1)
    temp = [[i] for i in all_vals]
    for _ in range(1, dim):
        temp = extend_list(temp, all_vals)
    final_list = [i for i in temp if sum(i) <= P]
    return final_list

class MultiPolynomial:

    def __init__(self, polynomials, is_dict=False):
        if not is_dict:
            self.polynomials = polynomials

            def extend_dict(d, polynomial):
                temp = {}
                for key in d:
                    for j in range(len(polynomial.coefficients)):
                        if d[key] * polynomial.coefficients[j] != 0:
                            temp[key + (j,)] = d[key] * polynomial.coefficients[j]
                return temp

            self.terms = {(i,): polynomials[0].coefficients[i] for i in range(len(polynomials[0].coefficients))}
            for i in range(1, len(self.polynomials)):
                self.terms = extend_dict(self.terms, polynomials[i])
        else:
            self.polynomials = []
            self.terms = polynomials

    def __mul__(self, other):
        temp = {key: other * self.terms[key] for key in self.terms}
        return MultiPolynomial(temp, is_dict=True)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return ''.join(
            f'{"+{:.2e}".format(self.terms[key]) if self.terms[key] > 0 else "{:.2e}".format(self.terms[key])}*{key}'
            for key in self.terms
        )

    def __repr__(self):
        return self.__str__()

    def __call__(self, x_vec):
        temp = 0
        for key in self.terms:
            small_temp = self.terms[key]
            for i in range(len(key)):
                small_temp *= pow(x_vec[i], key[i])
            temp += small_temp
        return temp

def multi_polynomial_sum(multi_list):
    new_terms = {}
    for m in multi_list:
        for key in m.terms:
            new_terms[key] = new_terms.get(key, 0) + m.terms[key]
    return MultiPolynomial(new_terms, is_dict=True)

def multi_polynomial_mul(m1, m2):  # Works only if the number of dimensions are the same!
    new_terms = {}
    for key1 in m1.terms:
        for key2 in m2.terms:
            new_key = tuple(key1[k] + key2[k] for k in range(len(key1)))
            new_terms[new_key] = new_terms.get(new_key, 0) + m1.terms[key1] * m2.terms[key2]
    return MultiPolynomial(new_terms, is_dict=True)

def monic_multi_polynomial(k_vec):
    polynomials_list = [monic_polynomial(k) for k in k_vec]
    return MultiPolynomial(polynomials_list)

class Legendre:
    def __init__(self):
        def w_x(x):
            return 0.5 if -1 <= x <= 1 else 0

        def a(n):
            return 0

        def b(n):
            return 1 if n == 0 else pow(n, 2) / (4 * pow(n, 2) - 1)

        self.name = "leg"
        self.w_x = w_x
        self.a = a
        self.b = b

class Jacobi:
    def __init__(self, alpha, beta):
        B = sp.gamma(alpha) * sp.gamma(beta) / sp.gamma(alpha + beta)

        def w_x(x):
            if -1 < x < 1:
                return pow(x + 1, alpha - 1) * pow(1 - x, beta - 1) / (pow(2, alpha + beta - 1) * B)
            else:
                return 0

        def a(n):
            if alpha == beta:
                return 0
            else:
                beta_ = alpha - 1
                alpha_ = beta - 1
                num = pow(alpha_, 2) - pow(beta_, 2)
                denom = (alpha_ + beta_ + 2 * n) * (alpha_ + beta_ + 2 * n + 2)
                return -num / denom

        def b(n):
            if n == 0:
                return 1
            else:
                beta_ = alpha - 1
                alpha_ = beta - 1
                num = 4 * n * (alpha_ + n) * (beta_ + n) * (alpha_ + beta_ + n)
                denom = pow(alpha_ + beta_ + 2 * n, 2) * (alpha_ + beta_ + 2 * n - 1) * (alpha_ + beta_ + 2 * n + 1)
                return num / denom

        self.name = "jaco"
        self.w_x = w_x
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta

class Hermite:
    def __init__(self):
        def w_x(x):
            return pow(2 * m.pi, -0.5) * pow(m.e, -0.5 * pow(x, 2))

        def a(n):
            return 0

        def b(n):
            return n / 2

        self.name = "herm"
        self.w_x = w_x
        self.a = a
        self.b = b

class Laguerre:
    def __init__(self):
        def w_x(x):
            return pow(m.e, -x) if x > 0 else 0

        def a(n):
            return 2 * n + 1

        def b(n):
            return n * (n + 1)

        self.name = "lag"
        self.w_x = w_x
        self.a = a
        self.b = b

if __name__ == "__main__":

    pol = Polynomial([0, 1, 1])
    m2 = MultiPolynomial([pol, pol, pol], is_dict = False)
    m1 = monic_multi_polynomial([0, 2, 3])
    print(m1)
    print(m2)
    print(multi_polynomial_mul(m2, m2))
