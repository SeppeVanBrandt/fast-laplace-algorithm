import numpy as np
import matplotlib.pyplot as plt
from multicoefficient import *

import time

#This code implements the Fast Laplace Algorithm as it is described in
#[1] S. D. Babacan, R. Molina and A. K. Katsaggelos, "Bayesian Compressive Sensing Using Laplace Priors,"
#[2] Tipping, Michael E., and Anita C. Faul. "Fast marginal likelihood maximisation for sparse Bayesian models."


class FastLaplaceAlgorithm:
    def __init__(self, P, dim, q, k_vec_basis, experimental_design, y_vec, type_list, name, max_iterations = 300, output=True):
        self.output = output
        self.max_iterations = max_iterations
        self.name = name
        self.M = len(experimental_design)
        self.N = len(k_vec_basis)
        self.y_vec = y_vec
        self.experimental_design = experimental_design
        self.uni_polynomials = []
#Construct univariable polynomials
        for d in range(dim):
            a_list = [type_list[d].a(i) for i in range(P+1)]
            b_list = [type_list[d].b(i) for i in range(P+1)]
            self.uni_polynomials.append(build_polynomials(a_list, b_list))
#Construct coefficients
        self.coefficients = [MultiCoefficient(k_vec) for k_vec in k_vec_basis]
        for c in self.coefficients:
            c.multi_polynomial = MultiPolynomial([self.uni_polynomials[i][c.k_vec[i]] for i in range(dim)])
#Construct full design matrix containing ALL basis functions
        self.full_design_matrix = np.zeros((self.M, self.N))
        for n in range(self.N):
            self.coefficients[n].design_vector = np.zeros((self.M, 1))
            for m in range(self.M):
                self.coefficients[n].design_vector[m] = self.coefficients[n].multi_polynomial(self.experimental_design[m])
                self.full_design_matrix[m,n] = self.coefficients[n].multi_polynomial(self.experimental_design[m])
        self.h_vec = np.diag(self.full_design_matrix@np.linalg.inv(self.full_design_matrix.T@self.full_design_matrix)@self.full_design_matrix.T)
        self.l = 0
        self.nu = 0 #Set nu to 0 for the entire simulation
        #self.beta = 1e5
        self.beta = (0.01*pow(np.linalg.norm(y_vec),2)) #Estimate that is suggested in [1]
        self.current_Sigma_matrix = None
        self.current_design_matrix = None
        self.final_multi_polynomial = None

    def y_and_design_to_file(self): #print y vector and design matrix to a file to use later
        filey = open("test_data/y_test.txt","w")
        filed = open("test_data/design_matrix_test.txt","w")
        for y in self.y_vec:
            filey.write(f"{y}\n")
        for m in range(self.M):
            for n in range(self.N-1):
                filed.write(f"{self.full_design_matrix[m,n]}\t")
            filed.write(f"{self.full_design_matrix[m,self.N-1]}\n")

    def final_polynomial_to_file(self): #print the final polynomial to a file
        file = open(f"test_data/poly_{self.name}.txt","w")
        for key in self.final_multi_polynomial.termen:
            tekst = f'{self.final_multi_polynomial.termen[key]}\t'
            for i in key:
                tekst += f'{i}\t'
            tekst += '\n'
            file.write(tekst)

    def calc_mu(self): #Calculate mu vector according to (24) in [1]
        coe_list = []
        for coe in self.coefficients:
            if coe.gamma != 0:
                coe_list.append(coe)
        for i in range(len(coe_list)):
            coe_list[i].mu = (self.beta*self.current_Sigma_matrix@self.current_design_matrix.T@self.y_vec)[i]
        return 0

    def calc_Sigma_matrix(self): #Calculate Sigma matrix according to (25) in [1]
        inv_gamma_vec = []
        for coe in self.coefficients:
            if coe.gamma != 0:
                inv_gamma_vec.append(1/coe.gamma)
        self.current_Sigma_matrix = np.linalg.inv(self.beta*self.current_design_matrix.T@self.current_design_matrix+np.diag(inv_gamma_vec))

    def calc_l(self): #Update l according to (35) in [1]
        gamma_som = 0
        N_t = 0
        for coe in self.coefficients:
            gamma_som += coe.gamma
            if coe.gamma !=0: N_t += 1
        self.l = (N_t - 1)/(0.5*gamma_som)

    def update_beta(self): #Update beta according to (36) in [1]
        w_vec = []
        m_teller = 0
        gamma_list = []
        for coe in self.coefficients:
            if coe.gamma != 0:
                w_vec.append(coe.mu)
                m_teller += 1
                gamma_list.append(coe.gamma)
        w_vec = np.asarray(w_vec)
        delta = pow(np.linalg.norm(self.current_design_matrix@w_vec - self.y_vec), 2)
        som = 0
        for m in range(len(gamma_list)):
            som += self.current_Sigma_matrix[m, m]*gamma_list[m]
        if (self.N - m_teller + som)/verschil > 1e5:    #Upper limit to beta
            self.beta = 1e5
        else:
            self.beta = (self.N - m_teller + som)/delta

    def calc_current_design_matrix(self): #Build design matrix from basis functions with non-zero coefficients
        check = False
        for coe in self.coefficients:
            if coe.gamma != 0:
                if check == False:
                    temp_design_matrix = coe.design_vector
                    check = True
                else:
                    temp_design_matrix = np.concatenate((temp_design_matrix, coe.design_vector), axis=1)
        self.current_design_matrix = temp_design_matrix

    def calc_s_q(self): #Update s_i and q_i according to (51-54) in [1]
        for coe in self.coefficients:
            coe.S = (self.beta*coe.design_vector.T@coe.design_vector - pow(self.beta,2)*coe.design_vector.T@self.current_design_matrix@self.current_Sigma_matrix@self.current_design_matrix.T@coe.design_vector).item()
            coe.Q = (self.beta*coe.design_vector.T@self.y_vec - pow(self.beta,2)*coe.design_vector.T@self.current_design_matrix@self.current_Sigma_matrix@self.current_design_matrix.T@self.y_vec).item()
            coe.s = coe.S/(1-coe.gamma*coe.S)
            coe.q = coe.Q/(1-coe.gamma*coe.S)

    def update_Sigma_matrix(self, keyword, current_coe, last_gamma): #Update Sigma matrix according to (28) in [2] (doesn't get used)
        if keyword == "add        ":
            sigma_ii = pow(1/current_coe.gamma + current_coe.S, -1)
            left_upper = self.current_Sigma_matrix + pow(self.beta,2)*sigma_ii*self.current_Sigma_matrix@self.current_design_matrix.T@current_coe.design_vector@current_coe.design_vector.T@self.current_design_matrix@self.current_Sigma_matrix
            right_upper = -pow(self.beta,2)*sigma_ii*self.current_Sigma_matrix@self.current_design_matrix.T@current_coe.design_vector
            left_lower = -pow(self.beta,2)*sigma_ii*(self.current_Sigma_matrix@self.current_design_matrix.T@current_coe.design_vector).T
            right_lower = np.asarray([[sigma_ii]])
            upper = np.concatenate((left_upper, right_upper), axis=1)
            lower = np.concatenate((left_lower, right_lower), axis=1)
            self.current_Sigma_matrix = np.concatenate((upper, lower), axis=0)
        if keyword == "re-estimate":
            sigma_ii = pow(1/current_coe.gamma + current_coe.S, -1)
            kappa = pow(sigma_ii + pow(last_gamma - current_coe.gamma, -1), -1)

    def run(self, tolerance = 0.01):
    #BEGIN
        if self.output == True:
            print(f"t\tkeyword\t\tmax_increase\tlast_gamma\tnew_gamma\tpolynomial\t\t\tresidue\t\tLOO error")
        begin = time.time()
        run_data = [[], [], []]
        max_coe, max_projection = None, 0
        for coe in self.coefficients:
            if pow(coe.design_vector.T@self.y_vec,2)/pow(np.linalg.norm(coe.design_vector),2) > max_projection:
                max_coe = coe
                max_projection = pow(coe.design_vector.T@self.y_vec,2)/pow(np.linalg.norm(coe.design_vector),2)
        max_coe.gamma = ((max_projection - pow(self.beta, -1))/pow(np.linalg.norm(max_coe.design_vector),2)).item()
        self.calc_current_design_matrix()
        self.calc_Sigma_matrix()
        self.calc_mu()
        self.calc_s_q()
        self.calc_l()
        if self.output == True:
            self.calc_final_multi_polynomial()
            res = self.get_residues()
            LOO = self.get_LOO_error()
            print(f"init\tadd        \t\t\t{0:.2e}\t{max_coe.gamma:.2e}\t{max_coe.k_vec}\t\t{res:.4e}\t{LOO:.4e}")
    #LOOP
        t = 0
        err = 1
        last_err = 1
        while True:
        #CHOICE OF BASIS FUNCTION
            tijd1 = time.time()
            max_increase, current_coe = -np.inf, None
            for coe in self.coefficients:
                inc = coe.calc_likelihood_increase(self.l)
                if inc > max_increase: #Select the basis function with the highest possible marginal likelihood increase
                    max_increase = inc
                    current_coe = coe
            tijd2 = time.time()
        #GAMMA CALCULATION
            if pow(current_coe.q,2)-current_coe.s > self.l and current_coe.gamma == 0:
                last_gamma = 0
                last_mu = 0
                current_coe.gamma = current_coe.calc_gamma(self.l)
                keyword = "add        "
            elif pow(current_coe.q,2)-current_coe.s > self.l and current_coe.gamma != 0:
                last_gamma = current_coe.gamma
                last_mu = current_coe.mu
                current_coe.gamma = current_coe.calc_gamma(self.l)
                keyword = "re-estimate"
            elif pow(current_coe.q,2)-current_coe.s <= self.l and current_coe.gamma != 0:
                last_gamma = current_coe.gamma
                last_mu = current_coe.mu
                current_coe.gamma = 0
                keyword = "prune      "
            elif pow(current_coe.q,2)-current_coe.s <= self.l and current_coe.gamma == 0:
                keyword = "skip       "
        #UPDATE STATS
            self.calc_current_design_matrix()
            self.calc_Sigma_matrix()
            self.calc_mu()
            self.calc_s_q()
            self.calc_l()
            #self.update_beta() #According to [1], adjusting beta dynamically is not a good idea, see page 58
            if self.output == True:
                self.calc_final_multi_polynomial()
                res = self.get_residues()
                LOO = self.get_LOO_error()
                print(f"{t}\t{keyword}\t{max_increase:.2e}\t{last_gamma:.2e}\t{current_coe.gamma:.2e}\t{current_coe.k_vec}\t\t{res:.4e}\t{LOO:.4e}")
            if max_increase == 0: #Break loop if likelihood can no longer be increased
                eind = time.time()
                if self.output == True:
                    print(f"Likelihood can no longer be increased. Time = {eind-begin:.1f}s")
                break
            if t == self.max_iterations: #Break loop if maximum amount of iterations has been reached
                eind = time.time()
                if self.output == True:
                    print(f"Maximum amount of iterations has been reached. Time = {eind-begin:.1f}s")
                break
            t += 1

    def calc_final_multi_polynomial(self):  #Calculate the final polynomial according to the non-zero coefficients
        temp = [coe.mu*coe.multi_polynomial for coe in self.coefficients]
        multi = multi_polynomial_sum(temp)
        for key in multi.terms:
            multi.terms[key] = float(multi.terms[key])
        self.final_multi_polynomial = multi

    def get_LOO_error(self): #Get leave-one-out error
        som = 0
        for i in range(self.M):
            teller = self.y_vec[i] - self.final_multi_polynomial(self.experimental_design[i, :])
            noemer = 1 - self.h_vec[i]
            som += pow(teller/noemer, 2)
        return som/(self.M*np.var(self.y_vec))

    def get_residues(self): #Get residues of fit
        som = 0
        for i in range(self.M):
            teller = self.y_vec[i] - self.final_multi_polynomial(self.experimental_design[i, :])
            som += abs(teller)
        return som/self.M

    def calc_non_zero_coefficients(self): #Calculate the number of non-zero coefficients, gives a measure of the sparseness
        total = 0
        for coe in self.coefficients:
            if coe.mu != 0.0:
                total += 1
        return total

    def print_final_multi_polynomial(self): #Display final polynomial in output
        print("\nFinal Polynomial:\n")
        print(self.final_multi_polynomial)

    def print_coefficients(self, non_zero = True): #Display all or non-zero coefficients in output
        print("\nFinal Coefficients\n")
        print("k_vec\t\t\tgamma\t\tmu\t\tmulti_polynomial")
        for c in self.coefficients:
            if c.gamma != 0 or non_zero != True:
                print(f'{c.k_vec}\t{c.gamma:.2e}\t{c.mu:.2e}\t{c.multi_polynomial}')
