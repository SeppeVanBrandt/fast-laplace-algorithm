from fastlaplace import *
from pyDOE import lhs
import scipy.stats as st

type = Hermite()
P = 5  # Degree of polynomial expansion
dim = 6  # Number of dimensions (number of random variables)
q = 0.7  # q-value for hyperbolic truncation

# Generate k-vectors for all polynomials with highest degree P
k_vec_list = calc_k_vec_list(dim, P)
k_vec_list_test = []

# Hyperbolic truncation
for k in k_vec_list:
    sum_val = 0
    for x in k:
        sum_val += pow(abs(x), q)
    if pow(sum_val, 1 / q) <= P:
        k_vec_list_test.append(k)

# Define test function, in this case a fifth-degree polynomial
def test_function_6d(x):
    return x[0]**5 + x[1] * x[2]**2 + 15 * x[3] * x[4] * x[5]

# Create the experimental design via Latin Hypercube Sampling
ED_test = lhs(dim, samples=100)  # LHS Sampling from a uniform hypercube
y_test = []
for ed in ED_test:
    for i in range(len(ed)):
        ed[i] = st.norm.ppf(ed[i], loc=0, scale=1)  # Transform uniform LHS samples to Gaussian samples
    y_test.append(test_function_6d(ed))

ED_test = np.array(ED_test)
y_test = np.array(y_test)

# Initialize Fast Laplace Algorithm
algo = FastLaplaceAlgorithm(
    P, dim, q, k_vec_list_test, ED_test, y_test,
    [Hermite(), Hermite(), Hermite(), Hermite(), Hermite(), Hermite()],
    "test", output=True
)
algo.run()  # Run
algo.calc_final_multi_polynomial()  # Calculate final polynomial
algo.print_final_multi_polynomial()  # Print final polynomial
algo.print_coefficients()  # Print coefficients
