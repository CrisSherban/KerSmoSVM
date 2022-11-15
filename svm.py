"""
MIT License

Copyright (c) July 2021 Serban Cristian Tudosie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import re

import numpy as np
import random as rnd
from dataclasses import dataclass, field
from kernel import Kernel


@dataclass(init=True)
class SVM:
    """
    Support Vector Machine that solves the optimization problem related to
    the training of the maximum margin hyperplane by the usage of the
    Sequential Minimal Algorithm.

    In this implementation is recommended to run the "dunder" signed methods
    only from inside the class as they will result hidden from outside and
    are meant to be encapsulated inside this class.

    What to expect on the modification of the default parameter C:
    The higher the C parameter the less tolerant the SVM becomes
    and the decision boundary is more severe.
    Check: https://chrisalbon.com/code/machine_learning/support_vector_machines/svc_parameters_using_rbf_kernel
    for more details on the C parameter and gamma if using RBF kernel
    """

    # //@formatter:off
    # attribute         type          default_value                                 comment
    kernel:             callable    = Kernel(kernel_type='linear').kernel           # kernel function to use
    name:               str         = 'SVM_Classifier'                              # name of the SVM
    max_iter:           int         = 1000                                          # maximum number of iterations
    iters_done:         int         = 0                                             # real number of iterations done
    C:                  float       = 10                                            # regularization parameter
    epsilon:            float       = 1e-3                                          # tolerance
    seed:               int         = 42                                            # seed to repeat experiments
    b:                  float       = 0.0                                           # threshold
    verbose:            bool        = False                                         # to print progress
    star_idx:           np.array    = field(default_factory=lambda: np.array([]))   # found alphas stars indexes
    X:                  np.array    = field(default_factory=lambda: np.array([]))   # training data matrix
    y:                  np.array    = field(default_factory=lambda: np.array([]))   # training labels
    alphas:             np.array    = field(default_factory=lambda: np.array([]))   # found alphas
    alphas_sv:          np.array    = field(default_factory=lambda: np.array([]))   # found alphas for sup vectors
    X_sv:               np.array    = field(default_factory=lambda: np.array([]))   # support vectors
    y_sv:               np.array    = field(default_factory=lambda: np.array([]))   # support vectors labels
    gram_matrix:        np.array    = field(default_factory=lambda: np.array([]))   # Gram matrix
    # //@formatter:on

    def __post_init__(self):
        if self.seed:
            rnd.seed(a=self.seed)
        if re.search(r"\'.*\'", str(self.kernel)).group(0) == "'poly'":
            # high Cs take a lot for poly kernel, setting it down for
            # faster computation, remove this line if you are cross validating
            self.C = 1

    def __str__(self) -> str:
        ker_type = re.search(r"\'.*\'", str(self.kernel)).group(0)
        return f"{self.name}:(Kernel: {ker_type}; C: {self.C}; eps: {self.epsilon}; max_iter: {self.max_iter})"

    def __create_gram_matrix(self, X: np.ndarray):
        """
            Creates the Gram Matrix with the given kernel of the SVM.
        :param X: training data matrix
        """
        if self.verbose:
            print("Computing Gram Matrix")
        self.gram_matrix = np.resize(self.gram_matrix, (len(X), len(X)))
        for i, xi in enumerate(X):
            for j, xj in enumerate(X):
                self.gram_matrix[i, j] = self.kernel(xi, xj)

    def __pred(self, x_index: int) -> int:
        """
            In this implementation this prediction function
            has to be run only inside the class since it uses a precomputed
            Gram matrix instead of kernel function re-computation
        :param x_index: index of the sample
        """
        return int(np.sign(np.sum(
            np.array([self.alphas[i] * self.y[i] * self.gram_matrix[i, x_index] for i in range(len(self.X))])
            + self.b)))

    def __pred_error(self, y_k: int, k: int) -> float:
        return self.__pred(k) - y_k

    def __compute_b(self, e_i, e_j, a_old_i, a_old_j, i, j) -> float:
        b1 = self.b - e_i - self.y[i] * (self.alphas[i] - a_old_i) * self.gram_matrix[i, i] \
             - self.y[j] * (self.alphas[j] - a_old_j) * self.gram_matrix[i, j]

        b2 = self.b - e_j - self.y[i] * (self.alphas[i] - a_old_i) * self.gram_matrix[i, j] \
             - self.y[j] * (self.alphas[j] - a_old_j) * self.gram_matrix[j, j]

        if 0 < self.alphas[i] < self.C:
            return b1
        elif 0 < self.alphas[j] < self.C:
            return b2
        else:
            return (b1 + b2) / 2

    def __compute_LH(self, C, alpha_j, alpha_i, y_j, y_i) -> (float, float):
        if y_i == y_j:
            return max(0, alpha_i + alpha_j - C), min(C, alpha_i + alpha_j)
        else:
            return max(0, alpha_j - alpha_i), min(C, C - alpha_i + alpha_j)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
            Fits the Support Vector Machine by optimizing the lagrangian
            with the usage of the Sequential Minimal Optimization algorithm
            credits:    1)  http://cs229.stanford.edu/materials/smo.pdf
                        2)  Platt, John.Fast Training of Support Vector Machines
                            using Sequential Minimal Optimization

            summary:
                        1) select two parameters αi and αj and optimize the
                           objective value jointly for both these αs.
                        2) adjusts the b parameter based on the new αs.
                        3) repeat until the αs converge.
        :param X: training data matrix
        :param y: labels for training data
        """

        # in case the training data is passed only in fit() method
        # and not in the constructor parameters
        if len(self.X) == 0 or len(self.y) == 0:
            self.X = np.resize(self.X, len(y))
            self.y = np.resize(self.y, len(y))

        num_samples, num_features = X.shape[0], X.shape[1]
        self.alphas = np.resize(self.alphas, num_samples)
        self.alphas = np.zeros(len(self.alphas))
        self.__create_gram_matrix(X)

        if self.verbose:
            print("Started fitting")
        for _ in range(self.max_iter):
            alpha_prev = copy.deepcopy(self.alphas)

            for i in range(0, num_samples):
                error_i = self.__pred_error(y[i], i)

                # find and select an j!=i
                # TODO: implement heuristics for the choice of j
                j = self.rnd_in_range(0, num_samples, i)
                error_j = self.__pred_error(y[j], j)

                # this represents the second order derivative of the objective function
                ker_ij = self.gram_matrix[i, i] + self.gram_matrix[j, j] - 2 * self.gram_matrix[i, j]
                if ker_ij == 0:  # because of division by 0 we counteract by trying another j
                    continue

                # compute bounds
                alpha_old_j, alpha_old_i = self.alphas[j], self.alphas[i]
                (L, H) = self.__compute_LH(self.C, alpha_old_j, alpha_old_i, y[j], y[i])
                if L == H:
                    continue

                # updating values with optimal ones
                self.alphas[j] = alpha_old_j + float(y[j] * (error_i - error_j)) / ker_ij
                # clipping alpha to lie in [L, H]
                self.alphas[j] = max(self.alphas[j], L)
                self.alphas[j] = min(self.alphas[j], H)
                self.alphas[i] = alpha_old_i + y[i] * y[j] * (alpha_old_j - self.alphas[j])

                self.b = self.__compute_b(error_i, error_j, alpha_old_i, alpha_old_j, i, j)

            self.iters_done += 1
            # convergence
            if np.linalg.norm(self.alphas - alpha_prev, ord=2) < self.epsilon:
                break

        assert self.iters_done <= self.max_iter, \
            f"Max number of iterations reached: {self.max_iter}"

        # support vectors indexes
        star_idx = np.where(self.alphas > 0)[0]
        self.star_idx = np.resize(self.star_idx, len(star_idx))
        self.X_sv = np.resize(self.star_idx, len(star_idx))
        self.y_sv = np.resize(self.star_idx, len(star_idx))
        self.alphas_sv = np.resize(self.star_idx, len(star_idx))

        self.star_idx = star_idx
        self.X_sv = X[star_idx]
        self.y_sv = y[star_idx]
        self.alphas_sv = self.alphas[star_idx]

    def predict(self, x: np.ndarray) -> int:
        """
        :param x: a new sample
        :return: prediction of the classified sample x, can be 1 or -1
        """
        assert len(self.star_idx) > 0, "Fit the SVM before predicting"
        kers_tmp = np.array([self.kernel(xi, x) for xi in self.X_sv])
        return int(np.sign(np.dot(self.alphas_sv * self.y_sv, kers_tmp)))

    @staticmethod
    def rnd_in_range(a: int, b: int, j: int) -> int:
        """
        :param a: lower bound, included
        :param b: upper bound, not included
        :param j: return value will be different from this
        :return: random int in range [a, b) different from j
        """
        i = 0
        while i == j:
            i = rnd.randint(a, b - 1)
        return i

    @staticmethod
    def accuracy(y_pred: list or np.ndarray, y_true: list or np.ndarray) -> float:
        """
        :param y_pred: 1-D array-like of predicted y
        :param y_true: 1-D array-like of true y
        :return: accuracy
        """
        return np.sum(np.array(y_pred) == np.array(y_true)) / len(y_true)
