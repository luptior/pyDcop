# BSD-3-Clause License
#
# Copyright 2017 Orange
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
This part of code is from http://krasserm.github.io/2018/03/19/gaussian-processes/

"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


"""
def divide_sleep(size):
    if np.random.ranf() >= loss(size):
        # successfully transmitted
        sleep(tran_time(size))
    else:
        # divide and send, timeout 10s
        sleep(5)
        divide_sleep(size / 2)
        divide_sleep(size / 2)


def proactive(size):
    # given a know relationship between size and expectation of delivery time
    # precalculated the function argmin = 7.9 ,so take 8 here
    if size <= 8:
        while np.random.ranf() < loss(size):
            sleep(5)
        else:
            sleep(tran_time(size))
    else:
        num = size / 8
        last = size % 8
        for i in range(num):
            # for the packages of size 8
            while np.random.ranf() < loss(8):
                sleep(5)
            else:
                sleep(tran_time(8))
        # for the last package
        if last != 0:
            while np.random.ranf() < loss(last):
                sleep(5)
            else:
                sleep(tran_time(last))
"""
# def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
#     K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
#     K_s = kernel(X_train, X_s, l, sigma_f)
#     K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
#     K_inv = inv(K)
#
#     mu_s = K_s.T.dot(K_inv).dot(Y_train)
#
#     cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
#
#     return mu_s, cov_s


# def kernel(X1, X2, l=1.0, sigma_f=1.0):
#     sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
#     return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()


# def log_likelihood(y, X_s, theta, sigma=0.5):
#     l = theta[0]
#     sigma_f = theta[1]
#     sigma_y = sigma
#     K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
#     K_s = kernel(X_train, X_s, l, sigma_f)
#     K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
#     K_inv = inv(K)
#
#     mu_s = K_s.T.dot(K_inv).dot(Y_train)
#     cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
#
#     return ((y - mu_s).T.dot(K_inv).dot(y - mu_s) / 2 - np.log(np.linalg.det(K)) / 2)[0][0]


def predication(train: list, x_pred: float, noise=0.1) -> float:

    # train should be in format as [[...], [...]]
    try:
        X_train = np.array(train[0]).reshape(-1, 1)
        Y_train = np.array(train[1]).reshape(-1, 1)
    except:
        sys.exit("train should be in format as [list1, list2]")

    # setting up the kernel, we currently use the most common one RBF
    rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise**2)

    gpr.fit(X_train, Y_train)

    # prediction of new points
    x_pred = np.asarray([x_pred]).reshape(-1, 1)
    mu_pred, cov_pred = gpr.predict(x_pred, return_cov=True)

    y=mu_pred[0][0]

    return y

if __name__ == '__main__':
    # Finite number of points
    X = np.arange(-5, 5, 0.2).reshape(-1, 1)

    # training data Finite number of points
    X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
    Y_train = np.sin(X_train)

    print(predication([X_train, Y_train], 1.5))


# if __name__ == '__main__':
#
#     # Finite number of points
#     X = np.arange(-5, 5, 0.2).reshape(-1, 1)
#
#     # training data Finite number of points
#     X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
#     Y_train = np.sin(X_train)
#     noise = 0.1
#
#
#     # setting up the kernel, we currently use the most common one RBF
#     rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
#     gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise**2)
#
#     # Reuse training data from previous 1D example
#     gpr.fit(X_train, Y_train)
#
#     # Compute posterior predictive mean and covariance
#     mu_s, cov_s = gpr.predict(X, return_cov=True)
#
#     # Obtain optimized kernel parameters
#     # l = gpr.kernel_.k2.get_params()['length_scale']
#     # sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])
#
#     # Plot the results
#     plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
#
#
#     # prediction of new points
#     x_pred = np.asarray([1.5]).reshape(-1, 1)
#     mu_pred, cov_pred = gpr.predict(x_pred, return_cov=True)
#
#     x=x_pred[0][0]
#     y=mu_pred[0][0]
#
#     print(x, y)
#
#     plt.show()