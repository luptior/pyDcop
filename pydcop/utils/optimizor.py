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
To be add

how to proporly split packages into smaller parts based on the constrains to speed up



"""

import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from sklearn.linear_model import LinearRegression


def func1(size: float) -> float:
    return 20 * size


def func2(size: float) -> float:
    return 2 * size + 10


def sending_time_size(size: float) -> float:
    """the transmission time based on the input file size in MB, return time in second"""
    return size / 100 + 0.01


def sending_time_size_with_lossp(size: float, lossprop: float = 0.1, timeout: float = 10.) -> float:
    """
    The transmission time based on the input file size in MB, return time in second
    if lost, the time return will be timeout +1, so the the action when timeout happen will occured.
    """
    if lossprop <= np.random.ranf():
        return size / 100 + 0.01
    else:
        return timeout + 1


def linear_optimze(x: list, y: list, x_pred):
    """
    x: input msg sizes in MB
    y: The transmission times
    """
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)

    model = LinearRegression().fit(x, y)

    y_pred = model.predict(x_pred)

    return y_pred