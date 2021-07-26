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

from dataclasses import dataclass, field
import numpy as np
import numexpr as ne


@dataclass(init=True)
class Kernel:
    """
        Kernel class that exposes methods that compute kernel
        functions depending on the kernel_type that is passed
        and the parameters corresponding to the kernel.
        The RBF in particular uses the numexpr package to speed up computation

        The peculiarity of the usage of kernel functions
        is the ability to compute directly the relation K(x1, x2) = ø(x1)·ø(x2)
        without computing explicitly the higher dimentional features given by
        the ø.
        This combined with the Gram Matrix allows a faster fitting and predicting
        of an SVM
    """
    # //@formatter:off
    # attribute         type          default_value                     comment
    kernel_type:        str         = 'linear'                          # type of kernel
    gamma:              float       = 1                                 # parameter for RBF kernel
    dim:                int         = 2                                 # parameter for Polynomial kernel
    # //@formatter:on

    def __post_init__(self):
        # this is decided inside the class, the 'client' does not have to
        # modify this encapsulated information
        self.__available_kernels = ['linear', 'poly', 'rbf']

    def kernel(self, x1, x2):
        assert self.kernel_type in self.__available_kernels, \
            f"Incorrect kernel type, choose between: {str(self.__available_kernels)}"
        return getattr(self, self.kernel_type)(x1, x2)

    def linear(self, x1, x2):
        return np.dot(x1, x2.T)

    def poly(self, x1, x2):
        return ne.evaluate('(dot + 1) ** d', {
            'dot': np.dot(x1, x2.T),
            'd': self.dim,
        })

    def rbf(self, x1, x2):
        return ne.evaluate('exp(-g * norm)', {
            'norm': np.linalg.norm(x1 - x2, ord=2) ** 2,
            'g': self.gamma,
        })
