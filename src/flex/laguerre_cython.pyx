import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def laguerre_eval(int n, np.ndarray[double, ndim=1] x_vals):
    cdef Py_ssize_t i
    cdef int k
    cdef double x, L, Lnm1, Lnm2
    cdef int size = x_vals.shape[0]
    cdef np.ndarray[double, ndim=1] result = np.empty(size)

    for i in range(size):
        x = x_vals[i]
        if n == 0:
            result[i] = 1.0
        elif n == 1:
            result[i] = 2.0 - x
        else:
            Lnm2 = 1.0
            Lnm1 = 2.0 - x
            for k in range(2, n + 1):
                L = ((2 * k - x) * Lnm1 - k * Lnm2) / k
                Lnm2 = Lnm1
                Lnm1 = L
            result[i] = Lnm1
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def laguerre_eval_generic(int n, double alpha, np.ndarray[double, ndim=1] x_vals):
    cdef Py_ssize_t i
    cdef int k
    cdef double x, L, Lnm1, Lnm2
    cdef int size = x_vals.shape[0]
    cdef np.ndarray[double, ndim=1] result = np.empty(size)

    for i in range(size):
        x = x_vals[i]
        if n == 0:
            result[i] = 1.0
        elif n == 1:
            result[i] = 1.0 + alpha - x
        else:
            Lnm2 = 1.0
            Lnm1 = 1.0 + alpha - x
            for k in range(2, n + 1):
                L = ((2 * k - 1 + alpha - x) * Lnm1 - (k - 1 + alpha) * Lnm2) / k
                Lnm2 = Lnm1
                Lnm1 = L
            result[i] = Lnm1
    return result

