"""
Chi-Square for two samples.

Three functions.
"""


import numpy as np
import timeit
import random
from random import randint
import time


def chisq_1(x, y):
    """
    Calculate a Chi-Square for two samples.

    Arguments:
    x:  list
        sample 1
    y:  list
        sample 2

    Returns
    chi_s:  float
            value of Chi-Square result
    """
    n, m = len(x), len(y)
    z = x + y
    u = np.unique(z)
    p = [float(z.count(i)) / float(m+n) for i in u]
    E_k = [n * j for j in p]
    O_k = [x.count(k) for k in u]
    chi_s = sum([float((O_k[i] - E_k[i])**2)/float(E_k[i])
                 for i in range(len(u))])
    return chi_s


def chisq_2(x, y):
    """
    Calculate a Chi-Square for two samples.

    Keyword arguments:
    x -- list sample 1
    y -- list sample 2
    """
    n, m = len(x), len(y)
    z = np.append(x, y)
    u, p = np.unique(z, return_counts=True)
    p = np.true_divide(p, (n+m))
    E_k = p*n
    O_k = [x.count(k) for k in u]
    chi_s = np.nansum(np.true_divide(np.square(O_k - E_k), E_k))
    return chi_s


def chisq_3(x, y):
    """
    Calculate a Chi-Square for two samples.

    Keyword arguments:
    x -- list sample 1
    y -- list sample 2
    """
    n, m = len(x), len(y)
    z = x + y
    u = [z[0]]
    for i in z:
        if i not in u:
            u += [i]
    p = []
    for j in u:
        count = 0
        for k in z:
            if j == k:
                count += 1
        p += [float(count)/float(n+m)]
    E_k = []
    for freq in p:
        E_k += [freq*n]
    O_k = []
    for r in u:
        count = 0
        for q in x:
            if r == q:
                count += 1
        O_k += [count]
    chi_s = 0
    for s in range(len(u)):
        chi_s += float((O_k[s]-E_k[s])**2) / float(E_k[s])
    return chi_s


# Establishing the simulation
fixtest_x = [1, 1, 2, 2, 2, 3, 4, 4, 4, 5]
fixtest_y = [2, 2, 3, 4, 4, 5, 5, 5]

random.seed(10)
sample_x = [randint(1, 6) for i in range(100000)]
sample_y = [randint(1, 6) for i in range(10000)]

coin_x = [randint(0, 1) for i in range(1000)]
coin_y = [randint(0, 1) for i in range(100)]


# %timeit -n 20 chisq_1(sample_x, sample_y)
# %timeit -n 20 chisq_2(sample_x, sample_y)
# %timeit -n 20 chisq_3(sample_x, sample_y)

"""
From the results of timing of single function, one can see the chisq_1 is the
slowest, chisq_2 is the second, and chisq_3 is the fastest. The chisq_1 is
expected to be the slowest since it used list and for each step it use list
comprehesion if need. However, the chisq_1 is easy to read since for each
element required, it has a separate line. The chisq_2, instead, using array as
a data structure and by using some default numpy function it saves some time
for instance, by using 'return_counts=True' one does not need to compute counts
separately. Like chisq_1, it is also easy to read except some results are
inside for saving some time. The chisq_3 does not use any functions from other
packages it uses all for loop for computing needed results. Among all three
functions, it is the hardest to read, since using all for loops people need to
check each loop to understand.
"""


def test_method_1():
    """
    Test chisq_1.

    Testing chisq_1 by using first samples
    """
    assert round(chisq_1(sample_x, sample_y), 5) == 0.36701


def test_method_2():
    """
    Test chisq_2.

    Testing chisq_2 by using first samples
    """
    assert round(chisq_2(sample_x, sample_y), 5) == 0.36701


def test_method_3():
    """
    Test chisq_3.

    Testing chisq_3 by using first samples
    """
    assert round(chisq_3(sample_x, sample_y), 5) == 0.36701


def test_fixed_1():
    """
    Test chisq_1.

    Testing chisq_1 by using first fixed samples
    """
    assert round(chisq_1(fixtest_x, fixtest_y), 5) == 1.43


def test_fixed_2():
    """
    Test chisq_2.

    Testing chisq_2 by using first fixed samples
    """
    assert round(chisq_2(fixtest_x, fixtest_y), 5) == 1.43


def test_fixed_3():
    """
    Test chisq_3.

    Testing chisq_3 by using first fixed samples
    """
    assert round(chisq_3(fixtest_x, fixtest_y), 5) == 1.43
