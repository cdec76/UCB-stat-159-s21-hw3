"""
Fish accept.

Test and check algorithms for Fisher accept
"""

import numpy as np
from scipy.stats import hypergeom
import pytest


def fisher_accept(N, G, n, alpha=0.05):
    """
    Acceptance region for randomized hypergeometric test.

    Find the acceptance region for a randomized, exact level alpha test of
    the null hypothesis X~Hypergeometric(N, G, n). The acceptance region is
    the smallest possible. (And not, for instance, symmetric.)

    If a non-randomized, conservative test is desired, use the union of I and J
    as the acceptance region.

    Parameters
    ----------
    N:  integer
        population size
    G:  integer
        number of "good" items in the population
    n:  integer
        sample size
    alpha : float
        desired significance level

    Returns
    --------
    I:  list
        values for which the test never rejects
    J:  list
        values for which the test sometimes rejects
    gamma : float
        probability the test does not reject when the value is in J
    """
    x = np.arange(0, n+1)          # all possible values of X
    posout = list(x)
    # start with all possible outcomes, then remove some
    pmf = hypergeom.pmf(x, N, G, n)   # hypergeometric pmf
    bottom = 0                     # smallest outcome still in I
    top = n                        # largest outcome still in I
    J = []
    p_J = 0                        # probability of the randomized outcome
    p_tail = 0                     # probability of outcomes excluded from I
    while p_tail < alpha:
        # still need to remove outcomes from the acceptance region
        pb = pmf[bottom]
        pt = pmf[top]
        if pb < pt:             # the lower possibility has smaller probability
            J = [bottom]
            p_J = pb
            bottom += 1
        elif pb > pt:           # the upper possibility has smaller probability
            J = [top]
            p_J = pt
            top -= 1
        else:
            if bottom < top:    # the two possibilities have equal probability
                J = [bottom, top]
                p_J = pb+pt
                bottom += 1
                top -= 1
            else:                  # there is only one possibility left
                J = [bottom]
                p_J = pb
                bottom += 1
        p_tail += p_J
        for j in J:
            posout.remove(j)
    gamma = (p_tail-alpha)/p_J
    # probability of accepting H_0 when X in J to get exact level alpha
    return posout, J, gamma


def fisher_accept2(N, G, n, alpha=0.05):
    """
    Acceptance region for randomized hypergeometric test.

    Find the acceptance region for a randomized, exact level alpha test of
    the null hypothesis X~Hypergeometric(N, G, n). The acceptance region is
    the smallest possible. (And not, for instance, symmetric.)

    If a non-randomized, conservative test is desired, use the union of I and J
    as the acceptance region.

    Parameters
    ----------
    N:  integer
        population size
    G:  integer
        number of "good" items in the population
    n:  integer
        sample size
    alpha : float
        desired significance level

    Returns
    --------
    I:  list
        values for which the test never rejects
    J:  list
        values for which the test sometimes rejects
    gamma : float
        probability the test does not reject when the value is in J
    """
    assert N > n
    assert N > G
    x = np.arange(0, n+1)          # all possible values of X
    posout = list(x)
    # start with all possible outcomes, then remove some
    pmf = hypergeom.pmf(x, N, G, n)   # hypergeometric pmf
    bottom = 0                     # smallest outcome still in I
    top = n                        # largest outcome still in I
    J = []
    p_J = 0                        # probability of the randomized outcome
    p_tail = 0                     # probability of outcomes excluded from I
    while p_tail < alpha:
        # still need to remove outcomes from the acceptance region
        pb = pmf[bottom]
        pt = pmf[top]
        if pb < pt:             # the lower possibility has smaller probability
            J = [bottom]
            p_J = pb
            bottom += 1
        elif pb > pt:           # the upper possibility has smaller probability
            J = [top]
            p_J = pt
            top -= 1
        else:
            if bottom < top:    # the two possibilities have equal probability
                J = [bottom, top]
                p_J = pb+pt
                bottom += 1
                top -= 1
            else:                  # there is only one possibility left
                J = [bottom]
                p_J = pb
                bottom += 1
        p_tail += p_J
        for j in J:
            posout.remove(j)
    gamma = (p_tail-alpha)/p_J
    # probability of accepting H_0 when X in J to get exact level alpha
    return posout, J, gamma


def test_1():
    """
    first test for Fisher acceptance.

    testing for a regular case
    """
    assert fisher_accept(N=10, G=2, n=5) == ([1], [0, 2], 0.8875000000000001)


def test_2():
    """
    second test for Fisher acceptance.

    testing if G > N
    """
    try:
        fisher_accept2(N=10, G=11, n=6)
    except AssertionError as e:
        pytest.fail(e, pytrace=True)
