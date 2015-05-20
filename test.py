#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import unittest

from nelder_mead import *
import numpy as np
import numpy.testing as npt

def f(x):
    return np.sin(x[0])*np.cos(x[1])*(1./(np.abs(x[2])+1))

class Test(unittest.TestCase):
    def test(self):
        res = nelder_mead(f, list(generate_simplex(np.array([0.,0.,0.]))))
        npt.assert_allclose(res[0], np.array([ -1.58089710e+00,  -2.39020317e-03,   1.39669799e-06]))

    def test_simplex(self):
        d = 3
        xs = list(generate_simplex(np.zeros(d)))
        self.assertEqual(len(xs), d+1)
        npt.assert_allclose(xs[-1], np.array([0.,0,.1]))

    def test_centroid(self):
        d = 3
        xs = np.array(list(generate_simplex(np.zeros(d), step=1.)))
        x0 = centroid(xs)
        npt.assert_allclose(x0, np.array([1./(d+1)]*d)) # centroid of the canonical simplex

