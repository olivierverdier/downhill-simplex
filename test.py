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
        N = NelderMead(f, list(generate_simplex(np.array([0.,0.,0.]))))
        res = N.run()
        npt.assert_allclose(res[0], np.array([ -1.58089710e+00,  -2.39020317e-03,   1.39669799e-06]))

    def test_simplex(self):
        d = 3
        x0 = np.zeros(d)
        xs = make_simplex(x0, step=1.)
        self.assertEqual(len(xs), d+1)
        npt.assert_allclose(xs[-1], np.array([0.,0,1.]))

    def test_centroid(self):
        d = 3
        xs = np.array(list(generate_simplex(np.zeros(d), step=1.)))
        x0 = centroid(xs)
        npt.assert_allclose(x0, np.array([1./(d+1)]*d)) # centroid of the canonical simplex

    def test_reflection(self):
        def f(x): return x[0]
        pts = make_simplex(np.zeros(2), step=1.)
        N = NelderMead(f, pts)
        ## res = [(pt,f(pt)) for pt in pts]
        ## res.sort(key = lambda x: x[1])
        res = N.make_score(N.points)
        res = N.sort(res)
        lpts = np.array([tup[0] for tup in res[:-1]])
        x0 = centroid(lpts)
        new_res = N.reflection(res, x0, 1., 1.)
        self.assert_(new_res)
        npt.assert_allclose(new_res[-1][0], np.array([-2., 1.5]))
