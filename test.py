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
        res = nelder_mead(f, np.array([0.,0.,0.]))
        npt.assert_allclose(res[0], np.array([ -1.58089710e+00,  -2.39020317e-03,   1.39669799e-06]))


