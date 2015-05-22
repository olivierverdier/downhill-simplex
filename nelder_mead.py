#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

'''
    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''

import numpy as np

def generate_simplex(x0, step=0.1):
    """
    Create a simplex based at x0
    """
    yield x0.copy()
    for i,_ in enumerate(x0):
        x = x0.copy()
        x[i] += step
        yield x

def make_simplex(x0, step=0.1):
    return np.array(list(generate_simplex(x0, step)))

def centroid(points):
    """
    Compute the centroid of a list points given as an array.
    """
    return np.mean(points, axis=0)



class NelderMead(object):

    refl = 1.
    ext = 1.
    cont = 0.5
    red = 0.5

    # max_stagnations: break after max_stagnations iterations with an improvement lower than no_improv_thr
    no_improve_thr=10e-6
    max_stagnations=10

    max_iter=1000
    
    def __init__(self, f, points):
        '''
            f: (function): function to optimize, must return a scalar score 
                and operate over a numpy array of the same dimensions as x_start
            points: (numpy array): initial position
        '''
        self.f = f
        self.points = points

    def step(self, res):
        # centroid of the lowest face
        pts = np.array([tup[0] for tup in res[:-1]])
        x0 = centroid(pts)

        new_res = self.reflection(res, x0, self.refl)
        if new_res is not None:
            exp_res = self.expansion(new_res, x0, self.ext)
            if exp_res is not None:
                new_res = exp_res
        else:
            new_res = self.contraction(res, x0, self.cont)
            if new_res is None:
                new_res = self.reduction(self.red)
        return new_res

    def run(self):
        # initialize
        self.prev_best = self.f(self.points[0])
        self.stagnations = 0
        res = self.make_score(self.points)

        # simplex iter
        for iters in range(self.max_iter):
            res = self.sort(res)
            best = res[0][1]

            # break after max_stagnations iterations with no improvement
            if best < self.prev_best - self.no_improve_thr:
                self.stagnations = 0
                self.prev_best = best
            else:
                self.stagnations += 1
        
            if self.stagnations >= self.max_stagnations:
                return res[0]

            # Nelder–Mead algorithm
            new_res = self.step(res)

            res = new_res
        else:
            raise Exception("No convergence after {} iterations".format(iters))


    def sort(self, res):
        """
        Order the points according to their value.
        """
        return sorted(res, key = lambda x: x[1])

    def reflection(self, res, x0, refl):
        """
        Reflection-extension step.
        refl: refl = 1 is a standard reflection
        """
        # reflected point and score
        xr = x0 + refl*(x0 - res[-1][0])
        rscore = self.f(xr)

        new_res = res[:]

        progress = rscore < new_res[-2][1]
        if progress: # if this is a progress, we keep it
            new_res[-1] = (xr, rscore)
            return new_res
        return None

    def expansion(self, res, x0, ext):
        """
        ext: the amount of the expansion; ext=0 means no expansion
        """
        xr, rscore = res[-1]
        # if it is the new best point, we try to expand
        if rscore < res[0][1]:
            xe = xr + ext*(xr - x0)
            escore = self.f(xe)
            if escore < rscore:
                new_res = res[:]
                new_res[-1] = (xe, escore)
                return new_res
        return None

    def contraction(self, res, x0, cont):
        """
        cont: contraction parametre: should be between zero and one
        """
        xc = x0 + cont*(res[-1][0] - x0)
        cscore = self.f(xc)

        new_res = res[:]

        progress = cscore < new_res[-1][1]
        if progress:
            new_res[-1] = (xc, cscore)
            return new_res
        return None

    def reduction(self, red):
        """
        red: reduction parametre: should be between zero and one
        """
        dirs = pts - pts[0]
        reduced_points = pts[0] + red*dirs
        new_res = self.make_score(reduced_points)
        return new_res

    def make_score(self, points):
        res = [(pt, self.f(pt)) for pt in points]
        return res
