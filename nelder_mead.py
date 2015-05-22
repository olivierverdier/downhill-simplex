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

    # no_improv_break: break after no_improv_break iterations with an improvement lower than no_improv_thr
    no_improve_thr=10e-6
    no_improv_break=10

    max_iter=1000
    
    def __init__(self, f, points):
        '''
            f: (function): function to optimize, must return a scalar score 
                and operate over a numpy array of the same dimensions as x_start
            points: (numpy array): initial position
        '''
        self.f = f
        self.points = points

    def run(self):
        # initialize
        self.prev_best = self.f(self.points[0])
        self.no_improv = 0
        self.res = self.make_score(self.points)

        # simplex iter
        for iters in range(self.max_iter):
            self.sort()
            best = self.res[0][1]

            # break after no_improv_break iterations with no improvement
            if best < self.prev_best - self.no_improve_thr:
                self.no_improv = 0
                self.prev_best = best
            else:
                self.no_improv += 1
        
            if self.no_improv >= self.no_improv_break:
                return self.res[0]

            # centroid of the lowest face
            pts = np.array([tup[0] for tup in self.res[:-1]])
            x0 = centroid(pts)

            # Nelder–Mead algorithm
            if not self.reflection(x0, self.refl, self.ext):
                if not self.contraction(x0, self.cont):
                    self.reduction(self.red)
        else:
            raise Exception("No convergence after {} iterations".format(iters))


    def sort(self):
        """
        Order the points according to their value.
        """
        self.res.sort(key = lambda x: x[1])

    def reflection(self, x0, refl, ext):
        """
        Reflection-extension step.
        refl: refl = 1 is a standard reflection
        ext: the amount of the expansion; ext=0 means no expansion
        """
        # reflected point and score
        xr = x0 + refl*(x0 - self.res[-1][0])
        rscore = self.f(xr)

        progress = rscore < self.res[-2][1]
        if progress: # if this is a progress, we keep it
            self.res[-1] = (xr, rscore)
            # if it is the new best point, we try to expand
            if rscore < self.res[0][1]:
                xe = xr + ext*(xr - x0)
                escore = self.f(xe)
                if escore < rscore:
                    self.res[-1] = (xe, escore)
        return progress

    def contraction(self, x0, cont):
        """
        cont: contraction parametre: should be between zero and one
        """
        xc = x0 + cont*(self.res[-1][0] - x0)
        cscore = self.f(xc)
        progress = cscore < self.res[-1][1]
        if progress:
            self.res[-1] = (xc, cscore)
        return progress

    def reduction(self, red):
        """
        red: reduction parametre: should be between zero and one
        """
        dirs = pts - pts[0]
        reduced_points = pts[0] + red*dirs
        self.res = self.make_score(self.f, reduced_points)

    def make_score(self, points):
        res = [(pt, self.f(pt)) for pt in points]
        return res
