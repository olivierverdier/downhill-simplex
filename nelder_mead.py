
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

def reflection(f, x0, res, alpha, gamma):
    """
    Reflection step.
    alpha: alpha = 1 is a standard reflection
    gamma: the amount of the expansion; gamma=0 means no expansion
    """
    # reflected point and score
    xr = x0 + alpha*(x0 - res[-1][0])
    rscore = f(xr)

    progress = rscore < res[-2][1]
    if progress: # if this is a progress, we keep it
        res[-1] = (xr, rscore)
        # if it is the new best point, we try to expand
        if rscore < res[0][1]:
            xe = xr + gamma*(xr - x0)
            escore = f(xe)
            if escore < rscore:
                res[-1] = (xe, escore)
    return progress

def contraction(f, x0, res, rho):
    """
    rho: contraction parametre: should be between zero and one
    """
    xc = x0 + rho*(res[-1][0] - x0)
    cscore = f(xc)
    progress = cscore < res[-1][1]
    if progress:
        res[-1] = (xc, cscore)
    return progress


def nelder_mead(f, points, 
         no_improve_thr=10e-6, no_improv_break=10, max_iter=0,
        alpha = 1., gamma = 1., rho = 0.5, sigma = 0.5):
    '''
        @param f (function): function to optimize, must return a scalar score 
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with 
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm 
            (see Wikipedia page for reference)
    '''

    # init
    dim = len(points[0])

    res = []
    for i,x in enumerate(points):
        score = f(x)
        res.append((x,score))

    prev_best = f(points[0])
    no_improv = 0

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key = lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1
    
        if no_improv >= no_improv_break:
            return res[0]

        # centroid of the lowest face
        pts = np.array([tup[0] for tup in res[:-1]])
        x0 = centroid(pts)

        if reflection(f, x0, res, alpha, gamma):
            continue

        if contraction(f, x0, res, rho):
            continue

        # reduction
        dirs = pts - pts[0]
        reduced_points = pts[0] + sigma*dirs
        res = [(pt,f(pt)) for pt in reduced_points]


