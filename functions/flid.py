"""
This file is used for manipulating FLID models from
Tschiatschek, S., Djolonga, J., and Krause, A. 
Learning probabilistic submodular diversity models via noise contrastive estimation. 
In Proc. International Conference on Artificial Intelligence and Statistics (AISTATS), 2016.

This file is borrowed from Sebastian Tschiatschek <stschia@gmail.com>.
"""

from __future__ import division, print_function
from itertools import chain, combinations
import numpy as np
from scipy.misc import logsumexp as lse
from scipy.special import expit
import random
_debug = True

class Function(object):
    def sample(self, n):
        raise NotImplementedError

    def __call__(self, S):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError

    def gradient(self, S):
        """Gradients of the log-likelihood wrt the parameters."""
        raise NotImplementedError

    def project_parameters(self):
        raise NotImplementedError


class DiversityFun(Function):
    
    def __init__(self, V, n_dim):
        self.V = V
        self.n_dim = n_dim

        self.utilities = np.zeros(len(V),  dtype=np.double)  # Utilities.
        self.W = 1e-3 * np.random.rand(len(V), n_dim).astype(dtype=np.double)  # Weights matrix.
        self.n_logz = np.array([0.], dtype=np.double)  # The "normalizer".

    @property
    def parameters(self):
        return [self.utilities, self.W, self.n_logz]

    def project_parameters(self):
        negInd = self.W < 0
        self.W[negInd] = 1e-3 * np.random.rand(np.sum(negInd))

    def __call__(self, S): # S: index of the set 
        if len(S) == 0:
            return self.n_logz[0]   # TODO

        slc = self.W[S, :]

        return (
                self.n_logz[0] +
                np.sum(self.utilities[S]) +
                np.sum((np.max(slc, axis=0) - np.sum(slc, axis=0)))
            )

    def all_singleton_adds(self, S):
        """
        Compute all function values resulting from adding a single
        value to S
        """
        Wutilities = -np.sum(self.W, axis=1)

        val = self.n_logz[0]
        val += np.sum(self.utilities[S])
        val += np.sum(Wutilities[S])
        vals = val + np.zeros(len(self.V))

        # now add the gain of adding a single value
        vals += self.utilities
        vals += Wutilities
        tmp = np.max(self.W[S, :], axis=0)

        # memory intense
        tmp2 =  np.repeat(tmp.reshape((1, self.n_dim, 1)), len(self.V), axis=0)
        vals += np.sum(np.max(np.concatenate((self.W.reshape((len(self.V), self.n_dim, 1)), tmp2), axis=2), axis=2), axis=1)

        vals[S] = 0  # no gain for things that were already there

        return vals


    def gradient(self, A):
        A = np.asarray(A)
        grad_util = np.zeros_like(self.utilities)
        grad_util[A] = 1

        grad_W = np.zeros_like(self.W)
        indices = list(np.argmax(self.W[A, :], axis=0))
        grad_W[A, :] -= 1
        grad_W[A[indices], range(self.n_dim)] += 1

        grad_n_logz = np.ones_like(self.n_logz)

        return [grad_util, grad_W, grad_n_logz]

    def _estimate_LL_exact(self, data):
        ''' return log likelihood  '''
        logZ = self.logZ_fast()
        return sum(self(d) - logZ for d in data) / len(data)  # why you need to divide by len(data)?   

    def logZ_FacLoc(self, ind, order):
        """ Compute FacLoc using the given indices... """
        inc = set()
        
        for (d, i) in enumerate(ind): 
            i,d = int(i), int(d)
            inc.add(order[int(i),d])
            

        rem = set()
        for (d, i) in enumerate(ind):
            i,d = int(i), int(d)
            r = order[:i,d]
            rem = rem.union(set(r))

        given = inc

        if len(inc.intersection(rem)) > 0:
            return - float('inf')

        W = self.W
        N = len(self.V)
        U = self.utilities - np.sum(W, axis=1)

        val = 0
        D = self.n_dim
        for d in range(D):
            val += np.max(W[list(given), d])
        val += sum(U[list(given)])  # np.prod(np.exp(U[list(given)]))
        rest = list(set(range(N)).difference(rem).difference(given))
        val += np.sum(np.log1p(np.exp(U[rest]))) # np.prod(1 + np.exp(U[rest]))

        return val

    def logZ_FacLoc_star(self, args):
        partial_list, order = args
        logZ_part = -float('inf')
        for ind in partial_list:
            logZ_part += lse([logZ_part, self.logZ_FacLoc(ind, order)])
        return logZ_part  # self.logZ_FacLoc(*args)


    def logZ_fast(self, parallel=False):
        if _debug:
            print('starting logZ_fast')
            
        logZ = self([]) - self.n_logz[0]

        W = self.W
        N = len(self.V)
        D = self.n_dim

        order = np.argsort(W, axis=0)
        for d in range(D):
            order[:,d] = order[::-1,d]

        import time
        time1 = time.time()
        if parallel:
            ind = np.zeros(D)
            ind_list = []
            for k in range(N ** D):
                ind_list.append(list(ind))

                # increase indices
                ind[0] += 1
                for l in range(D):
                    if ind[l] >= N:
                        if l + 1 < D:
                            ind[l+1] += 1
                        ind[l] = 0

            import multiprocessing
            import itertools
            N_CPU = multiprocessing.cpu_count()
            n_per_cpu = int(np.ceil(len(ind_list) / float(N_CPU)))
            partial_list = []
            for i in range(N_CPU):
                start = n_per_cpu * i
                end = n_per_cpu * (i + 1)
                end = min(end, len((ind_list)))
                if start >= end:
                    continue
                partial_list.append(ind_list[start:end])

            pool = multiprocessing.Pool(N_CPU)
            logZ_list = pool.map(self.logZ_FacLoc_star, zip(partial_list, itertools.repeat(order)))
            pool.close()
            pool.join()

            for res in logZ_list:
                logZ = lse([logZ, res])
        else:
            ind = np.zeros(D)
            for k in range(N ** D):
                logZ = lse([logZ, self.logZ_FacLoc(ind, order)])

                # increase indices
                ind[0] += 1
                for l in range(D):
                    if ind[l] >= N:
                        if l + 1 < D:
                            ind[l+1] += 1
                        ind[l] = 0
        time2 = time.time()
        
        print("It took %f seconds." % ((time2 - time1)))

        return logZ + self.n_logz[0]



