"""
Yatao (An) Bian <yatao.bian@gmail.com>
bianyatao.com 
May 13, 2019.
"""
import numpy as np 
import time 
import itertools as it 
import pickle 
import math 
import os
from functions.exp_specs import (flid_model_path, NM_BINS)

def supermakedirs(path, mode):
    if not path or os.path.exists(path):
        return []
    (head, tail) = os.path.split(path)
    res = supermakedirs(head, mode)
    os.mkdir(path)
    os.chmod(path, mode)
    res += [path]
    return res

class parameter():
    '''dummy class to act as struct'''
    pass

def xilogxi(x):
    """ x:  scalar """
    if 0 == x:
        return 0
    else:
        return x*np.log(x)
    
def entropyi(x):
    """x is a scalar"""
    return - xilogxi(x) - xilogxi(1 - x)

def xlogx(x):
    """ x:  nx1 vector """
    n = len(x)
    out = np.zeros((n, 1))
    
    for i in range(n):
        out[i] = xilogxi(x[i])
            
    return out 


def flid_multilinear(x, param):
#  x must be vector: nx1
    n = param.n
    D = param.D
    minusx = 1 - x
    
    f =  np.asscalar( np.dot(param.u_prime, x)[0])
    
    for d in np.arange(D):
       xid = x[ param.I[:,d] ]
       tmp = xid * np.expand_dims(param.Y[:,d], axis=1)
       prod_minusx = np.ones( (n,1) )
       
       for l in np.arange(n-2, -1, -1 ): # 0-index
           prod_minusx[l] = prod_minusx[l+1] * minusx[param.I[l+1,d] ]
       
       f += np.dot(tmp.T, prod_minusx)
        
    return np.asscalar(f[0])


def flid_pa_multilinear(x, param):
# x must be vector: nx1
# beta * multilinear1 + beta * multilinear2
    n = param.n
    D = param.D
    minusx = 1 - x
    
    num_folds = len( param.u)
    re = 0
    
    for fold_id in range(num_folds):
        f =  np.asscalar( np.dot(param.u_prime[fold_id], x)[0])
        
        for d in np.arange(D):
           xid = x[ param.I[fold_id][:,d] ]
           tmp = xid* np.expand_dims( param.Y[fold_id][:,d], axis=1)
           prod_minusx = np.ones( (n,1) )
           
           for l in np.arange(n-2, -1, -1 ): # 0-index
               prod_minusx[l] = prod_minusx[l+1] * minusx[param.I[fold_id][l+1,d]]
           
           f += np.dot(tmp.T, prod_minusx)
        
        re +=np.asscalar(f[0] * param.beta)
    
    return re

def flid_elbo(x, param):
#    x must be vector: nx1
    minusx = 1-x
    
    f = flid_multilinear(x, param)
    f = f - np.sum(xlogx(x)) - np.sum(xlogx(minusx) )
    return f

def flid_pa_elbo(x, param):
#    x must be vector: nx1
    minusx = 1 - x
    f = flid_pa_multilinear(x,param)
    f = f - np.sum(xlogx(x)) - np.sum(xlogx(minusx))
    
    return f

# calculate i-th partial derivative 
def flid_multilinear_gradi(x,i, param):
# gradi: 1*1 
    f = flid_multilinear

    xplus = x.copy(); xplus[i]=1
    xminus = x.copy(); xminus[i]=0
    gradi = f(xplus, param) - f(xminus, param)

    return gradi

# calculate i-th partial derivative for PA 
def flid_pa_multilinear_gradi(x, i, param):
# gradi: 1*1 
    n = param.n; f = flid_pa_multilinear

    xplus = x.copy(); xplus[i]=1
    xminus = x.copy(); xminus[i]=0
    gradi = f(xplus, param) - f(xminus, param)

    return gradi

def flid_multilinear_grad(x, param):
# grad: n*1 
    n = param.n
    f = flid_multilinear
    grad = np.zeros( (n,1) )
    
    for i in np.arange(n):
       xplus = x.copy(); xplus[i]=1
       xminus = x.copy(); xminus[i]=0
       grad[i] = f(xplus, param) - f(xminus, param)

    return grad 

def flid_pa_multilinear_grad(x, param):
# grad: n*1 
    n = param.n
    f = flid_pa_multilinear
    grad = np.zeros( (n,1) )
    
    for i in np.arange(n):
       xplus = x.copy(); xplus[i]=1
       xminus = x.copy(); xminus[i]=0
       grad[i] = f(xplus, param) - f(xminus, param)

    return grad 

def load_flid_data(dataset_id, D_, n_, data_fig_path, category, folds_):
    """      
        category: string, category name
        Flid amazon, single fold models 
    """
    if 1 == dataset_id:
        D = D_
        fold = folds_[0] #  1~10
        
        result_flid_f = \
                '{0}/{1}_flid_d_{2}_fold_{3}.pkl'.format(\
                        flid_model_path, category, D, fold)
                
        fhandle = open(result_flid_f, 'rb')
        results_flid = pickle.load(fhandle, fix_imports=True)
        
        flid = results_flid['model']
        
        W= flid.parameters[1]
        u = flid.parameters[0]
        n = u.shape[0]
        
        Y = np.sort(W, axis=0) # sort  according to axis 0
        I = np.argsort(W, axis=0)
        
        u_prime = u - np.sum(W, axis=1)
        
        param = parameter()
        param.logZ = - flid.n_logz
        param.n = n
        param.D = D
        param.W=W
        param.Y=Y
        param.I = I
        param.u = u
        param.u_prime = u_prime
        param.ub = np.ones( [n, 1] )
        param.lb = np.zeros( [n, 1] )
        
        f = flid_elbo   #  function handle  
        grad = flid_multilinear_grad
        gradi = flid_multilinear_gradi
        param.multilinear = flid_multilinear     
        
        
    if 2 == dataset_id:  #  PA-ELBO on FLID data
        
        beta = 1
        D = D_
        folds = folds_
        W =[]; I = []; Y = []; u = []; u_prime=[]; FV=[]
        logz = 0
        
        for fold in folds:
            result_flid_f = '{0}/{1}_flid_d_{2}_fold_{3}.pkl'.format(\
                            flid_model_path, category, D, fold)
                    
            results_flid = pickle.load(open(result_flid_f, 'rb'), \
                                       fix_imports=True)        
            
            flid = results_flid['model']
            logz -= beta*flid.n_logz
            w = flid.parameters[1]
            W.append (w )
            u.append(flid.parameters[0] )
            
            y = np.sort(w, axis = 0)
            Y.append ( y) # sort  according to axis 0
            I.append( np.argsort(w, axis=0))
            
            u_prime.append (flid.parameters[0] - np.sum(w, axis=1) )
            
            fv = np.sum(flid.parameters[0] ) -  \
            np.sum( np.sum( y[:-1, :], axis=0 ), axis=0)
            FV.append(fv)
        
        n = u[0].shape[0]
        
        param=parameter()
        
        param.logZ = logz
        param.beta = beta
        param.n = n
        param.D = D
        param.W=W
        param.Y=Y
        param.I = I
        param.u = u
        param.u_prime = u_prime
        param.FV=FV
        param.ub = np.ones( [n, 1] )
        param.lb = np.zeros( [n, 1] )
        
        f=flid_pa_elbo  #  function handle  
        grad = flid_pa_multilinear_grad
        gradi = flid_pa_multilinear_gradi
        param.multilinear = flid_pa_multilinear
    
    return f, grad, gradi, param


def x2marginals(x):
    # it is one identity mapping here
    return x

def sigmoid(u):
    return 1.0/( 1.0 + np.exp(-u) )

"""DR-DoubleGreedy. Algorithm 1 in the paper."""
def solver_dr_double_greedy(f, grad, gradi, param, max_iter, **kwargs):
    """Accelerated using the multilinear nature of the multilinear objective."""
  
    co_order = None
    if 'co_order' in kwargs:
        co_order = kwargs['co_order']
        
    a=param.lb; b = param.ub
    
    x = a.copy(); y = b.copy()
    fsx = []; fsy = []
    
    fvx = f(x, param); fsx.append(fvx)
    fvy = f(y, param); fsy.append(fvy)
    id_seq = co_order[0]
    
    t = time.time()
    for i in id_seq:
    
        gradix = gradi(x, i, param)
        ua = sigmoid(gradix)
        
        gradiy = gradi(y, i, param)
        ub = sigmoid(gradiy)
        
        delta_a = (ua - x[i]) * gradix + entropyi(ua) - entropyi(x[i])
        delta_b = (ub - y[i]) * gradiy + entropyi(ub) - entropyi(y[i])
        delta_a = max(0, delta_a); delta_b = max(0, delta_b)
        
        if 0 == delta_a and 0 == delta_b:
            ra = 1
        else:
            ra = delta_a/(delta_a + delta_b)
        
        u = ra*ua + (1-ra)*ub
        fvx = fsx[-1] + (u - x[i]) * gradix + entropyi(u) - entropyi(x[i])
        fvy = fsy[-1] + (u - y[i]) * gradiy + entropyi(u) - entropyi(y[i])
        
        x[i] = u; y[i]=u
        fsx.append(fvx); fsy.append(fvy)
        
    run_time = time.time() - t
    
    opt_f = fsx[-1]
    fs = max(fsx, fsy)
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time


"""Multiepoch extension of DR-DoubleGreedy"""
def solver_dr_double_greedy_multiepoch(f, grad, gradi, param, extra_epoch, **kwargs):

    co_order = None
    if 'co_order' in kwargs:
        co_order = kwargs['co_order']

    a=param.lb; b = param.ub
    
    x = a.copy(); y = b.copy()
    fsx = [];fsy = []
    
    fvx = f(x, param); fsx.append(fvx)
    fvy = f(y, param); fsy.append(fvy)
    
    id_seq = co_order[0]
    
    t = time.time()
    for i in id_seq:
    
        gradix = gradi(x, i, param)
        ua = sigmoid(gradix)
        
        gradiy = gradi(y, i, param)
        ub = sigmoid(gradiy)
        
        delta_a = (ua - x[i]) * gradix + entropyi(ua) - entropyi(x[i])
        delta_b = (ub - y[i]) * gradiy + entropyi(ub) - entropyi(y[i])
        delta_a = max(0, delta_a); delta_b = max(0, delta_b)
        
        if 0 == delta_a and 0 == delta_b:
            ra = 1
        else:
            ra = delta_a/(delta_a + delta_b)
        
        u = ra*ua + (1-ra)*ub
        fvx = fsx[-1] + (u - x[i]) * gradix + entropyi(u) - entropyi(x[i])
        fvy = fsy[-1] + (u - y[i]) * gradiy + entropyi(u) - entropyi(y[i])
        
        x[i] = u; y[i]=u
        fsx.append(fvx); fsy.append(fvy)
        
    fsx = max(fsx, fsy)
    
    for epoch in np.arange(extra_epoch):
        id_seq = co_order[epoch+1]
        for i in id_seq:
            fvx = fsx[-1]
            gradix = gradi(x,i,param)
            ua = sigmoid(gradix)
            
            x[i] = ua
            delta_a = f(x, param) - fvx
            
            fvx = delta_a + fvx
            fsx.append(fvx)
    
    run_time = time.time() - t
    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time


def disc2value(values, disc):
    arr = np.array( [ row[i] for row, i in zip(values, disc) ] )
    return arr[:, None]

"""Algorithm 1 in:
Soma, T. and Yoshida, Y. Non-monotone dr-submodular
function maximization. In AAAI, volume 17, pp. 898–904, 2017.
"""
def solver_SY_alg1(f, grad, gradi, param, max_iter, **kwargs):
    n = param.n
    co_order = None
    if 'co_order' in kwargs:
        co_order = kwargs['co_order']
    else:
        co_order = list(range(n)) * (1 + extra_epoch)
    a=param.lb; b = param.ub
    
    values = np.linspace(np.squeeze(a), np.squeeze(b), NM_BINS + 1).T 
    
    x = np.zeros(n, dtype=int)
    y = NM_BINS * np.ones(n, dtype=int)
        
    fsx = []; fsy = []
    
    fvx = f(disc2value(values, x), param); fsx.append(fvx)
    fvy = f(disc2value(values, y), param); fsy.append(fvy)
    id_seq = co_order[0]
    
    t = time.time()
    for id, i in enumerate(id_seq):
    
        fvx = fsx[-1]; fvy = fsy[-1]
        while x[i] < y[i]:
            xtmp = x.copy(); xtmp[i] += 1
            delta_a = f(disc2value(values, xtmp), param) - fvx
            
            ytmp = y.copy(); ytmp[i] -= 1
            delta_b = f(disc2value(values, ytmp), param) - fvy
            
            if delta_b < 0:
                x[i] += 1
            elif delta_a < 0:
                y[i] -= 1
            else:
                if 0 == delta_a and 0 == delta_b:
                    x[i] += 1
                else:
                    ra = delta_a/(delta_a + delta_b)
                    choice = np.random.random() < ra
                    if choice: 
                        x[i] += 1
                    else:
                        y[i] -= 1
            fvx = f(disc2value(values, x), param); fvy = f(disc2value(values, y), param)
        fsx.append(fvx); fsy.append(fvy)
        
    run_time = time.time() - t
    
    opt_f = fsx[-1]
    fs = max(fsx, fsy)
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time

"""
Algorithm 4 in
Niazadeh, R., Roughgarden, T., and Wang, J. 
Optimal algorithms for continuous non-monotone submodular and
dr-submodular maximization. 
In Advances in Neural Information Processing Systems (NeurIPS), 
pp. 9617–9627. 2018.
"""
def solver_bscb(f, grad, gradi, param, extra_epoch, **kwargs):
    epsilon = 1e-3
    n = param.n
    co_order = None
    if 'co_order' in kwargs:
        co_order = kwargs['co_order']
    else:
        co_order = list(range(n)) * (1 + extra_epoch)
        
    a=param.lb; b = param.ub
    
    x = a.copy(); y = b.copy()
    fsx = []; fsy = []
    
    fvx = f(x, param); fsx.append(fvx)
    fvy = f(y, param); fsy.append(fvy) 
    id_seq = co_order[0]
    
    t = time.time()
    for i in id_seq:
        xi = a[i]; yi = b[i]
        fvx = fsx[-1]; fvy = fsy[-1]
        
        gradix = gradi(x, i, param)
        gradiy = gradi(y, i, param)
        
        # For ELBO and PA-ELBO, the first two cases would never happen.
        # find the root of gradix * (1-u) + gradiy * u + log ((1-u)/u) = 0
        u = 0
        while yi - xi > epsilon/n:
          u = (xi + yi)/2
          if gradix * (1-u) + gradiy * u + math.log ((1-u)/u) > 0:
            xi = u 
          else:
            yi = u
        x[i] = u; y[i] = u
        fvx = f(x, param); fvy = f(y, param)
        fsx.append(fvx); fsy.append(fvy)
    
    run_time = time.time() - t
    opt_f = fsx[-1]
    fs = max(fsx, fsy)
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time

"""Multiepoch extension of BSCB"""
def solver_bscb_multiepoch(f, grad, gradi, param, extra_epoch, **kwargs):
    epsilon = 1e-3
    n = param.n
    co_order = None
    if 'co_order' in kwargs:
        co_order = kwargs['co_order']
    else:
        co_order = list(range(n)) * (1 + extra_epoch)
        
    a=param.lb; b = param.ub
    
    x = a.copy(); y = b.copy()
    fsx = []; fsy = []
    
    fvx = f(x, param); fsx.append(fvx)
    fvy = f(y, param); fsy.append(fvy) 
    id_seq = co_order[0]
    
    t = time.time()
    for i in id_seq:
        xi = a[i]; yi = b[i]
        fvx = fsx[-1]; fvy = fsy[-1]
        
        gradix = gradi(x, i, param)
        gradiy = gradi(y, i, param)
        
        # For ELBO and PA-ELBO, the first two cases would never happen.
        # find the root of gradix * (1-u) + gradiy * u + log ((1-u)/u) = 0
        u = 0
        while yi - xi > epsilon/n:
          u = (xi + yi)/2
          if gradix * (1-u) + gradiy * u + math.log ((1-u)/u) > 0:
            xi = u 
          else:
            yi = u
        x[i] = u; y[i] = u
        fvx = f(x, param); fvy = f(y, param)
        fsx.append(fvx); fsy.append(fvy) 
         
    fsx = max(fsx, fsy)
    
    for epoch in np.arange(extra_epoch):
        id_seq = co_order[epoch+1]
        for i in id_seq:
            fvx = fsx[-1]
            gradix = gradi(x,i,param)
            ua = sigmoid(gradix) 
            
            x[i] = ua
            delta_a = f(x, param) - fvx
            
            fvx = delta_a + fvx
            fsx.append(fvx)    
    
    run_time = time.time() - t
    opt_f = fsx[-1]
    fs = fsx
    opt_x = x 
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time
  

def _shrunken_lmo(grad_t, param, x):
  a = param.lb; b = param.ub; n = param.n
  ub = b - x
  vm = np.zeros([n, 1])
  for i in range(n):
    vm[i] = ub[i] if grad_t[i] >= 0 else a[i]
  
  return vm

def solver_shrunken_frank_wolfe(f, grad, gradi, param, max_iter, **kwargs): 

    a=param.lb; n = param.n
    
    x = a.copy()
    fsx = []
    fvx = f(x, param); fsx.append(fvx)
    
    more_epochs = 10
    num_epochs = more_epochs*(max_iter +1)
 
    tt = time.time()
    gamma_cons = 1/num_epochs

    t = 0
    while t < 1:
        grad_t = grad(x, param)
        vm = _shrunken_lmo(grad_t, param, x)
        gamma = min(gamma_cons, 1-t)
        x = x + gamma*vm
        t = t + gamma
        fvx = f(x, param)
        fsx.append(fvx)

    run_time = time.time() - tt
    
    opt_f = fsx[-1]
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fsx, margs, run_time


def solver_submodular_double_greedy(f, grad, gradi, param, extra_epoch, **kwargs):

    n = param.n
    co_order = None
    if 'co_order' in kwargs:
        co_order = kwargs['co_order']
    else:
        co_order = list(range(n)) * (1 + extra_epoch)
        
    a=param.lb; b = param.ub
    
    x = a.copy(); y = b.copy()
    
    fsx = []
    fsy = []    
    fvx =  f(x, param)
    fsx.append(fvx)
    
    fvy = f(y, param)
    fsy.append(fvy)
    id_seq = co_order[0]
    
    t = time.time()
    for i in id_seq:
    
        fvx = fsx[-1]; fvy = fsy[-1]
        gradix = gradi(x,i,param)
        ua = sigmoid(gradix)
        
        gradiy = gradi(y, i,param)
        ub = sigmoid(gradiy)
        
        xtmp = x.copy()
        xtmp[i] = ua;
        delta_a = f(xtmp, param) - fvx
        
        ytmp = y.copy();
        ytmp[i] = ub
        delta_b = f(ytmp, param) - fvy;
        
        if delta_a >= delta_b:
            x[i] = ua; y[i] = ua
            fvx = fvx + delta_a
            fsx.append(fvx)
            fsy.append ( f(y, param))
        else:
            y[i] = ub; x[i] = ub
            fvy = fvy + delta_b
            fsy.append(fvy)
            fsx.append ( f(x, param) )

    run_time = time.time() - t
    
    opt_f = fsx[-1]
    fs = max(fsx, fsy)
    opt_x = x
    margs = x2marginals(opt_x)
    
    return opt_x, opt_f, fs, margs, run_time

def solver_submodular_double_greedy_multiepoch(f, grad, gradi, param, extra_epoch, **kwargs):

    n = param.n
    co_order = None
    if 'co_order' in kwargs:
        co_order = kwargs['co_order']
    else:
        co_order = list(range(n)) * (1 + extra_epoch)
        
    a=param.lb; b = param.ub
    
    x = a.copy(); y = b.copy()
    fsx = []
    fsy = []
    
    fvx =  f(x, param)
    fsx.append(fvx)
    
    fvy = f(y, param)
    fsy.append(fvy)
    id_seq = co_order[0]
    
    t =time.time()
    for i in id_seq:
    
        fvx = fsx[-1]; fvy = fsy[-1]
        gradix = gradi(x,i,param)
        ua = sigmoid(gradix)
        
        gradiy = gradi(y, i,param)
        ub = sigmoid(gradiy)
        
        xtmp = x.copy()
        xtmp[i] = ua
        delta_a = f(xtmp, param) - fvx
        
        ytmp = y.copy()
        ytmp[i] = ub
        delta_b = f(ytmp, param) - fvy
        
        if delta_a >= delta_b:
            x[i] = ua; y[i] = ua
            fvx = fvx + delta_a
            fsx.append(fvx)
            fsy.append ( f(y, param))
        else:
            y[i] = ub; x[i] = ub
            fvy = fvy+ delta_b
            fsy.append(fvy)
            fsx.append ( f(x, param) )
    
    fsx = max(fsx, fsy)
    for epoch in range(extra_epoch):
        id_seq = co_order[1+epoch]
        for i in id_seq:
            
            gradix = gradi(x,i,param)
            ua = sigmoid(gradix)
            
            x[i] = ua
            delta_a = f(x, param) - fvx
            
            fvx = delta_a + fvx
            fsx.append(fvx);

    run_time = time.time() - t
    
    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)
    
    return opt_x, opt_f, fs, margs, run_time

def solver_coordinate_ascent_0(f, grad, gradi, param, extra_epoch, **kwargs):
    a = param.lb
    return subroutine_coordinate_ascent(f, grad, gradi, param, extra_epoch, a.copy(), **kwargs)

def subroutine_coordinate_ascent(f, grad, gradi, param, extra_epoch, init, **kwargs):
    n = param.n
    co_order = None
    if 'co_order' in kwargs:
        co_order = kwargs['co_order']
    else:
        co_order = list(range(n)) * (1 + extra_epoch)
        
    x = init 
    fsx = []
    fvx = f(x, param); fsx.append(fvx)
    id_seq = co_order[0]
    
    t = time.time()
    for i in id_seq:
        fvx = fsx[-1]
        gradix = gradi(x, i, param)
        ua = sigmoid(gradix)
        
        x[i] = ua
        fsx.append(f(x, param))
    
    for epoch in range(extra_epoch):
        id_seq = co_order[1+epoch]
        for i in id_seq:
            gradix = gradi(x,i,param);
            ua = sigmoid(gradix)
            
            x[i] = ua
            fsx.append(f(x, param))
                
    run_time = time.time() - t
    
    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time    


def solver_coordinate_ascent_1(f, grad, gradi, param, extra_epoch, **kwargs):
    b = param.ub
    return subroutine_coordinate_ascent(f, grad, gradi, param, extra_epoch, b.copy(), **kwargs)


def solver_coordinate_ascent_random(f, grad, gradi, param, extra_epoch, **kwargs):
    a = param.lb; b = param.ub
    x = np.random.random([param.n, 1])*(b-a) + a
    return subroutine_coordinate_ascent(f, grad, gradi, param, extra_epoch, x, **kwargs)


"""
using exhausitive search 
opt_x:  the marginals 
opt_f:  log Z
complexity:  O(2^n)
"""
def solver_exhaustive_search(f, grad, gradi, param, extra_epoch, **kwargs):

    n = param.n
    
    fmultilinear = param.multilinear
    t = time.time()
    Z = 0
    
    ground = [ [0,1] for _ in range(n) ]
    
    powerset = it.product(*ground)
    for dd in powerset:
        
        x = np.array(dd)[:,np.newaxis]
        Z = Z + np.exp( fmultilinear(x,param) )
    
    run_time = time.time() - t
    
    opt_f = np.log(Z)
    fs = opt_f
    
    # calculate margs 
    opt_x = np.zeros( [n,1] ); margs = opt_x
    
    for i  in range(n):
        Zi = 0; 
        ground = [ [0,1] for _ in range(n) ]
        ground[i] = [1];
        powerset = it.product(*ground)

        for dd in powerset:
            x = np.array(dd)[:,np.newaxis];
            Zi += np.exp( fmultilinear(x,param) )
        
        margs[i] = Zi/Z
    
    return opt_x, opt_f, fs,  margs, run_time


""" 
complexity:  O(n^(D+1))
Using method from: 
Tschiatschek, S., Djolonga, J., and Krause, A. 
Learning probabilistic submodular diversity models via noise contrastive estimation. 
In Proc. International Conference on Artificial Intelligence and Statistics (AISTATS), 2016.
"""
def solver_ground_truth_flid(f, grad, gradi, param, extra_epoch, **kwargs):

    n = param.n; D = param.D
    u_prime = param.u_prime
    
    t = time.time()
    I = param.I; Y=param.Y; W=param.W
    Z = 0
    
    id_seq = list (range(n))
        
    ground = [ id_seq for _ in range(D) ]
    powerset = it.product(*ground)
    
    for dd in powerset:
        
        idI = dd
        idW = np.zeros( [D ,1], dtype=int ) 
        
        for j in range(D):
            idW[j] = I[idI[j], j]
        
        II = np.unique(idW)
    
        X = []
        for j in range(D):
#            print(I[ idI[j]+1:, j])
            X =np.union1d(X, (I[ idI[j]+1:, j]) )
        
        if  np.intersect1d(II,X).size > 0:
            continue
        
        tmp_sum = 0;
        for d  in range(D):
            tmp_sum = tmp_sum + W[idW[d],d]
    
        
        for i in range (len(II)):
            tmp_sum = tmp_sum + u_prime[II[i]]
        
        Vprime = np.setdiff1d(id_seq, II)
        Vprime = np.setdiff1d(Vprime, X)
        
        tmp_prod = 1
        if Vprime.size >0:  
            for i in range (len(Vprime)):
                tmp_prod = tmp_prod*(1+np.exp(u_prime[Vprime[i]] ))
                
        Z = Z + ( np.exp(tmp_sum)*tmp_prod)[0]
    
    Z = Z +1
    run_time = time.time() - t 
    
    opt_f = np.log(Z)
    fs = opt_f
    opt_x = np.zeros( [n,1] )
    fm = param.multilinear
    
    for i in range (n):
        xtmp = np.zeros( [n,1] )
        xtmp[i] = 1
        opt_x[i] = fm(xtmp,param)/Z
    
    margs = opt_x
    
    return opt_x, opt_f, fs, margs,  run_time


def launch_solver(f, grad, gradi, param, method, extra_epoch, 
                  co_order=None):
    
    # co_order:  coordinate order, length = n*(extra_epoch + 1)
    
    func_names = (
    'solver_dr_double_greedy',                          # 0
    'solver_none',                                      # 1
    'solver_shrunken_frank_wolfe',                      # 2
    'solver_submodular_double_greedy',                  # 3
    'solver_dr_double_greedy_multiepoch',               # 4
    'solver_coordinate_ascent_0',                       # 5
    'solver_coordinate_ascent_1',                       # 6
    'solver_ground_truth_flid',                         # 7
    'solver_exhaustive_search',                         # 8
    'solver_coordinate_ascent_random',                  # 9
    'solver_submodular_double_greedy_multiepoch',       # 10
    'solver_bscb',                                      # 11
    'solver_bscb_multiepoch',                           # 12
    'solver_SY_alg1'                                    # 13
    )
    
    func = eval(func_names[method])
    
    if co_order is not None:
        [x_opt, opt_f, fs, margs, runtime] \
            = func(f, grad, gradi, param, extra_epoch,
                   co_order=co_order)
    else:
        [x_opt, opt_f, fs, margs, runtime] \
            = func(f, grad, gradi, param, extra_epoch)
    
    return x_opt, opt_f, fs, margs, runtime
