# This code is from
# Multi-Task Learning as Multi-Objective Optimization
# Ozan Sener, Vladlen Koltun
# Neural Information Processing Systems (NeurIPS) 2018 
# https://github.com/intel-isl/MultiObjectiveOptimization

import numpy as np

from .min_norm_solvers_numpy import MinNormSolver


adam_params = {
    "beta1": 0.9,
    "beta2": 0.999,
    "decay": 0,
    "m": 0 ,
    "v": 0
}

def adam_update(x, g, step_size):
    # warning: you need to reset m and v in `adam_params` with every new training
    if adam_params["decay"] != 0:
        g = g + adam_params["decay"] * x
    adam_params["m"] = adam_params["m"] * adam_params["beta1"] + \
                        (1 - adam_params["beta1"]) * g
    adam_params["v"] = adam_params["beta2"] * adam_params["v"] + \
                        (1 - adam_params["beta2"]) * g**2
    m_bar = adam_params["m"] / (1-adam_params["beta1"])
    v_bar = adam_params["v"] / (1-adam_params["beta2"])
    # default not amsgrad
    x = x - step_size * m_bar / (np.sqrt(v_bar) + 1e-6)
    return x


def moo_mtl_search(multi_obj_fg, x=None,
                   max_iters=200, n_dim=20, step_size=1, use_adam=False):
    """
    MOO-MTL
    """
    # x = np.random.uniform(-0.5,0.5,n_dim)
    x = np.random.randn(n_dim) if x is None else x
    fs = [multi_obj_fg(x)[0]]
    xs = [x]
    for t in range(max_iters):
        f, f_dx = multi_obj_fg(x)

        weights = get_d_moomtl(f_dx)

        if not use_adam:
            x = x - step_size * np.dot(weights.T, f_dx).flatten()
        else:
            x = adam_update(x, np.dot(weights.T, f_dx).flatten(), step_size)
        fs.append(f)
        xs.append(x)

    res = {'ls': np.stack(fs), 'xs': np.stack(xs)}
    return x, res


def get_d_moomtl(grads):
    """
    calculate the gradient direction for MOO-MTL
    """

    nobj, dim = grads.shape
    if nobj <= 1:
        return np.array([1.])

#    # use cvxopt to solve QP
#    P = np.dot(grads , grads.T)
#
#    q = np.zeros(nobj)
#
#    G =  - np.eye(nobj)
#    h = np.zeros(nobj)
#
#
#    A = np.ones(nobj).reshape(1,2)
#    b = np.ones(1)
#
#    cvxopt.solvers.options['show_progress'] = False
#    sol = cvxopt_solve_qp(P, q, G, h, A, b)
    # print(f'grad.shape: {grads.shape}')
    # use MinNormSolver to solve QP
    sol, nd = MinNormSolver.find_min_norm_element(grads)

    return sol
