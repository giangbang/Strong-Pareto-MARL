from .moo_mtl import get_d_moomtl

import numpy as np


def moo_mtl_pp_search(multi_obj_fg, x=None,
                   max_iters=200, n_dim=20, step_size=1, eps=1e-1):
    """
    MGDA++
    eps is the `eps` in the paper.
    """
    # x = np.random.uniform(-0.5,0.5,n_dim)
    x = np.random.randn(n_dim) if x is None else x
    fs = [multi_obj_fg(x)[0]]
    xs = [x]
    for t in range(max_iters):
        f, f_dx = multi_obj_fg(x)
        shape = f_dx.shape
        
        lf = len(f_dx)
        f_dx = list(filter(lambda x : np.linalg.norm(x) > eps, f_dx))
        lr = len(f_dx)
      
        f_dx = np.array(f_dx)
        if len(f_dx) == 0: f_dx = np.zeros(shape)

        weights = get_d_moomtl(f_dx)

        x = x - step_size * np.dot(weights.T, f_dx).flatten()
        fs.append(f)
        xs.append(x)

    res = {'ls': np.stack(fs), 'xs': np.stack(xs)}
    return x, res
