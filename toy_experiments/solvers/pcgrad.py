import numpy as np
import copy, random


def pcgrad_search(multi_obj_fg, x=None,
           max_iters=200, n_dim=20, step_size=1):
    x = np.random.randn(n_dim) if x is None else x
    fs = [multi_obj_fg(x)[0]]
    xs = [x]
    for t in range(max_iters):
        f, f_dx = multi_obj_fg(x)

        x = x - step_size * pcgrad_project_conflicting(f_dx)
        
        fs.append(f)
        xs.append(x)

    res = {'ls': np.stack(fs), 'xs': np.stack(xs)}
    return x, res


def pcgrad_project_conflicting(f_dx):
    pc_grad, num_task = copy.deepcopy(f_dx), len(f_dx)
    
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = np.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= (g_i_g_j) * g_j / (np.linalg.norm(g_j)**2)
    
     # follow the original paper with `sum`
    merged_grad = np.sum(pc_grad, axis=0)
    return merged_grad