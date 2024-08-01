import numpy as np

from problems.toy_biobjective import *
from solvers import (
    epo_search, 
    pareto_mtl_search, 
    linscalar, 
    moo_mtl_search, 
    moo_mtl_pp_search,
    pcgrad_search
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 18})



colors = list(mcolors.TABLEAU_COLORS.values())

if __name__ == '__main__':
    K = 50       # Number of trajectories
    n = 2       # dim of solution space
    m = 2       # dim of objective space

    rs = circle_points(K)  # preference

    pmtl_K = 5
    pmtl_refs = circle_points(pmtl_K, 0, np.pi / 2)
    methods = ['MGDA', 'MGDA++']

    ss, mi = 0.1, 100
    pf = create_pf()
    ps = np.linspace(-1 / np.sqrt(2), 1 / np.sqrt(2), K+1)
    i = 0
    random_pts = [np.array([x0_, x0_]) + np.random.uniform(-1, 1, 2) for x0_ in ps]

    fig, axs = plt.subplots(figsize=(6, 6))
    ax=axs
    fig.subplots_adjust(left=.12, bottom=.12, right=.97, top=.97)
    ax.plot(pf[:, 0], pf[:, 1], lw=3, c='k', label='Pareto Front', linestyle='--', alpha=0.7)
    for i_method, method in enumerate(methods):
        i=0
        xs = []
        ax = axs
        
        last_ls = []
        for k, r in enumerate(rs):
            
            x0 = np.zeros(n)
            x0[range(0, n, 2)] = 0.3
            x0[range(1, n, 2)] = -.3
            x0 += 0.1 * np.random.randn(n)
            x0 = np.random.uniform(-1, 1, n) if method in ["MOOMTL", "LinScalar"] else x0

            x0 = ps[i] + np.random.uniform(-1, 1, 2)
            i+=1

            l0, _ = concave_fun_eval(x0)
            if k == 0 and i_method == 0:
                ax.scatter([l0[0]], [l0[1]], c='g', s=70, alpha=0.9,
                    zorder=2, label=r'$F(x_0)$')

            ax.scatter([l0[0]], [l0[1]], c='g', s=70, alpha=0.9,
                    zorder=2)
            if method == 'EPO':
                _, res = epo_search(concave_fun_eval, r=r, x=x0,
                                    step_size=ss, max_iters=100)
            if method == 'PMTL':
                _, res = pareto_mtl_search(concave_fun_eval,
                                           ref_vecs=pmtl_refs, r=r_inv, x=x0,
                                           step_size=0.2, max_iters=150)
            if method == 'LinScalar':
                _, res = linscalar(concave_fun_eval, r=r, x=x0,
                                   step_size=ss, max_iters=mi)
            if method == 'MOOMTL' or method == "MGDA":
                _, res = moo_mtl_search(concave_fun_eval, x=x0,
                                        step_size=0.2, max_iters=150)
            if method == 'MOOMTL_adam' or method == "MGDA_adam":
                _, res = moo_mtl_search(concave_fun_eval, x=x0,
                                        step_size=0.2, max_iters=150, use_adam=True)
            if method == "MGDA++":
                _, res = moo_mtl_pp_search(concave_fun_eval, x=x0,
                                        step_size=0.2, max_iters=150, eps=1e-2)
            if method == "PCGrad":
                _, res = pcgrad_search(concave_fun_eval, x=x0,
                                        step_size=0.2, max_iters=150)

            last_ls.append(res['ls'][-1])
            ls = res['ls']
            xs.append(res['xs'])
    
            # colorline(ax, ls[:, 0], ls[:, 1], cmap=plt.get_cmap('jet'), linewidth=2, alpha=0.9)
            plt.plot(ls[:, 0], ls[:, 1], linewidth=3, alpha=0.35, c=['red', 'blue'][i_method])

        last_ls = np.stack(last_ls)
        ax.scatter(last_ls[:, 0], last_ls[:, 1], s=100, c=['red', 'blue'][i_method], alpha=0.95, 
                marker=(5, 2), zorder=3, label=f"{method} output")
        # ax.scatter(0, 0, c='r', marker=(5, 1), label="Pareto Optimal", s=115, zorder=3)
        ax.set_xlabel(r'$F_1$')
        ax.set_ylabel(r'$F_2$')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.legend(loc='best')
        fig.set_size_inches(6, 5.5)
        fig.tight_layout(h_pad=0, w_pad=0)
    
    fig.savefig('figures/mgda_and_mgdapp_convergence' + '.pdf')
    plt.show()
