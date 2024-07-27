import numpy as np

from problems.toy_biobjective2 import *
from solvers import epo_search, pareto_mtl_search, linscalar, moo_mtl_search, moo_mtl_pp_search

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from plt_ultils import colorline

import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 18})


colors = list(mcolors.TABLEAU_COLORS.values())

if __name__ == '__main__':
    K = 25       # Number of trajectories
    n = 2      # dim of solution space
    m = 2       # dim of objective space

    rs = circle_points(K)  # preference

    pmtl_K = 5
    pmtl_refs = circle_points(pmtl_K, 0, np.pi / 2)
    # methods = ['MGDA', 'MGDA++']
    methods = ['MGDA_adam', 'MGDA++', 'MGDA']
    # latexify(fig_width=2., fig_height=1.55)
    ss, mi = 0.1, 100
    pf = create_pf(m=m)
    for method in methods:
        xs = []
        fig, axs = plt.subplots(2, 1, figsize=(6, 12))
        ax = axs[0]
        fig.subplots_adjust(left=.12, bottom=.12, right=.97, top=.97)
        ax.plot(pf[:, 0], pf[:, 1], lw=2, c='k', label='Pareto Stationary', linestyle='--', alpha=0.7)
        last_ls = []
        for k, r in enumerate(rs):
            
            x0 = np.random.uniform(-9, 9, n) 
            l0, _ = concave_fun_eval(x0)
            if k == 0:
                ax.scatter([l0[0]], [l0[1]], c='g', s=100, alpha=0.9,
                    zorder=2, label=r'$F(x_0)$')  
                axs[1].scatter([x0[0]], [x0[1]], c='g', s=100, alpha=0.9,
                    zorder=2, label=r'$F(x_0)$')  
            ax.scatter([l0[0]], [l0[1]], c='g', s=100, alpha=0.9,
                    zorder=2)
            axs[1].scatter([x0[0]], [x0[1]], c='g', s=100, alpha=0.9,
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
                                        step_size=0.1, max_iters=50)
            if method == 'MOOMTL_adam' or method == "MGDA_adam":
                _, res = moo_mtl_search(concave_fun_eval, x=x0,
                                        step_size=0.1, max_iters=50, use_adam=True)
            if method == "MGDA++":
                _, res = moo_mtl_pp_search(concave_fun_eval, x=x0,
                                        step_size=0.1, max_iters=50)
            last_ls.append(res['ls'][-1])
            ls = res['ls']
            xs.append(res['xs'])
    
            colorline(ax, ls[:, 0], ls[:, 1], cmap=plt.get_cmap('jet'), linewidth=3, alpha=0.9)

        last_ls = np.stack(last_ls)
        ax.scatter(last_ls[:, 0], last_ls[:, 1], s=100, c=colors[1], alpha=0.99, 
                marker=(5, 2), zorder=3, label=f"{method} output")
        ax.scatter(0, 0, c='r', marker=(5, 1), label="Pareto Optimal", s=115, zorder=3)
        ax.set_xlabel(r'$F_1$')
        ax.set_ylabel(r'$F_2$')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.legend(loc='best')
        

        # the second plot
        
        ax = axs[1]
        delta = 0.025
        x = np.arange(-10.0, 10.0, delta)
        y = np.arange(-10.0, 10.0, delta)
        X, Y = np.meshgrid(x, y)
        inp = np.stack((X[..., None], Y[..., None]), axis=-1)

        inp = inp.reshape(-1, 2)
        # print(f1(inp).shape, f2(inp).shape)
        z = np.stack([f1(inp), f2(inp)], axis=-1).reshape(*X.shape, 2)
        Z=z

        CS = ax.contour(X, Y, z[..., 1], colors='b', label=r'$l_1(\theta)$', lw=2, alpha=0.4)
        ax.clabel(CS, inline=False, fontsize=5)
        CS = ax.contour(X, Y, z[..., 0], colors='g', label=r'$l_1(\theta)$', lw=2, alpha=0.4)
        ax.clabel(CS, inline=False, fontsize=5)

        xs = np.array(xs)

        pf1 = np.array([[0, 5], [0, -5]])
        pf2 = np.array([[-5, 0], [5, 0]])
        ax.plot(pf1[:, 0], pf1[:, 1], lw=2, c='k', label='Pareto Stationary', linestyle='--', alpha=0.7)
        ax.plot(pf2[:, 0], pf2[:, 1], lw=2, c='k', linestyle='--', alpha=0.7)
        ax.scatter(0, 0, c='r', marker=(5, 1), label="Pareto Optimal", s=115, zorder=3)

        ax.scatter(xs[..., -1, 0], xs[..., -1, 1], s=125, c=colors[1], alpha=0.99, 
                marker=(5, 2), zorder=3, label=f"{method} output")
        
        for line in xs:
            colorline(ax, line[:, 0], line[:, 1], cmap=plt.get_cmap('jet'), linewidth=3, alpha=0.9)

        plt.tight_layout()
        # ax.legend(loc='best')

        fig.set_size_inches(6, 11)
        fig.tight_layout(h_pad=0, w_pad=0)

        fig.savefig('figures/' + method + '.pdf')

    plt.show()
