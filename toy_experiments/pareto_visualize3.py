import numpy as np

from problems.toy_biobjective3 import *
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

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from plt_ultils import colorline


colors = list(mcolors.TABLEAU_COLORS.values())

if __name__ == '__main__':
    K = 1       # Number of trajectories
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

    # zoom in 

    xs_for_axins = []
    line_for_axins = []

    eps=0.01

    for i_method, method in enumerate(methods):
        i=0
        xs = []
        ax = axs
        
        last_ls = []
        for k, r in enumerate(rs):
            
            x0 = np.zeros(n)
            x0[range(0, n, 2)] = 0.3
            x0[range(1, n, 2)] = -.3
            x0 += np.random.randn(n)

            x0 = np.random.uniform(2, 2.95, 2)
            if k == len(rs)-1:
                x0 = np.array([1.5,2])
            i+=1

            l0, _ = concave_fun_eval(x0)
            if k == 0 and i_method == 0:
                axs.scatter([x0[0]], [x0[1]], c='g', s=70, alpha=0.9,
                    zorder=2, label=r'$x_0$')  
            
            axs.scatter([x0[0]], [x0[1]], c='g', s=70, alpha=0.9,
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
                                        step_size=0.01, max_iters=500)
            if method == 'MOOMTL_adam' or method == "MGDA_adam":
                _, res = moo_mtl_search(concave_fun_eval, x=x0,
                                        step_size=0.01, max_iters=500, use_adam=True)
            if method == "MGDA++":
                _, res = moo_mtl_pp_search(concave_fun_eval, x=x0,
                                        step_size=0.01, max_iters=500, eps=eps)
            if method == "PCGrad":
                _, res = pcgrad_search(concave_fun_eval, x=x0,
                                        step_size=0.01, max_iters=500)
            
            last_ls.append(res['ls'][-1])
            ls = res['ls']
            xs.append(res['xs'])

        last_ls = np.stack(last_ls)

        # the second plot
        
        ax = axs
        delta = 0.025
        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x, y)
        inp = np.stack((X[..., None], Y[..., None]), axis=-1)

        inp = inp.reshape(-1, 2)
        print(inp.shape)
        output1 = np.array([f1(ip) for ip in inp])
        output2 = np.array([f2(ip) for ip in inp])
        print(output1.shape, output2.shape)
        z = np.stack([output1, output2], axis=-1).reshape(*X.shape, 2)
        Z=z

        CS = ax.contour(X, Y, z[..., 1], colors='b', label=r'$F_1$', lw=2, alpha=0.2)
        ax.clabel(CS, inline=True, fontsize=10)
        CS = ax.contour(X, Y, z[..., 0], colors='g', label=r'$F_2$', lw=2, alpha=0.2)
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_xlabel(r'$x^1$')
        ax.set_ylabel(r'$x^2$')

        xs = np.array(xs)

        ax.scatter(xs[..., -1, 0], xs[..., -1, 1], s=80, c=['red', 'blue'][i_method], alpha=0.8, 
                marker=(5, 2), zorder=3, label=f"{method}")
        
        xs_for_axins.append((xs))
        
        for line in xs:
            plt.plot(line[:, 0], line[:, 1], linewidth=3, alpha=0.35, c=['red', 'blue'][i_method])
            line_for_axins.append((line, i_method))

        # pf2 = np.array([[-5, 0], [5, 0]])

    pf1 = np.array([[1, 1], [-1, -1]])
    ax.scatter([1,-1], [1, -1], c='k', s=115, zorder=3, alpha=0.3)
    ax.plot(pf1[:, 0], pf1[:, 1], lw=2, c='k', label='Pareto Set', linestyle='--', alpha=0.7, zorder=10)


    axins = zoomed_inset_axes(ax, 24, loc="lower right") 
    # axins.imshow(Z2, extent=(-3,3,-3,3))
    axins.scatter([1,-1], [1, -1], c='k', s=115, zorder=3, alpha=0.3)
    axins.plot(pf1[:, 0], pf1[:, 1], lw=2, c='k', linestyle='--', alpha=0.7, zorder=10)

    for i_xs, xs in enumerate(xs_for_axins):
        axins.scatter(xs[..., -1, 0], xs[..., -1, 1], s=250, c=['red', 'blue'][i_xs], alpha=1, 
                marker=(5, 2), zorder=11)
        
    for lines in (line_for_axins):
        line, i_line = lines
        axins.plot(line[:, 0], line[:, 1], linewidth=5, alpha=0.5, c=['red', 'blue'][i_line], zorder=2)

    circle2 = plt.Circle((1, 1), eps, color='blue', alpha=0.6, linestyle='--', fill=False)
    axins.add_patch(circle2)

    # sub region of the original image
    x1, x2, y1, y2 = 0.945, 1.055, 0.945, 1.055
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.setp(axins.get_xticklabels(), visible=False)
    plt.setp(axins.get_yticklabels(), visible=False)

    mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")

    fig.set_size_inches(5.5, 5.5)
    fig.tight_layout()
    ax.legend(loc='upper left')
    # fig.subplots_adjust(bottom=0.3)
    fig.savefig('figures/mgda_and_mgdapp_convergence2' + '.pdf')
    plt.show()
