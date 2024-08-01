import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt
import matplotlib as mpl

def f(x):
    return np.sum(np.square(x), axis=-1)

t = 5
# def f(x):
#     return np.linalg.norm(x,axis=-1)**2

def fi(x, i):
    x = x.copy()
    xi = x[..., i]
    x[..., i][np.logical_and(xi > -t, xi < t)] = 0
    x[..., i][xi <= -t] += t 
    x[..., i][xi >= t] -= t 
    # if xi > -t and xi < t:
    #     x[..., i] = 0
    # elif xi <= -t:
    #     x[..., i] += t
    # else: 
    #     x[..., i] -= t
    return f(x)

def f1(x):
    return fi(x, 0)

def f2(x):
    return fi(x, 1)

# calculate the gradients using autograd
f_dx = grad(f)

def fi_dx(x, i):
    x = x.copy()
    xi = x[..., i]
    x[..., i][np.logical_and(xi > -t, xi < t)] = 0
    x[..., i][xi <= -t] += t 
    x[..., i][xi >= t] -= t 
    return f_dx(x)

def f1_dx(x):
    return fi_dx(x, 0)

def f2_dx(x):
    return fi_dx(x, 1)

def concave_fun_eval(x):
    """
    return the function values and gradient values
    """
    ret = np.stack([f1(x), f2(x)]), np.stack([f1_dx(x), f2_dx(x)])
    if np.isnan(ret[1]).any():
        print("nan", x)
    return ret


# ### create the ground truth Pareto front ###
def create_pf(side_nonpf=False, m=3):
    """
    if `side_nonpf` is True, then the boundary of attainable objectives,
    which lie adjacent to the PF, is also returned.
    """
    pf = np.array([[t**2, 0], [0, 0], [0, t**2]])

    return pf


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y] * 2


def add_interval(ax, xdata, ydata,
                 color="k", caps="  ", label='', side="both", lw=2):
    line = ax.add_line(mpl.lines.Line2D(xdata, ydata))
    line.set_label(label)
    line.set_color(color)
    line.set_linewidth(lw)
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 12,
        'color': line.get_color()
    }
    a = []
    if side in ["left", "both"]:
        a0 = ax.annotate(caps[0], xy=(xdata[0], ydata[0]), zorder=2, **anno_args)
        a.append(a0)
    if side in ["right", "both"]:
        a1 = ax.annotate(caps[1], xy=(xdata[1], ydata[1]), zorder=2, **anno_args)
        a.append(a1)
    return (line, tuple(a))


if __name__ == '__main__':
    # latexify(fig_width=2.25, fig_height=1.8)
    delta = 0.025
    plt.rcParams.update({'font.size': 18})
    x = np.arange(-10.0, 10.0, delta)
    y = np.arange(-10.0, 10.0, delta)
    X, Y = np.meshgrid(x, y)
    inp = np.stack((X[..., None], Y[..., None]), axis=-1)

    inp = inp.reshape(-1, 2)
    z = np.stack([f1(inp), f2(inp)], axis=-1).reshape(*X.shape, 2)
    Z=z


    fig, ax = plt.subplots()
    # CS = ax.contour(X, Y, z[..., 0])
    CS = ax.contour(X, Y, z[..., 1], colors='b', label=r'$l_1(\theta)$', lw=2)
    ax.clabel(CS, inline=True, fontsize=10)
    CS = ax.contour(X, Y, z[..., 0], colors='g', label=r'$l_1(\theta)$', lw=2)
    ax.clabel(CS, inline=True, fontsize=10)
    # plt.legend()
    # ax.set_title('Simplest default with labels')
    # plt.imshow(Z[..., 1])
    ax.set_xlabel(r'$x^1$')
    ax.set_ylabel(r'$x^2$')
    fig.set_size_inches(6, 5)
    fig.tight_layout(h_pad=0, w_pad=0)
    plt.tight_layout()

    plt.savefig('../figures/moo_synthetic.pdf')   # for paper
    # plt.savefig('../figures/moo_synthetic_ppt_just_losses.pdf')     # for ppt
    plt.show()

