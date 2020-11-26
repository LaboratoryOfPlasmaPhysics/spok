import matplotlib.pyplot as plt
import numpy as np


def layout_EarthEnv_3planes(**kwargs):
    # figsize=(15,4.5),
    # to be fixed!
    xlim = (-30, 30)
    ylim = (-30, 30)
    zlim = (-30, 30)
    alpha = 0.5

    figsize = kwargs.get("kwargs", (15, 4.5))  # kwargs?!

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=figsize, constrained_layout=True)

    ax0.set_xlabel('X (Re)')
    ax0.set_ylabel('Y (Re)')
    ax1.set_xlabel('X (Re)')
    ax1.set_ylabel('Z (Re)')
    ax2.set_xlabel('Y (Re)')
    ax2.set_ylabel('Z (Re)')
    ax0.axhline(0, color='k', ls='dotted', alpha=alpha)
    ax0.axvline(0, color='k', ls='dotted', alpha=alpha)
    ax1.axhline(0, color='k', ls='dotted', alpha=alpha)
    ax1.axvline(0, color='k', ls='dotted', alpha=alpha)
    ax2.axhline(0, color='k', ls='dotted', alpha=alpha)
    ax2.axvline(0, color='k', ls='dotted', alpha=alpha)
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax1.set_xlim(xlim)
    ax1.set_ylim(zlim)
    ax2.set_xlim(ylim)
    ax2.set_ylim(zlim)

    return ax0, ax1, ax2


# plot_boundaries(MP, BS, slice_x=22, slice_y=24, slice_z=0)


def make_YZ_plan(pos):
    a = np.linspace(0, 2 * np.pi, 100)
    r = abs(pos[(pos.X ** 2).argmin():  (pos.X ** 2).argmin() + 1].Y.values)
    return r * np.cos(a), r * np.sin(a)


def plot_boundaries(MP, BS, **kwargs):
    style = kwargs.get("style", ['--k', '--k'])
    alpha = kwargs.get("alpha", 0.6)
    axes = kwargs.get("axes", layout_EarthEnv_3planes(**kwargs))

    if "slice_x" in kwargs:
        axes[0].plot()

    axes[0].plot(MP.X, MP.Y, style[0], alpha=alpha)
    axes[0].plot(BS.X, BS.Y, style[1], alpha=alpha)

    axes[1].plot(MP.X, MP.Z, style[0], alpha=alpha)
    axes[1].plot(BS.X, BS.Z, style[1], alpha=alpha)

    axes[2].plot(make_YZ_plan(MP)[0], make_YZ_plan(MP)[1], style[0], alpha=alpha)
    axes[2].plot(make_YZ_plan(BS)[0], make_YZ_plan(BS)[1], style[1], alpha=alpha)
