
import matplotlib.pyplot as plt
import numpy as np


def _make_figure(**kwargs):
    ncols = np.sum([1 for arg in ['x_slice', 'y_slice', 'z_slice'] if arg in kwargs])
    if ncols == 0:
        ncols = 1
    figsize = kwargs.get('figsize', (5 * ncols, 4.5))
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, constrained_layout=True)
    return fig, ax


def _set_infos_earth_env(fig, ax, **kwargs):
    if isinstance(ax, np.ndarray) is False:
        ax = [ax]
    c = 0
    if 'z_slice' in kwargs:
        ax[c].set_xlabel(kwargs.get('x_label', 'X (Re)'))
        ax[c].set_ylabel(kwargs.get('y_label', 'Y (Re)'))
        ax[c].set_xlim(kwargs.get('x_lim', (-20, 15)))
        ax[c].set_ylim(kwargs.get('y_lim', (-30, 30)))
        ax[c].axhline(0, color='k', ls='dotted', alpha=0.4)
        ax[c].axvline(0, color='k', ls='dotted', alpha=0.4)
        c += 1
    if 'y_slice' in kwargs:
        ax[c].set_xlabel(kwargs.get('x_label', 'X (Re)'))
        ax[c].set_ylabel(kwargs.get('y_label', 'Z (Re)'))
        ax[c].set_xlim(kwargs.get('x_lim', (-20, 15)))
        ax[c].set_ylim(kwargs.get('z_lim', (-30, 30)))
        ax[c].axhline(0, color='k', ls='dotted', alpha=0.4)
        ax[c].axvline(0, color='k', ls='dotted', alpha=0.4)
        c += 1
    if 'x_slice' in kwargs:
        ax[c].set_xlabel(kwargs.get('y_label', 'Y (Re)'))
        ax[c].set_ylabel(kwargs.get('z_label', 'Z (Re)'))
        ax[c].set_xlim(kwargs.get('y_lim', (-30, 30)))
        ax[c].set_ylim(kwargs.get('z_lim', (-30, 30)))
        ax[c].axhline(0, color='k', ls='dotted', alpha=0.4)
        ax[c].axvline(0, color='k', ls='dotted', alpha=0.4)
    if 'title' in kwargs:
        fig.suptitle(kwargs.get('title'))
    return fig, ax


def _find_theta_for_x_slice(magnetosheat, bd, phi, **kwargs):
    if bd == 'mp':
        def eq(t, x):
            return magnetosheat.magnetopause(t, x[0], base='spherical')[0] * np.cos(t) - x[1]
    elif bd == 'bs':
        def eq(t, x):
            return magnetosheat.bow_shock(t, x[0], base='spherical')[0] * np.cos(t) - x[1]
    n_pts = kwargs.get('n_pts', 300)
    xs = np.ones(n_pts) * kwargs.get('x_slice', 0)
    x0 = np.ones(n_pts) * np.pi / 3
    return root(eq, args=[phi, xs], x0=x0, jac=False, method='lm').x


def _find_phi_for_z_slice(magnetosheat, bd, theta, **kwargs):
    if bd == 'mp':
        def eq(p, x):
            if (p <= np.pi / 2) & (p >= -np.pi / 2):
                return magnetosheat.magnetopause(x[0], p, base='cartesian', **kwargs)[2] - x[1]
            else:
                return 1000
    elif bd == 'bs':
        def eq(p, x):
            if (p <= np.pi / 2) & (p >= -np.pi / 2):
                return magnetosheat.bow_shock(x[0], p, base='cartesian', **kwargs)[2] - x[1]
            else:
                return 1000
    xs = kwargs.get('z_slice', 0)
    phi = np.array(
        [root_scalar(eq, args=[t, xs], x0=-np.sign(t) * np.pi / 3, x1=-np.sign(t) * np.pi / 4, method='secant').root for
         t in theta])
    return phi


def _find_phi_for_y_slice(magnetosheat, bd, theta, **kwargs):
    if bd == 'mp':
        def eq(p, x):
            if (np.sin(p) >= 0) & (p <= np.pi) & (p >= 0):
                return magnetosheat.magnetopause(x[0], p, base='cartesian', **kwargs)[1] - x[1]
            else:
                return 1000

    elif bd == 'bs':
        def eq(p, x):
            if (np.sin(p) >= 0) & (p <= np.pi) & (p >= 0):
                return magnetosheat.bow_shock(x[0], p, base='cartesian', **kwargs)[1] - x[1]
            else:
                return 1000
    xs = kwargs.get('y_slice', 0)
    phi = np.array([root_scalar(eq, args=[t, xs], x0=np.pi / 3, x1=np.pi / 4, method='secant').root for t in theta])

    return phi


def _find_theta_lim_y_slice(magnetosheat, bd, **kwargs):
    if bd == 'mp':
        def eq(t, a):
            return magnetosheat.magnetopause(t, a[0], base='cartesian', **kwargs)[1] - a[1]

    if bd == 'bs':
        def eq(t, a):
            return magnetosheat.bow_shock(t, a[0], base='cartesian', **kwargs)[1] - a[1]

    t_lim1 = root_scalar(eq, args=[0, kwargs['y_slice']], x0=0.01, x1=np.pi / 3, method='secant').root
    t_lim2 = root_scalar(eq, args=[np.pi, kwargs['y_slice']], x0=0.01, x1=np.pi / 3, method='secant').root
    return t_lim1, t_lim2


def _find_theta_lim_z_slice(magnetosheat, bd, **kwargs):
    if bd == 'mp':
        def eq(t, a):
            return magnetosheat.magnetopause(t, a[0], base='cartesian', **kwargs)[2] - a[1]

    if bd == 'bs':
        def eq(t, a):
            return magnetosheat.bow_shock(t, a[0], base='cartesian', **kwargs)[2] - a[1]

    t_lim1 = root_scalar(eq, args=[np.pi / 2, kwargs['z_slice']], x0=0.01, x1=np.pi / 3, method='secant').root
    t_lim2 = root_scalar(eq, args=[-np.pi / 2, kwargs['z_slice']], x0=0.01, x1=np.pi / 3, method='secant').root
    return t_lim1, t_lim2


def make_theta_and_phi(magnetosheat, bd, fct_theta, fct_phi, **kwargs):
    n_pts = kwargs.get('n_pts', 300)
    if fct_phi is not None:
        theta_lim = np.array(fct_theta(magnetosheat, bd, **kwargs))
        theta_p = np.linspace(0.9 * np.pi, max(theta_lim), int(np.ceil(n_pts / 2)))
        theta_n = np.linspace(min(theta_lim), -0.9 * np.pi, int(np.floor(n_pts / 2)))
        theta = np.concatenate((theta_p, theta_n), axis=0)
        phi = fct_phi(magnetosheat, bd, theta, **kwargs)

    else:
        phi = np.linspace(0, 2 * np.pi, n_pts)
        theta = fct_theta(magnetosheat, bd, phi, **kwargs)
    return theta, phi


def check_validity_slice(magnetosheat, **kwargs):
    if 'z_slice' in kwargs:
        if kwargs.get('magnetopause', True) is True:
            z_max = max(magnetosheat.magnetopause(np.pi / 2, np.linspace(np.pi / 4, 3 * np.pi / 4, 100), **kwargs)[2])
            z_min = min(magnetosheat.magnetopause(np.pi / 2, np.linspace(-np.pi / 4, -3 * np.pi / 4, 100), **kwargs)[2])
            if (kwargs['z_slice'] > z_max) | (kwargs['z_slice'] < z_min):
                raise ValueError(
                    f" z_slice value must be between [{round(z_min, 2)},{round(z_max, 2)}] to be able to plot the magnetopause, else set magnetopause=False in kwargs ")
        if kwargs.get('bow_shock', True) is True:
            z_max = max(magnetosheat.bow_shock(np.pi / 2, np.linspace(np.pi / 4, 3 * np.pi / 4, 100), **kwargs)[2])
            z_min = min(magnetosheat.bow_shock(np.pi / 2, np.linspace(-np.pi / 4, -3 * np.pi / 4, 100), **kwargs)[2])
            if (kwargs['z_slice'] > z_max) | (kwargs['z_slice'] < z_min):
                raise ValueError(
                    f" z_slice value must be between [{round(z_min, 2)},{round(z_max, 2)}] to be able to plot the bow_shock, else set bow_shock=False in kwargs ")
    if 'y_slice' in kwargs:
        if kwargs.get('magnetopause', True) is True:
            y_max = max(magnetosheat.magnetopause(np.pi / 2, np.linspace(-np.pi / 4, np.pi / 4, 100), **kwargs)[1])
            y_min = min(
                magnetosheat.magnetopause(np.pi / 2, np.linspace(3 * np.pi / 4, 5 * np.pi / 4, 100), **kwargs)[1])
            if (kwargs['y_slice'] > y_max) | (kwargs['y_slice'] < y_min):
                raise ValueError(
                    f" y_slice value must be between [{round(y_min, 2)},{round(y_max, 2)}] to be able to plot the magnetopause, else set magnetopause=False in kwargs ")
        if kwargs.get('bow_shock', True) is True:
            y_max = max(magnetosheat.bow_shock(np.pi / 2, np.linspace(-np.pi / 4, np.pi / 4, 100), **kwargs)[1])
            y_min = min(magnetosheat.bow_shock(np.pi / 2, np.linspace(3 * np.pi / 4, 5 * np.pi / 4, 100), **kwargs)[1])
            if (kwargs['y_slice'] > y_max) | (kwargs['y_slice'] < y_min):
                raise ValueError(
                    f" y_slice value must be between [{round(y_min, 2)},{round(y_max, 2)}] to be able to plot the bow_shock, else set bow_shock=False in kwargs ")


def _plot_boundaries(magnetosheat, fig, ax, **kwargs):
    c = 0
    check_validity_slice(magnetosheat, **kwargs)
    if 'z_slice' in kwargs:
        if kwargs.get('magnetopause', True) is True:
            theta, phi = make_theta_and_phi(magnetosheat, 'mp', _find_theta_lim_z_slice, _find_phi_for_z_slice,
                                            **kwargs)
            x, y = magnetosheat.magnetopause(theta, phi, **kwargs)[:2]
            ax[c].plot(x, y, kwargs.get('style_mp', '-k'))

        if kwargs.get('bow_shock', True) is True:
            theta, phi = make_theta_and_phi(magnetosheat, 'bs', _find_theta_lim_z_slice, _find_phi_for_z_slice,
                                            **kwargs)
            x, y = magnetosheat.bow_shock(theta, phi, **kwargs)[:2]
            ax[c].plot(x, y, kwargs.get('style_bs', '-k'))
        c += 1

    if 'y_slice' in kwargs:
        if kwargs.get('magnetopause', True) is True:
            theta, phi = make_theta_and_phi(magnetosheat, 'mp', _find_theta_lim_y_slice, _find_phi_for_y_slice,
                                            **kwargs)
            x, z = magnetosheat.magnetopause(theta, phi, **kwargs)[::2]
            ax[c].plot(x, z, kwargs.get('style_mp', '-k'))

        if kwargs.get('bow_shock', True) is True:
            theta, phi = make_theta_and_phi(magnetosheat, 'bs', _find_theta_lim_y_slice, _find_phi_for_y_slice,
                                            **kwargs)
            x, z = magnetosheat.bow_shock(theta, phi, **kwargs)[::2]
            ax[c].plot(x, z, kwargs.get('style_bs', '-k'))
        c += 1

    if 'x_slice' in kwargs:
        if kwargs.get('magnetopause', True) is True:
            theta, phi = make_theta_and_phi(magnetosheat, 'bs', _find_theta_for_x_slice, None, **kwargs)
            y, z = magnetosheat.magnetopause(theta, phi, **kwargs)[1:]
            ax[c].plot(y, z, kwargs.get('style_mp', '-k'))
        if kwargs.get('bow_shock', True) is True:
            theta, phi = make_theta_and_phi(magnetosheat, 'bs', _find_theta_for_x_slice, None, **kwargs)
            y, z = magnetosheat.bow_shock(theta, phi, **kwargs)[1:]
            ax[c].plot(y, z, kwargs.get('style_bs', '-k'))

    return fig, ax


def layout_earth_env(magnetosheat, **kwargs):
    fig, ax = _make_figure(**kwargs)
    fig, ax = _set_infos_earth_env(fig, ax, **kwargs)
    fig, ax = _plot_boundaries(magnetosheat, fig, ax, **kwargs)
    return fig, ax
