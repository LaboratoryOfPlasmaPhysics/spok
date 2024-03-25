import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar, root


def _make_figure(**kwargs):
    ncols = np.sum([1 for arg in ['x_slice', 'y_slice', 'z_slice'] if arg in kwargs])
    if ncols == 0:
        ncols = 1

    if ('axes' in kwargs) and ('figure' in kwargs):
        figure = kwargs.pop("figure")
        axes = kwargs.pop("axes")
        if isinstance(axes, np.ndarray) is False:
            axes = np.array([[axes]])
        elif len(axes.shape) == 1:
            axes = np.array([axes])
        assert axes.shape[1] == ncols, "The axes given as input must have the same length as the asked number of cuts."
        print('Figure given as parameter.')
    else:
        figsize = kwargs.get('figsize', (5 * ncols, 4.5))
        figure, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, constrained_layout=True)

    if 'color_background' in kwargs:
        if isinstance(axes, np.ndarray):
            for i in range(len(axes)):
                axes[i].set_facecolor('xkcd:{}'.format(kwargs['color_background']))
        else:
            axes.set_facecolor('xkcd:{}'.format(kwargs['color_background']))

    return figure, axes


def _set_infos_earth_env(fig, axis, **kwargs):
    if isinstance(axis, np.ndarray) is False:
        axis = np.array([[axis]])
    elif len(axis.shape) == 1:
        axis = np.array([axis])

    for ax in axis:
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
            ax[c].set_ylabel(kwargs.get('z_label', 'Z (Re)'))
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
    return fig, axis


def _find_theta_for_x_slice(boundary_model, phi, **kwargs):
    def eq(t, x):
        return boundary_model(t, x[0], coord_sys='spherical')[0] * np.cos(t) - x[1]

    n_pts = kwargs.get('n_pts', 300)
    xs = np.ones(n_pts) * kwargs.get('x_slice', 0)
    x0 = np.ones(
        n_pts) * np.pi / 3  # initial guess of theta found empirically, will enable to
    # find a positive value of theta to make the x_slice with phi between [0;2pi]
    return root(eq, args=[phi, xs], x0=x0, jac=False, method='lm').x


def _find_phi_for_z_slice(boundary_model, theta, **kwargs):
    def eq(p, x):
        if (p <= np.pi) & (p >= 0):
            # cos(phi) must be positive because y=r*sin(theta)*cos(phi) and it's
            # easier to get z positive or negative with theta than with phi (multiple roots
            # with cos(roots) positive or negative)
            return boundary_model(x[0], p, coord_sys='cartesian', **kwargs)[2] - x[1]
        else:
            return 1000

    xs = kwargs.get('z_slice', 0)
    phi = np.array([root_scalar(eq, args=[t, xs], x0=np.pi / 3,
                                x1=np.pi / 4, method='secant').root for t in
                    theta])  # x0, x1 initial guesses of phi that will enable
    # to find a cos(phi) positive. -sign(theta) will allow to have an initial
    # guess closer to the wanted root. Found empirically
    return phi


def _find_phi_for_y_slice(boundary_model, theta, **kwargs):
    def eq(p, x):
        if (p <= np.pi / 2) & (p >= -np.pi / 2):
            # sin(phi) must be positive because z=r*sin(theta)*sin(phi)
            # and it's simpler to get z positive or negative with theta than with phi
            # (multiple roots with sin(roots) positive or negative)
            return boundary_model(x[0], p, coord_sys='cartesian', **kwargs)[1] - x[1]
        else:
            return 1000

    xs = kwargs.get('y_slice', 0)
    phi = np.array(
        [root_scalar(eq, args=[t, xs], x0=-np.sign(t) * np.pi / 3,
                     x1=-np.sign(t) * np.pi / 4, method='secant').root for
         t in theta])  # x0, x1 initial guesses of phi that will enable to
    # find a sin(phi) positive, found empirically
    return phi


def _find_theta_lim_y_slice(boundary_model,
                            **kwargs):  # find the smallest value of theta allowed in the
    # considerate y_slice
    def eq(t, a):
        return boundary_model(t, a[0], coord_sys='cartesian', **kwargs)[1] - a[1]

    phi_p = np.pi / 2  # phi for y positive : r*cos(theta) - y_slice = 0
    phi_n = -np.pi / 2  # phi for y negative : -r*cos(theta) - y_slice = 0
    t_lim1 = root_scalar(eq, args=[phi_p, kwargs['y_slice']], x0=0.01, x1=np.pi / 3,
                         method='secant').root  # x0, x1 initial guesses of theta that
    # will enable to find the smallest theta giving a y positive
    t_lim2 = root_scalar(eq, args=[phi_n, kwargs['y_slice']], x0=0.01, x1=np.pi / 3,
                         method='secant').root  # x0, x1 initial guesses of theta that
    # will enable to find the smallest theta giving a y negative
    return t_lim1, t_lim2


def _find_theta_lim_z_slice(boundary_model,
                            **kwargs):  # find the smallest value of theta allowed in the considerate z_slice
    def eq(t, a):
        return boundary_model(t, a[0], coord_sys='cartesian', **kwargs)[2] - a[1]

    phi_p = 0  # phi for z positive : r*cos(theta) - z_slice = 0
    phi_n = np.pi  # phi for z negative : -r*cos(theta) - z_slice = 0
    t_lim1 = root_scalar(eq, args=[phi_p, kwargs['z_slice']], x0=0.01, x1=np.pi / 3,
                         method='secant').root  # x0, x1 initial guesses of theta that
    # will enable to find the smallest theta giving a z positive
    t_lim2 = root_scalar(eq, args=[phi_n, kwargs['z_slice']], x0=0.01, x1=np.pi / 3,
                         method='secant').root  # x0, x1 initial guesses of theta that
    # will enable to find the smallest theta giving a z negative
    return t_lim1, t_lim2


def make_theta_and_phi(boundary_model, fct_theta, fct_phi, **kwargs):
    n_pts = kwargs.get('n_pts', 300)
    if fct_phi is not None:
        theta_lim = np.array(fct_theta(boundary_model, **kwargs))
        theta_p = np.linspace(0.8 * np.pi, max(theta_lim), int(np.ceil(n_pts / 2)))
        theta_n = np.linspace(min(theta_lim), -0.8 * np.pi, int(np.floor(n_pts / 2)))
        theta = np.concatenate((theta_p, theta_n), axis=0)
        phi = fct_phi(boundary_model, theta, **kwargs)

    else:
        phi = np.linspace(0, 2 * np.pi, n_pts)
        theta = fct_theta(boundary_model, phi, **kwargs)
    return theta, phi


def check_validity_of_asked_slice(boundary_model, **kwargs):
    theta0 = np.pi / 2  # theta to have x=0  (terminator)
    if 'z_slice' in kwargs:
        range_phi = np.linspace(-np.pi / 4, np.pi / 4, 100)
        # range of phi containing the greatest value of z for x=0
        z_max = np.max(boundary_model(theta0, range_phi, **kwargs)[2])
        z_min = np.min(boundary_model(-theta0, range_phi, **kwargs)[2])
        if (kwargs['z_slice'] > z_max) | (kwargs['z_slice'] < z_min):
            raise ValueError(f"z_slice value must be between [{round(z_min, 2)},{round(z_max, 2)}] to be able to plot "
                             f"the boundary")

    if 'y_slice' in kwargs:
        range_phi = np.linspace(np.pi / 4, 3 * np.pi / 4, 100)
        # range of phi containing the greatest value of y for x=0
        y_max = np.max(boundary_model(theta0, range_phi, **kwargs)[1])
        y_min = np.min(boundary_model(-theta0, range_phi, **kwargs)[1])
        if (kwargs['y_slice'] > y_max) | (kwargs['y_slice'] < y_min):
            raise ValueError(f" y_slice value must be between [{round(y_min, 2)},{round(y_max, 2)}] to be able to plot "
                             f"the boundary")


def plot_boundary(boundary_model, fig, axis, **kwargs):
    if isinstance(axis, np.ndarray) is False:
        axis = np.array([[axis]])
    elif len(axis.shape) == 1:
        axis = np.array([axis])

    for ax in axis:
        c = 0
        check_validity_of_asked_slice(boundary_model, **kwargs)
        if 'z_slice' in kwargs:  # cut z axis to plot xy plane
            theta, phi = make_theta_and_phi(boundary_model, _find_theta_lim_z_slice, _find_phi_for_z_slice, **kwargs)
            x, y = boundary_model(theta, phi, **kwargs)[:2]
            ax[c].plot(x, y, kwargs['style'],  alpha=kwargs['alpha'])
            c += 1

        if 'y_slice' in kwargs:  # cut y axis to plot xz plane
            theta, phi = make_theta_and_phi(boundary_model, _find_theta_lim_y_slice, _find_phi_for_y_slice, **kwargs)
            x, z = boundary_model(theta, phi, **kwargs)[::2]
            ax[c].plot(x, z, kwargs['style'],  alpha=kwargs['alpha'])
            c += 1

        if 'x_slice' in kwargs:  # cut x axis to plot yz plane
            theta, phi = make_theta_and_phi(boundary_model, _find_theta_for_x_slice, None, **kwargs)
            y, z = boundary_model(theta, phi, **kwargs)[1:]
            ax[c].plot(y, z, kwargs['style'],  alpha=kwargs['alpha'])

    return fig, axis


def layout_earth_env(magnetosheath, **kwargs):
    fig, ax = _make_figure(**kwargs)
    fig, ax = _set_infos_earth_env(fig, ax, **kwargs)
    if kwargs.get('magnetopause', True) is True:
        kwargs['style'] = kwargs.get('style_mp', '-k')
        kwargs['alpha'] = kwargs.get('alpha_mp', 1)
        fig, ax = plot_boundary(magnetosheath.magnetopause, fig, ax, **kwargs)
    if kwargs.get('bow_shock', True) is True:
        kwargs['style'] = kwargs.get('style_bs', '-k')
        kwargs['alpha'] = kwargs.get('alpha_bs', 1)
        fig, ax = plot_boundary(magnetosheath.bow_shock, fig, ax, **kwargs)
    plt.tight_layout()
    return fig, ax
