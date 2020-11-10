=====
space
=====


.. image:: https://img.shields.io/pypi/v/space.svg
        :target: https://pypi.python.org/pypi/space

.. image:: https://img.shields.io/travis/nicolasaunai/space.svg
        :target: https://travis-ci.com/nicolasaunai/space

.. image:: https://readthedocs.org/projects/space/badge/?version=latest
        :target: https://space.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Space Plasma Analysis CodE

.. image:: https://user-images.githubusercontent.com/3200931/98716891-978e2180-238c-11eb-9487-07c66221e5bb.png

.. code:: python
   import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  
        import numpy as np
        from space.models.planetary import Magnetosheath

        theta = np.linspace(0, np.pi/2, 200)
        phi  = np.linspace(0, 2*np.pi, 400)
        th, ph = np.meshgrid(theta, phi, indexing="ij")

        msh = Magnetosheath(magnetopause = "mp_formisano1979", bow_shock="bs_formisano1979")

        xbs,ybs,zbs = msh.bow_shock(th, ph)
        xmp,ymp,zmp = msh.magnetopause(th, ph)


        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d', azim=-120, elev=40)
        ax.plot_surface(xbs,ybs,zbs, color="y", alpha=0.4)
        ax.plot_surface(xmp,ymp,zmp, color="b", alpha=0.9)
        ax.plot(np.linspace(0, 20, 100),np.zeros(100), lw=2, color="k")
        ax.plot(np.zeros(100),np.linspace(0, 30, 100),np.zeros(100), lw=2, color="k")
        ax.plot(np.zeros(100),np.zeros(100),np.linspace(0, 40, 100), lw=2, color="k")
        ax.plot(data[:,0], data[:,1], data[:,2], color="r", lw=2)

        ax.set_title("MMS orbit")
        ax.set_xlabel(r"$x/Re$")
        ax.set_ylabel(r"$y/Re$")
        ax.set_zlabel(r"$z/Re$")
        ax.set_xlim((-10,20))

* Free software: GNU General Public License v3
* Documentation: https://space.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
