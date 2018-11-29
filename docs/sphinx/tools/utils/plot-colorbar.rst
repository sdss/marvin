.. _marvin-utils-plot-colorbar:

Colorbar (:mod:`marvin.utils.plot.colorbar`)
============================================


.. _marvin-utils-plot-colorbar-intro:

Introduction
------------
:mod:`marvin.utils.plot.colorbar` cooontains utility functions for making colorbars for Marvin maps. These functions are mostly for internal use, however, the :func:`~marvin.utils.plot.colorbar.linearlab` function returns the ``linearlab`` colormap, whose properties are described `here <https://mycarta.wordpress.com/2012/12/06/the-rainbow-is-deadlong-live-the-rainbow-part-5-cie-lab-linear-l-rainbow/>`_.



.. _marvin-utils-plot-colorbar-getting-started:

Getting Started
---------------


.. _marvin-utils-plot-colorbar-using:

Using :mod:`~marvin.utils.plot.colorbar`
----------------------------------------

Get the ``linearlab`` colormap that is the default for many MaNGA maps:

.. code-block:: python

    import marvin.utils.plot.colorbar as colorbar
    linearlab, linearlab_r = colorbar.linearlab()

.. plot::
    :align: center
    :include-source: False

    import marvin.utils.plot.colorbar as colorbar
    linearlab, linearlab_r = colorbar.linearlab()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=2, figsize=(6, 1))
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.25, right=0.99)
    gradient = np.array([np.linspace(0, 1, 256)])

    for ax, cmap in zip(axes, (linearlab, linearlab_r)):
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        pos = list(ax.get_position().bounds)
        y_text = pos[1] + pos[3] / 2.
        fig.text(0.02, y_text, cmap.name, va='center', ha='left', fontsize=16)
        ax.set_axis_off()


Reference/API
-------------

.. rubric:: Module

.. autosummary:: marvin.utils.plot.colorbar

.. rubric:: Functions

.. autosummary::

    marvin.utils.plot.colorbar.linearlab

|
