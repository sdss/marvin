.. _marvin-utils-plot-colorbar:

=======================================
Map (:mod:`marvin.utils.plot.colorbar`)
=======================================


.. _marvin-utils-plot-colorbar-intro:

Introduction
------------
:mod:`marvin.utils.plot.colorbar` contains utility functions for making colorbars for Marvin maps. The main function is :func:`~marvin.utils.plot.colorbar.draw_colorbar`, which makes the colorbar.


.. _marvin-utils-plot-colorbar-getting-started:

Getting Started
---------------

.. TODO add text here

Boilerplate

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    import marvin.utils.plot.colorbar as colorbar

    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    nocov_mask = mapplot.no_coverage_mask(mask=ha.mask, bit=0)
    nocov = np.ma.array(np.ones(ha.value.shape), mask=~nocov_mask)

    extent = mapplot.set_extent(cube_size=ha.value.shape, sky_coords=False)
    imshow_kws = {'extent': extent, 'interpolation': 'nearest', 'origin': 'lower'}


Create gray background where there is no IFU coverage by using :func:`~marvin.utils.plot.colorbar.colorbar.one_color_cmap` to create a colormap with a single color.

.. code-block:: python

    # create a colormap with a single color
    A8A8A8 = colorbar.one_color_cmap(color='#A8A8A8')

    fig, ax = plt.subplots()
    ax.imshow(nocov, cmap=A8A8A8, zorder=1, **imshow_kws)

.. image:: ../_static/one_color_cmap.png



Example of how :meth:`marvin.tools.map.Map.plot` draws a colorbar.

.. code-block:: python

    linearlab = colorbar.linearlab()[0]
    cb_kws = {'cmap': linearlab,
              'percentile_clip': [5, 95],
              'sigma_clip': False,
              'symmetric': False,
              'label': getattr(ha, 'unit'),
              'label_kws': {'size': 16},
              'tick_params_kws': {'labelsize': 16}}
    
    params = mapplot.get_plot_params(dapver, prop)
    badData = params['bitmasks']['badData']
    bad_data = mapplot.bad_data_mask(ha.mask, badData)
    low_snr = mapplot.low_snr_mask(value, ivar, snr_min)
    log_cb_mask = np.zeros(value.shape, dtype=bool)

    # final masked array to show
    good_spax = select_good_spaxels(value, nocov_mask, bad_data, low_snr, log_cb_mask)

    im = np.ma.array(ha.value, mask=nocov_mask)
    cb_kws = colorbar._set_cb_kws(cb_kws)
    cb_kws = colorbar._set_cbrange(im, cb_kws)

    p1 = ax.imshow(ha.masked, cmap=linearlab, zorder=10, **imshow_kws)
    colorbar.draw_colorbar(fig=fig, mappable=p1, ax=ax, **cb_kws)


.. colorbar ticklabels are smushed


.. _marvin-utils-plot-colorbar-using:

Using :mod:`~marvin.utils.plot.colorbar`
----------------------------------------

:ref:`marvin-utils-plot-map-default-params`


Create a Discrete Colormap



Reference/API
-------------

.. rubric:: Module

.. autosummary:: marvin.utils.plot.colorbar

.. rubric:: Functions

.. autosummary::

    marvin.utils.plot.colorbar.linearlab

|