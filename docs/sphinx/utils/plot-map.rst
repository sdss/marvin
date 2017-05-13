.. _marvin-utils-plot-map:

===================================
Plot (:mod:`marvin.utils.plot.map`)
===================================

.. _marvin-utils-plot-map-intro:

Introduction
------------
:mod:`marvin.utils.plot.map` are Marvin plotting utility functions.


.. _marvin-utils-plot-map-getting-started:

Getting Started
---------------

Plot map using default parameters

.. code-block:: python

    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    fig, ax = mapplot.plot(dapmap=ha)  # == ha.plot()

.. image:: ../_static/quick_map_plot.png


Use your own :attr:`~marvin.utils.plot.map.plot.value`, :attr:`~marvin.utils.plot.map.plot.ivar`, and/or :attr:`~marvin.utils.plot.map.plot.mask`.

.. code-block:: python

    import numpy as np
    fig, ax = mapplot.plot(value=np.random.random((34, 34)), mask=ha.mask)

If you do not provide a ``dapmap``, then the you need to manually set the ``title`` and ``cblabel``. However, you can provide a ``dapmap`` and a :attr:`~marvin.utils.plot.map.plot.value`, :attr:`~marvin.utils.plot.map.plot.ivar`, and/or :attr:`~marvin.utils.plot.map.plot.mask`, which will override the corresponding ``dapmap`` attribute.


.. code-block:: python

    fig, ax = mapplot.plot(dapmap=ha, value=ha.value * 10.)

This is especially useful for passing in a custom mask, such as one created with the :meth:`~marvin.tools.maps.Maps.get_bpt` method. For more explanation of the mask manipulation in this specific example, see :ref:`marvin-plotting-map-starforming`.

.. code-block:: python

    from marvin.tools.maps import Maps
    masks, __ = maps.get_bpt(show_plot=False)
    mask_non_sf = ~masks['sf']['global'] * 2**30  # non-star-forming spaxels
    mask = ha.mask | mask_non_sf
    ha.plot(mask=mask)  # == mapplot.plot(dapmap=ha, mask=mask)


.. TODO explain datamodel defaults
.. TODO stellar velocity has no SNR min

.. TODO multi-panel plots

.. _marvin-utils-plot-map-using:

Using :mod:`~marvin.utils.plot.map`
-----------------------------------

For more in-depth discussion of using :mod:`~marvin.utils.plot.map`, please see the following sections:

Plotting Tutorial
`````````````````

* :doc:`../tutorials/plotting`
  
  * :ref:`marvin-plotting-quick-map`
  * :ref:`marvin-plotting-multipanel-single`
  * :ref:`marvin-plotting-multipanel-multiple`
  * :ref:`marvin-plotting-custom-map-axes`
  * :ref:`marvin-plotting-map-starforming`




Reference/API
-------------

.. rubric:: Module

.. autosummary:: marvin.utils.plot.map

.. rubric:: Functions

.. autosummary::

    marvin.utils.plot.map.plot


.. rubric:: Module

.. autosummary:: marvin.utils.plot.colorbar

.. rubric:: Functions

.. autosummary::

    marvin.utils.plot.colorbar.draw_colorbar
