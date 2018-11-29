.. _marvin-utils-plot-map:

Map (:mod:`marvin.utils.plot.map`)
==================================

.. _marvin-utils-plot-map-intro:

Introduction
------------
:mod:`marvin.utils.plot.map` contains utility functions for plotting Marvin maps.  The main function in this module is :func:`~marvin.utils.plot.map.plot`, which is thinly wrapped by the :meth:`~marvin.tools.quantities.Map.plot` method in the :class:`~marvin.tools.quantities.Map` class for convenience.


.. _marvin-utils-plot-map-getting-started:

Getting Started
---------------

:mod:`~marvin.utils.plot.map.plot` makes plotting a publication-quality MaNGA map easy with its carefully chosen default parameters.

.. plot::
    :include-source: false
    :context:

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot


.. plot::
    :align: center
    :include-source: True
    :context: close-figs

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    fig, ax = mapplot.plot(dapmap=ha)  # == ha.plot()


However, you may want to do further processing of the map, so you can override the default DAP :attr:`~marvin.utils.plot.map.plot.value`, :attr:`~marvin.utils.plot.map.plot.ivar`, and/or :attr:`~marvin.utils.plot.map.plot.mask` with your own arrays.

.. code-block:: python

    fig, ax = mapplot.plot(dapmap=ha, value=ha.value * 10.)


A :attr:`~marvin.utils.plot.map.plot.dapmap` object is not even necessary for :mod:`~marvin.utils.plot.map.plot`, though if you do not provide a :attr:`~marvin.utils.plot.map.plot.dapmap` object, then you will need to set a :attr:`~marvin.utils.plot.map.plot.value`. You will also need to provide other attributes, such as :attr:`~marvin.utils.plot.map.plot.title` and :attr:`~marvin.utils.plot.map.plot.cblabel`, that are by default set from attributes of the :attr:`~marvin.utils.plot.map.plot.dapmap` object.

.. code-block:: python

    import numpy as np
    fig, ax = mapplot.plot(value=np.random.random((34, 34)), mask=ha.mask)


This flexibilty is especially useful for passing in a custom mask, such as one created with the :meth:`~marvin.tools.maps.Maps.get_bpt` method. For more explanation of the mask manipulation in this specific example, see the :ref:`plotting tutorial <marvin-plotting-map-starforming>`.

.. plot::
    :align: center
    :include-source: True
    :context: close-figs

    from marvin.tools.maps import Maps
    masks, __, __ = maps.get_bpt(show_plot=False)

    # Create a bitmask for non-star-forming spaxels by taking the
    # complement (`~`) of the BPT global star-forming mask (where True == star-forming)
    # and mark those spaxels as "DONOTUSE".
    mask_non_sf = ~masks['sf']['global'] * ha.pixmask.labels_to_value('DONOTUSE')

    # Do a bitwise OR between DAP mask and non-star-forming mask.
    mask = ha.mask | mask_non_sf
    fig, ax = mapplot.plot(dapmap=ha, mask=mask)  # == ha.plot(mask=mask)


:mod:`~marvin.utils.plot.map.plot` lets you build multi-panel plots because it accepts pre-defined `matplotlib.figure <http://matplotlib.org/api/figure_api.html>`_ and `matplotlib.axes <http://matplotlib.org/api/axes_api.html>`_ objects.

.. plot::
    :align: center
    :include-source: True
    :context: close-figs

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')  # set matplotlib style sheet

    plateifus = ['8485-1901', '7443-12701']
    mapnames = ['stellar_vel', 'stellar_sigma']

    rows = len(plateifus)
    cols = len(mapnames)
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for row, plateifu in zip(axes, plateifus):
        maps = Maps(plateifu=plateifu)
        for ax, mapname in zip(row, mapnames):
            mapplot.plot(dapmap=maps[mapname], fig=fig, ax=ax, title=' '.join((plateifu, mapname)))

    fig.tight_layout()


.. _marvin-utils-plot-map-using:

Using :mod:`~marvin.utils.plot.map`
-----------------------------------

For more in-depth discussion of using :mod:`~marvin.utils.plot.map`, please see the following sections:

Plotting Tutorial
`````````````````

* :ref:`marvin-plotting-tutorial`

  * :ref:`marvin-plotting-quick-map`
  * :ref:`marvin-plotting-multipanel-single`
  * :ref:`marvin-plotting-multipanel-multiple`
  * :ref:`marvin-plotting-custom-map-cbrange`
  * :ref:`marvin-plotting-custom-map-snr-min`
  * :ref:`marvin-plotting-custom-map-axes`
  * :ref:`Plot Halpha Map of Star-forming Spaxels <marvin-plotting-map-starforming>`
  * :ref:`Plot [NII]/Halpha Flux Ratio Map of Star-forming Spaxels <marvin-plotting-niiha-map-starforming>`
  * :ref:`marvin-plotting-qualitative-colorbar`
  * :ref:`marvin-plotting-custom-map-mask`


.. _marvin-utils-plot-map-default-params:

Default Plotting Parameters
```````````````````````````

====================  ====================  =========  ===============  ==================  ===========
MPL-5+
-------------------------------------------------------------------------------------------------------
Property Type         Bad Data Bitmasks     Colormap   Percentile Clip  Symmetric Colorbar  Minimum SNR
====================  ====================  =========  ===============  ==================  ===========
default               UNRELIABLE, DONOTUSE  linearlab  5, 95            False               1
velocities            UNRELIABLE, DONOTUSE  RdBu_r     10, 90           True                0\ :sup:`a`
velocity dispersions  UNRELIABLE, DONOTUSE  inferno    10, 90           False               1
====================  ====================  =========  ===============  ==================  ===========

:sup:`a` Velocities do not have a minimum SNR. This allows spaxels near the zero-velocity contour to be displayed, but users are cautioned that some spaxels could have arbitrarily low SNRs.

**Note**: MPL-4 uses the same default plotting parameters as MPL-5, except the Bad Data Bitmasks, which use bit 1 (roughly DONOTUSE) for all properties.


Masking
```````

Spaxels with Low Signal-to-Noise
::::::::::::::::::::::::::::::::

:meth:`~marvin.utils.plot.map.mask_low_snr` creates a mask of a map where the data is below a minimum signal-to-noise ratio.

.. code-block:: python

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    low_snr = mapplot.mask_low_snr(value=ha.value, ivar=ha.ivar, snr_min=1)

**Important**: In 2.1.4, the call signature is ``low_snr_mask(value, ivar, snr_min)``. In version 2.2.0, this changes to ``mask_low_snr(value, ivar, snr_min)``.


Spaxels with Negative Values
::::::::::::::::::::::::::::

:meth:`~marvin.utils.plot.map.mask_neg_values` creates a mask of a map where the values are negative.  This is necessary to avoid erros when using a logarithmic colorbar.

.. code-block:: python

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    neg_values = mapplot.mask_neg_values(value=ha.value)

**Important**: In 2.1.4, the call signature is ``log_colorbar_mask(value, log_cb)``. In version 2.2.0, this changes to ``mask_neg_values(value)``.



Set Title
:::::::::

:meth:`~marvin.utils.plot.map.set_title` sets the title of the axis object. You can directly specify the title or construct it from the property name (and channel name).

.. code-block:: python

    import marvin.utils.plot.map as mapplot
    title = mapplot.set_title(title=None, property_name=ha.datamodel.name, channel=ha.datamodel.channel.name)


Reference/API
-------------

.. rubric:: Module

.. autosummary:: marvin.utils.plot.map

.. rubric:: Functions

.. autosummary::

    marvin.utils.plot.map.ax_setup
    marvin.utils.plot.map.mask_low_snr
    marvin.utils.plot.map.mask_neg_values
    marvin.utils.plot.map.plot
    marvin.utils.plot.map.set_title
