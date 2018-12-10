
.. _marvin-bin:

Bins
====

MaNGA analyzes all of its datacubes with multiple binning schemes.  The main ones are: no binning, VORONOI binning to signal-to-noise 10, and a HYBRID binning where the emission-line properties are unbinned and the stellar properties are VORONOI binned to S/N~10.  Since the MaNGA Data Analysis Pipeline bins individual properties differently, each ``maps`` or ``modelcube`` property on a spaxel has a ``bin`` attribute, providing relevant bin information via a :ref:`BinInfo <marvin-tools-bin`> object.

Let's load a MAPS that uses a hybrid binning scheme, i.e. **HYB10**.
::

    >>> # grab a spaxel from a maps
    >>> maps = Maps('8485-1901')
    >>> maps
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

The emission-line properties are unbinned while the stellar properties are VORONOI binned.
::

    >>> # look at the maps of binning
    >>> bins = [maps.binid_stellar_continua, maps.binid_em_line_models]

.. plot::
    :align: center
    :include-source: True

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    for ax, binmap in zip(axes, bins):
        binmap.plot(fig=fig, ax=ax)

    fig.tight_layout()


::
    >>> spaxel = maps[12,19]
    >>> spaxel
    <Marvin Spaxel (plateifu=8485-1901, x=19, y=12; x_cen=2, y_cen=-5, loaded=maps)>

::

    >>> # access the bin info for stellar_velocity
    >>> stvel = spaxel.stellar_vel
    >>> stvel.bin
    <BinInfo (binid=0, n_spaxels=1)>

The central spaxel has a binid of 0, with this spaxel the only one belonging in that bin.  Let's look at the bin information for H-alpha flux.  This bin also only has one spaxel in it.
::

    >>> spaxel.emline_gflux_ha_6564.bin
    <BinInfo (binid=199, n_spaxels=1)>

The ``BinInfo`` also provides a convenience method, ``get_bin_spaxels``, for getting all spaxels belonging to that bin.  These spaxels are unloaded by default.
::

    >>> stvel.bin.get_bin_spaxels()
    [<Marvin Spaxel (x=17, y=17, loaded=False)]


.. _marvin-bin-api:

Reference/API
-------------

.. rubric:: Class Inheritance Diagram

.. inheritance-diagram:: marvin.tools.quantities.base_quantity.BinInfo

.. rubric:: Class

.. autosummary:: marvin.tools.quantities.base_quantity.BinInfo

.. rubric:: Methods

.. autosummary::

    marvin.tools.quantities.base_quantity.BinInfo.get_bin_spaxels
