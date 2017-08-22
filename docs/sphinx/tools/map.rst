.. _marvin-map:

=================================
Map (:mod:`marvin.tools.map.Map`)
=================================

.. _marvin-map-intro:

Introduction
------------
:mod:`~marvin.tools.map.Map` is a single map for a single galaxy. The main data that it contains are the :attr:`~marvin.tools.map.Map.value`, :attr:`~marvin.tools.map.Map.ivar`, and :attr:`~marvin.tools.map.Map.mask` arrays of the map. It also has some meta data and convenience functions, such as :meth:`~marvin.tools.map.Map.plot`, which wraps the :meth:`marvin.utils.plot.map.plot` method.

.. _marvin-map-getting-started:

Getting Started
---------------
To get a map, we first need to create a :mod:`marvin.tools.maps.Maps` object, which contains all of the maps for a galaxy.

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')


By default, :mod:`~marvin.tools.maps.Maps` returns the unbinned maps ``SPX``, but we can also choose from additional bintypes (see the `Technical Reference Manual <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/dap/GettingStarted#typeselection>`_ for a more complete description of each bintype and the associated usage warnings):

* ``SPX`` - spaxels are unbinned,
* ``VOR10`` - spaxels are Voronoi binned to a minimum continuum SNR of 10,
* ``NRE`` - spaxels are binned into two radial bins, binning all spectra from 0-1 and 1-2 (elliptical Petrosian) effective radii, and
* ``ALL`` - all spectra binned together.

.. code-block:: python

    maps = Maps(mangaid='1-209232', bintype='VOR10')


Once we have a :mod:`~marvin.tools.maps.Maps` object, we can "slice" it to get the H\ :math:`\alpha` (Gaussian-fitted) flux map.

.. code-block:: python

    ha = maps['emline_gflux_ha_6564']
    ha.plot()


.. image:: ../_static/quick_map_plot.png


Here ``maps['emline_gflux_ha_6564']`` is shorthand for ``maps.getMap('emline_gflux', channel='ha_6564')``, where the property and channel are joined by an underscore ("_"). For properties without channels, such as stellar velocity, just use the property name like ``maps['stellar_vel']``.

.. code-block:: python

    ha = maps.getMap('emline_gflux', channel='ha_6564')  # == maps['emline_gflux_ha_6564']
    stvel = maps.getMap('stellar_vel')                    # == maps['stellar_vel']

**New in 2.2.0**: You can guess at the map property name (and channel), and Marvin will return the map if there is a unique (and valid) property and channel.

.. code-block:: python

    maps['gflux ha']        # == maps['emline_gflux_ha_6564']
    maps['gvel oiii 5008']  # == maps[emline_gvel_oiii_5008]
    maps['stellar sig']     # == maps['stellar_sigma']
    
    # There are several properties of the Halpha line (velocity, sigma, etc.).
    maps['ha']  # ValueError
    
    # There are two [O III] lines.
    maps['gflux oiii']  # ValueError

The values, inverse variances, and bitmasks of the map can be accessed via the :attr:`~marvin.tools.map.Map.value`, :attr:`~marvin.tools.map.Map.ivar`, and :attr:`~marvin.tools.map.Map.mask` attributes, respectively.

**Important**: These arrays are ordered as ``[row, column]`` with the origin in the lower left, which corresponds to ``[y, x]``.

.. code-block:: python

    ha.value  # (34, 34) array
    ha.ivar   # (34, 34) array
    ha.mask   # (34, 34) array
    
    ha.value[17]  # get the middle row (i.e., "y")
    # array([  0.        ,   0.        ,   0.        ,   0.        ,
    #          0.        ,   0.        ,   0.03650022,   0.03789879,
    #          0.0838113 ,   0.16109767,   0.57484451,   1.42108019,
    #          2.98873795,   7.47787753,  14.08300415,  21.61707138,
    #         28.37593542,  31.47541953,  28.29092958,  20.82737156,
    #         13.33138178,   6.90730005,   3.70062335,   1.54131387,
    #          0.55510055,   0.34234428,   0.21906664,   0.18621548,
    #          0.1745672 ,   0.        ,   0.        ,   0.        ,
    #          0.        ,   0.        ])


The :attr:`~marvin.tools.map.Map.masked` attribute is a `numpy masked array <https://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html>`_ where the ``data`` is the :attr:`~marvin.tools.map.Map.value` array and the ``mask`` is a boolean array that is ``True`` for a given spaxel if any of the flags are set (i.e., where ``ha.mask > 0``).

.. code-block:: python

    ha.masked[17]
    # masked_array(data = [-- -- -- -- -- -- -- 0.03789878599602308 0.08381129696903318
    #                      0.1610976667261473 0.5748445110902572 1.421080190438372 2.988737954927168
    #                      7.477877525388817 14.083004151791611 21.61707138246288 28.37593542372677
    #                      31.475419531155 28.290929579722462 20.827371557790272 13.331381776434451
    #                      6.907300050577721 3.7006233506234203 1.5413138678320422 0.5551005467482618
    #                      0.3423442819444342 0.2190666373241594 0.18621548081774594
    #                      0.17456719770757587 -- -- -- -- --],
    #              mask = [ True  True  True  True  True  True  True False False False False False
    #                       False False False False False False False False False False False False
    #                       False False False False False  True  True  True  True  True],
    #              fill_value = 1e+20)
    
    (ha.masked.data == ha.value).all()                # True
    (ha.masked.mask == (ha.mask).astype(bool)).all()  # True


For more fine-grained data quality control, you can select spaxels based on the :attr:`~marvin.tools.map.Map.mask` attribute, which is an array of DAP spaxel `bitmasks <http://www.sdss.org/dr13/algorithms/bitmasks/>`_ that indicate issues with the data. The following table (lifted from the `MPL-5 Techincal Reference Manual <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/DAPMetaData#MANGA_DAPPIXMASK>`_) gives the meaning of each bit. For MPL-4, the bitmask is simply 0 = good and 1 = bad (which roughly corresponds to DONOTUSE).

===  ============  =============================================================
Bit	 Name	       Description
===  ============  =============================================================
0    NOCOV	       No coverage in this spaxel
1    LOWCOV	       Low coverage in this spaxel
2    DEADFIBER     Major contributing fiber is dead
3    FORESTAR      Foreground star
4    NOVALUE       Spaxel was not fit because it did not meet selection criteria
5    UNRELIABLE    Value is deemed unreliable; see TRM for definition
6    MATHERROR     Mathematical error in computing value
7    FITFAILED     Attempted fit for property failed
8    NEARBOUND     Fitted value is too near an imposed boundary; see TRM
9    NOCORRECTION  Appropriate correction not available
10   MULTICOMP     Multi-component velocity features present
30   DONOTUSE      Do not use this spaxel for science
===  ============  =============================================================

**Note**: For MPL-5, DONOTUSE is a consolidation of the flags NOCOV, LOWCOV, DEADFIBER, FORESTAR, NOVALUE, MATHERROR, FITFAILED, and NEARBOUND.

.. code-block:: python

    import numpy as np
    nocov     = (ha.mask & 2**0) > 0
    lowcov    = (ha.mask & 2**1) > 0
    deadfiber = (ha.mask & 2**2) > 0
    forestar  = (ha.mask & 2**3) > 0
    novalue   = (ha.mask & 2**4) > 0
    matherror = (ha.mask & 2**6) > 0
    fitfailed = (ha.mask & 2**7) > 0
    nearbound = (ha.mask & 2**8) > 0

    bad_data = np.logical_or.reduce((nocov, lowcov, deadfiber, forestar, novalue, matherror, fitfailed, nearbound))
    
    donotuse  = (ha.mask & 2**30) > 0
    
    (bad_data == donotuse).all()  # True


.. _marvin-map-using:

Using :mod:`~marvin.tools.map.Map`
----------------------------------

For more in-depth discussion of using :mod:`~marvin.tools.map.Map`, please see the following sections:

Map Plotting
````````````

* :doc:`../tutorials/plotting`
  
  * :ref:`marvin-plotting-quick-map`
  * :ref:`marvin-plotting-multipanel-single`
  * :ref:`marvin-plotting-multipanel-multiple`
  * :ref:`marvin-plotting-custom-map-axes`
  * :ref:`marvin-plotting-map-starforming`
  * :ref:`Plot [NII]/Halpha Flux Ratio Map of Star-forming Spaxels <marvin-plotting-niiha-map-starforming>`


Applying Bitmasks to a Map
``````````````````````````

* :doc:`../tutorials/bitmasks`


Map Arithmetic
``````````````

**New in 2.2.0** :mod:`~marvin.tools.map.Map` objects can be added, subtracted, multiplied, divided, or raised to a power.

.. code-block:: python

    ha = maps['emline_gflux_ha_6564']
    nii = maps['emline_gflux_nii_6585']
    
    sum_ = nii + ha
    diff = nii - ha
    prod = nii * ha
    quot = nii / ha
    pow_ = ha**0.5
    
    prod
    # <Marvin EnhancedMap '(emline_gflux_nii_6585 * emline_gflux_ha_6564)'>
    # array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #        ...,
    #        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #        [ 0.,  0.,  0., ...,  0.,  0.,  0.]]) 'erg2 / (cm4 s2 spaxel2)'

In addition to performing the arithmetic operation on the ``value``, the resulting :mod:`~marvin.tools.map.EnhancedMap` has correctly propagated ``ivar``, ``mask``, ``unit``, and ``scale``.  Instead of ``property`` and ``channel`` attributes, :mod:`~marvin.tools.map.EnhancedMap` objects have ``history`` and ``parent`` attributes about their creation operation(s) and parent :mod:`~marvin.tools.map.Map` object(s).

.. code-block:: python

    prod.history  # '(emline_gflux_nii_6585 * emline_gflux_ha_6564)'

    prod.parents
    # [<Marvin Map (plateifu='8485-1901', property='emline_gflux', channel=<Channel 'nii_6585' unit='km / s'>)>
    #  array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         ...,
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.]]) erg / (cm2 s spaxel),
    #  <Marvin Map (plateifu='8485-1901', property='emline_gflux', channel=<Channel 'ha_6564' unit='km / s'>)>
    #  array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         ...,
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.],
    #         [ 0.,  0.,  0., ...,  0.,  0.,  0.]]) erg / (cm2 s spaxel)]




Accessing the Parent Maps Object
````````````````````````````````

One of the most useful features of Marvin is the tight integration of the Tools. From a :mod:`~marvin.tools.map.Map` object we can access its parent :mod:`~marvin.tools.maps.Maps` object via the :attr:`~marvin.tools.map.Map.maps` attribute and meta data about the :class:`~marvin.utils.dap.datamodel.base.Property` via the :attr:`~marvin.tools.map.Map.property` attribute.

.. code-block:: python

    ha.maps == maps  # True
    
    ha.property
    # <Property 'emline_gflux', release='2.0.2', channel='ha_6564', unit='erg / (cm2 s spaxel)'>
    

Saving and Restoring a Map
``````````````````````````

Finally, we can :meth:`~marvin.tools.map.Map.save` our :mod:`~marvin.tools.map.Map` object as a MaNGA pickle file (``*.mpf``) and then :meth:`~marvin.tools.map.Map.restore` it.

.. code-block:: python

    from marvin.tools.map import Map
    ha.save(path='/path/to/save/directory/ha_8485-1901.mpf')
    zombie_ha = Map.restore(path='/path/to/save/directory/ha_8485-1901.mpf')


.. _marvin-map-reference:

Reference/API
-------------

.. rubric:: Class

.. autosummary:: marvin.tools.map.Map

.. rubric:: Methods

.. autosummary::

    marvin.tools.map.Map.save
    marvin.tools.map.Map.restore
    marvin.tools.map.Map.masked
    marvin.tools.map.Map.error
    marvin.tools.map.Map.snr
    marvin.tools.map.Map.plot
    marvin.tools.map.Map.inst_sigma_correction


|