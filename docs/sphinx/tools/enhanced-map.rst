.. _marvin-enhanced-map:

==================================================
Enhanced Map (:mod:`marvin.tools.map.EnhancedMap`)
==================================================

.. _marvin-enhanced-map-intro:

Introduction
------------
An :mod:`~marvin.tools.map.EnhancedMap` is a :mod:`~marvin.tools.map.Map` that has been modified by a map arithmetic operation (``+``, ``-``, ``*``, ``/``, or ``**``). It inherits most of the attributes of a :mod:`~marvin.tools.map.Map`. Notably, it lacks ``property`` and ``channel`` attributes in favor of ``history`` and ``parent`` attributes that describe its creation operation(s) and parent :mod:`~marvin.tools.map.Map` object(s).


.. _marvin-enhanced-map-getting-started:

Getting Started
---------------
We can create an :mod:`~marvin.tools.map.EnhancedMap` object by applying a map arithmetic operation to :mod:`~marvin.tools.map.Map` object(s).

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    nii = maps['emline_gflux_nii_6585']
    
    # All are EnhancedMap's.
    sum_ = nii + ha
    diff = nii - ha
    prod = nii * ha
    quot = nii / ha
    pow_ = ha**0.5


.. _marvin-enhanced-map-using:

Using :mod:`~marvin.tools.map.EnhancedMap`
------------------------------------------

For more in-depth discussion of :mod:`~marvin.tools.map.Map` methods and attributes, please see :ref:`marvin-map`.

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



.. _marvin-enhanced-map-reference:

Reference/API
-------------

.. rubric:: Class

.. autosummary:: marvin.tools.map.EnhancedMap

.. rubric:: Methods

.. autosummary::

    marvin.tools.map.EnhancedMap.save
    marvin.tools.map.EnhancedMap.restore
    marvin.tools.map.EnhancedMap.masked
    marvin.tools.map.EnhancedMap.error
    marvin.tools.map.EnhancedMap.snr
    marvin.tools.map.EnhancedMap.plot


|