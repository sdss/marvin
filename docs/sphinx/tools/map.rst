.. _marvin-map:

Map
===

:mod:`~marvin.tools.quantities.Map` is a single map for a single galaxy. The main data that it contains are the :attr:`~marvin.tools.quantities.Map.value`, :attr:`~marvin.tools.quantities.Map.ivar`, and :attr:`~marvin.tools.quantities.Map.mask` arrays of the map.

Initializing a Map
------------------

To get a `Map`, we first create a :mod:`marvin.tools.maps.Maps` object, which contains all of the maps for a galaxy.  Then we select an individual `Map` in one of four ways:

* exact key slicing,
* dot syntax,
* `getMap` method, or
* fuzzy key slicing.

.. code-block:: python

    >>> from marvin.tools import Maps
    >>> maps = Maps(plateifu='8485-1901')

    >>> # exact key slicing
    >>> ha = maps['emline_gflux_ha_6564']

    >>> # dot syntax
    >>> ha = maps.emline_gflux_ha_6564

    >>> # getMap()
    >>> ha = maps.getMap('emline_gflux_ha_6564')
    >>> # equivalently
    >>> ha = maps.getMap('emline_gflux', channel='ha_6564')

    >>> # fuzzy key slicing
    >>> ha = maps['gflux ha']


Fuzzy key slicing works if the input is unambiguously associated with a particular key:

.. code-block:: python

    >>> maps['gflux ha']        # == maps['emline_gflux_ha_6564']
    >>> maps['gvel oiii 5008']  # == maps[emline_gvel_oiii_5008]
    >>> maps['stellar sig']     # == maps['stellar_sigma']

    >>> # Ambiguous: there are several velocity properties (stellar and emission lines).
    >>> maps['vel']  # ValueError

    >>> # Ambiguous: there are two [O III] lines.
    >>> maps['gflux oiii']  # ValueError


.. _marvin-map-basic:

Basic Attributes
----------------

The values, inverse variances, and `bitmasks <http://www.sdss.org/dr13/algorithms/bitmasks/>`_ of the map can be accessed via the :attr:`~marvin.tools.quantities.Map.value`, :attr:`~marvin.tools.quantities.Map.ivar`, and :attr:`~marvin.tools.quantities.Map.mask` attributes, respectively.

.. code-block:: python

    >>> ha.value  # (34, 34) array
    >>> ha.ivar   # (34, 34) array
    >>> ha.mask   # (34, 34) array --- same as ha.pixmask.mask

    >>> ha.value[17]  # get the middle row (i.e., "y")
    array([ 0.       ,  0.       ,  0.       ,  0.       ,  0.       ,
        0.       ,  0.       ,  0.0360246,  0.0694705,  0.135435 ,
        0.564578 ,  1.44708  ,  3.12398  ,  7.72712  , 14.2869   ,
       22.2461   , 29.1134   , 32.1308   , 28.9591   , 21.4879   ,
       13.9937   ,  7.14412  ,  3.84099  ,  1.64863  ,  0.574292 ,
        0.349627 ,  0.196499 ,  0.144375 ,  0.118376 ,  0.       ,
        0.       ,  0.       ,  0.       ,  0.       ])

.. _marvin-map-access-spaxel:

Accessing an Individual Spaxel
------------------------------

Slicing a `Map` returns the property for a single spaxel:

.. code-block:: python

    >>> ha[17, 17]  # the Halpha flux value in the central spaxel
    <Marvin Map (property='emline_gflux_ha_6564')>
    30.7445 1e-17 erg / (cm2 s spaxel)


.. _marvin-map-access-maps:

Accessing the Parent Maps Object
--------------------------------
From a :mod:`~marvin.tools.quantities.Map` object we can access its parent :mod:`~marvin.tools.maps.Maps` object via the :attr:`~marvin.tools.quantities.Map.maps` attribute.

.. code-block:: python

    >>> ha.getMaps() == maps  # True


.. _marvin-map-arithmetic:

Map Arithmetic
--------------

:mod:`~marvin.tools.quantities.Map` objects can be added, subtracted, multiplied, divided, or raised to a power.  You can also take the logarithm of them.

.. code-block:: python

    >>> ha = maps['emline_gflux_ha_6564']
    >>> nii = maps['emline_gflux_nii_6585']

    >>> sum_ = nii + ha
    >>> diff = nii - ha
    >>> prod = nii * ha
    >>> quot = nii / ha
    >>> pow_ = ha**0.5
    >>> n2ha = np.log10(nii / ha)

In addition to performing the arithmetic operation on the ``value``, the resulting :mod:`~marvin.tools.quantities.map.EnhancedMap` has correctly propagated ``ivar``, ``mask``, ``pixmask``, ``unit``, and ``scale``.


.. _marvin-map-masking:

Masking
-------

The :attr:`~marvin.tools.quantities.Map.masked` attribute is a `numpy masked array <https://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html>`_. The ``data`` attribute is the :attr:`~marvin.tools.quantities.Map.value` array and the ``mask`` attribute is a boolean array.  ``mask`` is ``True`` for a given spaxel if any of the recommended bad data flags (NOCOV, UNRELIABLE, and DONOTUSE) are set.

.. code-block:: python

    >>> ha.masked[17]
    masked_array(data=[--, --, --, --, --, --, --, 0.0360246, 0.0694705,
                   0.135435, 0.564578, 1.44708, 3.12398, 7.72712, 14.2869,
                   22.2461, 29.1134, 32.1308, 28.9591, 21.4879, 13.9937,
                   7.14412, 3.84099, 1.64863, 0.574292, 0.349627,
                   0.196499, 0.144375, 0.118376, --, --, --, --, --],
             mask=[ True,  True,  True,  True,  True,  True,  True, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False,
                   False, False, False, False, False,  True,  True,  True,
                    True,  True],
       fill_value=1e+20)

For more fine-grained data quality control, you can select spaxels using :attr:`~marvin.tools.quantities.Map.pixmask`, which contains the :attr:`~marvin.tools.quantities.Map.mask` values, knows the ``MANGA_DAPPIXMASK`` schema, and has convenience methods for converting between mask values, bit values, and labels.

See :ref:`marvin-utils-maskbit` for details.

.. code-block:: python

    >>> ha.pixmask
    <Maskbit 'MANGA_DAPPIXMASK' shape=(34, 34)>

    >>> ha.pixmask.schema
        bit         label                                        description
    0     0         NOCOV                         No coverage in this spaxel
    1     1        LOWCOV                        Low coverage in this spaxel
    2     2     DEADFIBER                   Major contributing fiber is dead
    3     3      FORESTAR                                    Foreground star
    4     4       NOVALUE  Spaxel was not fit because it did not meet sel...
    5     5    UNRELIABLE  Value is deemed unreliable; see TRM for defini...
    6     6     MATHERROR              Mathematical error in computing value
    7     7     FITFAILED                  Attempted fit for property failed
    8     8     NEARBOUND  Fitted value is too near an imposed boundary; ...
    9     9  NOCORRECTION               Appropriate correction not available
    10   10     MULTICOMP          Multi-component velocity features present
    11   30      DONOTUSE                 Do not use this spaxel for science

    >>> ha.pixmask.mask    # == ha.mask
    >>> ha.pixmask.bits    # bits corresponding to mask array
    >>> ha.pixmask.labels  # labels corresponding to mask array

**Note**: For ``MANGA_DAPPIXMASK``, DONOTUSE is a consolidation of the flags NOCOV, LOWCOV, DEADFIBER, FORESTAR, NOVALUE, MATHERROR, FITFAILED, and NEARBOUND.

Common Masking Operations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    >>> # Spaxels not covered by the IFU
    >>> nocov = ha.pixmask.get_mask('NOCOV')

    >>> # Spaxels flagged as bad data
    >>> bad_data = ha.pixmask.get_mask(['UNRELIABLE', 'DONOTUSE'])

    >>> # Custom mask (flag data as DONOTUSE to hide in plotting)
    >>> custom_mask = (ha.value < 1e-17) * ha.pixmask.labels_to_value('DONOTUSE')

    >>> # Combine masks
    >>> my_mask = nocov | custom_mask


.. _marvin-map-plot:

Plotting
--------

`Map` can be easily plotted using the ``plot`` method.  Details on plotting parameters and defaults can be found :ref:`here<marvin-utils-plot-map>`.  For a guide about making different types of plots see the :ref:`marvin-plotting-tutorial`.

.. plot::
    :align: center
    :include-source: True

    >>> from marvin.tools import Maps
    >>> maps = Maps('8485-1901')
    >>> ha = maps.emline_gflux_ha_6564
    >>> ha.plot()  # plot the H-alpha flux map.


.. _marvin-map-save:

Saving and Restoring
--------------------

Finally, we can :meth:`~marvin.tools.quantities.Map.save` our :mod:`~marvin.tools.quantities.Map` object as a MaNGA pickle file (``*.mpf``) and then :meth:`~marvin.tools.quantities.Map.restore` it.

.. code-block:: python

    >>> from marvin.tools.quantities import Map
    >>> ha.save(path='/path/to/save/directory/ha_8485-1901.mpf')
    >>> zombie_ha = Map.restore(path='/path/to/save/directory/ha_8485-1901.mpf')


.. _marvin-map-reference:

Reference/API
-------------

.. rubric:: Class Inheritance Diagram

.. inheritance-diagram:: marvin.tools.quantities.Map

.. rubric:: Class

.. autosummary:: marvin.tools.quantities.Map

.. rubric:: Methods

.. autosummary::

    marvin.tools.quantities.Map.error
    marvin.tools.quantities.Map.inst_sigma_correction
    marvin.tools.quantities.Map.masked
    marvin.tools.quantities.Map.pixmask
    marvin.tools.quantities.Map.plot
    marvin.tools.quantities.Map.restore
    marvin.tools.quantities.Map.save
    marvin.tools.quantities.Map.snr



.. _marvin-enhancedmap:

EnhancedMap
-----------

An :mod:`~marvin.tools.quantities.EnhancedMap` is a :mod:`~marvin.tools.quantities.Map` that has been modified by a map arithmetic operation (``+``, ``-``, ``*``, ``/``, ``**``, or ``np.log10()``). It inherits most of the attributes of a :mod:`~marvin.tools.quantities.Map`.

.. _marvin-enhanced-map-reference:

Reference/API
^^^^^^^^^^^^^

.. rubric:: Class Inheritance Diagram

.. inheritance-diagram:: marvin.tools.quantities.EnhancedMap

.. rubric:: Class

.. autosummary:: marvin.tools.quantities.EnhancedMap

.. rubric:: Methods

.. autosummary::

    marvin.tools.quantities.EnhancedMap.save
    marvin.tools.quantities.EnhancedMap.restore
    marvin.tools.quantities.EnhancedMap.masked
    marvin.tools.quantities.EnhancedMap.error
    marvin.tools.quantities.EnhancedMap.snr
    marvin.tools.quantities.EnhancedMap.plot

|
