.. _marvin-dap-tools:

DAP Tools
=========

There are four types of DAP Tools:

* :ref:`marvin-maps`
* :ref:`marvin-map`
* :ref:`marvin-enhancedmap`
* :ref:`marvin-modelcube`


.. _marvin-maps:

Maps
----

`~marvin.tools.maps.Maps` is a class to interact with the set of DAP maps for a galaxy. For a general introduction to Marvin Tools, check out the :ref:`galaxy-tools` section.  Here we will revisit those features and will expand on some specifics of the `~marvin.tools.maps.Maps` class.

.. _marvin-maps-initializing:

Initializing a Maps
^^^^^^^^^^^^^^^^^^^

A `Maps` can be initialized by filename, plateifu, or mangaid.

**Filename**:

.. code-block:: python

    >>> maps = Maps(filename='/Users/username/manga/spectro/analysis/v2_4_3/2.2.1/HYB10-GAU-MILESHC/8485/1901/manga-8485-1901-MAPS-HYB10-GAU-MILESHC.fits.gz')
    >>> maps
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='file', bintype='HYB10', template='GAU-MILESHC')>

Either the full path or the path relative to the current working directory is required.  A `Maps` initialized from a file will always be in `local` mode.

**Plateifu** or **Mangaid**:

.. code-block:: python

    >>> maps = Maps(plateifu='8485-1901')
    >>> maps
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

    >>> maps = Maps(mangaid='1-209232')
    >>> maps
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

Marvin first attempts to find the data in a local database, otherwise will retrieve the data in `remote` mode.

**Smart Galaxy Lookup**

You can also initialize a `Maps` without the `filename` or a galaxy identifier (`plateifu`/`mangaid`) keyword argument, and Marvin will attempt to parse the input and find the desired galaxy:

.. code-block:: python

    >>> maps = Maps('8485-1901')
    >>> maps
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

**Bintype**

The default `Maps` bintype is `HYB10`, where the stellar continuum analysis of spectra is Voronoi binned to S/N~10 for the stellar kinematics; however, the emission line measurements are performed on the individual spaxels.  You can specify a different binning scheme with the `bintype` keyword (currently, the only other option is `VOR10`, which does the stellar continuum and emission line analyses on spectra Voronoi binned to S/N~10):

.. code-block:: python

    >>> maps = Maps('8485-1901', bintype='HYB10')
    >>> maps
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

**Template**

Currently, the only template available is `GAU-MILESHC`, which is selected by default.


.. _marvin-maps-basic:

Basic Attributes
^^^^^^^^^^^^^^^^

Like `Cubes`, `Maps` come with some basic attributes attached (e.g., the full header, the WCS info, the bintype and template) plus the NSA and DAPall catalog parameters.

.. code-block:: python

    # access the header
    >>> maps.header

    # access the wcs
    >>> maps.wcs

    # the NSA catalog information
    >>> maps.nsa['z']
    0.0407447

    # the DAPall catalog info
    >>> maps.dapall['sfr_tot']
    0.132697

`Maps` also has the DAP data quality, targeting, and pixel masks available as the `quality_flag`, `target_flags`, and `pixmask` attributes, respectively.  These are represented as :ref:`Maskbit <marvin-utils-maskbit>` objects.


.. _marvin-maps-datamodel:

Maps DataModel
^^^^^^^^^^^^^^

The :ref:`DAP datamodel <marvin-datamodels>` is attached to `Maps` as the `datamodel` attribute.  The datamodel describes the contents of the MaNGA DAP Maps, for a given release, and contains a list of `Properties` associated with a `Maps`.  This is a subset of the full DAP datamodel only pertaining to Maps.

.. code-block:: python

    # display the datamodel for maps properties
    >>> maps.datamodel
    [<Property 'spx_skycoo', channel='on_sky_x', release='2.1.3', unit=u'arcsec'>,
     <Property 'spx_skycoo', channel='on_sky_y', release='2.1.3', unit=u'arcsec'>,
     <Property 'spx_ellcoo', channel='elliptical_radius', release='2.1.3', unit=u'arcsec'>,
     <Property 'spx_ellcoo', channel='elliptical_azimuth', release='2.1.3', unit=u'deg'>,
     <Property 'spx_mflux', channel='None', release='2.1.3', unit=u'1e-17 erg / (cm2 s spaxel)'>,
     <Property 'spx_snr', channel='None', release='2.1.3', unit=u''>,
     <Property 'binid', channel='binned_spectra', release='2.1.3', unit=u''>,
     <Property 'binid', channel='stellar_continua', release='2.1.3', unit=u''>,
     <Property 'binid', channel='em_line_moments', release='2.1.3', unit=u''>,
     <Property 'binid', channel='em_line_models', release='2.1.3', unit=u''>,
     <Property 'binid', channel='spectral_indices', release='2.1.3', unit=u''>,
     ...
     <Property 'specindex_corr', channel='tio2sdss', release='2.1.3', unit=u'Angstrom'>,
     <Property 'specindex_corr', channel='d4000', release='2.1.3', unit=u''>,
     <Property 'specindex_corr', channel='dn4000', release='2.1.3', unit=u''>,
     <Property 'specindex_corr', channel='tiocvd', release='2.1.3', unit=u''>]

Each `Property` in the datamodel describes an available `Map` inside the `Maps` container, and has a channel, units, and a description.  You can fuzzy search through the list to identify maps:

.. code-block:: python

    # find the H-alpha Gaussian flux property
    >>> maps.datamodel['gflux_ha']
    <Property 'emline_gflux', channel='ha_6564', release='2.1.3', unit=u'1e-17 erg / (cm2 s spaxel)'>


.. _marvin-maps-access-map:

Accessing an Individual Map
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Property`s provide an interface to extract and create an individual `Map`. You can select an individual `Map` in one of four ways:

* exact key slicing,
* dot syntax,
* `getMap` method, or
* fuzzy key slicing.

.. code-block:: python

    >>> from marvin.tools import Maps
    >>> maps = Maps(plateifu='8485-1901')

    # exact key slicing
    >>> ha = maps['emline_gflux_ha_6564']

    # dot syntax
    >>> ha = maps.emline_gflux_ha_6564

    # getMap()
    >>> ha = maps.getMap('emline_gflux_ha_6564')
    # equivalently
    >>> ha = maps.getMap('emline_gflux', channel='ha_6564')

    # fuzzy key slicing
    >>> ha = maps['gflux ha']


Fuzzy key slicing works if the input is unambiguously associated with a particular key:

.. code-block:: python

    # Unambiguous inputs
    >>> maps['gflux ha']        # == maps['emline_gflux_ha_6564']
    >>> maps['gvel oiii 5008']  # == maps[emline_gvel_oiii_5008]
    >>> maps['stellar sig']     # == maps['stellar_sigma']

    # Ambiguous inputs
    # There are several velocity properties (stellar and emission lines).
    >>> maps['vel']  # ValueError

    # There are two [O III] lines.
    >>> maps['gflux oiii']  # ValueError


.. _marvin-maps-access-spaxel:

Accessing an Individual Spaxel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Slicing a `Maps` returns a `Spaxel` object with all of its properties:

.. code-block:: python

    >>> sp = maps[9, 10]
    >>> print(sp)
    <Marvin Spaxel (plateifu=8485-1901, x=10, y=9; x_cen=-7, y_cen=-8, loaded=maps)>


.. _marvin-maps-binids:

Getting Bin IDs
^^^^^^^^^^^^^^^

For binned `Maps`, you can retrieve a `Map` of the bin IDs directly from the `binid_*` attributes.  There are five types of bin IDs, designated as `binid_[name]`.  You can list them from the datamodel:

.. code-block:: python

    >>> maps.datamodel.parent['binid']
    <MultiChannelProperty 'binid', release='2.2.1', channels=['binned_spectra', 'stellar_continua', 'em_line_moments', 'em_line_models', 'spectral_indices']>

They are available as attributes.

.. code-block:: python

    # get a Map of the binned_spectra binids
    >>> maps.binid_binned_spectra
    <Marvin Map (property='binid_binned_spectra')>
    [[-1. -1. -1. ..., -1. -1. -1.]
     [-1. -1. -1. ..., -1. -1. -1.]
     [-1. -1. -1. ..., -1. -1. -1.]
     ...,
     [-1. -1. -1. ..., -1. -1. -1.]
     [-1. -1. -1. ..., -1. -1. -1.]
     [-1. -1. -1. ..., -1. -1. -1.]]

You can also retrieve a 2-d array of the bin IDs using the `get_binid` method.  By default, `get_binid` will return the bin IDs for the `binned_spectra` channel of **BINID**.

.. code-block:: python

    # get the default binids
    >>> maps.get_binid()
    <Marvin Map (property='binid_binned_spectra')>
    [[-1. -1. -1. ... -1. -1. -1.]
     [-1. -1. -1. ... -1. -1. -1.]
     [-1. -1. -1. ... -1. -1. -1.]
     ...
     [-1. -1. -1. ... -1. -1. -1.]
     [-1. -1. -1. ... -1. -1. -1.]
     [-1. -1. -1. ... -1. -1. -1.]]

    # equivalent
    >>> stvel_binids = maps.get_binid(property=maps.datamodel.stellar_vel)


.. _marvin-maps-access-objects:

Accessing Other Marvin Objects for the Same Galaxy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can grab the associated DRP `Cube` with `getCube`:

.. code-block:: python

    >>> maps.getCube()
    <Marvin Cube (plateifu='8485-1901', mode='remote', data_origin='api')>

or the `Modelcube` object using the `getModelCube` method:

.. code-block:: python

    >>> maps.getModelCube()
    <Marvin ModelCube (plateifu='8485-1901', mode='remote', data_origin='api', bintype='HYB10', template='GAU-MILESHC')>


.. _marvin-maps-save:

Saving and Restoring
^^^^^^^^^^^^^^^^^^^^

You can save a `Maps` locally as a Python pickle object, using the `save` method:

.. code-block:: python

    >>> maps.save('mymaps.mpf')

Your saved `Maps` can be restored as a `Maps` object using the `restore` class method:

.. code-block:: python

    >>> from marvin.tools import Maps
    >>> maps = Maps.restore('mymaps.mpf')


.. _marvin-maps-bpt:

BPT Diagram
^^^^^^^^^^^
You can create a :ref:`BPT<marvin-bpt>` diagram:

.. code-block:: python

    >>> masks, fig, axes = maps.get_bpt()

Reference/API
^^^^^^^^^^^^^

.. rubric:: Class Inheritance Diagram

.. inheritance-diagram:: marvin.tools.maps.Maps

.. rubric:: Class

.. autosummary:: marvin.tools.maps.Maps

.. rubric:: Methods

.. autosummary::

    marvin.tools.maps.Maps.get_binid
    marvin.tools.maps.Maps.get_unbinned
    marvin.tools.maps.Maps.get_bpt
    marvin.tools.maps.Maps.getCube
    marvin.tools.maps.Maps.getModelCube
    marvin.tools.maps.Maps.getSpaxel
    marvin.tools.maps.Maps.getMap
    marvin.tools.maps.Maps.getMapRatio
    marvin.tools.maps.Maps.download
    marvin.tools.maps.Maps.save
    marvin.tools.maps.Maps.restore


.. _marvin-map:

Map
---

:mod:`~marvin.tools.quantities.Map` is a single map for a single galaxy. The main data that it contains are the :attr:`~marvin.tools.quantities.Map.value`, :attr:`~marvin.tools.quantities.Map.ivar`, and :attr:`~marvin.tools.quantities.Map.mask` arrays of the map.

Initializing
^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^


.. _marvin-map-access-spaxel:

Accessing an Individual Spaxel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Slicing a `Map` returns a single property

.. code-block:: python

    >>> ha[17, 17]  # the Halpha flux value in the central spaxel
    <Marvin Map (property='emline_gflux_ha_6564')>
    30.7445 1e-17 erg / (cm2 s spaxel)


.. _marvin-map-access-maps:

Accessing the Parent Maps Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. _marvin-map-arithmetic:

Map Arithmetic
^^^^^^^^^^^^^^

.. _marvin-map-masking:

Masks
^^^^^

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
`````````````````````````

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

Plotting a Map
^^^^^^^^^^^^^^
You can plot a map.  See :ref:`marvin-map` for how to use the `Map` object, and the :ref:`marvin-plotting-tutorial` for a guide into plotting.  Details on plotting parameters and defaults can be found :ref:`here<marvin-utils-plot-map>`.

.. code-block:: python

    # plot the H-alpha flux map.
    >>> ha.plot()

.. image:: ../../_static/quick_map_plot.png


.. _marvin-map-save:

Saving and Restoring
^^^^^^^^^^^^^^^^^^^^

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

Reference/API
^^^^^^^^^^^^^

.. _marvin-modelcube:

ModelCube
---------

The HYB10 bintype
^^^^^^^^^^^^^^^^^

Reference/API
^^^^^^^^^^^^^
