.. _marvin-maps:

Maps
====

`Maps` and `Map` objects form a hierarchy:

* :ref:`marvin-maps`: set of DAP maps for a galaxy (analogous to a DAP MAPS FITS file),
* :ref:`marvin-map`: an individual map, and
* :ref:`marvin-enhancedmap`: an individual map modified by map arithmetic.


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

|
