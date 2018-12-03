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
====

`~marvin.tools.maps.Maps` is a class to interact with the set of DAP maps for a galaxy. For a general introduction to Marvin Tools, check out the :ref:`galaxy-tools` section.  Here we will revisit those features and will expand on some specifics of the `~marvin.tools.maps.Maps` class.

.. _marvin-maps-initializing:

Initializing a Maps
-------------------

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
----------------

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

The DataModel
-------------

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


.. _marvin-map-access:

Accessing an Individual Map
---------------------------

The `Property`s provide an interface to extract and create an individual `Map`.  You can use fuzzy indexing to retrieve a `Map`.  All properties are also available as class attributes.  Or you can use the old-fashioned `getMap` method.  All three are equivalent.

.. code-block:: python

    # get the H-alpha Gaussian flux Map
    >>> ha = maps['gflux_ha']

    # or
    >>> ha = maps.emline_gflux_ha_6564

    # or
    >>> ha = maps.getMap('emline_gflux_ha_6564')

    >>> print(ha)
    <Marvin Map (property='emline_gflux_ha_6564')>
    [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ...,
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]] 1e-17 erg / (cm2 s spaxel)


.. _marvin-spaxel-access:

Accessing an Individual Spaxel
------------------------------

Slicing a `Maps` returns a `Spaxel` object with all of its properties:

.. code-block:: python

    >>> sp = maps[9, 10]
    >>> print(sp)
    <Marvin Spaxel (plateifu=8485-1901, x=10, y=9; x_cen=-7, y_cen=-8, loaded=maps)>


.. _marvin-maps-binids:

Getting Bin IDs
---------------

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



BPT Diagrams
------------
See :ref:`marvin-bpt`.

.. _marvin-map:

The Map quantity
----------------

You can plot a map.  See :ref:`marvin-map` for how to use the `Map` object, and the :ref:`marvin-plotting-tutorial` for a guide into plotting.  Details on plotting parameters and defaults can be found :ref:`here<marvin-utils-plot-map>`.

.. code-block:: python

    # plot the H-alpha flux map.
    >>> ha.plot()

.. image:: ../../_static/quick_map_plot.png

Accessing an Individual Spaxel
------------------------------

Slicing a `Map` returns a single property
.. code-block:: python

    # the ha-value in the central bin
    >>> ha[17, 17]
    <Marvin Map (property='emline_gflux_ha_6564')>
    30.7445 1e-17 erg / (cm2 s spaxel)

Reference/API
-------------

.. _marvin-enhanced-map:

The Enhanced Map quantity
-------------------------

Reference/API
-------------

.. _marvin-modelcube:

ModelCube
=========

The HYB10 bintype
-----------------

Reference/API
-------------
