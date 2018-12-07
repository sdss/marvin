.. _marvin-maps-deprecated:

Maps
====

:ref:`marvin-tools-maps` is a class to interact with a set of DAP maps/properties for a galaxy.

.. _marvin-maps_getstart:

Getting Started
---------------

`Maps` behaves in much the same way as a `Cube`.  To initialize a `Maps`, you can specify either a **mangaid**, **plateifu**, or **filename** as input.  Marvin will attempt to open the file locally from a file, a database, or remotely over the API.

::

    from marvin.tools.maps import Maps
    maps = Maps(mangaid='1-209232')

    print(maps)
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='SPX', template='GAU-MILESHC')>

By default, it will grab the unbinned maps.  You can specify a different binning with the `bintype` keyword.
::

    maps = Maps(mangaid='1-209232', bintype='HYB10')

    print(maps)
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

You can quickly grab an entire map by fuzzy indexing or as a class attribute.
::

    # grab the ha-flux map
    ha = maps['gflux_ha']

    ha = maps.emline_gflux_ha_6564

    print(ha)
    <Marvin Map (property='emline_gflux_ha_6564')>
    [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ...,
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]] 1e-17 erg / (cm2 s spaxel)

You can quickly grab a spaxel/bin by slicing the `Maps` like an array.  Slicing a `Maps` returns a `Spaxel` or `Bin`, with a full spectrum.
::

    # grab the bin of the central spaxel
    bin_cen = maps[17, 17]
    print(bin_cen)
    <Marvin Bin (plateifu=8485-1901, x=17, y=17; x_cen=0, y_cen=0, n_spaxels=1)>

`n_spaxels` tells us there is only one spaxel in this bin. See :ref:`marvin-bin` for more details on the `Bin` class. The binned`flux` in this bin is available as an attribute.  It is represented as a Marvin Spectrum, which is a Quantity.  To quickly plot the flux, use the `plot` method on the `flux`.
::

    # look at the binned flux
    bin_cen.flux
    <Spectrum [ 0.54676276, 0.46566465, 0.4622981 ,...,  0.        ,
                0.        , 0.        ] 1e-17 erg / (Angstrom cm2 s spaxel)>

    # plot the binned flux
    bin_cen.flux.plot()

.. image:: ../_static/spec_8485-1901_17-17.png


The `Maps` data quality and targeting flags are available as the `quality_flag`, and `target_flags`, respectively.  These are represented as a :ref:`Maskbit <marvin-utils-maskbit>` objects.  A **good** quality `Maps` has an empty (0) bit list.

::

    # check the quality and bits
    maps.quality_flag
    <Maskbit 'MANGA_DAPQUAL' []>

    maps.quality_flag.bits
    []

    # check the targeting flags
    maps.target_flags
    [<Maskbit 'MANGA_TARGET1' ['SECONDARY_v1_1_0', 'SECONDARY_COM2', 'SECONDARY_v1_2_0']>,
     <Maskbit 'MANGA_TARGET2' []>,
     <Maskbit 'MANGA_TARGET3' []>]

A single `Map` has a pixel mask, as the `pixmask` attribute.
::

    # retrieve the maps pixel mask
    ha.pixmask
    <Maskbit 'MANGA_DAPPIXMASK' shape=(34, 34)>

The DAPall information is accessible via the `dapall` attribute.  It is a dictionary of the all the parameters from the DAPall file available for this target.  Use `dapall.keys()` to see all of the available parameters.
::

    # grab the star-formation rate within the IFU field-of-view
    maps.dapall['sfr_tot']
    0.132697

    # and the mean surface brightness within 1 effective radius
    maps.dapall['sb_1re']
    0.738855

.. _marvin-maps-using:

Using Maps
----------

.. _marvin-maps-init:

Initializing a Maps
^^^^^^^^^^^^^^^^^^^

A `Maps` can be initialized in several ways, by **filename**, in which case it will always be in `local` mode.
::

    maps = Maps(filename='/Users/Brian/Work/Manga/analysis/v2_3_1/2.1.3/SPX-GAU-MILESHC/8485/1901/manga-8485-1901-MAPS-SPX-GAU-MILESHC.fits.gz')
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='file', bintype='SPX', template='GAU-MILESHC')>

by **plateifu** or **mangaid**, in which case it attempts to find a local database, otherwise will open it in `remote` mode.
::

    maps = Maps(plateifu='8485-1901', bintype='HYB10')
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

    maps = Maps(mangaid='1-209232', bintype='HYB10')
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

However you can also initialize a `Maps` without the keyword argument and Marvin will attempt to figure out what input you mean.
::

    maps = Maps('8485-1901', bintype='HYB10')
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

.. _marvin-maps-basic-deprecated:

Basic Attributes
^^^^^^^^^^^^^^^^

Like 'Cubes', `Maps` come with some basic attributes attached, e.g. the full header, the WCS info, the bintype and template, and the NSA and DAPall catalog parameters.
::

    # access the header
    maps.header

    # access the wcs
    maps.wcs

    # the NSA catalog information
    maps.nsa['z']
    0.0407447

    # the DAPall catalog info
    maps.dapall['sfr_tot']
    0.132697

`Maps` also has the DAP data quality, targeting, and pixel masks available as the `quality_flag`, `target_flags`, and `pixmask` attributes, respectively.  These are represented as a :ref:`Maskbit <marvin-utils-maskbit>` objects.

.. _marvin-maps-datamodel-deprecated:

The DataModel
^^^^^^^^^^^^^

The :ref:`DAP datamodel <marvin-datamodels>` is attached to `Maps` as the `datamodel` attribute.  The datamodel describes the contents of the MaNGA DAP Maps, for a given release, and contains a list of `Properties` associated with a `Maps`.  This is a subset of the full DAP datamodel only pertaining to Maps.

::

    # display the datamodel for maps properties
    maps.datamodel
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

Each `Property` in the datamodel describes an available `Map` inside the `Maps` container, and has a channel, units, and a description.  You can fuzzy search through the list to identify maps
::

    # find the H-alpha Gaussian flux property
    maps.datamodel['gflux_ha']
    <Property 'emline_gflux', channel='ha_6564', release='2.1.3', unit=u'1e-17 erg / (cm2 s spaxel)'>

.. _marvin-maps-props:

Properties and the Map
^^^^^^^^^^^^^^^^^^^^^^

The `Properties` provide an interface to extract and create an individual `Map`.  You can use fuzzy indexing to retrieve a map.  All properties are also available as class attributes.  Or you can use the old-fashioned `getMap` method.  All three are equivalent.
::

    # get the H-alpha Gaussian flux Map
    ha = maps['gflux_ha']

    # or
    ha = maps.emline_gflux_ha_6564

    # or
    ha = maps.getMap('emline_gflux_ha_6564')

    print(ha)
    <Marvin Map (property='emline_gflux_ha_6564')>
    [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ...,
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]] 1e-17 erg / (cm2 s spaxel)

You can plot a map.  See :ref:`marvin-map` for how to use the `Map` object, and the :ref:`marvin-plotting-tutorial` for a guide into plotting.  Details on plotting parameters and defaults can be found :ref:`here<marvin-utils-plot-map>`.
::

    # plot the H-alpha flux map.
    ha.plot()

.. image:: ../_static/quick_map_plot.png

Slicing a map returns a single property
::

    # the ha-value in the central bin
    ha[17,17]
    <Marvin Map (property='emline_gflux_ha_6564')>
    30.7445 1e-17 erg / (cm2 s spaxel)

.. _marvin-maps-getbins:

Getting the Binids
^^^^^^^^^^^^^^^^^^

For binned `Maps`, you can retrieve a `Map` of the binids directly from the `binid_*` attributes.  For MPL-5, there is only a single `binid`.  As of MPL-6, there are five types of binids, designated as `binid_[name]`.  You can list them from the datamodel
::

     maps.datamodel.parent['binid']
    <MultiChannelProperty 'binid', release='2.1.3', channels=['binned_spectra', 'stellar_continua', 'em_line_moments', 'em_line_models', 'spectral_indices']>

They are available as attributes.
::

    # get a Map of the binned_spectra binids
    maps.binid_binned_spectra
    <Marvin Map (property='binid_binned_spectra')>
    [[-1. -1. -1. ..., -1. -1. -1.]
     [-1. -1. -1. ..., -1. -1. -1.]
     [-1. -1. -1. ..., -1. -1. -1.]
     ...,
     [-1. -1. -1. ..., -1. -1. -1.]
     [-1. -1. -1. ..., -1. -1. -1.]
     [-1. -1. -1. ..., -1. -1. -1.]]

You can also retrieve a 2-d array of the binids using the `get_binid` method.  For MPL-5, `get_binid` returns the binids from the **BINID** extension in the DAP files, while for MPL-6, by default, `get_binid` will return the binids for the `binned_spectra` channel of **BINID**.
::

    # get the default binids
    maps.get_binid()
    array([[-1, -1, -1, ..., -1, -1, -1],
           [-1, -1, -1, ..., -1, -1, -1],
           [-1, -1, -1, ..., -1, -1, -1],
           ...,
           [-1, -1, -1, ..., -1, -1, -1],
           [-1, -1, -1, ..., -1, -1, -1],
           [-1, -1, -1, ..., -1, -1, -1]])

MPL-6 has new cubes using hybrid binning, **HYB10**, with alternate binning schemes.  `get_binid` can retrieve those with the `binid` keyword.
::

    # grab the binids for the emline_fit model
    emline_binids = maps.get_binid(binid=maps.datamodel.binid_binned_spectra)

    print(emline_binids)
    array([[-1, -1, -1, ..., -1, -1, -1],
       [-1, -1, -1, ..., -1, -1, -1],
       [-1, -1, -1, ..., -1, -1, -1],
       ...,
       [-1, -1, -1, ..., -1, -1, -1],
       [-1, -1, -1, ..., -1, -1, -1],
       [-1, -1, -1, ..., -1, -1, -1]])

.. _marvin-maps-extract:

Extracting Spaxels/Bins
^^^^^^^^^^^^^^^^^^^^^^^

If working with a unbinned `Maps`, slicing and `getSpaxel` will retrieve and return a :ref:`Spaxel <marvin-tools-spaxel>` object, and behaves exactly the same as a Marvin :ref:`Cube <marvin-cube-extract>`.  For binned objects, it's exactly like a Marvin :ref:`ModelCube<marvin-modelcube-extract>`.  Slicing and extracting returns a :ref:`marvin-bin` object instead, behaves exactly the same as `Spaxel`, except it now contains a list of spaxels belonging to that bin.

.. _marvin-maps-access:

Accessing Related Objects
^^^^^^^^^^^^^^^^^^^^^^^^^

You can grab the associated DRP `Cube` with `getCube`.
::

    maps.getCube()
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='db')>

or the `Modelcube` object using the `getModelcube` method.
::

    maps.getModelCube()
    <Marvin ModelCube (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

From a binned `Maps`, you can go back to the unbinned version with the `get_unbinned` method:
::

    print(maps)
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>

    maps.get_unbinned()
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='SPX', template='GAU-MILESHC')>

You can create a :ref:`BPT<marvin-bpt>` diagram.
::

    maps.get_bpt()


.. _marvin-maps-save-deprecated:

Saving and Restoring
^^^^^^^^^^^^^^^^^^^^

You can save a `Maps` locally as a Python pickle object, using the `save` method.
::

    maps.save('mymaps.mpf')

as well as restore a Maps pickle object using the `restore` class method
::

    from marvin.tools.maps import Maps

    maps = Maps.restore('mymaps.mpf')


.. _marvin-maps-api:

Reference/API
-------------

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
