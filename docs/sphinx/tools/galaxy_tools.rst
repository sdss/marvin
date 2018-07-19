
.. role:: green
.. role:: orange
.. role:: red
.. role:: purple

.. py:currentmodule:: marvin.tools

.. _galaxy-tools:

Galaxy Tools
============

Introduction
------------

Marvin Tools provide the core functionality accessing MaNGA data with Marvin. At their lowest level they are class wrappers around data products or elements (DRP datacubes, DAP maps, spaxels). Their goals is to provide a more natural way to interact with the data, unconstrained by specific data storage architectures such as files or databases. The tools are built on top of the :ref:`data access modes <marvin-dma>` which abstracts the data access regardless of their actual origin.

Marvin Tools provide:

.. todo:: Add links in this list once all the documentation is in place.

- Access DRP :ref:`Cubes <marvin-cube>` and their spectra.
- Access DAP :ref:`Maps <marvin-maps>` and :ref:`ModelCubes <marvin-modelcube>`.
- Convenient access to all the DRP and DAP properties for a given :ref:`Spaxel or Bin <marvin-subregion-tools>`.
- Data is delivered as :ref:`quantities <marvin-quantities>` with attached variance and mask.
- DAP Map arithmetic.
- Autocompletion of properties and channels (powered by a datamodel).
- Extract all spaxels within a region.
- Access to NSA and DRPall data.
- Easy data download.

Getting started
---------------

We call Marvin galaxy Tools to the three main classes (`~cube.Cube`, `~maps.Maps`, and `~modelcube.ModelCube`) associated to the analogous DRP and DAP data products, the `quantities <marvin-quantities>` representing multidimensional data, and a variety of utilities and mixins that provide additional functionality. Sub-region galaxy tools (`~spaxel.Spaxel` and `~spaxel.Bin`) are explained `in their own section <marvin-subregion-tools>`. The three main Tools classes inherit from `~core.core.MarvinToolsClass` and thus much of their functionality and logic is shared. In this section we will prominently use the `~cube.Cube` but most of what we explain here can also be applied to the `~maps.Maps` and `~modelcube.ModelCube`.

All the Tools classes can be accessed from the :ref:`marvin.tools <marvin-tools-ref>` module. Let's load a DRP cube ::

    >>> import marvin
    >>> my_cube = marvin.tools.Cube('7443-12703')
    >>> my_cube
    <Marvin Cube (plateifu='7443-12703', mode='local', data_origin='file')>

Depending on whether you have the file on disk or not, the access mode will be ``'local'`` or ``'remote'``. Regardless of that, the way we interact with the object will be the same. All tools provide quick access to some basic metadata ::

    >>> print(my_cube.filename, my_cube.plateifu, my_cube.mangaid, my_cube.release)
    /Users/albireo/Documents/MaNGA/mangawork/manga/spectro/redux/v2_3_1/7443/stack/manga-7443-12703-LOGCUBE.fits.gz 7443-12703 12-193481, MPL-6
    >>> print(my_cube.ra, my_cube.dec)
    229.525575871 42.7458424664

Similarly we can access the `header <astropy.io.fits.Header>` of the file and the `WCS <astropy.wcs.WCS>` object ::

    >>> my_cube.header
    XTENSION= 'IMAGE   '           / IMAGE extension
    BITPIX  =                  -32 / Number of bits per data pixel
    NAXIS   =                    3 / Number of data axes
    NAXIS1  =                   74 /
    NAXIS2  =                   74 /

    >>> my_cube.wcs
    WCS Keywords

    Number of WCS axes: 3
    CTYPE : 'RA---TAN'  'DEC--TAN'  'WAVE-LOG'
    CRVAL : 229.52558  42.745842  3.62159598486e-07
    CRPIX : 38.0  38.0  1.0
    CD1_1 CD1_2 CD1_3  : -0.000138889  0.0  0.0
    CD2_1 CD2_2 CD2_3  : 0.0  0.000138889  0.0
    CD3_1 CD3_2 CD3_3  : 0.0  0.0  8.33903304339e-11
    NAXIS : 74  74  4563

What is more, we can access the :ref:`datamodel <marvin-datamodel>` of the cube file, which show us what extensions are available, how they are named in Marvin, and what they contain ::

    >>> datamodel = my_cube.datamodel
    >>> datamodel
    <DRPDataModel release='MPL-6', n_datacubes=3, n_spectra=2>

    >>> datamodel.datacubes
    [<DataCube 'flux', release='MPL-6', unit='1e-17 erg / (Angstrom cm2 s spaxel)'>,
     <DataCube 'dispersion', release='MPL-6', unit='Angstrom'>,
     <DataCube 'dispersion_prepixel', release='MPL-6', unit='Angstrom'>]

    >>> datamodel.spectra
    [<Spectrum 'spectral_resolution', release='MPL-6', unit='Angstrom'>,
     <Spectrum 'spectral_resolution_prepixel', release='MPL-6', unit='Angstrom'>]

This tells us that this cube has two associated 3D datacubes, ``'flux'``, ``'dispersion'``, and ``'dispersion_prepixel'``, and two associated spectra, ``'spectral_resolution'`` and ``'spectral_resolution_prepixel'``, as well as their associated units. We can get a desciption of what each of them ::

    >>> datamodel.datacubes.flux.description
    'flux'
    >>> datamodel.datacubes.flux.description
    '3D rectified cube'

In ``my_cube``, we can use the name of each of these datacubes and spectra to access the associated data quantity. Let's get the cube flux ::

    >>> flux = my_cube.flux
    >>> flux
    <DataCube [[[0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                ...,
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.]]] 1e-17 erg / (Angstrom cm2 s spaxel)>

The flux is represented as a 3D array with units. We can also access the inverse variance and the mask using ``flux.ivar`` and ``flux.mask``, respectively. We can slice this datacube to get another datacube ::

    >>> flux[:, 50:60, 50:60]
    <DataCube [[[ 0.23239002,  0.21799691,  0.1915081 , ...,  0.06516988,
              0.03220467,  0.02613733],
            [ 0.2511523 ,  0.25672358,  0.24318442, ...,  0.07530793,
              0.0505379 ,  0.05970671],
            [ 0.24604724,  0.23915106,  0.24392547, ...,  0.1116344 ,
              0.08573902,  0.10379973],
            ...,
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,
              0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,
              0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        , ...,  0.        ,
              0.        ,  0.        ]]] 1e-17 erg / (Angstrom cm2 s spaxel)>

Or get a single spectrum and plot it::

    >>> spectrum = flux[:, 50, 55]
    >>> spectrum
    <Spectrum [0.1060614 , 0.07801704, 0.02460545, ..., 0.16328742, 0.13772544,
           0.        ] 1e-17 erg / (Angstrom cm2 s spaxel)>

    >>> spectrum.plot(show_std=True)

.. plot::
    :align: center

    import marvin

    my_cube = marvin.tools.Cube('7443-12703')
    flux = my_cube.flux
    spectrum = flux[:, 50, 55]
    ax = spectrum.plot(show_std=True)
    ax.set_xlim(6000, 8000)


We will talk more about quantities in the :ref:`marvin-quantities` section, and about more advance plotting in :ref:`marvin-plotting`.

From a DRP cube we can get the associated DAP `~marvin.tools.maps.Maps` object for a certain bintype ::

    >>> hyb_maps = my_cube.getMaps(bintype='HYB10')
    <Marvin Maps (plateifu='7443-12703', mode='local', data_origin='file', bintype='HYB10', template='GAU-MILESHC')>

A `~marvin.tools.maps.Maps` behaves very similarly to a `~marvin.tools.cube.Cube` and everything we have discussed above will still work. Instead of datacubes and spectra, a Maps object contains a set of 2D quantities called `~marvin.tools.quantities.map.Map`, each one of them representing a different ``property`` measured by the DAP. One can get a full list of all the properties available using the :ref:`datamodel <marvin-datamodel>` ::

    >>> hyb_maps.datamodel
    [<Property 'spx_skycoo', channel='on_sky_x', release='2.1.3', unit='arcsec'>,
     <Property 'spx_skycoo', channel='on_sky_y', release='2.1.3', unit='arcsec'>,
     <Property 'spx_ellcoo', channel='elliptical_radius', release='2.1.3', unit='arcsec'>,
     <Property 'spx_ellcoo', channel='r_re', release='2.1.3', unit=''>,
     <Property 'spx_ellcoo', channel='elliptical_azimuth', release='2.1.3', unit='deg'>,
     <Property 'spx_mflux', channel='None', release='2.1.3', unit='1e-17 erg / (cm2 s spaxel)'>,
     <Property 'spx_snr', channel='None', release='2.1.3', unit=''>,
     <Property 'binid', channel='binned_spectra', release='2.1.3', unit=''>,
     ...
    ]

Note that some properties such as ``'spx_skycoo'`` have multiple channels (in this case the on-sky x and y coordinates). We can get more information about a property ::

    >>> hyb_maps.datamodel.spx_skycoo_on_sky_x.description
    'Offsets of each spaxel from the galaxy center.'

See the :ref:`datamodel <marvin-datamodel>` section for more information on how to use this feature. We can retrieve the map associated to a specific property directly from the `~marvin.tools.maps.Maps` instance. For example, let's get the H :math:`\alpha` emission line flux (fitted by a Gaussian) ::

    >>> ha = hyb_maps.emline_gflux_ha_6564
    >>> ha
    <Marvin Map (property='emline_gflux_ha_6564')>
    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]] 1e-17 erg / (cm2 s spaxel)

.. hint:: In IPython, you can use tab-completion to autocomplete the name of the property. If you press tab after writing ``hyb_maps.emline_`` you will get a list of all the emission line properties available.

`~marvin.tools.quantities.map.Map` quantities are similar to `~marvin.tools.quantities.datacube.DataCube` but wrap a 2D array. We can plot the Map as ::

    >>> fig, ax = ha.plot()

.. plot::
    :align: center

    import marvin
    my_maps = marvin.tools.Maps('7443-12703', bintype='HYB10')
    my_maps.emline_gflux_ha_6564.plot()

Note that the `~marvin.tools.quantities.map.Map.plot` method returns the matplotlib `~matplotlib.figure.Figure` and `~matplotlib.axes.Axes` for the plot. We can use those to modify or save the plot. :ref:`Marvin plotting routines <marvin-plotting>` try to select the best parameters, colour maps, and dynamic ranges. You can modify those by passing extra arguments to `~marvin.tools.quantities.map.Map.plot`. You can learn more in the :ref:`Map plotting <marvin-utils-plot-map>` section. We will talk about the `~marvin.tools.quantities.map.Map` class in detail in :ref:`marvin-quantities` and in :ref:`marvin-map`.

- Targeting bits
- Quality bits
- Downloading a file
- Get spaxel + slicing

.. _marvin-quantities:

Working with quantities
-----------------------

TBD


Using the tools
---------------

Data access modes
^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    data-access-modes

Storing data
^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    downloads
    pickling

Accessing catalogue data
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    catalogues

Defining apertures
^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    aperture


Maskbits
^^^^^^^^

.. toctree::
    :maxdepth: 2

    utils/maskbit

Datamodels
^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    datamodel

Using Maps and ModelCubes
^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    dap_tools

The Plate class
^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    plate

Plotting
^^^^^^^^

.. toctree::
    :maxdepth: 2

    utils/plotting

Image utilities
^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    utils/images


.. _visual-guide:

Visual guide
------------

All **object-** and **search-based** tools in Marvin are linked together. To better understand the flow amongst the various Tools, here is a visual guide.

.. image:: ../../Marvin_Visual_Guide.png
    :width: 800px
    :align: center
    :alt: marvin visual guide

* The :red:`red squares` and :green:`green squares` indicate the set of Marvin Tools available.
* The :orange:`orange circles` highlight how each Tool links together via a method or an attribute. In each transition link, a lowercase Tool name represents an instantiation of that tool, e.g. ``cube = Cube()``. To go from a Marvin ``Cube`` to a Marvin ``Spaxel``, you can use the ``cube.getSpaxel`` method or the ``cube[x,y]`` notation. Conversely, to go from a ``Spaxel`` to a ``Cube``, you would use the ``spaxel.cube`` attribute. Single or bidirectional arrows tell you which directions you can flow to and from the various tools.
* :purple:`Purple circles` represent display endpoints. If you want to display something, this shows you how which tool the plotting command is connected to, and how to navigate there.


Reference
---------

Tools
^^^^^

.. autosummary::

   marvin.tools.cube.Cube
   marvin.tools.maps.Maps
   marvin.tools.modelcube.ModelCube

Quantities
^^^^^^^^^^

.. autosummary::

    marvin.tools.quantities.analysis_props.AnalysisProperty
    marvin.tools.quantities.spectrum.Spectrum
    marvin.tools.quantities.map.Map
    marvin.tools.quantities.datacube.DataCube

MixIns
^^^^^^

.. autosummary::

    marvin.tools.mixins.nsa.NSAMixIn
    marvin.tools.mixins.dapall.DAPAllMixIn
    marvin.tools.mixins.aperture.GetApertureMixIn
