
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

We call Marvin Galaxy Tools to the three main classes (`Cube`, `~marvin.tools.maps.Maps`, and `~marvin.tools.ModelCube`) All the Tools classes can be accessed from the :ref:`marvin.tools <marvin-tools-ref>` module. Let's load a DRP cube ::

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

What is more, we can access the datamodel of the cube file, which show us what extensions are available, how they are named in Marvin, and what they contain ::

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

The flux is represented as a 3D array with units. We can also get the inverse variance and the mask

.. _marvin-quantities:

Working with quantities
-----------------------


----------

.. toctree::
    :maxdepth: 1

    cube
    maps
    map
    modelcube
    plate

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
