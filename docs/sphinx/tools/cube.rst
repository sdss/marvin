.. _marvin-cube:

Cube
====

`~marvin.tools.cube.Cube` is a class to interact with a fully reduced DRP data cube for a galaxy. For a general introduction to Marvin Tools, check out the :ref:`galaxy-tools` section. Here we will revisit those features and will expand on some specifics of the `~marvin.tools.cube.Cube` class.


Initializing a Cube
^^^^^^^^^^^^^^^^^^^

A `~marvin.tools.cube.Cube` can be initialized in several ways, by **filename**, in which case it will always be in ``local`` mode:
::

    cube = Cube(filename='/Users/Brian/Work/Manga/redux/v2_3_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz')
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='file')>

by **plateifu** or **mangaid**, in which case it attempts to find a local database, otherwise will open it in ``remote`` mode:
::

    cube = Cube(plateifu='8485-1901')
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='db')>

    cube = Cube(mangaid='1-209232')
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='db')>

However you can also initialize a `~marvin.tools.cube.Cube` without the keyword argument and Marvin will attempt to figure out what input you mean.
::

    cube = Cube('8485-1901')
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='db')>


.. _marvin-cube-basic:

Basic Attributes
^^^^^^^^^^^^^^^^

Cubes come with some basic attributes attached, like the full header (as an Astropy Header object), cube RA and Dec, the WCS info (as an Astropy WCS object), and the NSA catalog information (as a dictionary).
::

    # access the header
    cube.header

    CHECKSUM= 'HLO1KLM1HLM1HLM1'   / HDU checksum updated 2017-10-17T06:02:42
    DATASUM = '3722061489'         / data unit checksum updated 2017-10-17T06:02:42
    EXTNAME = 'FLUX    '
    QUALDATA= 'MASK    '           / Mask extension name
    ERRDATA = 'IVAR    '           / Error extension name
    HDUCLAS2= 'DATA    '
    HDUCLAS1= 'CUBE    '
    HDUCLASS= 'SDSS    '           / SDSS format class
    CUNIT2  = 'deg     '
    CUNIT1  = 'deg     '
    CTYPE2  = 'DEC--TAN'
    CTYPE1  = 'RA---TAN'
    ...

    # the cube RA and Dec (the OBJRA and OBJDEC)
    cube.ra, cube.dec
    (232.544703894, 48.6902009334)

    # the NSA catalog information
    cube.nsa['z']
    0.0407447

    c.nsa['elpetro_ba']
    0.87454

The `~marvin.tools.cube.Cube` data quality and targeting flags are available as the ``quality_flag`` and ``target_flags`` attributes, respectively.  These are represented as :ref:`Maskbit <marvin-utils-maskbit>` objects.  A **good** quality `~marvin.tools.cube.Cube` has an empty (0) bit list. If you are not familiar with MaNGA's maskbits, check the `official documentation <https://www.sdss.org/algorithms/bitmasks/#MANGA_TARGET1>`__.

::

    # check the quality and bits
    cube.quality_flag
    <Maskbit 'MANGA_DRP3QUAL' []>

    cube.quality_flag.bits
    []

    # check the targeting flags
    cube.target_flags
    [<Maskbit 'MANGA_TARGET1' ['SECONDARY_v1_1_0', 'SECONDARY_COM2', 'SECONDARY_v1_2_0']>,
     <Maskbit 'MANGA_TARGET2' []>,
     <Maskbit 'MANGA_TARGET3' []>]


.. _marvin-cube-datamodel:

The DataModel
^^^^^^^^^^^^^

The :ref:`DRP datamodel <marvin-datamodels>` is attached to `~marvin.tools.cube.Cube` as the ``datamodel`` attribute.  The datamodel describes the contents of the MaNGA DRP Cube, for a given release.
::

    cube.datamodel
    <DRPDataModel release='MPL-6', n_datacubes=3, n_spectra=2>>

The DRP datamodel contains both 1-d (Spectra) and 3-d (DataCubes) representations of the data within a DRP Cube.
::

    # see the available Datacubes
    cube.datamodel.datacubes
    [<DataCube 'flux', release='MPL-6', unit=u'1e-17 erg / (Angstrom cm2 s spaxel)'>,
     <DataCube 'dispersion', release='MPL-6', unit=u'Angstrom'>,
     <DataCube 'dispersion_prepixel', release='MPL-6', unit=u'Angstrom'>]

     # see the available Spectra
    [<Spectrum 'spectral_resolution', release='MPL-6', unit=u'Angstrom'>,
     <Spectrum 'spectral_resolution_prepixel', release='MPL-6', unit=u'Angstrom'>]


.. _marvin-cube-datacubes:

DataCubes and Spectra
^^^^^^^^^^^^^^^^^^^^^

The datamodel provides `~marvin.tools.quantities.datacube.DataCube` and `~marvin.tools.quantities.spectrum.Spectrum` objects for each target for a given release.  These objects are :ref:`marvin quantities <marvin-quantities>`.  For example, in DR15, there are three available `DataCubes <marvin.tools.quantities.datacube.DataCube>`, the ``flux``, ``dispersion``, and ``dispersion_prepixel``, and two `Spectra <marvin.tools.quantities.spectrum.Spectrum>`, the ``spectral_resolution`` and ``spectral_resolution_prepixel``.
::

    # access the cube flux
    cube.flux
    <DataCube [[[ 0., 0., 0.,...,  0., 0., 0.],
                [ 0., 0., 0.,...,  0., 0., 0.],
                [ 0., 0., 0.,...,  0., 0., 0.],
                ...,
                [ 0., 0., 0.,...,  0., 0., 0.],
                [ 0., 0., 0.,...,  0., 0., 0.],
                [ 0., 0., 0.,...,  0., 0., 0.]]] 1e-17 erg / (Angstrom cm2 s spaxel)>

    type(cube.flux)
    marvin.tools.quantities.datacube.DataCube

You can always get back the numpy array values using the ``value`` attribute.
::

    # retrieve the underlying data
    cube.flux.value
    array([[[ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            ...,
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.]],

           [[ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            ...

DataCubes and Spectra behave as quantities, so may have available ``wavelength``, ``ivar`` and ``mask`` attached.
::

    # get the wavelength
    cube.flux.wavelength
    <Quantity [  3621.6 ,  3622.43,  3623.26,...,  10349.  , 10351.4 , 10353.8 ] Angstrom>

    # get the flux ivar and mask
    cube.flux.ivar
    array([[[ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            ...,
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.]],

           [[ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            ...

    cube.flux.mask
    array([[[1027, 1027, 1027, ..., 1027, 1027, 1027],
            [1027, 1027, 1027, ..., 1027, 1027, 1027],
            [1027, 1027, 1027, ..., 1027, 1027, 1027],
            ...,
            [1027, 1027, 1027, ..., 1027, 1027, 1027],
            [1027, 1027, 1027, ..., 1027, 1027, 1027],
            [1027, 1027, 1027, ..., 1027, 1027, 1027]],

           [[1027, 1027, 1027, ..., 1027, 1027, 1027],
            [1027, 1027, 1027, ..., 1027, 1027, 1027],
            [1027, 1027, 1027, ..., 1027, 1027, 1027],
            ...

You can manipulate the pixel mask using the ``pixmask`` attribute.
::

    cube.flux.pixmask
    <Maskbit 'MANGA_DRP3PIXMASK' shape=(4563, 34, 34)>


.. _marvin-cube-extract:

Extracting a Spaxel
^^^^^^^^^^^^^^^^^^^

From a `~marvin.tools.cube.Cube` you can access Marvin objects related to this particular target.  To access a `~marvin.tools.spaxel.Spaxel`, you can slice like an array
::

    # slice a cube by i, j
    spaxel = cube[17, 17]
    <Marvin Spaxel (plateifu=8485-1901, x=17, y=17; x_cen=0, y_cen=0)>

When slicing a `~marvin.tools.cube.Cube`, the xy origin is always the lower left corner of the array, `xyorig="lower"`.  Remember Numpy arrays are in row-major.  You can also use the `~marvin.tools.cube.Cube.getSpaxel` method, which provides addionional keyword options; ``cube[i, j]`` is a shorthand for ``cube.getSpaxel(x=j, y=i, xyorig='lower')``.
::

    # get the central spaxel
    spaxel = cube.getSpaxel(x=17, y=17, xyorig='lower')
    <Marvin Spaxel (plateifu=8485-1901, x=17, y=17; x_cen=0, y_cen=0)>

By default, the xy origin in ``getSpaxel`` is the center of the `~marvin.tools.cube.Cube`, `xyorig="center"`.
::

    spaxel = cube.getSpaxel(x=1, y=1)
    <Marvin Spaxel (plateifu=8485-1901, x=18, y=18; x_cen=1, y_cen=1)>


.. _marvin-cube-access:

Accessing Maps
^^^^^^^^^^^^^^

`~marvin.tools.maps.Maps` are also available from the `~marvin.tools.cube.Cube` object, using the `~marvin.tools.cube.Cube.getMaps` method.  By default, this grabs the `~marvin.tools.maps.Maps` with the default bintype. For more information about `~marvin.tools.maps.Maps` see :ref:`marvin-maps`.
::

    # grab the Marvin Maps object
    cube.getMaps()
    <Marvin Maps (plateifu='8485-1901', mode='local', data_origin='db', bintype='HYB10', template='GAU-MILESHC')>


.. _marvin-cube-save:

Saving and Restoring
^^^^^^^^^^^^^^^^^^^^

You can save a `~marvin.tools.cube.Cube` locally as a Python pickle object, using the `~marvin.tools.core.MarvinToolsClass.save` method.

::

    cube.save('mycube.mpf')

as well as restore a Cube pickle object using the `~marvin.tools.core.MarvinToolsClass.restore` class method

::

    from marvin.tools.cube.Cube import Cube

    cube = Cube.restore('mycube.mpf')


.. _marvin-cube-api:

Reference/API
^^^^^^^^^^^^^

Class Inheritance Diagram
-------------------------

.. inheritance-diagram:: marvin.tools.cube.Cube

Class
-----

.. autosummary:: marvin.tools.cube.Cube

Methods
-------

.. autosummary::

    marvin.tools.cube.Cube.getMaps
    marvin.tools.cube.Cube.getSpaxel
    marvin.tools.cube.Cube.download
    marvin.tools.cube.Cube.save
    marvin.tools.cube.Cube.restore
