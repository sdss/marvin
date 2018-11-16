.. py:currentmodule:: marvin.tools.rss

.. _marvin-rss:

Row-Stacked Spectra (RSS)
=========================

Introduction
------------

`Row-stacked spectra <https://www.sdss.org/dr15/manga/manga-data/data-model/#RSSFiles>`__ files (RSS) compile the fully-reduced spectra from all the fibres and exposures for a given galaxy or target. They are especially useful for projects that require access to the pre-cube data, such as stacking. The data is organised as a 2D array in which different rows correspond to different spectra, with the columns being the wavelength direction.

The RSS class
-------------

The `RSS` class provides access to row-stacked spectra data, either from a file, DB, or remotely via the Marvin API. While most of its functionality is shared with the other Tools that subclass from `~marvin.tools.core.MarvinToolsClass` (see the :ref:`introduction <gal-tools-getting-started>` to the Galaxy Tools), `RSS` has a number of specific features that we will discuss here.

As always, we can instantiate an `RSS` object using a plate-ifu or mangaid, or from a file. The :ref:`multi-modal <marvin-dma>` access system will retrieve the necessary data locally or remotely ::

    >>> rss = marvin.tools.RSS('8485-1901')
    >>> rss
    <Marvin RSS (mangaid='1-209232', plateifu='8485-1901', mode='local', data_origin='file')>

As usual, we can access attributes such as the ``header`` or the ``datamodel`` ::

    >>> rss.header
    XTENSION= 'IMAGE   '           / IMAGE extension
    BITPIX  =                  -32 / Number of bits per data pixel
    NAXIS   =                    2 / Number of data axes
    NAXIS1  =                 4563 /
    NAXIS2  =                  171 /
    PCOUNT  =                    0 / No Group Parameters
    GCOUNT  =                    1 / One Data Group
    AUTHOR  = 'Brian Cherinka & David Law <bcherin1@jhu.edu, dlaw@stsci.edu>' /
    VERSDRP2= 'v2_4_3  '           / MaNGA DRP version (2d processing)
    VERSDRP3= 'v2_4_3  '           / MaNGA DRP Version (3d processing)
    VERSPLDS= 'v2_52   '           / Platedesign Version
    VERSFLAT= 'v1_31   '           / Specflat Version
    VERSCORE= 'v1_6_2  '           / MaNGAcore Version
    VERSPRIM= 'v2_5    '           / MaNGA Preimaging Version
    VERSUTIL= 'v5_5_32 '           / Version of idlutils
    VERSIDL = 'x86_64 linux unix linux 7.1.1 Aug 21 2009 64 64 ' / Version of IDL
    BSCALE  =              1.00000 / Intensity unit scaling
    BZERO   =              0.00000 / Intensity zeropoint
    BUNIT   = '1E-17 erg/s/cm^2/Ang/fiber' / Specific intensity (per fiber-area)
    MASKNAME= 'MANGA_DRP2PIXMASK'  / Bits in sdssMaskbits.par used by mask extension
    TELESCOP= 'SDSS 2.5-M'         / Sloan Digital Sky Survey
    INSTRUME= 'MaNGA   '           / SDSS-IV MaNGA IFU
    SRVYMODE= 'MaNGA dither'       / Survey leading this observation and its mode
    PLATETYP= 'APOGEE-2&MaNGA'     / Type of plate (e.g. MANGA, APOGEE-2&MANGA)
    OBJSYS  = 'ICRS    '           / The TCC objSys
    EQUINOX =              2000.00 /
    RADESYS = 'FK5     '           /
    LAMPLIST= 'lamphgcdne.dat'     /
    TPLDATA = 'BOSZ_3000-11000A.fits' /

    >>> rss.datamodel
    <DRPRSSDataModel release='DR15', n_rss=3, n_spectra=2>
    >>> rss.datamodel.rss
    [<RSS 'flux', release='DR15', unit='1e-17 erg / (Angstrom cm2 fiber s)'>,
     <RSS 'dispersion', release='DR15', unit='Angstrom'>,
     <RSS 'dispersion_prepixel', release='DR15', unit='Angstrom'>]

We can use `RSS.getCube` to retrieve the corresponding `~marvin.tools.cube.Cube` ::

    >>> cube = rss.getCube()
    >>> cube
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='file')>


The `RSS.obsinfo` table
^^^^^^^^^^^^^^^^^^^^^^^

`RSS.obsinfo` provides access to an Astropy `~astropy.table.Table` with the observing information for this galaxy ::

    >>> rss.obsinfo
    <Table length=9>
            SLITFILE              METFILE      HARNAME ... PF_FWHM_R  PF_FWHM_I PF_FWHM_Z
            str25                 str17         str5  ...  float32    float32   float32
    ------------------------- ----------------- ------- ... ---------- --------- ----------
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ...  1.1196343 1.0926069  1.0622483
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ...  1.0522692 1.0284542    1.00053
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ...  1.0496484 1.0258191  0.9979574
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ...  1.0698904 1.0452466  1.0166885
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ... 0.98610526 0.9662201 0.94095564
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ...  0.9154704 0.8994158  0.8768676
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ... 0.96761906 0.9485599 0.92396384
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ...  1.1718149 1.1423621  1.1101378
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ...  1.1463778 1.1175543  1.0860871

In this case the file includes the spectra from nine observations. Since this is a 19-fibre IFU that means this RSS contains :math:`19 \times 9=171` flux spectra (and associated extensions). The full datamodel, with descriptions of the contents of each column can be found `here <https://data.sdss.org/datamodel/files/MANGA_SPECTRO_REDUX/DRPVER/PLATE4/stack/manga-RSS.html#hdu11>`__.


Accessing individual fibres
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to being a subclass of `~marvin.tools.core.MarvinToolsClass`, `RSS` is also a *list* of `RSSFiber` instances. Each `RSSFiber` contains the data and metadata associated with a single observation and fibre ::

    >>> isinstance(rss, list)
    True
    >>> len(rss)
    171

    >>> rssfiber = rss[0]
    >>> rssfiber
    <RSSFiber [ 0.64692211, -1.50171757,  0.54236597, ...,  0.        ,
                0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>

`RSSFiber` instances behave as `~marvin.tools.quantities.spectrum.Spectrum` quantities ::

    >>> rssfiber.snr
    array([0.22988   , 0.55315766, 0.18455871, ..., 0.        , 0.        , 0.        ])
    >>> rssfiber.unit
    Unit("1e-17 erg / (Angstrom cm2 fiber s)")
    >>> rssfiber.pixmask
    <Maskbit 'MANGA_DRP2PIXMASK' shape=(4563,)>

and they also contain the ``obsinfo`` data of the exposure associated with this fibre ::

    >>> rssfiber.obsinfo
    <Table length=1>
            SLITFILE              METFILE      HARNAME ... PF_FWHM_R PF_FWHM_I PF_FWHM_Z
            str25                 str17         str5  ...  float32   float32   float32
    ------------------------- ----------------- ------- ... --------- --------- ---------
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ... 1.1196343 1.0926069 1.0622483

From the `RSSFiber` we can access data associated with the fibre, for instance the ``dispersion`` or ``spectral_resolution`` ::

    >>> rssfiber.dispersion
    <Spectrum [1.0794843, 1.0798984, 1.0803117, ..., 2.187566 , 2.187566 , 2.187566 ] Angstrom>

Frequently we want to select all the fibres that were part of an exposure or a set. For that purpose we can use the `RSSFiber.select_fibers` method ::

    >>> rss.select_fibers(exposure_no=198571)
    [<RSSFiber [2.4875052 , 3.32200694, 2.87790442, ..., 0.        , 0.        ,
                0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    <RSSFiber [0.53248107, 1.67843473, 5.14122868, ..., 0.        , 0.        ,
                0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    <RSSFiber [-4.6951623 , -4.88117075, -4.4301815 , ...,  0.        ,
                0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    <RSSFiber [0.73094839, 0.5693031 , 1.83849978, ..., 0.        , 0.        ,
                0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    <RSSFiber [-2.55996156, -2.84130025,  3.60015202, ...,  0.        ,
                0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    ...
    <RSSFiber [ 0.9367817 , -3.32021999, -3.43391848, ...,  0.        ,
                0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    <RSSFiber [-3.31121993,  4.09930992,  1.47489429, ...,  0.        ,
                0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    <RSSFiber [-3.27046323, -2.24382639, -1.72951198, ...,  0.        ,
                0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    <RSSFiber [-1.70159054,  0.95310146, -1.33062816, ...,  0.        ,
                0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    <RSSFiber [ 1.59522855,  1.32217634, -1.26507163, ...,  0.        ,
                0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>,
    <RSSFiber [-4.0862627 , -3.05495214, -1.46317339, ...,  0.        ,
                0.        ,  0.        ] 1e-17 erg / (Angstrom cm2 fiber s)>]


Lazy loading and ``autoload``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the multiple `RSSFiber` instance that are part of an `RSS` object are *lazily* loaded. That means that while the object exists (you can, for example, do ``len(rss)`` and get the correct number of `RSSFiber` instances), the data inside each `RSSFiber` is only loaded when the object is accessed. This enables quick initialisation of the `RSS` objects at the expense of a certain overhead every time a fibre is accessed. Sometimes you may want to load all the fibres at once and then access them quickly you can do that by calling the `RSS.load_all` method or by instantiating the `RSS` object with ``autoload=True`` ::

    >>> rss = RSS('8485-1901', autoload=True)

Similarly, you can disable the autoload of fibres by setting ``rss.autoload = False``. In this case you can still access some information such as the ``obsinfo`` row ::

    >>> rss.autoload = False
    >>> unloaded_rss_fiber = rss[16]
    >>> unloaded_rss_fiber.value
    array([0., 0., 0., ..., 0., 0., 0.])  # All zeros. Not initialised.
    >>> unloaded_rss_fiber.obsinfo
    <Table length=1>
            SLITFILE              METFILE      HARNAME ... PF_FWHM_R PF_FWHM_I PF_FWHM_Z
            str25                 str17         str5  ...  float32   float32   float32
    ------------------------- ----------------- ------- ... --------- --------- ---------
    slitmap-8485-57132-01.par ma060-56887-1.par   ma060 ... 1.1196343 1.0926069 1.0622483


Reference/API
-------------

Class Inheritance Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: RSS
.. inheritance-diagram:: RSSFiber

Class
^^^^^

.. autosummary:: RSS

Methods
^^^^^^^

.. autosummary::

    RSS.load_all
    RSS.select_fibers
    RSSFiber.load
