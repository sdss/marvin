
.. _marvin-vacs:

Value-Added Catalogs (VACS)
---------------------------

While the core of SDSS data releases centers around its base projects' science deliverables, smaller teams frequently 
contribute added value to its core deliverables with additional science products.  These value-added data products or 
catalogs (VACS) are derived data products based on the core deliverables that are vetted, hosted, and released by SDSS 
in order to maximize the impact of SDSS data sets.

Marvin provides access to any VACs that have been contributed into Marvin's framework.  To contribute a MaNGA VAC to 
Marvin, see :ref:`how to contribute <marvin-contributing>`.  Marvin collects all contributed VACs for a given release and 
places them inside a ``VACContainer`` object.  This container is available as a ``vacs`` attribute on most Marvin Tools 
relevant to that VAC.

::

    # load a MaNGA cube and access any VAC information for 8485-1901
    from marvin.tools import Cube
    cube = Cube('8485-1901')
    cube.vacs  #  prints <VACContainer ('mangahi')>

    # it is also available from a Maps object
    maps = cube.getMaps()
    m.vacs  #  prints <VACContainer ('mangahi')>

The ``VACContainer`` will list the names of all available VACS for the current release.  These names are dottable 
attributes on the class.  For example, the MaNGA DR15 release contains the MaNGA-HI VAC, which has been integrated into 
Marvin.  It is available as ``mangahi``.

::

    # access the MaNGA-HI VAC
    hi = cube.vacs.mangahi
    print(hi)  #  prints HI(8485-1901)
    print(hi.data)  #  prints 'No HI data exists for 8485-1901'

Each VAC is different and thus may return entirely different data structures, depending on how that VAC was integrated 
into Marvin.  It may return e.g. a single number, ``dict``, ``class instance``, or ``FITS record``.  VAC integrations can 
be arbitrarily simple or complex.  Note also that not all VACs will be available for every MaNGA release, and not all targets 
will have VAC information available, as indicated in the above example.  Let's try a different galaxy

::


    cube = Cube('7443-12701')
    hi = cube.vacs.mangahi
    print(hi)  ## prints HI(7443-12701)
    print(hi.data)

    #  prints
    #  FITS_rec([('7443-12701', '12-98126', 230.5074624, 43.53234133, 6139, '16A-14', 767.4, 1.76, 8.82, -999., -999., -999., -999., -999, -999., -999, -999, -999, -999, -999, -999., -999., -999., -999., -999., -999.)],
    #  dtype=(numpy.record, [('plateifu', 'S10'), ('mangaid', 'S9'), ('objra', '>f8'), ('objdec', '>f8'), ('vopt', '>i2'), ('session', 'S12'), ('Exp', '>f4'), ('rms', '>f4'), ('logHIlim200kms', '>f4'), ('peak', '>f4'), ('snr', '>f4'), ('FHI', '>f4'), ('logMHI', '>f4'), ('VHI', '>i2'), ('eV', '>f4'), ('WM50', '>i2'), ('WP50', '>i2'), ('WP20', '>i2'), ('W2P50', '>i2'), ('WF50', '>i2'), ('Pr', '>f4'), ('Pl', '>f4'), ('ar', '>f4'), ('br', '>f4'), ('al', '>f4'), ('bl', '>f4')]))

This galaxy has HI data.  This VAC has also provided two convenience methods for quickly interacting with HI data, 
``plot_spectrum``, and ``plot_massfraction``.

.. plot::
    :align: center
    :include-source: True

    # plot the HI spectrum for 7443-12701
    hi.plot_spectrum()



