
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
relevant to that VAC as well as as a separate `~.VACs` tool class.  

.. _marvin-vacs-target:

Target Specific VAC Access
^^^^^^^^^^^^^^^^^^^^^^^^^^

When navigating individual targets in Marvin, e.g. via the Cube or Maps tools, to access any available VAC information 
specific to that targets, use the ``vacs`` attribute attached to most Marvin Tools.   

::

    >>> # enable Marvin for public data access
    >>> from marvin import config
    >>> config.setRelease('DR15')

    >>> # load a MaNGA cube and access any VAC information for 8485-1901
    >>> from marvin.tools import Cube
    >>> cube = Cube('8485-1901')
    >>> cube.vacs
        <VACContainer ('firefly', 'galaxyzoo', 'gema', 'HI', 'visual_morphology')>

    >>> # it is also available from a Maps object
    >>> maps = cube.getMaps()
    >>> m.vacs
        <VACContainer ('firefly', 'galaxyzoo', 'gema', 'HI', 'visual_morphology')>


The ``VACContainer`` will list the names of all available VACS for the current release.  These names are dottable 
attributes on the class.  For example, the MaNGA DR15 release contains the MaNGA Galaxy Zoo VAC, which has been integrated into 
Marvin.  It is available as ``galaxyzoo``.

::

    >>> # access the Galaxy Zoo VAC
    >>> gz = cube.vacs.galaxyzoo
    >>> print(gz)
        FITS_rec([(221394, 'J153010.73+484124.8', 19., 48.69020093, 232.54470389, '1-209232', 6318., 48.69020093, 232.54470389, 'original', 44., 0.93617021, 0.95360705, 44., 0.95631384, 2., 0.04255319, 0.22727273, 2., 0.04346881, 1., 0.0212766, 0., 0.01, 0.00021734, 47., 46.01, 0., 0., 0., 0., 0., 2., 1., 1., 2., 1., 2., 2., 0., 0., 0., 0., 0., 2., 1., 1., 2., 1., 2., 2., 0., 0., 0., 0., 0., 2., 1., 1., 2., 1., 2., 2., 0., 0., 0., 0., 0., 1., 0.5, 0.46666667, 1., 0.5, 0., 0., 0., 0., 0., 1., 0.5, 0.28125, 1., 0.5, 2., 2., 1., 0.02173913, 0.0952381, 1., 0.02173913, 45., 0.97826087, 0.84090909, 45., 0.97826087, 46., 46., 21., 0.47727273, 0.48571429, 21., 0.47727273, 23., 0.52272727, 0.33333333, 23., 0.52272727, 0., 0., 0., 0., 0., 44., 44., 0., nan, 1., 0., nan, 0., nan, 0.11111111, 0., nan, 0., nan, 0.22580645, 0., nan, 0., 0., 0., nan, 0., 0., nan, 0., nan, 1., 0., nan, 0., nan, 0.96525097, 0., nan, 0., 0., 0., nan, 0.4, 0., nan, 0., nan, 0.73809524, 0., nan, 0., nan, 0.12121212, 0., nan, 0., nan, 0.04651163, 0., nan, 0., nan, 0.1, 0., nan, 0., nan, 0., 0., nan, 0., 0.)],
            dtype=(numpy.record, [('nsa_id', '>i8'), ('IAUNAME', 'S19'), ('IFUDESIGNSIZE', '>f8'), ('IFU_DEC', '>f8'), ('IFU_RA', '>f8'), ('MANGAID', 'S8'), ('MANGA_TILEID', '>f8'), ('OBJECT_DEC', '>f8'), ('OBJECT_RA', '>f8'), ('survey', 'S77'), ('t01_smooth_or_features_a01_smooth_count', '>f8'), ('t01_smooth_or_features_a01_smooth_count_fraction', '>f8'), ('t01_smooth_or_features_a01_smooth_debiased', '>f8'), ('t01_smooth_or_features_a01_smooth_weight', '>f8'), ('t01_smooth_or_features_a01_smooth_weight_fraction', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_count', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_count_fraction', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_debiased', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_weight', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_weight_fraction', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_count', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_count_fraction', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_debiased', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_weight', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_weight_fraction', '>f8'), ('t01_smooth_or_features_count', '>f8'), ('t01_smooth_or_features_weight', '>f8'), ('t02_edgeon_a04_yes_count', '>f8'), ('t02_edgeon_a04_yes_count_fraction', '>f8'), ('t02_edgeon_a04_yes_debiased', '>f8'), ('t02_edgeon_a04_yes_weight', '>f8'), ('t02_edgeon_a04_yes_weight_fraction', '>f8'), ('t02_edgeon_a05_no_count', '>f8'), ('t02_edgeon_a05_no_count_fraction', '>f8'), ('t02_edgeon_a05_no_debiased', '>f8'), ('t02_edgeon_a05_no_weight', '>f8'), ('t02_edgeon_a05_no_weight_fraction', '>f8'), ('t02_edgeon_count', '>f8'), ('t02_edgeon_weight', '>f8'), ('t03_bar_a06_bar_count', '>f8'), ('t03_bar_a06_bar_count_fraction', '>f8'), ('t03_bar_a06_bar_debiased', '>f8'), ('t03_bar_a06_bar_weight', '>f8'), ('t03_bar_a06_bar_weight_fraction', '>f8'), ('t03_bar_a07_no_bar_count', '>f8'), ('t03_bar_a07_no_bar_count_fraction', '>f8'), ('t03_bar_a07_no_bar_debiased', '>f8'), ('t03_bar_a07_no_bar_weight', '>f8'), ('t03_bar_a07_no_bar_weight_fraction', '>f8'), ('t03_bar_count', '>f8'), ('t03_bar_weight', '>f8'), ('t04_spiral_a08_spiral_count', '>f8'), ('t04_spiral_a08_spiral_count_fraction', '>f8'), ('t04_spiral_a08_spiral_debiased', '>f8'), ('t04_spiral_a08_spiral_weight', '>f8'), ('t04_spiral_a08_spiral_weight_fraction', '>f8'), ('t04_spiral_a09_no_spiral_count', '>f8'), ('t04_spiral_a09_no_spiral_count_fraction', '>f8'), ('t04_spiral_a09_no_spiral_debiased', '>f8'), ('t04_spiral_a09_no_spiral_weight', '>f8'), ('t04_spiral_a09_no_spiral_weight_fraction', '>f8'), ('t04_spiral_count', '>f8'), ('t04_spiral_weight', '>f8'), ('t05_bulge_prominence_a10_no_bulge_count', '>f8'), ('t05_bulge_prominence_a10_no_bulge_count_fraction', '>f8'), ('t05_bulge_prominence_a10_no_bulge_debiased', '>f8'), ('t05_bulge_prominence_a10_no_bulge_weight', '>f8'), ('t05_bulge_prominence_a10_no_bulge_weight_fraction', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_count', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_count_fraction', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_debiased', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_weight', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_weight_fraction', '>f8'), ('t05_bulge_prominence_a12_obvious_count', '>f8'), ('t05_bulge_prominence_a12_obvious_count_fraction', '>f8'), ('t05_bulge_prominence_a12_obvious_debiased', '>f8'), ('t05_bulge_prominence_a12_obvious_weight', '>f8'), ('t05_bulge_prominence_a12_obvious_weight_fraction', '>f8'), ('t05_bulge_prominence_a13_dominant_count', '>f8'), ('t05_bulge_prominence_a13_dominant_count_fraction', '>f8'), ('t05_bulge_prominence_a13_dominant_debiased', '>f8'), ('t05_bulge_prominence_a13_dominant_weight', '>f8'), ('t05_bulge_prominence_a13_dominant_weight_fraction', '>f8'), ('t05_bulge_prominence_count', '>f8'), ('t05_bulge_prominence_weight', '>f8'), ('t06_odd_a14_yes_count', '>f8'), ('t06_odd_a14_yes_count_fraction', '>f8'), ('t06_odd_a14_yes_debiased', '>f8'), ('t06_odd_a14_yes_weight', '>f8'), ('t06_odd_a14_yes_weight_fraction', '>f8'), ('t06_odd_a15_no_count', '>f8'), ('t06_odd_a15_no_count_fraction', '>f8'), ('t06_odd_a15_no_debiased', '>f8'), ('t06_odd_a15_no_weight', '>f8'), ('t06_odd_a15_no_weight_fraction', '>f8'), ('t06_odd_count', '>f8'), ('t06_odd_weight', '>f8'), ('t07_rounded_a16_completely_round_count', '>f8'), ('t07_rounded_a16_completely_round_count_fraction', '>f8'), ('t07_rounded_a16_completely_round_debiased', '>f8'), ('t07_rounded_a16_completely_round_weight', '>f8'), ('t07_rounded_a16_completely_round_weight_fraction', '>f8'), ('t07_rounded_a17_in_between_count', '>f8'), ('t07_rounded_a17_in_between_count_fraction', '>f8'), ('t07_rounded_a17_in_between_debiased', '>f8'), ('t07_rounded_a17_in_between_weight', '>f8'), ('t07_rounded_a17_in_between_weight_fraction', '>f8'), ('t07_rounded_a18_cigar_shaped_count', '>f8'), ('t07_rounded_a18_cigar_shaped_count_fraction', '>f8'), ('t07_rounded_a18_cigar_shaped_debiased', '>f8'), ('t07_rounded_a18_cigar_shaped_weight', '>f8'), ('t07_rounded_a18_cigar_shaped_weight_fraction', '>f8'), ('t07_rounded_count', '>f8'), ('t07_rounded_weight', '>f8'), ('t09_bulge_shape_a25_rounded_count', '>f8'), ('t09_bulge_shape_a25_rounded_count_fraction', '>f8'), ('t09_bulge_shape_a25_rounded_debiased', '>f8'), ('t09_bulge_shape_a25_rounded_weight', '>f8'), ('t09_bulge_shape_a25_rounded_weight_fraction', '>f8'), ('t09_bulge_shape_a26_boxy_count', '>f8'), ('t09_bulge_shape_a26_boxy_count_fraction', '>f8'), ('t09_bulge_shape_a26_boxy_debiased', '>f8'), ('t09_bulge_shape_a26_boxy_weight', '>f8'), ('t09_bulge_shape_a26_boxy_weight_fraction', '>f8'), ('t09_bulge_shape_a27_no_bulge_count', '>f8'), ('t09_bulge_shape_a27_no_bulge_count_fraction', '>f8'), ('t09_bulge_shape_a27_no_bulge_debiased', '>f8'), ('t09_bulge_shape_a27_no_bulge_weight', '>f8'), ('t09_bulge_shape_a27_no_bulge_weight_fraction', '>f8'), ('t09_bulge_shape_count', '>f8'), ('t09_bulge_shape_weight', '>f8'), ('t10_arms_winding_a28_tight_count', '>f8'), ('t10_arms_winding_a28_tight_count_fraction', '>f8'), ('t10_arms_winding_a28_tight_debiased', '>f8'), ('t10_arms_winding_a28_tight_weight', '>f8'), ('t10_arms_winding_a28_tight_weight_fraction', '>f8'), ('t10_arms_winding_a29_medium_count', '>f8'), ('t10_arms_winding_a29_medium_count_fraction', '>f8'), ('t10_arms_winding_a29_medium_debiased', '>f8'), ('t10_arms_winding_a29_medium_weight', '>f8'), ('t10_arms_winding_a29_medium_weight_fraction', '>f8'), ('t10_arms_winding_a30_loose_count', '>f8'), ('t10_arms_winding_a30_loose_count_fraction', '>f8'), ('t10_arms_winding_a30_loose_debiased', '>f8'), ('t10_arms_winding_a30_loose_weight', '>f8'), ('t10_arms_winding_a30_loose_weight_fraction', '>f8'), ('t10_arms_winding_count', '>f8'), ('t10_arms_winding_weight', '>f8'), ('t11_arms_number_a31_1_count', '>f8'), ('t11_arms_number_a31_1_count_fraction', '>f8'), ('t11_arms_number_a31_1_debiased', '>f8'), ('t11_arms_number_a31_1_weight', '>f8'), ('t11_arms_number_a31_1_weight_fraction', '>f8'), ('t11_arms_number_a32_2_count', '>f8'), ('t11_arms_number_a32_2_count_fraction', '>f8'), ('t11_arms_number_a32_2_debiased', '>f8'), ('t11_arms_number_a32_2_weight', '>f8'), ('t11_arms_number_a32_2_weight_fraction', '>f8'), ('t11_arms_number_a33_3_count', '>f8'), ('t11_arms_number_a33_3_count_fraction', '>f8'), ('t11_arms_number_a33_3_debiased', '>f8'), ('t11_arms_number_a33_3_weight', '>f8'), ('t11_arms_number_a33_3_weight_fraction', '>f8'), ('t11_arms_number_a34_4_count', '>f8'), ('t11_arms_number_a34_4_count_fraction', '>f8'), ('t11_arms_number_a34_4_debiased', '>f8'), ('t11_arms_number_a34_4_weight', '>f8'), ('t11_arms_number_a34_4_weight_fraction', '>f8'), ('t11_arms_number_a36_more_than_4_count', '>f8'), ('t11_arms_number_a36_more_than_4_count_fraction', '>f8'), ('t11_arms_number_a36_more_than_4_debiased', '>f8'), ('t11_arms_number_a36_more_than_4_weight', '>f8'), ('t11_arms_number_a36_more_than_4_weight_fraction', '>f8'), ('t11_arms_number_a37_cant_tell_count', '>f8'), ('t11_arms_number_a37_cant_tell_count_fraction', '>f8'), ('t11_arms_number_a37_cant_tell_debiased', '>f8'), ('t11_arms_number_a37_cant_tell_weight', '>f8'), ('t11_arms_number_a37_cant_tell_weight_fraction', '>f8'), ('t11_arms_number_count', '>f8'), ('t11_arms_number_weight', '>f8')]))

Each VAC is different and thus may return entirely different data structures, depending on how that VAC was integrated 
into Marvin.  It may return e.g. a single number, ``dict``, ``class instance``, or ``FITS record``.  VAC integrations can 
be arbitrarily simple or complex.  Note also that not all VACs will be available for every MaNGA release, and not all targets 
will have VAC information available.  Please see the list of :ref:`marvin-available-vacs` for details on what each VAC returns.   

.. _marvin-vacs-whole:

Full Catalog VAC Access
^^^^^^^^^^^^^^^^^^^^^^^

To access the entirety of each available VAC catalogs, use the `~marvin.tools.vacs.VACs` Tool.  

::

    >>> from marvin.tools.vacs import VACs
    >>> v = VACs()
    >>> print(v)
        <VACs (firefly, galaxyzoo, gema, HI)>

This tool returns access to the underlying catalog data for each VAC with an available summary FITS file.  The ``data``
attribute contains the full HDUList of the VAC.  `~.VACs.info` prints file information.        

::
    >>> # access the complete Galaxy Zoo VAC catalogs
    >>> gz = v.galaxyzoo
    >>> print(gz)
        <GalaxyzooData(description=Returns Galaxy Zoo morphology, n_hdus=2)>

    >>> # access the data
    >>> gz.data
        [<astropy.io.fits.hdu.image.PrimaryHDU object at 0x2c95f10b8>, <astropy.io.fits.hdu.table.BinTableHDU object at 0x2c83a6e10>]

Let's try a different galaxy

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



