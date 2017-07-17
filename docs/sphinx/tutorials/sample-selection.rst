.. _marvin-sample-selection-tutorial:

=========================
Sample Selection Tutorial
=========================


Select Main Sample Galaxies
---------------------------

To select the Main Sample galaxies (Primary + Secondary + Color Enhanced samples), we need to download the `MPL-5 DRPall file <https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/drpall-v2_0_1.fits>`_ and put it in the expected directory ``$SAS_BASE_DIR/mangawork/manga/spectro/redux/v2_0_1/``. For more information on the location of your ``$SAS_BASE_DIR`` environment variable, see the :ref:`Getting Started <marvin-getting-started-sas-base-dir>` page.

Let's open the DRPall file:

.. code-block:: python

    import os
    from astropy.io import fits
    from marvin import config
    
    config.setRelease('MPL-5')
    drpver, __ = config.lookUpVersions()
    
    drpall_path = os.path.join(os.environ['SAS_BASE_DIR'], 'mangawork', 'manga', 'spectro', 'redux',
                               drpver, 'drpall-{}.fits'.format(drpver))
    drpall = fits.open(drpall_path)
    data = drpall[1].data


The Main Sample consists of the Primary, Secondary, and Color-Enhanced Samples, which correspond to `MNGTARG1 <http://www.sdss.org/dr13/algorithms/bitmasks/#MANGA_TARGET1>`_ bits 10, 11, and 12, respectively.

.. code-block:: python
    
    import numpy as np
    primary        = data['mngtarg1'] & 2**10                                 # 1278 galaxies
    secondary      = data['mngtarg1'] & 2**11                                 #  947 galaxies
    color_enhanced = data['mngtarg1'] & 2**12                                 #  447 galaxies
    
    main_sample = np.logical_or.reduce((primary, secondary, color_enhanced))  # 2672 galaxies
    
    plateifus = data['plateifu'][main_sample]


Now we can use the :func:`~marvin.utils.general.general.downloadList` function to download all of the files of type ``map`` (other valid ``dltypes``: ``map``, ``image``, ``rss``, ``mastar``, ``default``, or ``plate``).

.. code-block:: python

    from marvin.utils.general import downloadList
    downloadList(plateifus, dltype='map')


|