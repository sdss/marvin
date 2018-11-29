.. _marvin-sample-selection-tutorial:

=========================
Sample Selection Tutorial
=========================


Select Main Sample Galaxies
---------------------------

To select the Main Sample galaxies (Primary + Secondary + Color Enhanced samples), we can use the MaNGA DRPall summary file.  You can use the ``get_drpall_table`` utility to load the full DRPall table.  By default this will load the DRPall file for the current release set in Marvin.

Let's open the DRPall file:

.. code-block:: python

    from marvin.utils.general import get_drpall_table
    data = get_drpall_table()


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
