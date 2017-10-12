.. _marvin-utils-maskbit:

=====================================================
Maskbit (:mod:`marvin.utils.general.maskbit.Maskbit`)
=====================================================

.. _marvin-utils-maskbit-intro:

Introduction
------------
:mod:`~marvin.utils.general.maskbit.Maskbit` contains the schema for a MaNGA flag (e.g., ``DAPPIXMASK``) and provides convenience functions for translating amongst mask values, bits, and labels.  :mod:`~marvin.utils.general.maskbit.Maskbit` can be initialized with the name of a MaNGA flag but is more often encountered as an attribute on another :ref:`marvin-tools` object (e.g., :attr:`~marvin.tools.map.Map.pixmask` is an instance of :mod:`~marvin.utils.general.maskbit.Maskbit`).


.. _marvin-utils-maskbit-getting-started:

Getting Started
---------------

:mod:`~marvin.utils.general.maskbit.Maskbit` makes properly applying masks easy by providing the schema for a flag:

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')
    ha = maps['gflux ha']
    
    ha.manga_target1.description
    # 'Targeting bits for all galaxy targets.'
    
    ha.manga_target1.schema
    #     bit                  label                     description
    # 0     0                   NONE
    # 1     1       PRIMARY_PLUS_COM        March 2014 commissioning
    # 2     2          SECONDARY_COM        March 2014 commissioning
    # 3     3     COLOR_ENHANCED_COM        March 2014 commissioning
    # 4     4         PRIMARY_v1_1_0   First tag, August 2014 plates
    # 5     5       SECONDARY_v1_1_0   First tag, August 2014 plates
    # 6     6  COLOR_ENHANCED_v1_1_0   First tag, August 2014 plates
    # 7     7           PRIMARY_COM2         July 2014 commissioning
    # 8     8         SECONDARY_COM2         July 2014 commissioning
    # 9     9    COLOR_ENHANCED_COM2         July 2014 commissioning
    # 10   10         PRIMARY_v1_2_0
    # 11   11       SECONDARY_v1_2_0
    # 12   12  COLOR_ENHANCED_v1_2_0
    # 13   13                 FILLER                  Filler targets
    # 14   14                RETIRED            Bit retired from use


It also contains the mask value, the corresponding bits, and the corresponding labels for the :ref:`marvin-tools` object:

.. code-block:: python

    ha.manga_target1.mask    # 2336
    ha.manga_target1.bits    # [5, 8, 11]
    ha.manga_target1.labels  # ['SECONDARY_v1_1_0', 'SECONDARY_COM2', 'SECONDARY_v1_2_0']


Let's look at a flag with a mask that is an array and not just a single integer:

.. code-block:: python

    ha.pixmask
    # <Maskbit 'MANGA_DAPPIXMASK'
    # 
    #     bit         label                                        description
    # 0     0         NOCOV                         No coverage in this spaxel
    # 1     1        LOWCOV                        Low coverage in this spaxel
    # 2     2     DEADFIBER                   Major contributing fiber is dead
    # 3     3      FORESTAR                                    Foreground star
    # 4     4       NOVALUE  Spaxel was not fit because it did not meet sel...
    # 5     5    UNRELIABLE  Value is deemed unreliable; see TRM for defini...
    # 6     6     MATHERROR              Mathematical error in computing value
    # 7     7     FITFAILED                  Attempted fit for property failed
    # 8     8     NEARBOUND  Fitted value is too near an imposed boundary; ...
    # 9     9  NOCORRECTION               Appropriate correction not available
    # 10   10     MULTICOMP          Multi-component velocity features present
    # 11   30      DONOTUSE                 Do not use this spaxel for science>
    
    ha.pixmask.mask  # == ha.mask
    # array([[1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843,
    #     1073741843],
    #    ...,
    #    [1073741843, 1073741843, 1073741843, ..., 1073741843, 1073741843,
    #     1073741843]])

    ha.pixmask.bits
    # [[[0, 1, 4, 30],
    #   ...,
    # [0, 1, 4, 30]]]
    
    ha.pixmask.labels
    # [[['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE'],
    #   ...,
    # ['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE']]]

    ha.pixmask.mask[17, 32]    # 1073741843
    ha.pixmask.bits[17][32]    # [0, 1, 4, 30]
    ha.pixmask.labels[17][32]  # ['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE']


With ``MANGA_DAPPIXMASK``, you might want to translate individual mask values, bits, or labels:

.. code-block:: python

    ha.pixmask.values_to_bits(1073741843)  # [0, 1, 4, 30]
    ha.pixmask.values_to_labels(1073741843)  #['NOCOV', 'LOWCOV', 'NOVALUE', 'DONOTUSE']
    
    # Translate one label
    ha.pixmask.labels_to_value('NOCOV')  # 1
    ha.pixmask.labels_to_bits('NOCOV')   # [0]
    
    # Translate multiple labels
    ha.pixmask.labels_to_value(['NOCOV', 'UNRELIABLE'])  # 33
    ha.pixmask.labels_to_bits(['NOCOV', 'UNRELIABLE'])  # [0, 5]
    

You might want to produce a mask array (e.g., to produce a custom mask for plotting):

.. code-block:: python

    # Mask of regions with no IFU coverage 
    nocov = ha.pixmask.get_mask('NOCOV')
    
    # Mask of regions with low Halpha flux and marked as DONOTUSE
    low_ha = (ha.value < 1e-17) * ha.pixmask.labels_to_value('DONOTUSE')

    # Combine masks using bitwise OR (`|`)
    my_mask = nocov | low_ha

    # import marvin.utils.plot.map as mapplot
    # fig, ax = mapplot.plot(dapmap=ha, mask=my_mask)  # TODO BROKEN


.. .. image:: ../_static/custom_mask.png




.. INITIALIZING A MASKBIT

It's also possible to initialize a :mod:`~marvin.utils.general.maskbit.Maskbit` instance without a :ref:`marvin-tools` object:

.. code-block:: python

    from marvin.utils.general.maskbit import Maskbit
    mngtarg1 = Maskbit('MANGA_TARGET1')

    mngtarg1.schema
    #     bit                  label                     description
    # 0     0                   NONE
    # 1     1       PRIMARY_PLUS_COM        March 2014 commissioning
    # 2     2          SECONDARY_COM        March 2014 commissioning
    # 3     3     COLOR_ENHANCED_COM        March 2014 commissioning
    # 4     4         PRIMARY_v1_1_0   First tag, August 2014 plates
    # 5     5       SECONDARY_v1_1_0   First tag, August 2014 plates
    # 6     6  COLOR_ENHANCED_v1_1_0   First tag, August 2014 plates
    # 7     7           PRIMARY_COM2         July 2014 commissioning
    # 8     8         SECONDARY_COM2         July 2014 commissioning
    # 9     9    COLOR_ENHANCED_COM2         July 2014 commissioning
    # 10   10         PRIMARY_v1_2_0
    # 11   11       SECONDARY_v1_2_0
    # 12   12  COLOR_ENHANCED_v1_2_0
    # 13   13                 FILLER                  Filler targets
    # 14   14                RETIRED            Bit retired from use>


.. _marvin-utils-maskbit-using:

Using :mod:`~marvin.utils.plot.map`
-----------------------------------

For more in-depth discussion of using :mod:`~marvin.utils.plot.map`, please see the following sections:

Plotting Tutorial
`````````````````

* :doc:`../tutorials/plotting`
  
  * :ref:`marvin-plotting-quick-map`
  * :ref:`marvin-plotting-multipanel-single`
  * :ref:`marvin-plotting-multipanel-multiple`
  * :ref:`marvin-plotting-custom-map-axes`
  * :ref:`Plot Halpha Map of Star-forming Spaxels <marvin-plotting-map-starforming>`
  * :ref:`Plot [NII]/Halpha Flux Ratio Map of Star-forming Spaxels <marvin-plotting-niiha-map-starforming>`


.. _marvin-utils-maskbit-default-params:

Default Plotting Parameters
```````````````````````````

====================  ====================  =========  ===============  ==================  ===========
MPL-5
-------------------------------------------------------------------------------------------------------
Property Type         Bad Data Bitmasks     Colormap   Percentile Clip  Symmetric Colorbar  Minimum SNR
====================  ====================  =========  ===============  ==================  ===========
default               UNRELIABLE, DONOTUSE  linearlab  5, 95            False               1
velocities            UNRELIABLE, DONOTUSE  RdBu_r     10, 90           True                0\ :sup:`a`
velocity dispersions  UNRELIABLE, DONOTUSE  inferno    10, 90           False               1
====================  ====================  =========  ===============  ==================  ===========

:sup:`a` Velocities do not have a minimum SNR. This allows spaxels near the zero-velocity contour to be displayed, but users are cautioned that some spaxels could have arbitrarily low SNRs.

**Note**: MPL-4 uses the same default plotting parameters as MPL-5, except the Bad Data Bitmasks, which use bit 1 (rough DONOTUSE) for all properties.


Masking
```````

Spaxels Not Covered by the IFU
::::::::::::::::::::::::::::::

:meth:`~marvin.utils.plot.map.no_coverage_mask` creates a mask of a map where there is no coverage by the IFU.

.. code-block:: python

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    nocov = mapplot.no_coverage_mask(mask=ha.mask, bit=0, ivar=ha.ivar)


**Important**: In 2.1.3, the call signature is ``no_coverage_mask(value, ivar, mask, bit)``. In version 2.1.4, this changes to ``no_coverage_mask(mask, bit, ivar=None)``.


Spaxels Flagged as Bad Data
:::::::::::::::::::::::::::

:meth:`~marvin.utils.plot.map.bad_data_mask` creates a mask of a map where the data has been flagged by the DAP as UNRELIABLE or DONOTUSE.

.. code-block:: python

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    bad_data = mapplot.bad_data_mask(mask=ha.mask, bits={'doNotUse': 30, 'unreliable': 5})


**Important**: In 2.1.3, the call signature is ``bad_data_mask(mask, bits)``. In version 2.1.4, this changes to ``bad_data_mask(mask, bits)``.


Spaxels with Low Signal-to-Noise
::::::::::::::::::::::::::::::::

:meth:`~marvin.utils.plot.map.low_snr_mask` creates a mask of a map where the data is below a minimum signal-to-noise ratio.

.. code-block:: python

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    low_snr = mapplot.low_snr_mask(value=ha.value, ivar=ha.ivar, snr_min=1)


Spaxels with Negative Values
::::::::::::::::::::::::::::

:meth:`~marvin.utils.plot.map.log_colorbar_mask` creates a mask of a map where the values are negative.  This is necessary to avoid erros when using a logarithmic colorbar.

.. code-block:: python

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    log_cb_mask = mapplot.log_colorbar_mask(value=ha.value, log_cb=True)


Combine Various Undesirable Masks
:::::::::::::::::::::::::::::::::

:meth:`~marvin.utils.plot.map.select_good_spaxels` creates a `NumPy masked array <https://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_ that combines masks of undesirable spaxels (no IFU coverage, bad data, low signal-to-noise ratio, and negative values [if using a logarithmic colorbar]).

.. code-block:: python

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    good_spax = mapplot.select_good_spaxels(value=ha.value, nocov=nocov, bad_data=bad_data, low_snr=low_snr, log_cb_mask=log_cb_mask)


Set the Plotting Extent for `imshow <https://matplotlib.org/devdocs/api/_as_gen/matplotlib.axes.Axes.imshow.html>`_
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:meth:`~marvin.utils.plot.map.set_extent` returns the coordinates of the lower-left and upper-right corners of the map in cube coordinates (lower-left = (0, 0) and in units of spaxels) or sky coordinates (center = (0, 0) and in units of arcsec).

.. code-block:: python

    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    extent = mapplot.set_extent(cube_size=ha.value.shape, sky_coords=False)


Set Hatch Style
:::::::::::::::

:meth:`~marvin.utils.plot.map.set_patch_style` sets the style for the hatched region(s) that correspond to spaxels that are covered by the IFU but do not have usable data. :meth:`~marvin.utils.plot.map.plot` creates a single large hatched rectangle patch as the lowest layer and then places the gray background (no IFU coverage) and colored spaxels (good data) as higher layers.

.. code-block:: python

    import marvin.utils.plot.map as mapplot
    patch_kws = mapplot.set_patch_style(extent=extent, facecolor='#A8A8A8')


Axis Setup
::::::::::

:meth:`~marvin.utils.plot.map.ax_setup` sets the x- and y-labels and facecolor.


Set Title
:::::::::

:meth:`~marvin.utils.plot.map.set_title` sets the title of the axis object. You can directly specify the title or construct it from the property name (and channel name).

.. code-block:: python

    import marvin.utils.plot.map as mapplot
    title = mapplot.set_title(title=None, property_name=ha.property_name, channel=ha.channel)

Reference/API
-------------

.. rubric:: Module

.. autosummary:: marvin.utils.plot.map

.. rubric:: Functions

.. autosummary::

    marvin.utils.plot.map.ax_setup
    marvin.utils.plot.map.bad_data_mask
    marvin.utils.plot.map.log_colorbar_mask
    marvin.utils.plot.map.low_snr_mask
    marvin.utils.plot.map.no_coverage_mask
    marvin.utils.plot.map.plot
    marvin.utils.plot.map.select_good_spaxels
    marvin.utils.plot.map.set_extent
    marvin.utils.plot.map.set_patch_style
    marvin.utils.plot.map.set_title
