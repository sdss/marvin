
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
- Access :ref:`Row-stacked Spectra <marvin-rss>` files.
- Access DAP :ref:`Maps <marvin-maps>` and :ref:`ModelCubes <marvin-modelcube>`.
- Convenient access to all the DRP and DAP properties for a given :ref:`Spaxel or Bin <marvin-subregion-tools>`.
- The data is delivered as :ref:`quantities <marvin-quantities>` with attached variance and mask, as well as associated properties.
- Easy handling of maskbits and labels.
- DAP :ref:`Map arithmetic <marvin-enhanced-map>`.
- Autocompletion of properties and channels (powered by a datamodel).
- Extract all spaxels within a region.
- Access to NSA and DRPall data.
- Easy data download.


.. _marvin-quantities:

Working with quantities
-----------------------

Marvin presents scientific data in the form of `Astropy Quantities <http://docs.astropy.org/en/stable/units/quantity.html#quantity>`__. A Quantity is essentially a number with an associated physical unit. In Marvin we expand on that concept and extend the Quantities with a mask, an inverse variance (`why do we use ivar in MaNGA? <https://www.sdss.org/manga/manga-tutorials/manga-faq/#WhydoyououtputIVAR(inversevariance)insteadoferrors?>`__) and, when relevant, the associated wavelength. Marvin Quantities also provide useful methods to, for instance, calculate the SNR or plot the value. Marvin provides Quantities for 1D (`~marvin.tools.quantities.spectrum.Spectrum`, `~marvin.tools.quantities.analysis_props.AnalysisProperty`), 2D (`~marvin.tools.quantities.map.Map`), and 3D data (`~marvin.tools.quantities.datacube.DataCube`).

All Quantities behave similarly. Let's start by getting a datacube (3D Quantity) from a `~marvin.tools.cube.Cube` object ::

    >>> my_cube = marvin.tools.Cube('7443-12701')
    >>> flux = my_cube.flux
    >>> flux
    <DataCube [[[0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                ...,
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.]],

                ...,

               [[0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                ...,
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.]]] 1e-17 erg / (Angstrom cm2 s spaxel)>
    >>> flux.wavelength
    <Quantity [ 3621.6 ,  3622.43,  3623.26, ..., 10349.  , 10351.4 , 10353.8 ] Angstrom>

A slice of a `~marvin.tools.quantities.datacube.DataCube` is another datacube ::

    >>> flux_section = flux[1000:2000, 15:20, 15:20]
    >>> flux_section
    <DataCube [[[0.0484641 , 0.0455479 , 0.0421016 , 0.0391036 , 0.0412236 ],
                [0.048177  , 0.0437978 , 0.0384898 , 0.0335415 , 0.0345823 ],
                [0.0358995 , 0.0385949 , 0.0338827 , 0.0293836 , 0.0337355 ],
                [0.0177076 , 0.024134  , 0.0270703 , 0.0271202 , 0.0312836 ],
                [0.0052256 , 0.0119592 , 0.0181215 , 0.0243616 , 0.0311569 ]],

                ...,

               [[0.0448547 , 0.0435139 , 0.041652  , 0.0415161 , 0.0468557 ],
                [0.0408965 , 0.0431359 , 0.0441348 , 0.0448875 , 0.0507026 ],
                [0.0375406 , 0.0409193 , 0.0423735 , 0.0434993 , 0.0484709 ],
                [0.0306319 , 0.0335499 , 0.0357318 , 0.0381165 , 0.0422256 ],
                [0.0261617 , 0.0271262 , 0.0294177 , 0.033631  , 0.039794  ]]] 1e-17 erg / (Angstrom cm2 s spaxel)>

Note that in addition to the array the `~marvin.tools.quantities.datacube.DataCube` has associated units (:math:`{\rm 10^{-17}\,erg\,cm^{-2}\,s^{-1}\,spaxel}`). We can get the value, unit, and the scale as ::

    >>> flux_section.value
    array([[[0.0484641 , 0.0455479 , 0.0421016 , 0.0391036 , 0.0412236 ],
            [0.048177  , 0.0437978 , 0.0384898 , 0.0335415 , 0.0345823 ],
            [0.0358995 , 0.0385949 , 0.0338827 , 0.0293836 , 0.0337355 ],
            [0.0177076 , 0.024134  , 0.0270703 , 0.0271202 , 0.0312836 ],
            [0.0052256 , 0.0119592 , 0.0181215 , 0.0243616 , 0.0311569 ]],

           ...,

           [[0.0448547 , 0.0435139 , 0.041652  , 0.0415161 , 0.0468557 ],
            [0.0408965 , 0.0431359 , 0.0441348 , 0.0448875 , 0.0507026 ],
            [0.0375406 , 0.0409193 , 0.0423735 , 0.0434993 , 0.0484709 ],
            [0.0306319 , 0.0335499 , 0.0357318 , 0.0381165 , 0.0422256 ],
            [0.0261617 , 0.0271262 , 0.0294177 , 0.033631  , 0.039794  ]]])
    >>> flux_section.unit
    Unit("1e-17 erg / (Angstrom cm2 s spaxel)")
    >>> flux_section.unit.scale
    1e-17

It's important to pay attention to the scale to convert to physical units. If you prefer to have the scale included in the value you can use the `~marvin.tools.quantities.base_quantity.QuantityMixIn.descale` method ::

    >>> flux.value[1000, 15, 15]
    0.0484641
    >>> descaled = flux.descale()
    >>> descaled.value[1000, 15, 15]
    4.84641e-19

We can also access the associated inverse variance or convert it to error, as well as compute the signal-to-noise ratio ::

    >>> flux.ivar[1000, 15, 15]
    3654.32
    >>> flux.error[1000, 15, 15]
    <Quantity 0.01654233 1e-17 erg / (Angstrom cm2 s spaxel)>
    >>> flux.snr[1000, 15, 15]
    2.9297019457938314

The mask associated with the values is easily accessible via the ``mask`` attribute. We can also use the `~marvin.tools.quantities.base_quantity.QuantityMixIn.masked` method to return a Numpy `masked array <https://docs.scipy.org/doc/numpy/reference/maskedarray.html>`__ in which the values that should not be used have been masked away ::

    >>> flux_section.masked
    masked_array(
    data=[[[--, 0.0455479, 0.0421016, 0.0391036, 0.0412236],
           [0.048177, 0.0437978, 0.0384898, 0.0335415, 0.0345823],
           [0.0358995, 0.0385949, 0.0338827, 0.0293836, 0.0337355],
           [0.0177076, 0.024134, 0.0270703, 0.0271202, 0.0312836],
           [0.0052256, 0.0119592, 0.0181215, 0.0243616, 0.0311569]],

           ...,

           [[--, 0.0435139, 0.041652, 0.0415161, 0.0468557],
            [0.0408965, 0.0431359, 0.0441348, 0.0448875, 0.0507026],
            [0.0375406, 0.0409193, 0.0423735, 0.0434993, 0.0484709],
            [0.0306319, 0.0335499, 0.0357318, 0.0381165, 0.0422256],
            [0.0261617, 0.0271262, 0.0294177, 0.033631, 0.039794]]],
    mask=[[[ True, False, False, False, False],
           [False, False, False, False, False],
           [False, False, False, False, False],
           [False, False, False, False, False],
           [False, False, False, False, False]],

           ...,

           [[ True, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False]]],
    fill_value=1e+20)

Quantities have an associated `~marvin.tools.quantities.base_quantity.QuantityMixIn.pixmask`, which provides a simple way to interact with the mask bits (for more information, go to the :ref:`marvin-maskbit` section) ::

    >>> flux.pixmask
    <Maskbit 'MANGA_DRP3PIXMASK' shape=(4563, 72, 72)>

    >>> flux.pixmask.get_mask('NOCOV')  # Returns a mask of values with the NOCOV maskbit.
    array([[[1, 1, 1, ..., 1, 1, 1],
            [1, 1, 1, ..., 1, 1, 1],
            [1, 1, 1, ..., 1, 1, 1],
            ...,
            [1, 1, 1, ..., 1, 1, 1],
            [1, 1, 1, ..., 1, 1, 1],
            [1, 1, 1, ..., 1, 1, 1]],

            ...,

           [[1, 1, 1, ..., 1, 1, 1],
            [1, 1, 1, ..., 1, 1, 1],
            [1, 1, 1, ..., 1, 1, 1],
            ...,
            [1, 1, 1, ..., 1, 1, 1],
            [1, 1, 1, ..., 1, 1, 1],
            [1, 1, 1, ..., 1, 1, 1]]])

We can also slice a datacube to get a single spectrum ::

    >>> spectrum_20_20 = flux[:, 20, 20]
    >>> spectrum_20_20
    <Spectrum [0.0669153, 0.0599907, 0.0229852, ..., 0.       , 0.       ,
           0.       ] 1e-17 erg / (Angstrom cm2 s spaxel)>

In this case the returned Quantity is a 1D `~marvin.tools.quantities.spectrum.Spectrum`. This new Quantity behaves exactly as the `~marvin.tools.quantities.datacube.DataCube` but in this case we can also `~marvin.tools.quantities.spectrum.Spectrum.plot` the spectrum ::

    >>> spectrum_20_20.plot()
    <matplotlib.axes._subplots.AxesSubplot at 0x130a1d518>

.. plot::
    :align: center
    :include-source: False

    import marvin

    my_cube = marvin.tools.Cube('7443-12703')
    flux = my_cube[20, 20].flux
    flux.plot()

Let's now have a look at the Marvin 2D Quantity: the `~marvin.tools.quantities.map.Map`. ::

    >>> maps_obj = Maps('7443-12703')
    >>> ha = maps_obj.emline_gflux_ha_6564
    <Marvin Map (property='emline_gflux_ha_6564')>
    [[0. 0. 0. ... 0. 0. 0.]
    [0. 0. 0. ... 0. 0. 0.]
    [0. 0. 0. ... 0. 0. 0.]
    ...
    [0. 0. 0. ... 0. 0. 0.]
    [0. 0. 0. ... 0. 0. 0.]
    [0. 0. 0. ... 0. 0. 0.]] 1e-17 erg / (cm2 s spaxel)

We can still use all the tools we discussed above. For example, let's plot the signal-to-noise ratio ::

    >>> snr = ha.snr
    >>> plt.imshow(snr, origin='lower')

.. plot::
    :align: center
    :include-source: False

    import marvin
    import matplotlib.pyplot as plt
    maps_obj = marvin.tools.Maps('7443-12703')
    ha_snr = maps_obj.emline_gflux_ha_6564.snr
    plt.imshow(ha_snr, origin='lower')

Map objects are a bit especial, though, and we will discuss them in detail in :ref:`their own section <marvin-map>`. Here, let's see how we can do "Map arithmetic" by calculating the :math:`{\rm H\alpha/H\beta}` ratio ::

    >>> hb = maps_obj.emline_gew_hb_4862
    >>> ha_hb = ha / hb
    >>> ha_hb
    <Marvin EnhancedMap>
    array([[nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan]], dtype=float32) '1e-17 erg / (Angstrom cm2 s spaxel)'
    >>> ha_hb.plot()

.. plot::
    :align: center
    :include-source: False

    import marvin
    maps_obj = marvin.tools.Maps('7443-12703')
    ha = maps_obj.emline_gflux_ha_6564
    hb = maps_obj.emline_gew_hb_4862
    ha_hb = ha / hb
    ha_hb.plot()

`~marvin.tools.quantities.map.EnhancedMap` result from the arithmetic combination of two maps and take care of all the gritty details: error propagation, division by zero, maskbit propagation, etc.

Finally, `~marvin.tools.quantities.analysis_props.AnalysisProperty` are 1D quantities associated with a value for a single spaxel on a `~marvin.tools.quantities.map.Map`. We will discuss them in depth when we talk about :ref:`marvin-subregion-tools`.

Using the tools
---------------

Data access modes
^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    ../core/data-access-modes

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

Advanced use of Galaxy Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    cube
    rss
    dap_tools
    plate

Plotting
^^^^^^^^

.. toctree::
    :maxdepth: 2

    utils/plotting
    bpt

Image utilities
^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    image


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
   marvin.tools.rss.RSS
   marvin.tools.maps.Maps
   marvin.tools.modelcube.ModelCube

Quantities
^^^^^^^^^^

.. autosummary::

    marvin.tools.quantities.analysis_props.AnalysisProperty
    marvin.tools.quantities.spectrum.Spectrum
    marvin.tools.quantities.map.Map
    marvin.tools.rss.RSSFiber
    marvin.tools.quantities.datacube.DataCube

MixIns
^^^^^^

.. autosummary::

    marvin.tools.mixins.nsa.NSAMixIn
    marvin.tools.mixins.dapall.DAPallMixIn
    marvin.tools.mixins.aperture.GetApertureMixIn
