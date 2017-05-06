.. _marvin-lean-tutorial:

.. ipython:: python
   :suppress:

   import matplotlib
   matplotlib.style.use('seaborn-darkgrid')
   from marvin import config
   config.mode = 'remote'


A Lean Marvin Tutorial
======================

This tutorial runs through all of the steps for doing a project with Marvin from start-to-finish with no extra fat.

**Project**: Calculate the [NII]/Halpha ratio for star-forming spaxels in galaxies with stellar mass between :math:`10^{10}` and :math:`10^{11}`.


Sample Selection
----------------

Marvin uses a simplified query syntax (in both `Web <https://sas.sdss.org/marvin2/search/>`_ and local queries) that understands the MaNGA database schema, so you don't have to write complicated SQL queries.

**Goal**: Find galaxies with stellar mass between :math:`10^{10}` and :math:`10^{11}`.

Create the query and run it (limit to only 3 results for demo purposes):

.. ipython:: python

    from marvin.tools.query import doQuery
    q, r = doQuery(searchfilter='nsa.sersic_logmass >= 10 and nsa.sersic_logmass <= 11', limit=3)

**Tip** see :ref:`Example Queries <marvin-query-examples>` and :ref:`Marvin Query Syntax Tutorial <marvin-sqlboolean>` for help with designing search filters.

View the :ref:`marvin-results`:

.. ipython:: python

    df = r.toDF()
    df

Convert to :ref:`marvin-tools-maps` objects:

.. ipython:: python

    r.convertToTool('maps')
    r.objects
    galaxies = r.objects


Get Maps
--------

Alternatively, maybe we already knew our galaxy IDs, which we can use to create :ref:`marvin-tools-maps` objects:

.. ipython:: python

    from marvin.tools.maps import Maps
    mangaids = ['1-245458', '1-22301', '1-605884']
    galaxies = [Maps(mangaid=mangaid) for mangaid in mangaids]


Get the Halpha maps:

.. ipython:: python

    haflux_maps = [galaxy['emline_gflux_ha_6564'] for galaxy in galaxies]


Plot Halpha map of the second galaxy:

.. ipython:: python

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')
    haflux_map = haflux_maps[1]
    fig, ax = haflux_map.plot()

.. image:: ../_static/haflux_7992-6101.png


The dark blue region near the center of the galaxy looks suspicious, so let's take a look at the model fits of those spaxels.

The easiest way is to navigate to the `Galaxy page for 7992-6101 <https://sas.sdss.org/marvin2/galaxy/7992-6101>`_ and click on the red "Map/SpecView Off" button.

However, we can also plot the spectrum and model fits in Python. First, we can find the coordinates of a spaxel by moving our cursor around the interactive matplotlib plotting window. When the cursor is over the spaxel of interest, the coordinates will appear in the lower right.


Get Spectrum and Model Fit
--------------------------

Then we can create a :ref:`marvin-tools-spaxel` object from the :ref:`marvin-tools-map` object and retrieve the model fit.

.. ipython:: python

    spax = haflux_map.maps.getSpaxel(x=28, y=24, xyorig='lower', modelcube=True)


Now let's plot the spectrum and model fit:

.. ipython:: python

    fig, ax = plt.subplots()
    pObs = ax.plot(spax.spectrum.wavelength, spax.spectrum.flux)
    pModel = ax.plot(spax.spectrum.wavelength, spax.model.flux)
    ax.axis([7100, 7500, 0.3, 0.65]);
    plt.legend(pObs + pModel, ['observed', 'model']);
    ax.set_xlabel('observed wavelength [{}]'.format(spax.spectrum.wavelength_unit));
    ax.set_ylabel('flux [{}]'.format(spax.spectrum.units));

.. image:: ../_static/spec_7992-6101.png

Clearly something went horribly horribly wrong in the fit. In fact, the DAP did not even try to fit a emission line component to the Halpha and [NII] lines. This is unfortunate, but let's press on.



Plot BPT Diagram
----------------

The :meth:`~marvin.tools.maps.Maps.get_bpt` returns masks for spaxels of different ionization types and the Figure object.

.. ipython:: python

    masks, fig = haflux_map.maps.get_bpt()

.. image:: ../_static/bpt_7992-6101.png

For a detailed description see :ref:`marvin-bpt`.


Select Star-forming Spaxels
---------------------------

Select the star-forming spaxels that are in the star-forming region of each diagnostic diagram (hence the "global" keyword):

.. ipython:: python

    sf = masks['sf']['global']


Create the image to display and the background using the star-forming mask:

.. ipython:: python

    image = np.ma.array(haflux_map.value, mask=~sf)
    mask_nodata = np.ma.array(np.ones(haflux_map.value.shape), mask=sf)

If we wanted to do additional calculations instead of creating a plot, this masked array would also be the object on which we would perform operations.


Plot Star-forming Spaxels
-------------------------

Let's set the background to gray:

.. ipython:: python

    from marvin.utils.plot import colorbar
    A8A8A8 = colorbar.one_color_cmap(color='#A8A8A8')


Plot the star-forming spaxels:

.. ipython:: python

    fig, ax = plt.subplots()
    ax.imshow(mask_nodata, cmap=A8A8A8, origin='lower', zorder=1);
    p = ax.imshow(image, cmap='viridis', origin='lower', zorder=10)
    ax.set_xlabel('spaxel');
    ax.set_ylabel('spaxel');
    cb = fig.colorbar(p)
    cb.set_label('flux [{}]'.format(haflux_map.unit))

.. image:: ../_static/haflux_sf_7992-6101.png

.. We should make the map plotting a util.plot function instead of a tools.map function.
.. Want to be able to take in arbitrary masks (e.g., BPT masks)
.. Expose map._make_image and map._make_mask_no_measurement because they are useful for making masks
.. Factor out bitmask bits.


Plot [NII]/Halpha Flux Ratio for Star-forming Spaxels
-----------------------------------------------------

Calculate [NII]6585/Halpha flux ratio:

.. ipython:: python

    maps_7992_6101 = galaxies[1]
    nii_ha = maps_7992_6101.getMapRatio(property_name='emline_gflux', channel_1='nii_6585', channel_2='ha_6564')


Plot the [NII]/Halpha flux ratio for the star-forming spaxels:

.. ipython:: python

    fig, ax = plt.subplots()
    ax.imshow(mask_nodata, cmap=A8A8A8, origin='lower', zorder=1);
    ax.set_xlabel('spaxel');
    ax.set_ylabel('spaxel');
    p = ax.imshow(np.ma.array(nii_ha.value, mask=~sf), origin='lower', cmap='viridis', zorder=10)
    cb = fig.colorbar(p)
    cb.set_label('[NII]6585 / Halpha flux ratio')

.. image:: ../_static/niiha_sf_7992-6101.png


Next Steps
----------
- :ref:`Download Data <marvin-download-objects>` (avoid repeating the same remote API calls every time you run your script)
- :ref:`Jupyter Notebook Tutorials <marvin-jupyter>`
- :ref:`marvin-first-steps` (more general introduction to Marvin)


|