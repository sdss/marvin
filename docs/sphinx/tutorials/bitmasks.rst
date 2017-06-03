.. _marvin-bitmasks:

========
Bitmasks
========



.. warning::
    
    Page under heavy development


`Introduction to bitmasks <http://www.sdss.org/dr13/algorithms/bitmasks/>`_




Get a map:

.. code-block:: python


    from marvin.tools.maps import Maps
    galaxy = Maps(plateifu='8485-1901')
    ha = galaxy['emline_gflux_ha_6564']

Use the DAP bitmasks to flag spaxels that we don't want to include:

.. code-block:: python

    import numpy as np
    novalue = (ha.mask & 2**4) > 0
    badvalue = (ha.mask & 2**5) > 0
    matherror = (ha.mask & 2**6) > 0
    badfit = (ha.mask & 2**7) > 0
    donotuse = (ha.mask & 2**30) > 0
    no_data = np.logical_or.reduce((novalue, badvalue, matherror, badfit, donotuse))


Create masked arrays for the image and regions that are masked by the DAP:

.. code-block:: python

    image = np.ma.array(haflux_map.value, mask=~sf)
    mask_nodata = np.ma.array(np.ones(haflux_map.value.shape), mask=sf)


Let's set the background to gray:

.. code-block:: python

    from marvin.utils.plot import colorbar
    A8A8A8 = colorbar.one_color_cmap(color='#A8A8A8')


Use the masked image to make the plot:

.. code-block:: python

    fig, ax = plt.subplots()
    ax.imshow(mask_nodata, cmap=A8A8A8, origin='lower', zorder=1);
    p = ax.imshow(image, cmap='viridis', origin='lower', zorder=10)
    ax.set_xlabel('spaxel');
    ax.set_ylabel('spaxel');
    cb = fig.colorbar(p)
    cb.set_label('flux [{}]'.format(haflux_map.unit))
