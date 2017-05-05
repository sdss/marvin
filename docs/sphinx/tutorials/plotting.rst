.. _marvin-plotting-tutorial:

.. TODO 


Plotting with Marvin
====================

General Tips
------------

Choose a Matplotlib Style Sheet
```````````````````````````````
.. code-block:: python

    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')


Reset the Default Style
```````````````````````

.. code-block:: python

    import matplotlib
    matplotlib.rcdefaults()


Quick Map Plot
--------------

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    ha.plot()

.. image:: ../_static/quick_map_plot.png


Quick Spectrum Plot
-------------------

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')
    spax = maps[17, 17]
    spax.spectrum.plot()

.. image:: ../_static/quick_spectrum_plot.png


Quick Model Fit Plot
--------------------

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')
    spax = maps.getSpaxel(x=17, y=17, xyorig='lower', modelcube=True)
    ax = spax.spectrum.plot()
    ax.plot(spax.model.wavelength, spax.model.flux)
    ax.legend(list(ax.get_lines()), ['observed', 'model'])

.. image:: ../_static/quick_model_plot.png



BPT Plot
--------

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')
    masks, fig = maps.get_bpt()

.. image:: ../_static/bpt.png


Multi-panel Map Plot
--------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from marvin.tools.maps import Maps
    import marvin.utils.plot.map as mapplot
    plt.style.use('seaborn-darkgrid')  # set matplotlib style sheet

    maps = Maps(plateifu='8485-1901')
    stvel = maps['stellar_vel']
    ha = maps['emline_gflux_ha_6564']
    d4000 = maps['specindex_d4000']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, map_ in zip(axes, [stvel, ha, d4000]):
        mapplot.plot(dapmap=map_, fig=fig, ax=ax)

    fig.tight_layout()

.. image:: ../_static/multipanel.png


Custom Axis and Colorbar Locations for Map Plot
-----------------------------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from marvin.tools.maps import Maps
    plt.style.use('seaborn-darkgrid')  # set matplotlib style sheet
    
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']

    fig = plt.figure()
    ax = fig.add_axes([0.12, 0.1, 2 / 3., 5 / 6.])
    fig, ax = ha.plot(fig=fig, ax=ax, cb_kws={'axloc': [0.8, 0.1, 0.03, 5 / 6.]})

.. image:: ../_static/custom_axes.png


Custom Spectrum and Model Fit
-----------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    from marvin.tools.maps import Maps
    plt.style.use('seaborn-darkgrid')  # set matplotlib style sheet

    maps = Maps(mangaid='1-22301')
    spax = maps.getSpaxel(x=28, y=24, xyorig='lower', modelcube=True)

    fig, ax = plt.subplots()
    pObs = ax.plot(spax.spectrum.wavelength, spax.spectrum.flux)
    pModel = ax.plot(spax.spectrum.wavelength, spax.model.flux)
    ax.axis([7100, 7500, 0.3, 0.65])
    plt.legend(pObs + pModel, ['observed', 'model'])
    ax.set_xlabel('observed wavelength [{}]'.format(spax.spectrum.wavelength_unit))
    ax.set_ylabel('flux [{}]'.format(spax.spectrum.units))

.. image:: ../_static/spec_7992-6101.png


Map Using BPT Mask
-----------------

.. code-block:: python

    from marvin.tools.maps import Maps
    maps = Maps(plateifu='8485-1901')
    ha = maps['emline_gflux_ha_6564']
    masks, __ = maps.get_bpt(show_plot=False)
    ha.plot(mask=~masks['sf']['global'])


|
