.. _marvin-maps:

Maps
====

First Steps
-----------

Let's get the DAP maps of a galaxy by creating a :ref:`marvin-tools-maps` object from its ``mangaid``, ``plateifu``, or ``filename`` (you must specify the full file path). Now we can get the H\ :math:`\alpha` flux :ref:`marvin-tools-map` object and plot it.

::

    from marvin.tools.maps import Maps
    maps = Maps(mangaid='1-209232')
    haflux = maps['emline_gflux_ha_6564']
    fig, ax = haflux.plot()

.. image:: ../_static/haflux_8485-1901.png

Here ``maps['emline_gflux_ha_6564']`` is shorthand for ``maps.getMap('emline_gflux', channel='ha_6564')``, where the property and channel are joined by an underscore ('_'). For properties without channels, such as stellar velocity, just use the property name like ``maps['stellar_vel']``.

To save the plot:

::
    
    fig.savefig('haflux.pdf')

Version 2.1 introduces a completely refactoring of the :meth:`~marvin.tools.map.Map.plot` method. Please see the `Changelog <https://github.com/sdss/marvin/blob/master/CHANGELOG.md>`_ for a complete list of changes and new options available, but here a few critical default settings that are now used:

* Clip at 5th and 95th percentiles (10th and 90th percentiles for velocity and sigma plots).
* Velocity plots are symmetric about 0.
* **no data** (gray): either ``ivar = 0`` or DAP bitmasks ``NOVALUE``, ``BADVALUE``, ``MATHERROR``, ``BADFIT``, or ``DONOTUSE`` set.
* **no measurement** (hatched): pass **no data** criteria but DAP did not produce a measurement or  signal-to-noise ratio < 1.

The DAP map data is stored as 2-D arrays in the ``value``, ``ivar``, and ``mask`` attributes of the ``haflux`` :ref:`marvin-tools-map` object:

::

    haflux.value

We can also grab the DRP and DAP data on any single spaxel. Let's get the central spaxel (x=17, y=17):

::
    
    spax = maps[17, 17]

Here ``maps[17, 17]`` is shorthand for ``maps.getSpaxel(x=17, y=17, xyorig='lower')``, which has additional options that can be invoked by using the :meth:`~marvin.tools.maps.Maps.getSpaxel` method. For example, set (``modelcube=True``) to return the model spectrum (``spax.model``).

We can then get a dictionary of all of the DAP :ref:`marvin-tools-analprop`s with ``spax.properties`` and get the value of any one of them, such as stellar velocity, by using the appropriate key ("stellar_vel" in this case), and accessing the ``value`` attribute of the :ref:`marvin-tools-analprop` object:

::

    spax.properties['stellar_vel'].value

The beauty of Marvin is that you can link to other data about the same galaxy (see :ref:`visual-guide`). Let's see the spectrum.

::

    spax.spectrum.plot()

.. image:: ../_static/spec_8485-1901_17-17.png

Head on over to :ref:`marvin-cube` to learn more about :ref:`marvin-tools-cube` and
:ref:`marvin-tools-spectrum`-related operations.

Advanced Maps Options
---------------------

Bintype
```````

By default, :ref:`marvin-tools-maps` selects the unbinned maps ``SPX``, but we can also choose from additional bintypes (see the `MPL-5 Technical Reference Manual <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/dap/GettingStarted#typeselection>`_ for a more complete description of each bintype and the associated usage warnings):

* ``SPX`` - spaxels are unbinned,
* ``VOR10`` - spaxels are Voronoi binned to a minimum continuum SNR of 10,
* ``NRE`` - spaxels are binned into two radial bins, binning all spectra from 0-1 and 1-2 (elliptical Petrosian) effective radii, and
* ``ALL`` - all spectra binned together.

::
    
    maps = Maps(mangaid='1-209232', bintype='VOR10')

Download
````````

Download the maps using ``rsync`` via `sdss_access <https://github.com/sdss/sdss_access>`_ (see :ref:`marvin-sdss-depends`):

::
    
    maps.download()


Plotting Options
````````````````

Minimum Signal-to-Noise Ratio
:::::::::::::::::::::::::::::

Next Steps

|
