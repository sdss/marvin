.. _marvin-maps:

Maps
====

:ref:`marvin-tools-maps` is a class to interact with the DAP maps for a galaxy. First specify the galaxy that you want by creating a :ref:`marvin-tools-maps` object. Then you can use the :ref:`marvin-tools-maps` object to get an H\ :math:`\alpha` flux map as a :ref:`marvin-tools-map` object.

::

    from marvin.tools.maps import Maps
    maps = Maps(mangaid='1-209232')
    haflux = maps.getMap('emline_gflux', channel='ha_6564')
    fig, ax = haflux.plot()

.. image:: ../_static/haflux_8485-1901.png


The DAP data is stored as 2-D arrays in the ``value``, ``ivar``, and ``mask`` attributes of the ``haflux`` :ref:`marvin-tools-map` object.

::
    
    haflux.value


The beauty of Marvin is that you can link to other data about the same galaxy. Let's see the spectrum of the central spaxel (x=17, y=17).

::
    
    spec = maps.cube[17, 17].spectrum
    spec.plot()


.. image:: ../_static/spec_8485-1901_17-17.png


The spectrum data is stored as 1-D arrays in the ``flux``, ``ivar``, and ``mask`` attributes of ``spec`` :ref:`marvin-tools-spectrum` object.

::
    
    spec.flux


Head on over to :ref:`marvin-cube` to learn more about cube and spectrum-related operations.

|