.. _marvin-map:

Map
====

For the docstring see :ref:`marvin-tools-map`.

A class to interface with an individual MaNGA DAP map.

.. This class represents a fully reduced DRP data cube, initialised either from a
   file, a database, or remotely via the Marvin API.

.. filename, plate-IFU, mangaID


::
    
    from marvin.tools.maps import Maps
    maps = Maps(mangaid='1-209232')
    haflux = maps.getMap('emline_gflux', channel='ha_6564')
    fig, ax = haflux.plot()



|