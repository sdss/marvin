
.. _marvin-tools:

Tools
=====

Marvin provides tools that are convenience classes and functions for searching, accessing, interacting with, and visualzing MaNGA
data. The tools make up the common core components of Marvin and are utilized within the tools itself, as well as by the Web and API.

.. marvin-tools-classes:

Object-Based Tools
------------------

Marvin includes classes that correspond to different levels of MaNGA data
organization\:

- :ref:`marvin-tools-cube`: Interface to the MaNGA DRP Cube
- :ref:`marvin-tools-spaxel`: Explore individual spaxels
- :ref:`marvin-tools-spectrum`: The object containing the spaxel flux, ivar, and mask
- :ref:`marvin-tools-rss`: Interface to the MaNGA DRP RSS object
- :ref:`marvin-tools-maps`: Interface to the MaNGA DAP Maps
- :ref:`marvin-tools-map`: Access individual maps
- :ref:`marvin-tools-bin`: Explore a bin of spaxels
- :ref:`marvin-tools-modelcube`: Interface to the MaNGA DAP Modelcube object
- :ref:`marvin-tools-plate`: Explore all cubes for a given Plate

These classes follow the Manga :doc:`data-access-modes` when determining the data location.  Thus they seamlessly move between local FITS file and remote data via API.


.. marvin-tools-queries:

Search-Based Tools
------------------

Marvin provides tools for easily searching through the MaNGA dataset via queries, from within your own Python terminal.

- :doc:`query`: Perform Queries.
- :doc:`results`: Deal with Results from Queries.

|
