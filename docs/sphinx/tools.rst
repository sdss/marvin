.. _marvin-tools:

Tools
=====

Marvin provides tools that are convenience classes and functions for searching, accessing, interacting with, and visualzing MaNGA
data. The tools make up the common core components of Marvin and are utilized within the tools itself, as well as by the Web and API.

.. _marvin-tools-classes:

Object-Based Tools
------------------

Marvin includes classes that correspond to different levels of MaNGA data
organization\:

**Core Tools**: Directly importable and usable

- :ref:`marvin-tools-cube`: Interface to the MaNGA DRP Cube
- :ref:`marvin-tools-spaxel`: Explore individual spaxels
- :ref:`marvin-tools-rss`: Interface to the MaNGA DRP RSS object
- :ref:`marvin-tools-maps`: Interface to the MaNGA DAP Maps
- :ref:`marvin-tools-bin`: Explore a bin of spaxels
- :ref:`marvin-tools-modelcube`: Interface to the MaNGA DAP Modelcube object
- :ref:`marvin-tools-plate`: Explore all cubes for a given Plate

**Helper Tools**: Not importable but still usable

- :ref:`marvin-tools-spectrum`: The object containing the spaxel flux, ivar, and mask
- :ref:`marvin-tools-map`: Access individual maps
- :ref:`marvin-tools-mapsprop`: Access MAPS extension information

These classes follow the MaNGA :doc:`data-access-modes` when determining the data location.  Thus they seamlessly move between local FITS file and remote data via API.


.. _marvin-tools-queries:

Search-Based Tools
------------------

Marvin provides tools for easily searching through the MaNGA dataset via queries, from within your own Python terminal.

- :doc:`query`: Perform Queries.
- :doc:`results`: Deal with Results from Queries.

|


.. _marvin-visual-guide:

Visual Guide to Marvin Tools
----------------------------

All **object-** and **search-based** tools in Marvin are seamlessly linked together.  To better understand the flow amongst the various Tools, here is a visual guide.

|

.. image:: ../Marvin_Visual_Guide.png
    :width: 800px
    :align: center
    :alt: marvin visual guide

|


