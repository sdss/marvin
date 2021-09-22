.. _marvin-tools:

Tools (marvin.tools)
====================

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
- :doc:`tools/plate`: Explore all cubes for a given Plate
- :ref:`marvin-tools-image`: Interface to MaNGA images
- :ref:`marvin-vacs`: Interface to VACs in Marvin

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


.. role:: green
.. role:: orange
.. role:: red
.. role:: purple


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

* The :red:`red squares` and :green:`green squares` indicate the set of Marvin Tools available.
* The :orange:`orange circles` highlight how each Tool links together via a method or an attribute.  In each transition link, a ``lowercase`` Tool name represents an instantiation of that tool, e.g. ``cube = Cube()``.  To go from a ``Marvin Cube`` to a ``Marvin Spaxel``, you can use the ``cube.getSpaxel`` method or the ``cube[x,y]`` notation.  Conversely, to go from a ``Spaxel`` to a ``Cube``, you would use the ``spaxel.cube`` attribute.  Single- or Bi- directional arrows tell you which directions you can flow to and from the various tools.
* :purple:`Purple circles` represent display endpoints.  If you want to display something, this shows you how which tool the plotting command is connected to, and how to navigate there.

