
.. _marvin-tools:

Tools
=====

Marvin provides tools that are convenience classes and functions for searching, accessing, interacting with, and visualzing MaNGA
data. The tools make up the common core components of Marvin and are utilized within the tools itself, as well as by the Web and API.

.. marvin-tools-classes:

Object-Based Tools
------------------

Marvin includes classes that correspond to different levels of MaNGA data
organization\: :ref:`marvin-tools-spectrum`, :ref:`marvin-tools-spaxel`,
:ref:`marvin-tools-rss`, :ref:`marvin-tools-cube`, and :ref:`marvin-tools-plate`.  These classes follow the Manga :doc:`data-access-modes` when determining the data location.  Thus they seamlessly move between local FITS file and remote data via API.


.. marvin-tools-queries:

Search-Based Tools
------------------

Marvin provides tools for easily searching through the MaNGA dataset via queries, from within your own Python terminal.

- :doc:`query`: Perform Queries.
- :doc:`results`: Deal with Results from Queries.

|