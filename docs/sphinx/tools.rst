
.. _marvin-tools:

Tools
=====

Marvin-tools is an importable python package that provides convenience classes
and functions for searching, accessing, interacting with, and visualzing MaNGA
data. Since these capabilities are useful in both an exploratory environment,
such as in Marvin-web, and in science-grade analysis code, we factored out the
common elements into Marvin-tools.

Marvin-tools includes classes that correspond to different levels of MaNGA data
organization\: :ref:`marvin-tools-spectrum`, :ref:`marvin-tools-spaxel`,
:ref:`marvin-tools-rss`, :ref:`marvin-tools-cube`, and
:ref:`marvin-tools-plate`.  These classes have methods to retrieve the
appropriate data from a locally stored file, over the internet via Marvin-API,
or by downloading FITS files.

One of the most powerful aspects of the Marvin ecosystem is the ability to use
Marvin-API through Marvin-tools to query the MaNGA databases from within a
python script or terminal. With the Marvin-tools class :doc:`query` you can
build and execute a query. The results of your query are returned as an instance
of the :doc:`results` class, which has built-in methods for navigating and
presenting those results.

Performing Queries:

:doc:`query`

Dealing with Results from Queries:

:doc:`results`