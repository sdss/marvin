
.. currentmodule:: marvin.tools

.. _marvin-metadata:

Accessing metadata
==================

In addition to the raw data, Marvin provides additional metadata about the object via several attributes that are available for most Tools classes.

NSA
---

Data from the `NASA Sloan Atlas <http://www.nsatlas.org/data>`_ can be accessed for each `~cube.Cube`, `~maps.Maps`, and `modelcube.ModelCube` object via the `~marvin.core.core.NSAMixIn.nsa` attribute. The feature is implemented via the `~marvin.core.core.NSAMixIn` class. For example:

.. code-block:: python

    >>> my_cube = Cube('8485-1901', mode='remote')
    >>> my_cube.nsa
    {'iauname': 'J153010.73+484124.8',
    'subdir': '15h/p48/J153010.73+484124.8',
    'ra': 232.544703894,
    'dec': 48.6902009334,
    'isdss': 225926,
    'ined': 167868,
    'isixdf': -1,
    'ialfalfa': -1,
    'izcat': 373580,
    'itwodf': -1,
    'mag': 17.3247,
    'z': 0.0407447,
    'zsrc': 'sdss   ',
    'size': 0.07,
    ...

How much NSA information is available depends on the mode in which the object was created. If you opened a `~cube.Cube` or `~maps.Maps` from a file, Marvin will try to use the local `DRPall <https://www.sdss.org/dr15/manga/manga-tutorials/drpall/>`_ file, which contains only a subset of the NSA data. If loaded remotely, Marvin will make an API call to retrieve the full NSA information from the server. You can change this behaviour in runtime by setting the ``nsa_source`` attribute from ``'drpall'`` to ``'nsa'``.

DAPall
------

The `DAPall file <https://www.sdss.org/dr15/manga/manga-data/catalogs/#DAPALLFile>`_  contains galaxy-summed, derived information from the Data Analysis Pipeline, such as total fluxes, SFR, etc. Similarly to the `NSA`_ data, it can be accessed from `~maps.Maps` and `~modelcube.ModelCube` objects via the `~marvin.core.core.DAPallMixIn.dapall` attribute.

VACs
----

.. admonition:: Warning
    :class: warning

    Value Added Catalogues are supported by their owners. Please read the documentation for the VAC you are planning on using and make sure you understand the format of the data and any related caveat.

`Value Added Catalogues (VAC) <http://www.sdss.org/dr15/data_access/value-added-catalogs/>`_ provide analysis of MaNGA data created by collaborators and that are distributed as part of SDSS data releases. VAC owners can integrate their catalogues in Marvin following :ref:`this procedure <marvin-contributing-vacs>`. VACs can be accessed via the ``.vacs`` attribute. The first time you access a VAC (or, depending on the VAC, the first you access a VAC for a certain object) a download will be triggered and the catalogue will be downloaded to you local SAS. From that moment on the VAC will be accessible offline. Available VACs are listed below.


Reference
---------

Classes
^^^^^^^

.. autosummary::

    marvin.core.core.NSAMixIn
    marvin.core.core.DAPallMixIn
    marvin.contrib.vacs.base.VACMixIn

Available VACs
^^^^^^^^^^^^^^

.. include:: ../contributing/available_vacs.rst
