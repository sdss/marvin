.. currentmodule:: marvin.tools

.. _marvin-catalogues:

Catalogue mixins
================

In addition to the raw data, Marvin provides additional catalogue information about the object via several attributes that are available for most Tools classes.

.. _nsa:

DRPall / NSA
------------

Data from the `NASA Sloan Atlas <http://www.nsatlas.org/data>`_ can be accessed for each `~cube.Cube`, `~rss.RSS`, `~maps.Maps`, and `~modelcube.ModelCube` object via the `~marvin.tools.mixins.nsa.NSAMixIn.nsa` attribute. The feature is implemented as part of the `~marvin.tools.mixins.nsa.NSAMixIn` class. For example:

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

How much NSA information is available depends on the mode in which the object was created. If you instantiated the Tools object from a file, Marvin will try to use the local `DRPall <https://www.sdss.org/dr15/manga/manga-tutorials/drpall/>`_ file, which contains only a subset of the NSA data. If loaded remotely, Marvin will make an API call to retrieve the full NSA information from the server. You can change this behaviour during instantiation by setting the ``nsa_source`` attribute from ``'drpall'`` to ``'nsa'`` ::

    >>> my_cube_from_file = Cube('path/to/file/manga-8485-1901-LOGCUBE.fits.gz', nsa_source='nsa')

DAPall
------

The `DAPall file <https://www.sdss.org/dr15/manga/manga-data/catalogs/#DAPALLFile>`_  contains galaxy-summed, derived information from the Data Analysis Pipeline, such as total fluxes, SFR, etc. Similarly to the :ref:`NSA <nsa>` data, it can be accessed from `~maps.Maps` and `~modelcube.ModelCube` objects via the `~marvin.tools.mixins.dapall.DAPallMixIn.dapall` attribute in the `~marvin.tools.mixins.dapall.DAPallMixIn` ::

    >>> maps.dapall
    {'plate': 8485,
     'ifudesign': 1901,
     'plateifu': '8485-1901',
     'mangaid': '1-209232',
     'drpallindx': 3505,
     'mode': 'CUBE',
     'daptype': 'SPX-GAU-MILESHC',
     'dapdone': 'true',
     'objra': 232.545,
     'objdec': 48.6902,
     'ifura': 232.545,
     'ifudec': 48.6902,
     ...}


.. include:: vacs.rst


Reference
---------

Classes
^^^^^^^

.. autosummary::

    marvin.tools.mixins.nsa.NSAMixIn
    marvin.tools.mixins.dapall.DAPallMixIn


Available VACS
^^^^^^^^^^^^^^

.. include:: ../contributing/available_vacs.rst
