
.. _marvin-vacs:

Value-Added Catalogs (VACS)
---------------------------

While the core of SDSS data releases centers around its base projects' science deliverables, smaller teams frequently contribute added value to its core deliverables with additional science products.  These value-added data products or catalogs (VACS) are derived data products based on the core deliverables that are vetted, hosted, and released by SDSS in order to maximize the impact of SDSS data sets.

Marvin

::

    from marvin.tools import Cube

    cube = Cube('8485-1901')
    cube.vacs
    <VACContainer ('mangahi')>





