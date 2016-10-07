

.. _marvin-query-examples:

Example Filter Conditions
=========================

Return Galaxies
---------------

...below a redshift of 0.1::

    nsa.z < 0.1

...on plates 7815 and 8485::

    cube.plate == 7815 or cube.plate == 8485

...with a IFU size of 127::

    ifu.name = 127*

...with Halpha flux > 25 in more than 20% of their good spaxels::

    npergood(emline_gflux_ha_6564 > 25) >= 20


Return Spaxels
--------------

...that have an Halpha flux > 25::

    emline_gflux_ha_6564 > 25

End


