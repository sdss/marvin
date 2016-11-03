

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

...that have Halpha EW > 6 in galaxies between stellar mass 9.5-11 with sersic index < 2::

    nsa.sersic_logmass >= 9.5 and nsa.sersic_logmass < 11 and nsa.sersic_n < 2 and emline_ew_ha_6564 > 6

...that have NII/Halpha ratio > 0.1::

    nii_to_ha > 0.1

End


