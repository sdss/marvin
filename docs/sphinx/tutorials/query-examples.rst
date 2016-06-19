

.. _marvin-query-examples:

Example Queries
===============

Find galaxies
-------------

...below a redshift of 0.1::

    nsa.z < 0.1

...on plates 7815 and 8485::

    cube.plate == 7815 or cube.plate == 8485

...with a IFU size of 127::

    ifu.name = 127*

...that contain any spaxel with an Halpha flux > 25::

    emline_type.name == Ha and emline_parameter.name == GFLUX and emline.value > 25

...that contain any spaxel with a velocity > 250 km/s::

    stellar_kin_parameter.name == vel and stellar_kin.value > 250

End
