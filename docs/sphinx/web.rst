
Marvin Web
============

Description of Pages and Capabilities


.. _web-search:

Search
------

Pseudo-natural language search

Example searches
^^^^^^^^^^^^^^^^

Find galaxies...
""""""""""""""""

...below a redshift of 0.1::

    nsa.z < 0.1

...on plates 7815 and 8485::
    
    cube.plate == 7815 or cube.plate == 8485 

...with a IFU size of 127::
    
    ifu.name = 127*

...that contain a spaxel with an Halpha flux > 25::
    
    emline_type.name == Ha and emline_parameter.name == GFLUX and emline.value > 25

...that contain a spaxel with a velocity > 250 km/s::
    
    stellar_kin_parameter.name == vel and stellar_kin.value > 250


.. Search does not handle sub-queries yet

.. Find spaxels...
.. """""""""""""""

.. d ...with Halpha flux > 25::
    
..    emline_type.name == Ha and emline_parameter.name == GFLUX and emline.value > 25
    

.. d ...with [OIII]5008 velocity < 200 km/s:

..    emline_type.name == OIII and emline_typle.rest_wavelength == 5008 and emline_parameter.name == GVEL and emline.value < 200

    

Galaxy
------

The Galaxy page includes:

* basic observation details (such as coordinates and signal-to-noise\ :sup:`2`),
  quality flags, and targeting information,

* links to download the cube, RSS, or DAP FITS files and to view the galaxy in
  the `SDSS Skyserver <http://skyserver.sdss.org/dr12/en/home.aspx>`_,

* an image of the galaxy that can be clicked on to show the nearest spectrum,
  and

* an interactive spectrum display.


.. _web-spectrum:

Spectrum
^^^^^^^^

Spectra stuff