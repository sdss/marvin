
Marvin Web
============

Description of Pages and Capabilities


.. _web-search:

Search
------

Marvin-web's Search accepts boolean search strings that are parsed with a
`modified version <https://github.com/havok2063/SQLAlchemy-boolean-search>`_ of
`SQLAlchemy-boolean-search
<https://github.com/lingthio/SQLAlchemy-boolean-search>`_. Please see here for a
:doc:`boolean-search-tutorial`.


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

* **meta-data**: basic observation details (such as coordinates and
  signal-to-noise\ :sup:`2`), quality flags, and targeting information,

* **download link**: links to download the cube, RSS, or DAP FITS files and to
  view the galaxy in the `SDSS Skyserver
  <http://skyserver.sdss.org/dr12/en/home.aspx>`_,

* **galaxy image** that can be clicked on to show the nearest spectrum, and

* **interactive spectrum display**.


.. _web-spectrum:

Spectrum Display
^^^^^^^^^^^^^^^^

Enable the spectrum display by clicking on the Map/Spec View box.

* **Select Spectrum**: Click on the image to show the spectrum of the spaxel at
  a particular location (default is central spaxel), which is indicated by the
  red dot and whose coordinates are listed above the spectrum.

* **Zooming**: Zoom in by clicking and dragging either horizontally or
  vertically. Double click to unzoom.

* **Spectrum features**:

  * solid line: spectrum
  * shaded region: 1-sigma error range
  * cursor coordinates: wavelength and flux value


.