
Marvin Web
============

Description of Pages and Capabilities


.. _web-search:

Search
------

Marvin-web's Search accepts boolean search strings that are parsed with a
`modified version <https://github.com/havok2063/SQLAlchemy-boolean-search>`_ of
`SQLAlchemy-boolean-search
<https://github.com/lingthio/SQLAlchemy-boolean-search>`_.

See here for a :doc:`boolean-search-tutorial`.

See here for :doc:`query-examples`.

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