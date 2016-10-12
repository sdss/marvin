
.. _marvin-web:

Web
===

Here we describe the general content and functionality of all pages in
Marvin-Web.  Most of the functionality that you will find on the Web exists in
Marvin Tools. Indeed, a lot of the functionality simply uses existing Marvin
tools.

.. _web-main:

Main
----

The main Marvin splash page for the MaNGA survey contains a navgiation bar with links
to other Marvin pages, a Log-In window, general information about Marvin
and the development team, as well as quick contact information at the bottom.

**ID Search Box**:

This search box allows to search for MaNGA objects either by
plate-IFU, plateID, or mangaID.  It features autocompletion so just
start typing away.

Note: The dropdown menu does NOT display all available
designations given the typed entry.

**Version Dropdown Select**:

A dropdown select button indicating what version of MaNGA data you are
currently working with.

.. _web-help:

Help
----
A link to the Marvin Documentation page.

.. _web-random:

Image Roulette
--------------

Displays a random set of 16 images of galaxies from the MaNGA survey.  Each
thumbnail on the left is displayed in larger form on the right.  Click the giant
image to go the individual Galaxy Page.

.. _web-search:

Search
------

Search the MaNGA dataset using a simplified interface with pseudo-natural
language syntax.

**Return Parameters**:

A dropdown multiple select box indicating which parameters are available to query
on and/or return.  You may select multiple parameters.

**Query Parameters**:

An input box to type the parameters you wish to return.  Autocompletion is enabled.
Type a parameter, hit enter, and type again.

.. note:: Decide which format of the above two is most useful.

**Search Filter**:

An input string search filter box that accepts a pseudo-natural language format.

The search filter accepts boolean search strings that are parsed with a
`modified version <https://github.com/havok2063/SQLAlchemy-boolean-search>`_ of
`SQLAlchemy-boolean-search
<https://github.com/lingthio/SQLAlchemy-boolean-search>`_. Please see here for a
:doc:`tutorials/boolean-search-tutorial`.


.. _web-plate:

Plate
-----

The Plate page includes:

* **meta-data**: some basic information about the Plate

* **data link**: a link to the plate directory on the SAS

* **galaxy images**: the set of galaxies observed on this plate,
  that link to the individual galaxy pages

.. _web-galaxy:

Galaxy
------

The Galaxy page includes:

* **meta-data**: basic observation details (such as coordinates and
  signal-to-noise\ :sup:`2`), quality flags, and targeting information,

* **cube quality**: The quality of the cube as indicated by the DRP3QUAL
  bitmask. The bitmask is indicated.  Hover over the flags button to see a pop
  up of the flags contained in the given bitmask.

  * Color Code:

    * Green - Good Quality
    * Yellow - Poor Quality
    * Red - Critical Failures

* **manga target bits**: The MaNGA target bit masks for the current galaxy.  Indicates
  a Galaxy, Stellar, or Ancillary bit.  Hover over the flags button to see a pop up
  of the flags contained in the given bitmask.

* **download link**: links to download the cube, RSS, or the default DAP MAPS
  FITS files and to view the galaxy in the `SDSS Skyserver
  <http://skyserver.sdss.org/dr12/en/home.aspx>`_,

* **galaxy image** that can be clicked on to show the nearest spectrum, and

* **interactive spectrum display**.

* **interactive map display**.

.. _web-spectrum:

Spectrum Display
^^^^^^^^^^^^^^^^

Enable the spectrum display by clicking on the Map/Spec View box.  The spectrum display uses the
`DyGraphs <http://dygraphs.com/>`_ javascript library.

* **Select Spectrum**: Click on the image to show the spectrum of the spaxel at
  a particular location (default is central spaxel), which is indicated by the
  red dot and whose coordinates are listed above the spectrum.

* **Zooming**: Zoom in by clicking and dragging either horizontally or
  vertically. Double click to unzoom.  The zoomed region will remain as you
  click on different locations of the galaxy image.

* **Panning**: When zoomed in, hold shift and click and drag with the mouse to
  pan left and right.

* **Spectrum features**:

  * green solid line: spectrum
  * blue solid line: model fits (unbinned: SPX-MILESHC) (for MPLs >= 5)
  * shaded region: 1-sigma error range
  * cursor coordinates: wavelength, flux, and modelfit value

.. _web-maps:

Map Display
^^^^^^^^^^^

Enable the map display by clicking on the Map/Spec View box.  This displays a series of six maps.  Default maps are
the six emission line gflux maps [OIId, Hb, OIII5008, NII6585, Ha, SII6718].  These maps are generated using the
`HighCharts <http://www.highcharts.com/>`_ javascript library.

* **Map Dropdown**: Choose up to 6 maps from the dropdown list.  Click the Get Maps button to display them.
* **Bin-Template Selection**: Choose a binning and template option from the dropdown list.
* **Select Spaxel**: Click on an individual Spaxel to display it in the above Spectrum Viewer.
* **Hover**: Hover over a Spaxel to see the spaxel x and y, and the value of the map at the particular point
* **ColorAxis**: The color axis (right-side) is mapped to the min and max of the data series, after masked values
  have taken into account.
* **Map Colors**: The map colors are defined as follows:
  * Grey = Values with the "NoCoverage" maskbit set, or for MPL-4, a mask value of 1.
  * Hatched area = Values with
  * One-Tone Blue = All maps that have all values >= 0 (e.g. emission line flux maps)
  * Two-Tone Blue-to-Red = All maps that have a minimum value < 0 (e.g. velocity maps)





