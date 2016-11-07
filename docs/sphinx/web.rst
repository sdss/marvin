
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

Enable the spectrum display by toggling on the Map/Spec View box.  The spectrum display uses the
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

Enable the map display by toggling the red Map/Spec View box.  This displays a series of three maps by default,
with the ability to select up to six maps.  The default maps loaded are the stellar velocity map, the Ha emission line
flux map, and the d4000 spectral index map.  All maps are generated using the
`HighCharts <http://www.highcharts.com/>`_ javascript library.

* **Selecting Maps**: Choose Analysis Properties and Binning-Stellar Template combinations to show.

  * **Analysis Property Dropdown**: Choose up to 6 properties. *Default properties are the Halpha emission line flux (Gaussian fit), the stellar velocity, and the d4000 spectral index maps*
  * **Binning Scheme--Stellar Template Dropdown**: Choose a binning and stellar template set combination. *Default is SPX-GAU-MILESHC (i.e., spaxel binning (i.e., no binning) with the MILESHC stellar template set).*
  * **Get Maps**: Click to display maps.
  * **Reset Selection**: Clear your selected Analysis Properties (Binning Scheme and Stellar Template combination will remain the same.).

* **Map Color Schemes**:

  * **No Data and Bad Data**

    * Grey = Values with the "NoCoverage" maskbit set, or for MPL-4, a mask value of 1.
    * Hatched area = Values with mask bits (5,6,7,or 30) set or low S/N (S/N ratio < 1).

  * **Color Maps**

    * CIE Lab Linear L* (Black-Green-White) = Default color map for sequential values (e.g., emission line fluxes).
    * Inferno (Indigo-Red-White) = Alternative color map for sequential values.
    * Blue-White-Red = Diverging color map with Blue and Red symmetrically diverging from the midpoint color White.

  * **Color Axis**

    * The color axes are restricted to the following percentile ranges of the unmasked data to best display the relative patterns within each map without being skewed by outliers.
      * Velocity: 10-90th percentiles
      * Velocity dispersion: 10-90th percentiles
      * Emission line flux: 5-95th percentiles
      * Other: min-max

* **Hover**: Hover over a Spaxel to show its (x, y) coordinates and value (also indicated by an arrow next to the color axis).

* **Show Spectrum**: Click on an individual Spaxel to display it in the above Spectrum Viewer.

* **Saving a Map**: Click on the menu dropdown (three horizontal lines) just to the upper right of each map and select file format (PNG, JPG, PDF, SVG).



