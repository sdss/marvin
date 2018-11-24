
.. _marvin-web:

Web (marvin.web)
================

Here we describe the general content and functionality of all pages in
Marvin-Web.  Most of the functionality that you will find on the Web exists in
Marvin Tools. Indeed, a lot of the functionality simply uses existing Marvin
tools.

.. _web-main:

Main Page
---------

The main Marvin splash page for the MaNGA survey contains a navigation bar with links
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
currently working with.  The default available releases are public data releases.

**Login**:

The Marvin login box.  Log in using your personal SDSS credentials or the general SDSS username and password to
access collaboration data.


.. _web-random:

Image Roulette Page
-------------------

Displays a random set of 16 images of galaxies from the MaNGA survey.  Each
thumbnail on the left is displayed in larger form on the right.  Click the giant
image to go the individual Galaxy Page.

.. _web-search:

Search Page
-----------

Search the MaNGA dataset using a simplified interface with pseudo-natural
language syntax.

**Return Parameters**:

A dropdown multiple select box indicating which parameters are available to query
on and/or return.  You may select multiple parameters (up to 5). See :ref:`marvin-query-parameters`.

**Search Filter**:

An input string search filter box that accepts a pseudo-natural language format.

The search filter accepts boolean search strings that are parsed with a
`modified version <https://github.com/havok2063/SQLAlchemy-boolean-search>`_ of
`SQLAlchemy-boolean-search
<https://github.com/lingthio/SQLAlchemy-boolean-search>`_. Please see here for a
:doc:`tutorials/boolean-search-tutorial`.

**Guided Query Builder**:

If you need help building a query, you can use this to help design your SQL filter.  This feature uses the `Jquery Query Builder <http://querybuilder.js.org/>`_.

After a query is run a table of results is generated to navigate around.  Returned columns include some default target identification parameters, plus any parameters used in your search filter, or additional returned parameters.  Large result sets are paginated.  You can also sort on individual columns by clicking the column header.

**View Galaxies**:

After a search, the **View Galaxies** button will display the postage stamp images of all the galaxies in the results.  This page displays up to 16 galaxies at a time, with the big carousel image linking to the individual galaxy page.  There are pagination buttons to help you cycle through your pages.


.. _web-plate:

Plate Pages
-----------

A Plate page includes:

* **meta-data**: some basic information about the Plate

* **data link**: a link to the plate directory on the SAS

* **galaxy images**: the set of galaxies observed on this plate,
  that link to the individual galaxy pages

.. _web-galaxy:

Galaxy Pages
------------

A Galaxy page includes:

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

* **download link**: links to download the cube, RSS, or the default DAP maps
  FITS files

* **SDSS Skyserver link**: view the galaxy in the `SDSS Skyserver
  <http://skyserver.sdss.org/dr12/en/home.aspx>`_,

* **Map/Specview toggle button**: Click this toggle button to activate:

    * a **galaxy image** that can be clicked on to show the nearest spectrum,

    * an :ref:`interactive spectrum display <web-spectrum>`,

    * an :ref:`interactive map display <web-maps>`, and

    * an :ref:`interactive galaxy properties display <galaxy_properties>`.


.. _web-spectrum:

Spectrum Display
^^^^^^^^^^^^^^^^

Enable the spectrum display by toggling on the Map/SpecView box.  The spectrum display uses the `DyGraphs <http://dygraphs.com/>`_ javascript library.

* **Select Spectrum**: Click on the image or a :ref:`map <web-maps>` to show the spectrum of the spaxel at a particular location (default is central spaxel) whose coordinates are listed above the spectrum. The most recently selected location in the image is indicated by the red dot.

* **Zooming**: Zoom in by clicking and dragging either horizontally or
  vertically.  Double click to unzoom.  The zoomed region will remain as you
  click on different locations of the galaxy image.

* **Panning**: When zoomed in, hold shift and click and drag with the mouse to
  pan left and right.

* **Spectrum features**:

  * green solid line: spectrum (in observed frame)
  * green shaded region: 1-sigma error range
  * blue solid line: full model fit (HYB10-MILESHC for Data Release >= 15 and MPLs >= 7)
  * cursor coordinates: wavelength, flux, and model fit values


.. _web-maps:

Map Display
^^^^^^^^^^^

Enable the map display by toggling the red Map/SpecView box.  This displays a series of three maps by default, with the ability to select up to six maps.  The default maps loaded are the stellar velocity map, the Halpha emission line flux map, and the d4000 spectral index map.  All maps are generated using the `HighCharts <http://www.highcharts.com/>`_ javascript library.

* **Selecting Maps**: Choose Analysis Properties and Binning-Stellar Template combinations to show.

  * **Analysis Property Dropdown**: Choose up to 6 properties. *Default properties are the Halpha emission line flux (Gaussian fit), the stellar velocity, and the d4000 spectral index maps.*
  * **Binning Scheme--Stellar Template Dropdown**: Choose a binning and stellar template set combination. *Default is HYB10-GAU-MILESHC* (i.e., hybrid binning scheme with stellar continuum fit in Voronoi bins with signal-to-noise ratio >= 10 and emission lines fit in each spaxel with the MILESHC stellar template set).
  * **Get Maps**: Click to display maps.
  * **Reset Selection**: Clear your selected Analysis Properties (Binning Scheme and Stellar Template combination will remain the same.).

* **Sigma Corrections**:
  When selecting the ``stellar_sigma`` or ``emline_sigma`` maps, we automatically apply the relevant sigma correction.  A corrected map is indicated via the **Corrected: [name]** map title.  Uncorrected maps, for example, in MPL-6, retain the original title name.

* **Map Color Schemes**:

  * **No Data and Bad Data**

    * Grey = Values with the "NoCoverage" maskbit set, or for MPL-4, a mask value of 1.
    * Hatched area = Values with mask bits (5, 6, 7, or 30) set or low S/N (S/N ratio < 1; not used for velocity maps).

  * **Color Maps**

    * CIE Lab Linear L* (Black-Green-White): default color map for sequential values (e.g., emission line fluxes).
    * Inferno (Indigo-Red-White): alternative color map for sequential values used for velocity dispersion maps.
    * Blue-White-Red: diverging color map with Blue and Red symmetrically diverging from the midpoint color White used for velocity maps.

  * **Color Axis**

    * The color axes are restricted to the following percentile ranges of the unmasked data to best display the relative patterns within each map without being skewed by outliers.

      * Velocity: 10-90th percentiles
      * Velocity dispersion: 10-90th percentiles
      * Emission line flux: 5-95th percentiles
      * Other: min-max

* **Hover**: Hover over a Spaxel to show its (x, y) coordinates and value (also indicated by an arrow next to the color axis).

* **Show Spectrum**: Click on an individual Spaxel to display it in the above Spectrum Viewer.

* **Saving a Map**: Click on the menu dropdown (three horizontal lines) just to the upper right of each map and select file format (PNG, JPG, PDF, SVG).


.. _galaxy_properties:

Galaxy Properties Display
^^^^^^^^^^^^^^^^^^^^^^^^^

Clicking the Galaxy Properties tab will show you the `NASA-Sloan Atlas (NSA) catalog <https://www.sdss.org/dr13/manga/manga-target-selection/nsa/>`_ information for this galaxy in a table format.  In addition, there are two tabs for interactive display.

* **NSA table**: Most of the NSA galaxy properties are displayed in this table (paginated by default).  Click the arrow in the upper right corner to toggle the pagination and view all parameters at once.

* **Scatter Plot**: This tab provides two convenient scatter plots highlighting the relative location of the specific galaxy amongst the NSA sample of MaNGA galaxies. The plots are interactive.  Click and drag to zoom in.  Hover over points to see pop up info. You can change the plotted parameters by dragging and dropping one of the twelve **bold** parameters displayed in the NSA table on the left.  As you drag, the drop location will be highlighted in red.

* **Box and Whisker**: This tab provides a simplified interface to the Scatter Plot tab.  It displays the galaxy NSA parameter relative the entire sample in a series of box-and-whisker plots.  By default, the twelve **bold** parameters in the NSA table are displayed here. Hover over the red dot or the outliers to see their values. Scroll horizontally to see more parameters.

In each `box-and-whisker <https://en.wikipedia.org/wiki/Box_plot>`_ plot:
 * The red dot is the galaxy parameter value.
 * The horizontal line is the median value of the NSA sample.
 * The lower and upper bounds of the box are the 25th and 75th percentiles.
 * The whiskers of the box are 1.5 \* interquartile range.
 * Outlier points are indicated as light grey open circles.

|
