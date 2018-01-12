.. _whats-new:

What's New in Marvin 2.2 (January 2018)
=======================================

Marvin 2.2.0 brings significant improvements in the way you interact with MaNGA data.  Try the :ref:`Jupyter Notebook<marvin-jupyter-new22>` for a small sample.

* `MPL-6 <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-6>`_ compatible
* New DRP, DAP, and Query :ref:`Datamodels <marvin-datamodels>`
* :ref:`Cubes<marvin-cube>`, :ref:`Maps<marvin-maps>`, and :ref:`ModelCubes<marvin-modelcube>` now use Astropy Quantities, i.e. encapsulating a measurement with its associated parameters (e.g., unit, mask, or inverse variance)
* Improved Bin class
* Fuzzy Searching and Tab Completion
* New access to DAPall data on `Maps` and `ModelCubes`
* :ref:`Scatter <marvin-utils-plot-scatter>` and :ref:`Histogram <marvin-utils-plot-hist>` Plotting
* Improved Query :ref:`Results <marvin-results>` Handling and Integrated :ref:`Plotting <marvin-results_plot>`
* New :ref:`MaskBit <marvin-utils-maskbit>` class

|

What's New in Marvin 2.1.4 (August 2017)
========================================

* Refactored the Query Page in Marvin Web: Adds more intuitive parameters naming in dropdown.  Adds Guided Marvin Query Builder, using `Jquery Query Builder <http://querybuilder.js.org/>`_.  See the Search page section of :doc:`Web Docs <web>`.

* Adds Galaxy Postage Stamp view of the result set from a Marvin Query in the Web

* Adds Rate Limiting for the Marvin API.  Adopts a limit of 200 requests/min on all routes and 60/min for queries.

* Adds new query_params object in Marvin Tools for improved navigation and selection of available query parameters.  See updated documentation for :doc:`Queries <query>` and :doc:`Query Params <query-params>`

* Adds ability for creating custom maps (using custom values and masks) with Marvin Plotting framework.  See updated :doc:`Plotting Tutorial <tutorials/plotting>`

* New Sidebar in Marvin Documentation for easier navigation.

* New Marvin :doc:`Getting Started <getting-started>` Page.

* New Marvin :doc:`Exercises <exercises>` for showcasing utilization of Marvin in science workflows

* Numerous bug fixes.  See `Changelog <https://github.com/sdss/marvin/blob/master/CHANGELOG.md>`_ for full account of all Github Issues closed.

|

What's New in Marvin 2.1.3 (May 2017)
=====================================

* Slicing in tool objects now behaves as in a Numpy array. That means that `cube[i, j]` returns the same result as `cube.getSpaxel(x=j, y=i, xyorig='lower')`.

* Now it is possible to query on absolute magnitude colours from NSA's `elpetro_absmag`. Absolute magnitudes are now the default for plotting on the web.

* The data file for the default colormap for Map.plot() ("linear_Lab") is now included in pip version of Marvin and does not throw invalid `FileNotFoundError` if the data file is missing.

* Query shortcuts are now only applied on full words, to avoid blind replacements. This fixes a bug that made parameters such as `elpetro_absmag_r` being replaced by `elpetro_absmaelpetro_mag_g_r`.

* Refactored :doc:`Map <tools/map>` plotting methods into :doc:`Utilities <utils/plot-map>`.

  * Map plotting now accepts user-defined ``value``, ``ivar``, and ``mask`` arrays (e.g., BPT masks).
  * It is possible to create multi-panel map plots.
  * All plotting code no longer overwrites matplotlib rcParams.
  * Map plotting has new default gray/hatching scheme for data quality (in tools and web):

    * gray: spaxels with NOCOV.
    * hatched: spaxels with bad data (UNRELIABLE and DONOTUSE) or S/N below some minimum value.
    * colored: good data.

  * Map plotting no longer masks spaxels near zero velocity contour because by default (in tools and web), there is no minimum signal-to-noise ratio for velocity plots.

* New tutorials: :doc:`tutorials/plotting` and :doc:`tutorials/lean-tutorial`.

|

What's New in Marvin 2.1 (February 2017)
========================================

* Marvin is now minimally compliant with Python 3.5+

* `<https://sas.sdss.org/marvin>`_ now points to Marvin 2 (instead of Marvin 1).

* The NSA catalog information is now available via **Cube.nsa** in Marvin Cubes.

* Marvin :ref:`marvin-web` now has a new :ref:`nsa_display` tab with interactive scatter, and box-and-whisker plots.

* Marvin :ref:`marvin-web` has more python tips for working with Marvin :ref:`marvin-tools` objects.

* Marvin now uses Sentry to catch and send errors.

* Marvin :ref:`marvin-maps` now include the ability to make and plot a :ref:`marvin-bpt` diagram.

* Marvin :ref:`marvin-maps` have updated plotting display and now include a new signal-to-noise (snr) attribute on each map.

* Check out the :ref:`visual-guide`.

* Marvin Spaxels now include ``ra`` and ``dec`` as properties.

* Streamlined list of query parameters both in the :ref:`marvin-web` and :ref:`marvin-tools`.  Added new parameter ``ha_to_hb`` ratio.

* Marvin has updated the :ref:`marvin-images` functions for downloading, showing, and locating.

* New **check_marvin** utility to provide some basic system checks with regards to Marvin

* Marvin :ref:`marvin-web` now has a "Provide Feedback" button in the navbar that directly links to a New Issue in Github.

* See `Changelog <https://github.com/sdss/marvin/blob/master/CHANGELOG.md>`_ for more.

|

What's New in Marvin 2.0 Beta (November 2016)
=============================================

* Brand new painless installation (pip install sdss-marvin)

* New Marvin Tools (Maps, Bin, ModelCube)

* Pickling of Marvin Tools, Queries, and Results (i.e. local save and restore)

* DAP Spaxel ("Zonal") Queries

* Dynamic DAP Map display in the web, with point-and-click spaxel

* For MPL-5+, display of model fits in spectrum view in the web

* Versions simplified from mpl, drp, dap down to release

* API :ref:`marvin-authentication`

|

What's New in Marvin 2.0 Alpha (June 2016)
==========================================

Marvin 2.0 is a complete overhaul of Marvin 1.0, converting Marvin into a full suite of interaction tools.

Marvin 2.0 introduces two new modes of operations, :doc:`tools` and :doc:`api`, to the Marvin
environment, and introduces an extensive redesign of the `Marvin web app
<https://sas.sdss.org/marvin/>`_.

The major improvements and additions in this release:

* :doc:`core/data-access-modes`: a new mode based navigation system that allows you to seamlessly interact with MaNGA data no matter where it is.

* :doc:`../tools`: a python package for accessing and interacting with MaNGA
  data, whether the files are in your computer or they need to be retrieved remotely via the
  API.

* :doc:`../api`: remotely grab the data you are looking for as JSONs to integrate directly into your local scripts

* :doc:`../query`: a tool to harness the full statistical power of the MaNGA
  data set by querying the :ref:`marvin-databases`.

* A completely overhauled :doc:`../web` interface, including:

  * A more powerful :ref:`web-search` with an intuitive pseudo-natural language
    search capability.

  * A simple and clean Plate and Galaxy detail page.

  * Interactive spectrum selection from the galaxy image.

  * An image roulette if you are feeling lucky.
