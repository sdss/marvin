.. _whats-new:


What's New
==========

.. toctree::
    :hidden:

    changelog

This section summarises the most important new features a bugfixes in Marvin. For the complete list of changes check the :ref:`marvin-changelog`.

2.7.0 (December 2021)
---------------------
- Support for `DR17 <https://www.sdss.org/dr17/manga/>`_.  See the :ref:`Marvin DR17 datamodel<datamodel-dr17>`.
- Updated VACs for DR17.  New VAC GalazyZoo3D (`~marvin.contrib.vacs.galaxyzoo3d.GZ3DVAC`).
- Adds support for DR17 DAPTYPES `SPX-MILESHC-MASTARSSP` and `VOR10-MILESHC-MASTARSSP` (see `DAP Output <https://www.sdss.org/dr17/manga/manga-data/data-model/#DataAnalysisPipelineOutput>`_).
- New `Explore <https://dr17.sdss.org/marvin/explore>`_ page in Marvin Web

  - Find it via the **Upload** button in the top navbar.
  - Upload a list of targets and choose which DAP map to display for all targets

- Improvements to the Marvin web interface
- New "Available VACs" information button on galaxy web page
- Change to default behaviour of Marvin's Query Result `~marvin.tools.results.Results.toJson` method

 - conversion to JSON now uses `.pandas.DataFrame.to_json` as default internal mechanism
 - the `toJson` method by default now returns an orientation of a list of records
 
- Adds "Export Data" button to the web query page
- Improves performance of data table in the web query page
- Removes web and API support for MPLs 8-10.  These MPLs are still accessible via local file-based mode in Marvin.

2.6.1 (November 2021)
---------------------
- Support for using Tools as context managers, allowing efficient access and closure of file handlers. This will make workflows relying on looping through lists of galaxies and performing similar actions less error-prone.

2.6.0 (March 2020)
------------------
- Adds support for MaNGA release :ref:`MPL-11 MaNGA data<datamodel-mpl11>`

2.5.0 (September 2020)
----------------------
- Adds support for MaNGA release :ref:`MPL-10 MaNGA data<datamodel-mpl10>`
- Removes support in the web and API (remote access) for MPLs 4, 5, and 6.
- Removes support for DAP SPX and VOR10 binning schemes in the web and API access.  Only the HYB10 binning scheme is available over the web/API.
- Note on deprecated support for MPLs and DAP binning schemes in the web/API:

  - MPLs 4, 5 and 6 are still supported via local file mode
  - Alternate DAP binning schemes (SPX, VOR10) are still supported via local file mode

- MPL-10 datamodel changes, compared to MPL-9:

  - DRP cubes/rss files

    - Extension `DISP/PREDISP` has been renamed to `LSFPOST/LSFPRE` in FITS files
    - Marvin MPL-10 Cube/RSS datamodels retain the `dispersion/dispersion_prepixel` nomenclature for backwards compatibility.

  - DAP maps/modelcube files

    - Removed MAPS extensions `SPECINDEX_BCEN`, `SPECINDEX_BCNT`, `SPECINDEX_RCEN`, `SPECINDEX_RCNT`
    - Added MAPS extensions `SPECINDEX_BF`, `SPECINDEX_BF_CORR`, `SPECINDEX_BF_MODEL`, `SPECINDEX_WGT`, `SPECINDEX_WGT_CORR`, `SPECINDEX_WGT_MODEL`
    - Added ModelCube (LOGCUBE) `LSF` extension.


2.4.1 (August 2020)
-------------------
- Adds support for switching to the Marvin mirror, `magrathea.sdss.org`, hosted at Johns Hopkins

2.4.0 (March 2020)
------------------
- Caching of remote Marvin Tools (cached database queries)
- Improved web performance

  - Adds caching for routes on plate and galaxy pages
  - Improved web spaxel/map handling

- Added utility functions, and improved handling, of DRPall and DAPall files.
- `Marvin Cube` now has a `get_available_bintypes` method which retrieves the available DAP binning-template schemes available for the given cube.
- Moving to more semantic versioning (**major.minor.patch**)

  - major release bumps for breaking changes to API
  - minor release bumps for feature changes and some patches
  - patch release bumps for bug fixes

- Added support for the Visual Morphology VAC
- Moved to internal use of `tree 3.0.0` and `sdss_access 1.0.0`

2.3.5 (December 2019)
---------------------

- Support for :ref:`MPL-9 MaNGA data<datamodel-mpl9>`
- Enhancements to the web spectrum display

    - You can now toggle common lines measured by the DAP (annotations placed using global NSA redshift)
    - You can now toggle between rest-frame and observed wavelength (shifted using global NSA redshift)
    - You can toggle a bad pixel mask. Pixel regions marked as DONOTUSE are now displayed as a red highlighted region.

- Improved image loading on Image Roulette page
- "Traditional" spaxel queries have been reimplemented until ongoing improvements can be completed

2.3.4 (December 2019)
---------------------

Along with plenty of bug fixes, the following things are new

- Support for DR16, which is the same as DR15.
- Added support for the following :ref:`MaNGA VACS<marvin-available-vacs>`

    - GEMA: Returns GEMA table data
    - Galaxy Zoo: Returns Galaxy Zoo morphologies
    - Firefly: Returns stellar population parameters fitted by FIREFLY
    - MaNGA HI: Returns HI summary data and spectra
- A new :ref:`Marvin VACs<marvin-vacs>` tool to access the full VAC FITS file.
- Complete support for Windows.  With an update to `sdss_access`, Marvin now fully works on Windows, using `curl` as the default download service.
- No more API rate limiting.  This has been removed until further notice.
- New info popups for "DAP Maps" and "Queries" with links to additional information.  Click the "DAP Maps" and "Learn More' labels with the question marks, on the galaxy and query pages, respectively.
- Improved galaxy web memory - the galaxy web page now remembers when you've selected the dynamic MapSpecView, as well your map and bintype selections. Quickly move from galaxy to galaxy without reselecting the same things over and over.
- Improved DataModel documentation

2.3.2 (February 2019)
---------------------

Support for MPL-8.  MPL-8 includes several changes to the DAP Datamodel.
Please see the :ref:`MPL-8 DataModel<datamodel-mpl8>` and the `MPL-8 TRM <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-8/DAPDataModel>`_ page (SDSS collaboration access required).  In summary, the changes include:

- The stellar template is now `MILESHC-MILESHC`, indicating the template used for the stellar kinematics fitting and the emission-lines, respectively.
- The binning types now include `SPX`, in addition to `HYB10` and `VOR10`.  `HYB10` is still the default bintype.
- Changes to ModelCubes

    - Removed `emline_base_fit` Model
    - New `stellar_fit` Model
- Changes to Maps

    - The `stellar_sigmacorr` now has two channels.
    - New `stellar_fom` multi-channel property.  This property contains figure-of-merit metrics for the stellar kinematics.
    - Moved `stellar_cont_fresid` and `stellar_cont_rchi2` into the stellar_fom property.
    - Three new emission-lines: `HeI 3889`, `NI 5199`, `NI 5201`
    - New `emline_ga` multi-channel property; the amplitude of the fitted Gaussian
    - New `emline_ganr` multi-channel property; the amplitude/noise of the fitted Gaussian
    - New `emline_fom` multi-channel property; full spectrum figure-of-merits for the emission lines
    - New `emline_lfom` multi-channel property; reduced chi-square figure-of-merits for the emission lines

- The DRPall file now has two data extensions, **MANGA**, and **MASTAR**.  The MANGA extension contains all MaNGA-lead galaxy observations.  The MASTAR extension contains all mini-bundles and MaStar (APOGEE-lead) targets of observation.  Marvin currently does not support MaStar observations.


2.3.0 (December 2018)
---------------------

.. todo:: Fix link to authentication.

We are excited to introduce Marvin 2.3.0, the first version of Marvin that provides support for public releases of MaNGA data. With this change, that coincides with the release of `SDSS DR15 <http://www.sdss.org>`__, Marvin ceases to be just an internal collaboration tool and becomes available to the whole astronomical and educational communities.

While Marvin now allows unrestricted access to DR15 data, it still supports access to proprietary MaNGA data (MPL-4 to MPL-7). This double access mode makes necessary the implementation of a new :ref:`authentication <marvin-authentication>` framework.

This version brings new and exciting features, such as an improved `~marvin.tools.rss.RSS` and a reimplemented `~marvin.tools.mixins.aperture.GetApertureMixIn.getAperture` method. We have also restructured (and in many cases rewritten) the `online documentation <http://sdss-marvin.readthedocs.io/en/stable/>`__; we hope this new structure will lower the learning curve for new users, and make advance features easily accessible to those who are more proficient. Finally, many (many) bugs have been squashed and we have implemented numerous small improvements. A full list is available in the :ref:`changelog <marvin-2.3.0>`.

The following subsections describe some of the major changes in detail. As always, full documentation for the new features is available.

Support for public data releases
********************************

Reimplemented `~marvin.tools.rss.RSS`
*************************************

The :ref:`RSS Tool <marvin-rss>` had never been a totally functional :ref:`Galaxy Tool <galaxy-tools>`. In this version we have completely refactored the `RSS class <marvin.tools.rss.RSS>` and it is now working great! Instances of `~marvin.tools.rss.RSS` consist of a list of `~marvin.tools.rss.RSSFiber` (basically a `~marvin.tools.quantities.spectrum.Spectrum` with additional attributes), one for each IFU fibre and observation. Observing information is available via the `~marvin.tools.rss.RSS.obsinfo` attribute. See the :ref:`documentation <marvin-rss>` for further details.

The new `~marvin.tools.rss.RSS` class is well tested but given the magnitude of the refactoring we are offering it in beta state. We appreciate your bug reports and any suggestions on how to improve it.

Extracting all the spaxels in an aperture
*****************************************

Early versions of Marvin included a ``Cube.getAperture`` method that allowed to extract the spaxels contained in a geometrical aperture. That feature was deemed not science-ready and removed in following releases. In this version we are reintroducing it as a mixin, `~marvin.tools.mixins.aperture.GetApertureMixIn`, that provides the `~marvin.tools.mixins.aperture.GetApertureMixIn.getAperture` method to Cubes, Maps, and ModelCubes. The mixin makes heavy use of `photutils <http://photutils.readthedocs.io/en/stable/>`_ to define geometric regions (elliptical, circular, rectangular) either in the image frame or using on-sky coordinates. Selecting the spaxels within a circular region around a set of coordinates is now as easy as doing ::

    >>> ap = cube.getAperture((232.546173, 48.6892288), 5, aperture_type='circular')
    >>> spaxels = ap.getSpaxels()

Full documentation is available :ref:`here <marvin-get-aperture>`. As with the `~marvin.tools.rss.RSS` class, please double check any result before using it for science publications. We welcome any feedback on how to improve this feature.

A new tool for handling MaNGA images
************************************

A new :ref:`Marvin Image <marvin-image>` Tool is now available for interacting with MaNGA optical images.  This tool utilizes the Marvin MMA system for easier access to images locally or remotely.  The new tool provides plotting as a Matplotlib figure, which includes the image WCS information, as well as options for overlaying individual IFU or sky fibers, and customizing the IFU hexagon.  You can also

  >>> from marvin.tools import Image
  >>> im = Image('8485-1901')

The old image utility functions documented :ref:`here <marvin-images>` remain but have now been deprecated by the new `~marvin.tools.image.Image` class.  Replacement utility functions are now available, with more information located :ref:`here<image-utils>`.

|

2.2.6 (August 2018)
-------------------

.. attention:: This is a critical bugfix release that corrects a problem that could affect your science results. Please update as soon as possible and check whether your analysis has been impacted by this bug.

This version fixes a critical bug when retrieving the spaxels associated with a bin, as well as a problem with the calculation of the inverse variance for deredden datacubes. It also simplifies the library namespace allowing for easier access to the most used Tools.

Critical bugfixes
*****************

Spaxels associated with a bin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In version 2.2 we introduced the concept of :ref:`Bin <marvin-bin>` as a collection of spaxels that belong to the same binning unit. As part of the API, one can use the `~marvin.tools.spaxel.Bin.spaxels` attribute to access a list of the spaxels that are included in the bin. The bug now fixed caused a list of incorrect spaxels to be associated with the bin, due to an inversion in the ``(x, y)`` order of the spaxels. *Before* 2.2.6 one would get ::

    >>> cube = Cube('8485-1901')
    >>> maps = cube.getMaps('HYB10')
    >>> bb = maps[22, 14]
    >>> bb.spaxels
    [<Marvin Spaxel (x=21, y=13, loaded=False),
     <Marvin Spaxel (x=21, y=14, loaded=False),
     <Marvin Spaxel (x=22, y=13, loaded=False),
     <Marvin Spaxel (x=22, y=14, loaded=False)]

where the x and y values should be

.. code-block:: console

    [<Marvin Spaxel (x=13, y=21, loaded=False),
     <Marvin Spaxel (x=14, y=21, loaded=False),
     <Marvin Spaxel (x=13, y=22, loaded=False),
     <Marvin Spaxel (x=14, y=22, loaded=False)]

Inverse variance for deredden datacubes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `~marvin.tools.quantities.datacube.DataCube` quantity includes a `~marvin.tools.quantities.datacube.DataCube.deredden` method that applies the reddening correction to the flux and inverse variance in the datacube. The inverse variance associated to the derredden flux had a bug in its calculation and was incorrect in all cases. That has now been fixed. It also fixes the spelling of ``deredden`` (😅).

MPL-7 now available
*******************

Starting with this release, Marvin provides access to MPL-7, the latest MaNGA Product Launch. Please read the `release notes (SDSS collaboration access required) <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-7/whatsnew>`__ for this version to understand what has changed. MPL-7 is now the default release when Marvin is imported.

In MPL-7 we have made ``HYB10`` the default bintype. Hybrid-binned ``Maps`` and ``ModelCubes`` use different binning schemes depending on the property measured. Before using ``HYB10`` make sure you read `the documentation <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-7/dap/GettingStarted#HYB10-GAU-MILESHC>`__ and understand how to use the data. For ``HYB10`` the `~marvin.tools.spaxel.Bin` class is somehow limited, since it does not allow for different binning schemes depending on the measured quantity. We are planning a major reimplementation of how bins are handled, which we will release with Marvin 2.3.0. In the meantime, please be aware of these limitations when using ``HYB10``.

Simplifying the namespace
**************************

Prior to 2.2.6 accessing different Tools classes was inconvenient since one would need to import them independently (e.g., ``from marvin.tools.cube import Cube``, ``from marvin.tools.maps import Maps``, etc.) This version makes access easier by exposing all the Tools from the ``marvin.tools`` namespace so that you can now do ::

    import marvin
    cube = marvin.tools.Cube('8485-1901')
    maps = marvin.tools.Maps('7443-12701')

Passing keyword arguments to `Spectrum.plot <marvin.tools.quantities.spectrum.Spectrum.plot>`
*********************************************************************************************

Extra arguments passed to `Spectrum.plot <marvin.tools.quantities.spectrum.Spectrum.plot>` are now redirected to `matplotlib.axes.Axes.plot`. This provides extra flexibility for your plots. For instance, you can now set labels for the legend associated with your plot ::

    ax = spectrum.plot(use_std=True, label='flux')
    ax.plot(spectrum.wavelength, model_flux, label='model')
    ax.legend()

Stellar Sigma Correction
************************

For MPL-6, we now raise an explicit error when attempting to apply the correction to ``stellar_sigma``, using the ``inst_sigma_correction`` method.  The error message now suggests to upgrade to MPL-7 data.  For the web display of the ``stellar_sigma`` and ``emline_gsigma`` maps, we now apply the sigma correction automatically.  The corrected map is indicated via **Corrected: stellar_sigma** map title.

|

2.2 (January 2018)
------------------

Marvin 2.2.0 brings significant improvements in the way you interact with MaNGA data.  Try the :ref:`Jupyter Notebook<marvin-exercises>` for a small sample.

* `MPL-6 <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-6>`_ compatible (SDSS collaboration access required)
* New DRP, DAP, and Query :ref:`Datamodels <marvin-datamodels>`
* :ref:`Cubes<marvin-cube>`, :ref:`Maps<marvin-maps>`, and :ref:`ModelCubes<marvin-modelcube>` now use Astropy Quantities, i.e. encapsulating a measurement with its associated parameters (e.g., unit, mask, or inverse variance)
* Improved Bin class
* Fuzzy Searching and Tab Completion
* New access to DAPall data on `Maps` and `ModelCubes`
* :ref:`Scatter <marvin-utils-plot-scatter>` and :ref:`Histogram <marvin-utils-plot-hist>` Plotting
* Improved Query :ref:`Results <marvin-results>` Handling and Integrated :ref:`Plotting <marvin-results_plot>`
* New :ref:`MaskBit <marvin-maskbit>` class

|

2.1.4 (August 2017)
-------------------

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

2.1.3 (May 2017)
----------------

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

* New tutorials: :doc:`tutorials/plotting-tutorial` and :doc:`tutorials/lean-tutorial`.

|

2.1 (February 2017)
-------------------

* Marvin is now minimally compliant with Python 3.5+

* `<https://sas.sdss.org/marvin>`_ now points to Marvin 2 (instead of Marvin 1).

* The NSA catalog information is now available via **Cube.nsa** in Marvin Cubes.

* Marvin :ref:`marvin-web` now has a new :ref:`galaxy_properties` tab with interactive scatter, and box-and-whisker plots.

* Marvin :ref:`marvin-web` has more python tips for working with Marvin :ref:`marvin-query-tools` objects.

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

2.0 Beta (November 2016)
------------------------

* Brand new painless installation (pip install sdss-marvin)

* New Marvin Tools (Maps, Bin, ModelCube)

* Pickling of Marvin Tools, Queries, and Results (i.e. local save and restore)

* DAP Spaxel ("Zonal") Queries

* Dynamic DAP Map display in the web, with point-and-click spaxel

* For MPL-5+, display of model fits in spectrum view in the web

* Versions simplified from mpl, drp, dap down to release

* :ref:`marvin-authentication`

|

2.0 Alpha (June 2016)
---------------------

Marvin 2.0 is a complete overhaul of Marvin 1.0, converting Marvin into a full suite of interaction tools.

Marvin 2.0 introduces two new modes of operations, :doc:`tools/index` and :doc:`api`, to the Marvin
environment, and introduces an extensive redesign of the `Marvin web app
<https://sas.sdss.org/marvin/>`_.

The major improvements and additions in this release:

* :doc:`core/data-access-modes`: a new mode based navigation system that allows you to seamlessly interact with MaNGA data no matter where it is.

* :doc:`../tools/index`: a python package for accessing and interacting with MaNGA
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
