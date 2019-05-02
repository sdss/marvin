Marvin's Change Log
===================

[2.3.3] - unreleased
--------------------

Fixed
^^^^^
- Issue :issue:`658` - modelcube datamodel units missing angstrom


[2.3.2] - 2019/02/27
--------------------

Added
^^^^^
- Support for MPL-8

Fixed
^^^^^
- Issue :issue:`629` - web table of search results broken
- Issue :issue:`627` - web query displaying error message
- Issue :issue:`630` - broken web links on query page
- Issue :issue:`591` - broken query streaming return all results 

[2.3.1] - 2018/12/10
--------------------

Refactored
^^^^^^^^^^
- The entire Sphinx documentation


[2.3.0] - 2018/12/03
--------------------

Breaking changes
^^^^^^^^^^^^^^^^
- Removed ``Bin`` class. Bin information is now available on a per-quantity basis (:issue:`109`). See :ref:`What's new? <whats-new>` and documentation for details.
- Syntax on the inputs to the ``Query`` and ``Results`` tools has been changed.
- DAP spaxel queries have been disabled due to performance issues. We expect to bring them back soon. Metadata queries (those querying the NSA or DAPall tables) are still available.
- ``getSpaxel`` now only loads the quantities from the parent object (that means that, for example, ``Maps.getSpaxel`` only loads ``Maps`` properties by default). Additional quantities can be loaded using `~marvin.tools.spaxel.Spaxel.load`.
- ``getSpaxel`` accepted arguments have been changed to ``cube``, ``maps``, and ``modelcube``. The formerly accepted arguments (``drp``, ``properties``, ``model(s)``) now raise a deprecation error.

Added
^^^^^
- Added cheatsheet to docs.
- New Web authentication using Flask-Login
- New API authentication using Flask-JWT-Extended
- Adds MPL-7 / DR15 datamodel
- New config.access attribute indicating public or collab access
- New config.login method to handle token-based login
- New marvin.yml config file for customization of configuration options
- Adds User table into the history schema of mangadb.  Tracks user logins.
- ``Map`` has a new method ``getSpaxel`` to retrieve an ``Spaxel`` using the parent ``Maps``.
- New configuration option in ``marvin.yml``, ``default_release``, to set the release to be used when Marvin gets imported (:issue:`463`).
- Applying a NumPy ufunc (except ``np.log10()``) raises ``NotImplementedError`` because ivar propagation is not implemented yet.
- New ``Marvin Image`` Tool to load optical images using the MMA (:issue:`22`)
- New ``Bundle`` and ``Cutout`` utility classes
- New ``MMAMixIn`` for providing multi-modal data access
- ``qual_flag`` and ``pixmask_flag`` are now stored in the datamodel (:issue:`479,482`).
- ``Query`` tool now accepts a new ``targets`` and ``quality`` keyword argument which enables querying on target or quality maskbit labels. (:issue:`485`)
- Added a new API route for streaming large query results.  This uses a generator to stream large results back to the client to minimize memory usage and bulk responses.

Changed
^^^^^^^
- Integrated datamodel plotting params into actual datamodel structures
- Moved netrc checks into the Brain
- Expanded sasurl into public and collab urls
- Changes personal emails to sdss helpdesk email in web
- Added rawsql and return_params columns to history.query table in mangadb
- Extra keyword arguments passed to ``Spectrum.plot`` are now forwarded to ``Axes.plot``.
- Tools (e.g., ``Cube``, ``Maps``) can now be accessed from the ``marvin`` namespace (e.g., ``marvin.tools.Cube`` or ``marvin.tools.cube.Cube``).
- Map plotting ``ax_setup()`` function is now hidden.
- Moved ``yanny.py`` to ``extern/`` and added a readme file for the external packages (:issue:`468`).
- `~marvin.tools.quantities.Spectrum.plot` now only masks part of the spectrum that have the ``DONOTUSE`` maskbit set (:issue:`455`).
- ``pixmask`` is now available for all quantities (except ``AnalysisProprty``). The property ``masked`` now uses the bit ``DONOTUSE`` to determine what values must be masked out (:issue:`462`).
- Raises error when applying ``inst_sigma_correction`` on ``stellar_sigma`` MPL-6 maps.  Applies correction to stellar_sigma and emline_sigma for web maps with added 'Corrected' title (:issue:`478`)
- Removes targeting bits from ``Spaxel`` and ``Bin`` (:issue:`465`).
- The name of the channel is now shown in the ``Property`` description (:issue:`424`).
- Replaced inconsistent parameter ``model`` in `~marvin.tools.maps.Maps.getSpaxel`. Use ``models`` instead.
- MarvinError now accepts an optional `ignore_git` keyword to locally turn off the git addition to the message
- Using the `return_all` keyword in ``Query`` or `getAll` in ``Results`` now calls the streaming API route instead.
- When `~marvin.tool.cube.Cube` or `~marvin.tool.modelcube.ModelCube` are instantiated from file, gunzip the file to a temporary location to speed up subsequent access (:issue:`525`).
- Convert MMA warnings to debug messages (:issues:`580`).

Fixed
^^^^^
- Issue :issue:`421` - query returning spaxel props returns wrong total count
- Bugfix - Python 3 xrange syntax bug in buildImageDict
- ``Bin._create_spaxels`` instantiating spaxels with the wrong ``(i,j)`` values for the bin. The ``(i, j)`` values from the ``binid`` map were being swapped twice before sending them to ``SpaxelBase`` (:issue:`457`).
- A bug in the calculation of the deredden inverse variance in a `~marvin.tools.quantities.datacube.DataCube`.
- Issue with setting drpall path on initial import/set of release before tree has been planted
- Issue :issue:`456` - spectrum web display shows incorrect RA, Dec
- Issue :issue:`422` - ensuring config auto checks access to netrc
- Issue :issue:`423` - adding marvin.yml documentation
- Issue :issue:`431` - adding login documentation
- Issue :issue:`151` - adding web spectrum tooltips
- Issue :issue:`548` - failed to retrieve ModelCube extension in remote mode
- Fixed typo by in method name ``Spectrum.derredden -> Spectrum.deredden``.
- Fixed `#305 <https://github.com/sdss/marvin/issues/305>`_ - adding ivar propogation for np.log10(Map)
- A bug when explicitly returning default parameters in a query (:issue:`484`)
- Fixed `#510 <https://github.com/sdss/marvin/issues/510>`_ - fixes incorrect conversion to sky coordinates in map plotting.
- Fixed `#563 <https://github.com/sdss/marvin/issues/563>`_ - fail retrieving Query datamodels in Python 3.6+.
- Fixes bug with sasurl not properly being set to api.sdss.org on initial import
- Incorrect setting of the default bintype to download from web (:issue:`531`).
- Fixed :issue:`536`, :issue:`537`, :issue:`538`.  Added modelcube to downloadList.
- Incorrect mismatch warning between MPL-7 and DR15 (:issue:`495`).
- Incorrect handling of maskbits when the mask does not contain any of the bits in the list (:issue:`507`).
- Fixed :issue:`534` - flipped axes in NSA scatterplot when plotting absmag colors
- Fixed :issue:`559` - bug in check_marvin when marvindb is None
- Fixed :issue:`579` - bug in MMA with marvindb preventing files from opening
- Fixed :issue:`543`, :issue:`552`, :issue:`553` - bugs with various Query handlings
- Fixed :issue:`575` - cannot access maps due to bug in login and authentication in Interaction class
- Fixed :issue:`539` - print downloadList target directory
- Fixed :issue:`566` - made error message for web query with non-unique parameters name more specific

Refactored
^^^^^^^^^^
- Moved `marvin.core.core` to `marvin.tools.core` and split the mixins into `marvin.tools.mixins`.
- Reimplemented `~marvin.tools.mixins.aperture.GetApertureMixIn.getAperture` as a mixin using photutils apertures (:issue:`3,315`).
- Reimplemented `~marvin.tools.rss.RSS` as a list of `~marvin.tools.rss.RSSFiber` objects (:issue:`27,504`).
- Moved pieces of MarvinToolsClass into `marvin.tools.mixins`.
- Reimplemented `~marvin.tools.query.Query` to remove local query dependencies from remote mode usage.


[2.2.5] - 2018/04/26
--------------------

Changed
^^^^^^^
- Galaxy Web page spaxel loading to be robust when no modelspaxels are present in the database.


[2.2.4] - 2018/04/04
--------------------

Fixed
^^^^^
- Issue `#400 <https://github.com/sdss/marvin/issues/400>`_: SII in BPT diagram should use sum of 6717 and 6732.


[2.2.3] - 2018/03/20
--------------------

Added
^^^^^

- Added tests for `emline_gflux_ha_6564` and fixed values in galaxy_test_data.

Fixed
^^^^^

- Issue `#182 <https://github.com/sdss/marvin/issues/182>`_
- Issue `#202 <https://github.com/sdss/marvin/issues/202>`_
- Issue `#319 <https://github.com/sdss/marvin/issues/319>`_
- Issue `#322 <https://github.com/sdss/marvin/issues/322>`_
- Issue `#334 <https://github.com/sdss/marvin/issues/334>`_
- Issue `#339 <https://github.com/sdss/marvin/issues/339>`_
- Issue `#341 <https://github.com/sdss/marvin/issues/341>`_
- Issue `#342 <https://github.com/sdss/marvin/issues/342>`_
- Issue `#348 <https://github.com/sdss/marvin/issues/348>`_
- Issue `#352 <https://github.com/sdss/marvin/issues/352>`_
- Issue `#354 <https://github.com/sdss/marvin/issues/354>`_
- Issue `#355 <https://github.com/sdss/marvin/issues/355>`_
- Issue `#362 <https://github.com/sdss/marvin/issues/362>`_
- Issue `#366 <https://github.com/sdss/marvin/issues/366>`_
- Issue `#367 <https://github.com/sdss/marvin/issues/367>`_
- Issue `#368 <https://github.com/sdss/marvin/issues/368>`_
- Issue `#369 <https://github.com/sdss/marvin/issues/369>`_
- Issue `#372 <https://github.com/sdss/marvin/issues/372>`_
- Issue `#375 <https://github.com/sdss/marvin/issues/375>`_
- Issue `#378 <https://github.com/sdss/marvin/issues/378>`_
- Issue `#379 <https://github.com/sdss/marvin/issues/379>`_
- Issue `#383 <https://github.com/sdss/marvin/issues/383>`_
- Issue `#385 <https://github.com/sdss/marvin/issues/385>`_
- Issue `#386 <https://github.com/sdss/marvin/issues/386>`_
- Issue `#374 <https://github.com/sdss/marvin/issues/374>`_: Cube units do not persist under axis reordering.
- Fixed some problems with test_spaxel tests.
- Issue `#382 <https://github.com/sdss/marvin/issues/382>`_: Is fuzzywuzzy too fuzzy?
- Fixed an issue with Astropy 3 in `get_nsa_data()`.
- Fixed some issues with query results tests
- Issue `#391 <https://github.com/sdss/marvin/issues/391>`_
- Issue `#387 <https://github.com/sdss/marvin/issues/387>`_
- Issue `#384 <https://github.com/sdss/marvin/issues/384>`_
- Issue `#380 <https://github.com/sdss/marvin/issues/380>`_
- Issue `#376 <https://github.com/sdss/marvin/issues/376>`_
- Issue `#373 <https://github.com/sdss/marvin/issues/373>`_
- Issue `#371 <https://github.com/sdss/marvin/issues/371>`_
- Issue `#370 <https://github.com/sdss/marvin/issues/370>`_
- Issue `#363 <https://github.com/sdss/marvin/issues/363>`_
- Issue `#361 <https://github.com/sdss/marvin/issues/361>`_
- Issue `#360 <https://github.com/sdss/marvin/issues/360>`_
- Issue `#359 <https://github.com/sdss/marvin/issues/359>`_
- Issue `#358 <https://github.com/sdss/marvin/issues/358>`_
- Issue `#357 <https://github.com/sdss/marvin/issues/357>`_
- Issue `#353 <https://github.com/sdss/marvin/issues/353>`_
- Issue `#351 <https://github.com/sdss/marvin/issues/351>`_
- Issue `#349 <https://github.com/sdss/marvin/issues/349>`_
- Issue `#346 <https://github.com/sdss/marvin/issues/346>`_
- Issue `#345 <https://github.com/sdss/marvin/issues/345>`_
- Issue `#344 <https://github.com/sdss/marvin/issues/344>`_
- Issue `#343 <https://github.com/sdss/marvin/issues/343>`_
- Issue `#340 <https://github.com/sdss/marvin/issues/340>`_
- Issue `#337 <https://github.com/sdss/marvin/issues/337>`_
- Issue `#336 <https://github.com/sdss/marvin/issues/336>`_
- Issue `#335 <https://github.com/sdss/marvin/issues/335>`_
- Issue `#333 <https://github.com/sdss/marvin/issues/333>`_
- Issue `#331 <https://github.com/sdss/marvin/issues/331>`_
- Issue `#330 <https://github.com/sdss/marvin/issues/330>`_
- Issue `#329 <https://github.com/sdss/marvin/issues/329>`_
- Issue `#328 <https://github.com/sdss/marvin/issues/328>`_
- Issue `#327 <https://github.com/sdss/marvin/issues/327>`_
- Issue `#326 <https://github.com/sdss/marvin/issues/326>`_
- Issue `#325 <https://github.com/sdss/marvin/issues/325>`_
- Issue `#324 <https://github.com/sdss/marvin/issues/324>`_
- Issue `#320 <https://github.com/sdss/marvin/issues/320>`_
- Issue `#307 <https://github.com/sdss/marvin/issues/307>`_
- Issue `#395 <https://github.com/sdss/marvin/issues/395>`_
- Issue `#390 <https://github.com/sdss/marvin/issues/390>`_


Removed
^^^^^^^

- The banner that showed up in Safari has been removed since most versions should now work properly.


[2.2.2] - 2018/02/25
--------------------

Fixed
^^^^^

- MPL-6 issue with all H-alpha extensions mapped to NII instead.  Indexing issue in MPL-6 datamodel.
- MPL-6 issue with elliptical coordinate extensions;  missing R/Reff channel in MPL-6 datamodel.
- Issue `#324 <https://github.com/sdss/marvin/issues/324>`_
- Issue `#325 <https://github.com/sdss/marvin/issues/325>`_
- Issue `#326 <https://github.com/sdss/marvin/issues/326>`_
- Issue `#327 <https://github.com/sdss/marvin/issues/327>`_
- Issue `#330 <https://github.com/sdss/marvin/issues/330>`_
- Issue `#333 <https://github.com/sdss/marvin/issues/333>`_
- Issue `#335 <https://github.com/sdss/marvin/issues/335>`_
- Issue `#336 <https://github.com/sdss/marvin/issues/336>`_
- Issue `#343 <https://github.com/sdss/marvin/issues/343>`_
- Issue `#351 <https://github.com/sdss/marvin/issues/351>`_
- Issue `#353 <https://github.com/sdss/marvin/issues/353>`_
- Issue `#357 <https://github.com/sdss/marvin/issues/357>`_
- Issue `#358 <https://github.com/sdss/marvin/issues/358>`_
- Issue `#360 <https://github.com/sdss/marvin/issues/360>`_
- Issue `#363 <https://github.com/sdss/marvin/issues/363>`_
- Issue `#373 <https://github.com/sdss/marvin/issues/373>`_


[2.2.1] - 2018/01/12
--------------------

Fixed
^^^^^

- bugfix in MPL-6 datamodel for gew OII lines

[2.2.0] - 2018/01/12
--------------------

Added
^^^^^

-  Added ``Maskbit`` class for easy conversion between mask values, bits, and
   labels.
-  Better BPT documentation, in particular in the ``Modifying the plot``
   section.
-  A hack function ``marvin.utils.plot.utils.bind_to_figure()`` that
   replicate the contents of a matplotlib axes in another figure.
-  New scatter and histogram plotting utility functions
-  Integrated scatter and histogram plotting into query Results
-  New methods for easier query Results handling
-  New Pythonic DRP, DAP, and Query DataModels
-  Access to DAPall data

Changed
^^^^^^^

-  Issue `#190 <https://github.com/sdss/marvin/issues/190>`_: ``Maps.get_bpt()`` and
   ``marvin.utils.dap.bpt.bpt_kewley06()`` now also return a list of
   axes. Each axes contains a method pointing to the
   ``marvin.utils.plot.utils.bind_to_figure()`` function, for easily
   transfer the axes to a new figure.
-  All Cubes, Maps, and Modelcubes use Astropy Quantities
-  Refactored to the Bin class
-  Bin and Spaxel are now subclassed from SpaxelBase

Fixed
^^^^^

- Issue `#24 <https://github.com/sdss/marvin/issues/24>`_
- Issue `#99 <https://github.com/sdss/marvin/issues/99>`_
- Issue `#110 <https://github.com/sdss/marvin/issues/110>`_
- Issue `#111 <https://github.com/sdss/marvin/issues/111>`_
- Issue `#131 <https://github.com/sdss/marvin/issues/131>`_
- Issue `#133 <https://github.com/sdss/marvin/issues/133>`_
- Issue `#173 <https://github.com/sdss/marvin/issues/173>`_
- Issue `#178 <https://github.com/sdss/marvin/issues/178>`_
- Issue `#180 <https://github.com/sdss/marvin/issues/180>`_
- Issue `#190 <https://github.com/sdss/marvin/issues/190>`_
- Issue `#191 <https://github.com/sdss/marvin/issues/191>`_
- Issue `#233 <https://github.com/sdss/marvin/issues/233>`_
- Issue `#235 <https://github.com/sdss/marvin/issues/235>`_
- Issue `#246 <https://github.com/sdss/marvin/issues/246>`_
- Issue `#248 <https://github.com/sdss/marvin/issues/248>`_
- Issue `#261 <https://github.com/sdss/marvin/issues/261>`_
- Issue `#263 <https://github.com/sdss/marvin/issues/263>`_
- Issue `#269 <https://github.com/sdss/marvin/issues/269>`_
- Issue `#279 <https://github.com/sdss/marvin/issues/279>`_
- Issue `#281 <https://github.com/sdss/marvin/issues/281>`_
- Issue `#286 <https://github.com/sdss/marvin/issues/286>`_
- Issue `#287 <https://github.com/sdss/marvin/issues/287>`_
- Issue `#290 <https://github.com/sdss/marvin/issues/290>`_
- Issue `#291 <https://github.com/sdss/marvin/issues/291>`_
- Issue `#294 <https://github.com/sdss/marvin/issues/294>`_
- Issue `#295 <https://github.com/sdss/marvin/issues/295>`_
- Issue `#296 <https://github.com/sdss/marvin/issues/296>`_
- Issue `#297 <https://github.com/sdss/marvin/issues/297>`_
- Issue `#299 <https://github.com/sdss/marvin/issues/299>`_
- Issue `#301 <https://github.com/sdss/marvin/issues/301>`_
- Issue `#302 <https://github.com/sdss/marvin/issues/302>`_
- Issue `#303 <https://github.com/sdss/marvin/issues/303>`_
- Issue `#304 <https://github.com/sdss/marvin/issues/304>`_
- Issue `#308 <https://github.com/sdss/marvin/issues/308>`_
- Issue `#311 <https://github.com/sdss/marvin/issues/311>`_
- Issue `#312 <https://github.com/sdss/marvin/issues/312>`_


[2.1.4] - 2017/08/02
--------------------

Added
^^^^^

-  Added new query_params object, for easier navigation of available
   query parameters. Added new tests.
-  Added a new guided query builder using Jquery Query Builder to the
   Search page
-  Added a View Galaxies link on the web results to view postage stamps
   of the galaxies in the results
-  Added Route Rate Limiting. Adopts a limit of 200/min for all api
   routes and 60/minute for query api calls and web searches

Changed
^^^^^^^

-  Changed call signature for
   :meth:``marvin.utils.plot.map.no_coverage_mask`` (removed ``value``
   arg because unused, added ``None`` as default value ``ivar``
   (``None``), and re-ordered args and kwargs).
-  Changed call signature for
   :meth:``marvin.utils.plot.map.bad_data_mask`` (removed ``value`` arg
   because unused).
-  Changed the Marvin web search page to use the new query_params and
   parameter grouping. Removed the autocomplete input box.
-  Updated the documentation on query and query_params.
-  Modified Guided Search operator options to remove options that could
   not be parsed by SQLA boolean_search
-  Refactored the web settings, route registration, extensions to enable
   extensibility
-  Issue `#282 <https://github.com/sdss/marvin/issues/282>`_: Improvements to "Go to CAS" link. Changed to Go To
   SkyServer and updated link to public up-to-date link

Fixed
^^^^^

-  Issue `#102 <https://github.com/sdss/marvin/issues/102>`_: problem with urllib package when attempting to retrieve
   the Marvin URLMap
-  Issue `#93 <https://github.com/sdss/marvin/issues/93>`_: safari browser does not play well with marvin
-  Issue `#155 <https://github.com/sdss/marvin/issues/155>`_: Contrails in Web Map
-  Issue `#174 <https://github.com/sdss/marvin/issues/174>`_: sdss_access may not be completely python 3 compatible
-  Issue `#196 <https://github.com/sdss/marvin/issues/196>`_: Bin not loading from local sas
-  Issue `#207 <https://github.com/sdss/marvin/issues/207>`_: Get Maps in MapSpecView of Galaxy page sometimes fails to
   return selected maps
-  Issue `#210 <https://github.com/sdss/marvin/issues/210>`_: pip upgrade may not install new things as fresh install
-  Issue `#209 <https://github.com/sdss/marvin/issues/209>`_: marvin version from pip install is incorrect
-  Issue `#268 <https://github.com/sdss/marvin/issues/268>`_: Cube flux from file error
-  Issue `#85 <https://github.com/sdss/marvin/issues/85>`_: Python does not start in Python 3
-  Issue `#273 <https://github.com/sdss/marvin/issues/273>`_: ha.value bug
-  Issue `#277 <https://github.com/sdss/marvin/issues/277>`_: Ticks for log normalized colorbar
-  Issue `#275 <https://github.com/sdss/marvin/issues/275>`_: logger crashes on warning when other loggers try to log
-  Issue `#258 <https://github.com/sdss/marvin/issues/258>`_: 422 Invalid Parameters
-  Issue `#271 <https://github.com/sdss/marvin/issues/271>`_: Problem in dowloading image.
-  Issue `#97 <https://github.com/sdss/marvin/issues/97>`_: sqlalchemy-boolean-search not found when installed from
   pip source
-  Issue `#227 <https://github.com/sdss/marvin/issues/227>`_: Marvin installation in python 3.6 (update setuptools to
   36)
-  Issue `#262 <https://github.com/sdss/marvin/issues/262>`_: problem with marvin update
-  Issue `#270 <https://github.com/sdss/marvin/issues/270>`_: BPT array sizing not compatible
-  Issue `#88 <https://github.com/sdss/marvin/issues/88>`_: Deployment at Utah requires automatisation
-  Issue `#234 <https://github.com/sdss/marvin/issues/234>`_: Add (and use) functions to the datamodel to determine
   plotting parameters
-  Issue `#278 <https://github.com/sdss/marvin/issues/278>`_: marvin_test_if decorator breaks in python 2.7
-  Issue `#274 <https://github.com/sdss/marvin/issues/274>`_: cube slicing to get a spaxel fails with maps error
-  Issue `#39 <https://github.com/sdss/marvin/issues/39>`_: implement more complete testing framework
-  Issue `#242 <https://github.com/sdss/marvin/issues/242>`_: Result object representation error with 0 query results
-  Issue `#159 <https://github.com/sdss/marvin/issues/159>`_: Marvin issues multiple warnings in PY3
-  Issue `#149 <https://github.com/sdss/marvin/issues/149>`_: Improve integrated flux maps display in web


[2.1.3] - 2017/05/18
--------------------

Added
^^^^^

-  Issue `#204 <https://github.com/sdss/marvin/issues/204>`_: added elpetro_absmag colours to mangaSampleDB models.
-  Issue `#253 <https://github.com/sdss/marvin/issues/253>`_: Plotting tutorial.
-  Issue `#223 <https://github.com/sdss/marvin/issues/223>`_: Easy multi-panel map plotting (with correctly placed
   colorbars).
-  Issue #232 and Issue `#251 <https://github.com/sdss/marvin/issues/251>`_: Uses matplotlib style sheets context
   managers for plotting (map, spectrum, and BPT) and restores previous
   defaults before methods finish.
-  Issue `#189 <https://github.com/sdss/marvin/issues/189>`_: Map plotting accepts user-defined value, ivar, and/or
   mask (including BPT masks).
-  Issue `#252 <https://github.com/sdss/marvin/issues/252>`_: Quantile clipping for properties other than velocity,
   sigma, or flux in web.
-  Added ``utils.plot.map`` doc page.
-  Added ``tools.map`` doc page.

Changed
^^^^^^^

-  Issue `#243 <https://github.com/sdss/marvin/issues/243>`_: inverted ``__getitem__`` behaviour for
   Cube/Maps/ModelCube and fixed tests.
-  Modified Flask Profiler File to always point to
   $MARVIN_DIR/flask_profiler.sql
-  Issue `#241 <https://github.com/sdss/marvin/issues/241>`_: Moved map plotting methods from tools/map to
   utils/plot/map
-  Issue #229 and Issue `#231 <https://github.com/sdss/marvin/issues/231>`_: Switch to new gray/hatching scheme (in
   tools and web):

   -  gray: spaxels with NOCOV.
   -  hatched: spaxels with bad data (UNRELIABLE and DONOTUSE) or S/N
      below some minimum value.
   -  colored: good data.

-  Issue `#238 <https://github.com/sdss/marvin/issues/238>`_: Move plot defaults to datamodel (i.e., bitmasks,
   colormaps, percentile clips, symmetric, minimum SNR).
-  Issue `#206 <https://github.com/sdss/marvin/issues/206>`_: SNR minimum to None (effectively 0) for velocity maps so
   that they aren't hatched near the zero velocity contour.
-  Simplified default colormap name to "linearlab."
-  Decreased map plot title font size in web so that it does not run
   onto second line and overlap plot.

Fixed
^^^^^

-  Interactive prompt for username in sdss_access now works for Python
   3.
-  Fixed `#195 <https://github.com/sdss/marvin/issues/195>`_: The data file for the default colormap for ``Map.plot()``
   ("linear_Lab") is now included in pip version of Marvin and does not
   throw invalid ``FileNotFoundError`` if the data file is missing.
-  Fixed `#143 <https://github.com/sdss/marvin/issues/143>`_: prevents access mode to go in to remote if filename is
   present.
-  Fixed `#213 <https://github.com/sdss/marvin/issues/213>`_: shortcuts are now only applied on full words, to avoid
   blind replacements.
-  Fixed `#206 <https://github.com/sdss/marvin/issues/206>`_: no longer masks spaxels close to zero velocity contour in
   web and tools map plots
-  Fixed `#229 <https://github.com/sdss/marvin/issues/229>`_: corrects web bitmask parsing for map plots
-  Fixed `#231 <https://github.com/sdss/marvin/issues/231>`_: hatch regions within IFU but without data in map plots
-  Fixed `#255 <https://github.com/sdss/marvin/issues/255>`_: Lean tutorial code cells did not work with the ipython
   directive, so they now use the python directive.
-  Highcharts draggable legend cdn.

Removed
^^^^^^^

-  Issue #232 and Issue `#251 <https://github.com/sdss/marvin/issues/251>`_: Automatic setting of matplotlib style
   sheets via seaborn import or ``plt.style.use()``.


[2.1.2] - 2017/03/17
--------------------

Added
^^^^^

-  API and Web argument validation using webargs and marshmallow. If
   parameters invalid, returns 422 status.

Changed
^^^^^^^

-  Per Issue `#186 <https://github.com/sdss/marvin/issues/186>`_: Switched to using the elpetro version of stellar
   mass, absolute magnitude i-band, and i-band mass-to-light ratio for
   NSA web display, from sersic values. (elpetro_logmass,
   elpetro_absmag_i, elpetro_mtol_i)
-  Issue `#188 <https://github.com/sdss/marvin/issues/188>`_: deprecated snr in favour of snr_min for get_bpt. snr can
   still be used.
-  Issue `#187 <https://github.com/sdss/marvin/issues/187>`_: Renamed NSA Display tab in web to Galaxy Properties.
   Added a link to the NASA-Sloan Atlas catalogue to the table title.
-  Moved our documentation to readthedocs for version control. Updated
   all Marvin web documenation links to point to readthedocs.

Fixed
^^^^^

-  A bug in the calculation of the composite mask for BPT.
-  Issue `#179 <https://github.com/sdss/marvin/issues/179>`_: Fixed a python 2/3 exception error compatibility with the
   2.1 release.


[2.1.1] - 2017/02/18
--------------------

Added
^^^^^

-  Added query runtime output in search page html. And a warning if
   query is larger than 20 seconds.

Changed
^^^^^^^

-  Removed the python 3 raise Exception in the check_marvin bin
-  Reverted the api/query return output from jsonify back to json.dumps

   -  This is an issue with python 2.7.3 namedtuple vs 2.7.11+

Fixed
^^^^^

-  Issue `#181 <https://github.com/sdss/marvin/issues/181>`_: web display of maps were inverted; changed to xyz[jj, ii,
   val] in heatmap.js
-  Added more code to handle MarvinSentry exceptions to fix #179.


[2.1.0] - 2017/02/16
--------------------

Added
^^^^^

-  Restructured documentation index page.
-  Improved installation documentation:

   -  Removed old installation text
   -  Added section on marvin SDSS dependencies and SAS_BASE_DIR
   -  Added section for FAQ about installation
   -  Added web browser cache issue into FAQ

-  Added traceback info in the API calls

   -  Added traceback attribute in Brain config
   -  Added hidden \_traceback attribute in Marvin config
   -  Only implemented in two Query API calls at the moment
   -  Added a few tests for traceback
   -  see usage in cube_query in marvin/api/query.py

-  Added the Ha_to_Hb ratio the DAP ModelClasses for querying
-  Added new script to perform somce basic system, os, and Marvin
   checks: bin/check_marvin
-  Added an alert banner when the user is using Safari. See #94.
-  Issue `#122 <https://github.com/sdss/marvin/issues/122>`_: added ra/dec to spaxel
-  Issue `#145 <https://github.com/sdss/marvin/issues/145>`_: Limited the number of query parameters in the web
-  Added more tests to Results for sorting, paging, and getting subsets
-  Added kwargs input for Spaxel when using Result.convertToTool
-  Added automatic Sentry error logging #147 into MarvinError, and
   Sentry in Flask for production mode
-  Added custom error handlers for the web, with potential user feedback
   form
-  Added Sentry tool for grabbing and displaying Sentry statistics
-  Added text to MarvinError with a Github Issues link and description
   of how to submit and issue
-  Added Results option to save to CSV
-  Added new parameters in Marvin Config to turn off Sentry error
   handling and Github Issue message
-  Added Python example code for getting a spectrum in galaxy page of
   web.
-  Added new test for image utilities getRandomImages, getImagesByPlate,
   getImagesByList
-  Added new documentation on Image Utilities
-  Added new image utility function showImage, which displays images
   from your local SAS
-  Added the Kewley+06 implementation of the BPT classification as
   ``Maps.get_bpt()``
-  Added quick access to the NSA information for a Cube/Maps either from
   mangaSampleDB or drpall.

Changed
^^^^^^^

-  When marvin is running from source (not dist), ``marvin.__version__``
   is ``dev``.
-  Removed the cleanUpQueries method to assess db stability
-  Switched dogpile.cache from using a file to python-memcached
-  Syntax changes and bug fixes to get Marvin Web working when Marvin
   run on 3.5
-  Got Queries and Results working in 3.5
-  Changed all convertToTool options in Results from mangaid to plateifu
-  Added release explicitly into api query routes
-  Modified the decision tree in query to throw an error in local mode
-  Modified convertToTool to accept a mode keyword
-  Modifed the MarvinError for optional Sentry exception catching, and
   github issue inclusion
-  Updated all Marvin tests to turn off Sentry exception catching and
   the github message
-  Updated some of the Tools Snippets on the web
-  Overhauled Map plotting

   -  uses DAP bitmasks (NOVALUE, BADVALUE, MATHERROR, BADFIT, and
      DONOTUSE)
   -  adds percentile and sigma clipping
   -  adds hatching for regions with data (i.e., a spectrum) but no
      measurement by the DAP
   -  adds Linear Lab color map
   -  adds option for logarithmic colorbar
   -  adds option to use sky coordinates
   -  adds map property name as title
   -  makes plot square
   -  sets plotting defaults:

      -  cmap is linear_Lab (sequential)
      -  cmap is RdBu_r (diverging) for velocity plots (Note: this is
         reversed from the sense of the default coolwarm colormap in
         v2.0---red for positive velocities and blue for negative
         velocities)
      -  cmap is inferno (sequential) for sigma plots
      -  clips at 5th and 95th percentiles
      -  clips at 10th and 90th percentiles for velocity and sigma plots
      -  velocity plots are symmetric about 0
      -  uses DAP bitmasks NOVALUE, BADVALUE, MATHERROR, BADFIT, and
         DONOTUSE
      -  also masks spaxels with ivar=0
      -  minimum SNR is 1

-  Changed Marvin Plate path back to the standard MarvinToolsClass use
-  Made sdss_access somewhat more Python 3 compatible
-  Modified the image utilities to return local paths in local/remote
   modes and url paths when as_url is True
-  downloadList utility function now downloads images
-  updated the limit-as parameter in the uwsgi ini file to 4096 mb from
   1024 mb for production environment

Fixed
^^^^^

-  Issue `#115 <https://github.com/sdss/marvin/issues/115>`_: drpall does not get updated when a tool sets a custom
   release.
-  Issue `#107 <https://github.com/sdss/marvin/issues/107>`_: missing os library under save function of Map class
-  Issue `#117 <https://github.com/sdss/marvin/issues/117>`_: hybrid colours were incorrect as they were being derived
   from petroth50_el.
-  Issue `#119 <https://github.com/sdss/marvin/issues/119>`_: test_get_spaxel_no_db fails
-  Issue `#121 <https://github.com/sdss/marvin/issues/121>`_: bugfix with misspelled word in downloadList utility
   function
-  Issue `#105 <https://github.com/sdss/marvin/issues/105>`_: query results convertToTool not robust when null/default
   parameters not present
-  Issue `#136 <https://github.com/sdss/marvin/issues/136>`_: BinTest errors when nose2 run in py3.5 and marvin server
   in 3.5
-  Issue `#137 <https://github.com/sdss/marvin/issues/137>`_: PIL should work in py2.7 and py3.5
-  Issue `#172 <https://github.com/sdss/marvin/issues/172>`_: broken mode=auto in image utilities
-  Issue `#158 <https://github.com/sdss/marvin/issues/158>`_: version discrepancy in setup.py


[2.0.9] - 2016/11/19
--------------------

Added
^^^^^

-  Docs now use ``marvin.__version__``.

Fixed
^^^^^

-  Fixed #100, `#103 <https://github.com/sdss/marvin/issues/103>`_: problem with getMap for properties without ivar.
-  Fixed `#101 <https://github.com/sdss/marvin/issues/101>`_: problem with marvin query.


[2.0.8] - 2016/11/18
--------------------

Fixed
^^^^^

-  Now really fixing #98

.. 207---20161118:


[2.0.7] - 2016/11/18
--------------------

Fixed
^^^^^

-  Fixed issue #98


[2.0.6] - 2016/11/17
--------------------

Fixed
^^^^^

-  Bug in Queries with dap query check running in remote mode. Param
   form is empty.


[2.0.5] - 2016/11/17
--------------------

Added
^^^^^

-  Added netrc configuration to installation documentation.
-  Added netrc check on init.

Fixed
^^^^^

-  Added mask to model spaxel.
-  Bug in Cube tool when a galaxy loaded from db does not have NSA info;
   no failure with redshift
-  Two bugs in index.py on KeyErrors: Sentry issues 181369719,181012809
-  Bug on plate web page preventing meta-data from rendering
-  Fixed installation in Python 3.
-  Fixed long_description in setup.py to work with PyPI.
-  Fixed a problem that made marvin always use the modules in extern

.. the-dark-ages---multiple-versions-not-logged:

[The dark ages] - multiple versions not logged.
-----------------------------------------------

[1.90.0]
--------

Changed
^^^^^^^

-  Full refactoring of Marvin 1.0
-  Refactored web

Added
^^^^^

-  Marvin Tools
-  Queries (only global properties, for now)
-  Point-and-click for marvin-web
-  RESTful API
-  Many more changes

Fixed
^^^^^

-  Issue albireox/marvin#2: Change how matplotlib gets imported.
