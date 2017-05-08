# Marvin's Change Log

## [2.1.4] - Under Development
### Added:
- Plotting tutorial.
- Easy multi-panel map plotting (with correctly placed colorbars).
- Uses matplotlib style sheets context managers for plotting (map, spectrum, and BPT) and restores previous defaults before methods finish.
- Map plotting accepts user-defined value, ivar, and/or mask.

### Changed:
- Moved map plotting methods from tools/map to utils/plot/map
- Switch to new gray/hatching scheme (in tools and web):
  - gray: spaxels with NOCOV.
  - hatched: spaxels with bad data (UNRELIABLE, MATHERROR, FITFAILED, DONOTUSE) or S/N below some minimum value.
  - colored: good data.
- Move plot defaults to datamodel (i.e., bitmasks, colormaps, percentile clips, symmetric, minimum SNR).
- SNR minimum to None (effectively 0) for velocity maps so that they aren't hatched near the zero velocity contour.
- Simplified default colormap name to "linearlab."
- Decreased map plot title font size in web so that it does not run onto second line and overlap plot.

### Fixed:
- Lean tutorial code cells did not work with the ipython directive, so they now use the python directive.
- Highcharts draggable legend cdn.

### Removed:
- Automatic setting of matplotlib style sheets via seaborn import or plt.style.use.


## [2.1.3] - Under Development
### Added:

### Changed:

### Fixed:
- Interactive prompt for username in sdss_access now works for Python 3.
- The data file for the default colormap for Map.plot() ("linearlab") is now included in pip version of Marvin and does not throw invalid FileNotFoundError if the data file is missing.


## [2.1.2] - 2017/03/17
### Added:
- API and Web argument validation using webargs and marshmallow.  If parameters invalid, returns 422 status.

### Changed:
- Per Issue #186: Switched to using the elpetro version of stellar mass, absolute magnitude i-band, and i-band
  mass-to-light ratio for NSA web display, from sersic values. (elpetro_logmass, elpetro_absmag_i, elpetro_mtol_i)
- Issue #188: deprecated snr in favour of snr_min for get_bpt. snr can still be used.
- Issue #187: Renamed NSA Display tab in web to Galaxy Properties.  Added a link to the NASA-Sloan Atlas catalogue to the table title.
- Moved our documentation to readthedocs for version control.  Updated all Marvin web documenation links to point to readthedocs.

### Fixed:
- A bug in the calculation of the composite mask for BPT.
- Issue #179: Fixed a python 2/3 exception error compatibility with the 2.1 release.

## [2.1.1] - 2017/02/18
### Added:
- Added query runtime output in search page html. And a warning if query is larger than 20 seconds.

### Changed:
- Removed the python 3 raise Exception in the check_marvin bin
- Reverted the api/query return output from jsonify back to json.dumps
    - This is an issue with python 2.7.3 namedtuple vs 2.7.11+

### Fixed:
- Issue #181: web display of maps were inverted; changed to xyz[jj, ii, val] in heatmap.js
- Added more code to handle MarvinSentry exceptions to fix #179.


## [2.1.0] - 2017/02/16
### Added:
- Restructured documentation index page.
- Improved installation documentation:
    - Removed old installation text
    - Added section on marvin SDSS dependencies and SAS_BASE_DIR
    - Added section for FAQ about installation
    - Added web browser cache issue into FAQ
- Added traceback info in the API calls
    - Added traceback attribute in Brain config
    - Added hidden _traceback attribute in Marvin config
    - Only implemented in two Query API calls at the moment
    - Added a few tests for traceback
    - see usage in cube_query in marvin/api/query.py
- Added the Ha_to_Hb ratio the DAP ModelClasses for querying
- Added new script to perform somce basic system, os, and Marvin checks: bin/check_marvin
- Added an alert banner when the user is using Safari. See #94.
- Issue #122: added ra/dec to spaxel
- Issue #145: Limited the number of query parameters in the web
- Added more tests to Results for sorting, paging, and getting subsets
- Added kwargs input for Spaxel when using Result.convertToTool
- Added automatic Sentry error logging #147 into MarvinError, and Sentry in Flask for production mode
- Added custom error handlers for the web, with potential user feedback form
- Added Sentry tool for grabbing and displaying Sentry statistics
- Added text to MarvinError with a Github Issues link and description of how to submit and issue
- Added Results option to save to CSV
- Added new parameters in Marvin Config to turn off Sentry error handling and Github Issue message
- Added Python example code for getting a spectrum in galaxy page of web.
- Added new test for image utilities getRandomImages, getImagesByPlate, getImagesByList
- Added new documentation on Image Utilities
- Added new image utility function showImage, which displays images from your local SAS
- Added the Kewley+06 implementation of the BPT classification as `Maps.get_bpt()`
- Added quick access to the NSA information for a Cube/Maps either from mangaSampleDB or drpall.

### Changed:
- When marvin is running from source (not dist), `marvin.__version__` is `dev`.
- Removed the cleanUpQueries method to assess db stability
- Switched dogpile.cache from using a file to python-memcached
- Syntax changes and bug fixes to get Marvin Web working when Marvin run on 3.5
- Got Queries and Results working in 3.5
- Changed all convertToTool options in Results from mangaid to plateifu
- Added release explicitly into api query routes
- Modified the decision tree in query to throw an error in local mode
- Modified convertToTool to accept a mode keyword
- Modifed the MarvinError for optional Sentry exception catching, and github issue inclusion
- Updated all Marvin tests to turn off Sentry exception catching and the github message
- Updated some of the Tools Snippets on the web
- Overhauled Map plotting
    - uses DAP bitmasks (NOVALUE, BADVALUE, MATHERROR, BADFIT, and DONOTUSE)
    - adds percentile and sigma clipping
    - adds hatching for regions with data (i.e., a spectrum) but no measurement by the DAP
    - adds Linear Lab color map
    - adds option for logarithmic colorbar
    - adds option to use sky coordinates
    - adds map property name as title
    - makes plot square
    - sets plotting defaults:
        - cmap is linear_Lab (sequential)
        - cmap is RdBu_r (diverging) for velocity plots (Note: this is reversed from the sense of the default coolwarm    colormap in v2.0---red for positive velocities and blue for negative velocities)
        - cmap is inferno (sequential) for sigma plots
        - clips at 5th and 95th percentiles
        - clips at 10th and 90th percentiles for velocity and sigma plots
        - velocity plots are symmetric about 0
        - uses DAP bitmasks NOVALUE, BADVALUE, MATHERROR, BADFIT, and DONOTUSE
        - also masks spaxels with ivar=0
        - minimum SNR is 1
- Changed Marvin Plate path back to the standard MarvinToolsClass use
- Made sdss_access somewhat more Python 3 compatible
- Modified the image utilities to return local paths in local/remote modes and url paths when as_url is True
- downloadList utility function now downloads images
- updated the limit-as parameter in the uwsgi ini file to 4096 mb from 1024 mb for production environment

### Fixed:
- Issue #115: drpall does not get updated when a tool sets a custom release.
- Issue #107: missing os library under save function of Map class
- Issue #117: hybrid colours were incorrect as they were being derived from petroth50_el.
- Issue #119: test_get_spaxel_no_db fails
- Issue #121: bugfix with misspelled word in downloadList utility function
- Issue #105: query results convertToTool not robust when null/default parameters not present
- Issue #136: BinTest errors when nose2 run in py3.5 and marvin server in 3.5
- Issue #137: PIL should work in py2.7 and py3.5
- Issue #172: broken mode=auto in image utilities
- Issue #158: version discrepancy in setup.py

## [2.0.9] - 2016/11/19
### Added:
- Docs now use `marvin.__version__`.

### Fixed:
- Fixed #100, #103: problem with getMap for properties without ivar.
- Fixed #101: problem with marvin query.


## [2.0.8] - 2016/11/18
### Fixed:
- Now really fixing #98


## [2.0.7] - 2016/11/18
### Fixed:
- Fixed issue #98


## [2.0.6] - 2016/11/17
### Fixed:
- Bug in Queries with dap query check running in remote mode.  Param form is empty.


## [2.0.5] - 2016/11/17
### Added:
- Added netrc configuration to installation documentation.
- Added netrc check on init.

### Fixed:
- Added mask to model spaxel.
- Bug in Cube tool when a galaxy loaded from db does not have NSA info; no failure with redshift
- Two bugs in index.py on KeyErrors: Sentry issues 181369719,181012809
- Bug on plate web page preventing meta-data from rendering
- Fixed installation in Python 3.
- Fixed long_description in setup.py to work with PyPI.
- Fixed a problem that made marvin always use the modules in extern


## [The dark ages] - multiple versions not logged.


## [1.90.0]
### Changed
- Full refactoring of Marvin 1.0
- Refactored web

### Added
- Marvin Tools
- Queries (only global properties, for now)
- Point-and-click for marvin-web
- RESTful API
- Many more changes

### Fixed
- Issue albireox/marvin#2: Change how matplotlib gets imported
