
.. _marvin-general:

General Info
============

This web page describes some of the core functionality of the Marvin python package.

.. _marvin-config-info:

Marvin Configuration Class
--------------------------
The Marvin :ref:`marvin-config-class` class controls the global configuration of Marvin.  It has
several attributes that control how you interact with MaNGA data.

* **mode**:
    The mode you want to run Marvin in. Mode can be either 'local', 'remote', or 'auto'. This determines how
    you interact with the MaNGA data.  Either through local FITS files on your own filesystem, or remotely using an
    API, if you do not have the data stored locally.  For more information, see :ref:`marvin-modes` below.

* **download**:
    Setting this attribute to True tells Marvin that whenever you don't have a file locally, to use
    sdss_access to download it via rsync.  See :ref:`marvin-download-objects` below.

* **release**:
    The Config class has a single attribute referencing the version of MaNGA data you are working with.
    **release** refers to either MaNGA Product Launches (MPLs), e.g. "MPL-5", or Data Releases (DRs), e.g. "DR13".  See :ref:`marvin-set-versions` below, to see how to set the version of the data you want.

* **use_sentry**:
    Marvin uses `Sentry <https://sentry.io>`_ to handle real-time logging of errors.  This setting toggles this feature.  The default value is set to **True**.  Set to **False** to disable.

* **add_github_message**:
    Marvin appends a message to every error instructing you on how to submit a new Github Issue regarding the error you just experienced.  If you wish to disable this message, set this value to **False**.  The default value is **True**.

* **db**:
    This attribute lets Marvin know if you have a database that it can be connected to.  If you have no database, this
    attribute will be set to None.  This attribute is set automatically and **you do not have to do anything with this attribute**.

* **sasurl**:
    This attribute tells Marvin what the SAS base url is when accessing the Marvin API. This attribute
    is set automatically and **you do not have to do anything with this attribute**.

* **urlmap**:
    This attribute is a lookup dictionary for all of the API routes that Marvin uses internally.
    This attribute is set automatically.  If you not using the Marvin API :ref:`marvin-api-interaction` class directly,
    then **you do not have to do anything with this attribute**.

.. _marvin-modes:

Marvin Modes
------------
* **Local mode** - Use this mode to deal with local FITS files either in your local SAS or through explicit file locations.
* **Remote mode** - Use this mode to deal with data remotely.  The data is retrieved from Utah using the API.  Depending on your use,
  it may be returned as a JSON object or used to complete the Tool function you are using.
* **Auto mode** - Use this mode to have Marvin attempt to automatically handle the modes.  Marvin starts in this mode. It is recommended to leave Marvin in this mode, and let him handle your data access.
* See :doc:`data-access-modes` for more detailed information.

.. _marvin-set-versions:

Setting Versions
----------------
You can globally control the version of the data you want to access using several convienence methods in the :ref:`marvin-config-class` class. Setting the release will also internally set up the appropriate DRP and DAP versions.

:py:meth:`~marvin.Config.setRelease`: main method to set the release version (e.g. MPL-5 or DR13).  Accepts either MPL or DR strings.

::

    from marvin import config
    config.setRelease('MPL-5')

You can also individually set MPLs or DRs separately with similar methods.

:py:meth:`~marvin.Config.setMPL`: set the version using the MPL designation (e.g. MPL-4)

::

    from marvin import config
    config.setMPL('MPL-4') # sets the global version to MPL-4

:py:meth:`~marvin.Config.setDR`: set the version using the DR designation (e.g. DR13)

::

    from marvin import config
    config.setDR('DR13') # sets the global version to DR13


.. _marvin-download-objects:

Downloading Objects
-------------------
Marvin allows you to download objects in several ways, when acting in **LOCAL** mode. Marvin downloads all objects
using the SDSS Python package **sdss_access**.  When downloading files, Marvin places the files in your local
SAS as specified by the ``$SAS_BASE_DIR`` environment variable.

* **Via Config**:
    Setting the Marvin config.download attribute to True

* **Via Tools**:
    Initializing Marvin objects with the download=True flag.

* **Via Query Results**:
    Calling the download method from a set of Query results

* **Via Explicit Call**:
    Calling the downloadList utility function

Download Authentication
^^^^^^^^^^^^^^^^^^^^^^^
Downloading with sdss_access requires authentication to the SAS, using a .netrc file placed in your local home directory.
::

    # create a .netrc file if you do not already have one
    cd ~/
    touch .netrc

    # using a text editor, place the following text inside your .netrc file.
    machine data.sdss.org
        login sdss
        password replace_with_sdss_password

**Note**: For API Authentication, please go to :ref:`marvin-authentication`

Via Config
^^^^^^^^^^
::

    from marvin import config
    from marvin.tools.cube import Cube

    # set config attributes and turn on global downloads
    config.setRelease('MPL-4')
    config.mode = 'local'
    config.download = True

    # instantiate Cube objects
    cc = Cube(plateifu='8485-1901')
    cc = Cube(mangaid='12-98126')

Both cubes will be downloaded and placed in
::

    $SAS_BASE_DIR/mangawork/manga/spectro/redux/v1_5_1/8485/stack/
    $SAS_BASE_DIR/mangawork/manga/spectro/redux/v1_5_1/7443/stack/

Via Tools
^^^^^^^^^^
::

    from marvin import config
    from marvin.tools.cube import Cube
    config.mode = 'local'

    # instantiate Cube objects
    cc = Cube(plateifu='8485-1901', download=True)
    cc = Cube(mangaid='12-98126')

The cube for 8485-1901 will be explicitly downloaded but the cube for 12-98126 will not be.

Via Query Results
^^^^^^^^^^^^^^^^^
::

    from marvin.tools.query import Query

    # Make a query
    searchfilter = 'nsa.z < 0.2'
    q = Query(searchfilter=searchfilter)

    # Run the query and retrieve the results
    r = q.run()

    # Download the results
    r.download()

All cubes from the query results will be downloaded and placed in their respective locations in your local SAS.

Via Explicit Call
^^^^^^^^^^^^^^^^^
**downloadList** lets you download the files for cubes, images, maps, rss, mastar cubes, or the entire plate directory.
::

    # Import the downloadList utility function
    from marvin import config
    from marvin.utils.general import downloadList
    config.setRelease('MPL-4')

    # Make a list of plate-IFUs
    gallist = ['8485-1901', '7443-12701']

    # Download cubes for the objects in your list
    downloadList(gallist, dltype='cube')

All cubes from your list will be downloaded and placed in their respective locations in your local SAS.

