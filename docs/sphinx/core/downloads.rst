
.. _marvin-download-objects:

===================
Downloading Objects
===================
Marvin allows you to download objects in several ways, when acting in **LOCAL** mode. Marvin downloads all objectsusing the SDSS Python package `sdss_access <https://github.com/sdss/sdss_access>`_.  When downloading files, Marvin places the files in your local SAS hierarchy as specified by the ``$SAS_BASE_DIR`` environment variable.

* **Via Config**:
    Setting the Marvin ``config.download`` attribute to ``True``.

* **Via Tools**:
    Initializing Marvin objects with the ``download=True`` flag.

* **Via Query Results**:
    Calling the download method from a set of query :class:`~marvin.tools.query.results.Results`.

* **Via Explicit Call**:
    Calling the :meth:`~marvin.utils.general.general.downloadList` utility function.


Download Authentication
-----------------------
Downloading with sdss_access requires authentication to the SAS, using a ``.netrc`` file placed in your local home directory.

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
----------

.. code-block:: python

    from marvin import config
    from marvin.tools.cube import Cube

    # set config attributes and turn on global downloads
    config.setRelease('MPL-5')
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
---------

.. code-block:: python

    from marvin import config
    from marvin.tools.cube import Cube
    config.mode = 'local'

    # instantiate Cube objects
    cc = Cube(plateifu='8485-1901', download=True)
    cc = Cube(mangaid='12-98126')


The cube for 8485-1901 will be explicitly downloaded but the cube for 12-98126 will not be.


Via Query Results
-----------------

.. code-block:: python

    from marvin.tools.query import Query

    # Make a query
    search_filter = 'nsa.z < 0.2'
    q = Query(search_filter=search_filter)

    # Run the query and retrieve the results
    r = q.run()

    # Download the results
    r.download()


All cubes from the query results will be downloaded and placed in their respective locations in your local SAS.


Via Explicit Call
-----------------
:meth:`~marvin.utils.general.general.downloadList` lets you download the files for cubes, images, maps, rss, mastar cubes, or the entire plate directory.

.. code-block:: python

    # Import the downloadList utility function
    from marvin import config
    from marvin.utils.general import downloadList
    config.setRelease('MPL-5')

    # Make a list of plate-IFUs
    gallist = ['8485-1901', '7443-12701']

    # Download cubes for the objects in your list
    downloadList(gallist, dltype='cube')


All cubes from your list will be downloaded and placed in their respective locations in your local SAS.

**Tip**: if you want to download all of the MaNGA Main Sample galaxies, check out the :doc:`../tutorials/sample-selection`.

|
