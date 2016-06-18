Data Access Modes
=================

.. _local-mode:

Local Mode
----------

In Local mode, :doc:`marvin_tools` will access data stored locally on your
machine. If you specify a file name, then it will open that file. Alternatively,
you can provide a MaNGA-ID or a plate-IFU identifier. Marvin will check to see
if you have a database (e.g., running :doc:`marvin_tools` at Utah) and use that
if possible. More likely, you will not have access to the database, so it will
look for a FITS file. If neither of those options is successful, then it will
download the requested data if automatic downloading is enabled.

Explicit Filename
::

    import marvin
    from marvin.tools.cube import Cube
    marvin.config.mode = 'local'

    # loads a cube from an explicit filepath
    cc = Cube(filename='/Users/Brian/mycubes/manga-8485-1901-LOGCUBE.fits.gz')

Local SAS
::

    import marvin
    from marvin.tools.cube import Cube
    marvin.config.mode = 'local'

    # checks for file in your local SAS. If found, loads it.  If not, may download it.
    cc = Cube(plateifu='8485-1901')

.. _remote-mode:

Remote Mode
-----------

In Remote mode, :doc:`marvin_tools` will retrieve the data remotely via the
:doc:`api`.  When dealing with Marvin Tools like Cubes or Spaxels, if a MaNGA-ID or a plate-IFU identifier
is provided, a remote API call is made to Marvin running at Utah, where it retrieves the data requested
and returns it as a JSON object.  Once the data has been acquired, the Marvin Tool object you are dealing with
will work as if you had the data locally.

::

    import marvin
    from marvin.tools.cube import Cube
    marvin.config.mode = 'remote'

    # grabs the necessary information from Utah, returns it to instantiate the Cube tool
    cc = Cube(plateifu='8485-1901')

.. _auto-mode:

Auto Mode
---------

In Auto mode, Marvin first tries `Local Mode`_, but if that attempt fails, it
automatically switches to `Remote Mode`_ .


Mode Decision Tree
------------------

|

.. image:: ../Mode_Decision_Tree.png
    :width: 800px
    :align: center
    :alt: Mode decision tree

|
