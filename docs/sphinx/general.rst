
General Info
============

This page describes some of the core functionality of the Marvin python package.

Marvin Configuration Class
^^^^^^^^^^^^^^^^^^^^^^^^^^
The Marvin :ref:`marvin-config-class` class controls the global configuration of Marvin.  It has
several attributes that control how you interact with MaNGA data.

* **mode**:
    The mode you want to run Marvin in. Mode can be either 'local', 'remote', or 'auto'. This determines how
    you interact with the MaNGA data.  Either through local FITS files on your own filesystem, or remotely using an
    API, if you do not have the data stored locally.  For more information, see :ref:`marvin-modes` below.

* **download**:
    Setting this attribute to True tells Marvin that whenever you don't have a file locally, to use
    sdss_access to download it via rsync.  See :ref:`marvin-download-objects` below.

* **version attributes**:
    The Config class has three attributes referencing the version of MaNGA data you are working with: **mplver**,
    **drpver**, and **dapver**.  You can set these versions manually but to avoid versions conflicts, it is
    recommended to always use one of the conveneince functions provided.  See :ref:`marvin-set-versions` below.

* **db**:
    This attribute is lets Marvin know if you have a database that can be connected to.  If you have no database, this
    attribute will be set to None.

.. _marvin-modes:

Marvin Modes
^^^^^^^^^^^^
See :doc:`data-access-modes`

.. _marvin-set-versions:

Setting Versions
^^^^^^^^^^^^^^^^
You can globally control the version of the data you want to access using several convienence methods in the :ref:`marvin-config-class` class.

:py:meth:`~marvin.Config.setMPL`: set the version using the MPL designation (e.g. MPL-4)

::

    from marvin import config
    config.setMPL('MPL-4') # sets the global version to MPL-4; also sets the DRP and DAP versions to v1_5_1 and 1.1.1, respectively

:func:`setVersions <marvin.Config.setVersions>`: set the version using the DRP and DAP versions (e.g. v1_5_1, 1.1.1)

::

    from marvin import config
    config.setVersions(drpver='v1_5_1', dapver='1.1.1') # sets the global DRP and DAP versions; also sets the MPL version to MPL-4

.. _marvin-download-objects:

Downloading Objects
^^^^^^^^^^^^^^^^^^^
Marvin allows you to download objects in several ways, when acting in **LOCAL** mode.

* Setting the Marvin config.download attribute to True
* Initializing Marvin objects with the download=True flag.
* Calling the download method from a set of Query results

It places the file in your local sas specifed by the
    $SAS_BASE_DIR environment variables
