
.. _marvin-config-info:

Configuration Class (marvin.config)
===================================
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

Data Access Modes
=================
* **Local mode** - Use this mode to deal with local FITS files either in your local SAS or through explicit file locations.
* **Remote mode** - Use this mode to deal with data remotely.  The data is retrieved from Utah using the API.  Depending on your use,
  it may be returned as a JSON object or used to complete the Tool function you are using.
* **Auto mode** - Use this mode to have Marvin attempt to automatically handle the modes.  Marvin starts in this mode. It is recommended to leave Marvin in this mode, and let him handle your data access.
* See :ref:`marvin-dma` for more detailed information.

.. _marvin-set-versions:

Setting Versions
================
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

