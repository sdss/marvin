
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

* **access**:
    This attribute informs Marvin of the type of access it has.  The allowed values are either "public" or "collab", for public and collaborationa access, respectively.  Public access provides access only to MaNGA public Data Releases, while collaboration access provides access to all MaNGA data release, DRs and MPLs.  The default value is **public**.

* **login**:
    This method

.. _marvin-modes:

Data Access Modes
=================
* **Local mode** - Use this mode to deal with local FITS files either in your local SAS or through explicit file locations.
* **Remote mode** - Use this mode to deal with data remotely.  The data is retrieved from Utah using the API.  Depending on your use,
  it may be returned as a JSON object or used to complete the Tool function you are using.
* **Auto mode** - Use this mode to have Marvin attempt to automatically handle the modes.  Marvin starts in this mode. It is recommended to leave Marvin in this mode, and let him handle your data access.
* See :ref:`marvin-dam` for more detailed information.

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
    config.setDR('DR15') # sets the global version to DR15

.. _marvin-access:

Public vs Collab Access
=======================

There are two access modes in Marvin, one for the public, and one for members of the SDSS collaboration.  The default access mode in Marvin is **public**.  Public access only provides access to MaNGA released in public Data Releases (DRs).

::

    # the default release with public access is the latest DR
    from marvin import config
    INFO: No release version set. Setting default to DR15

    # the data release is DR15
    config.release
    'DR15'

If you are a member of the SDSS collaboration and have properly set up your netrc authentication, you can switch access to **collab** and get access to MaNGA collaboration data, MPLs, as well as DRs.

::

    # switch to collaboration access
    from marvin import config
    config.access = 'collab'

    # switch to an MPL
    config.setRelease('MPL-7')

.. _marvin-api-login:

Logging In
==========

The Marvin API requires authentication with a token for access.  To receive a token, use the ``login`` method
::

    # to receive an API token
    config.login()

    # check you have a token
    config.token

A valid token lasts for 300 days.  When your token expires, you will need to login again to receive a new token.
::

    # receive a fresh token
    config.login(refresh=True)

Note, by default your token will disappear upon exiting your iPython terminal session.  You will need to login again within a new session.  To preserve your token between iPython sessions, copy your token into the **use_token** attribute of your custom Marvin config file.

Attempting to use Marvin with invalid credentials or an invalid token will produce the following error.
::

    # access a cube remotely without proper credentials
    cube = Cube('8485-1901', mode='remote')

    MarvinError: found a problem when checking if remote cube exists: API Authentication Error: Token has expired. Please check your token or login again for a fresh one.


.. _marvin_custom_yaml:

Marvin Custom Configuration File
================================

Most Marvin configuration options can be set using the Marvin ``config`` object within iPython.  You can also set configuration parameters using a custom YAML configuration file, ``marvin.yml``.  This file must be placed in the following directory in your HOME, ``~/.marvin/marvin.yml``.  Currently configurable options are:

* **check_access**:
    Set to **True** to have Marvin automatically check for proper netrc collaboration access and switch to **collab** access mode on startup.  Default is **False**.

* **use_sentry**:
    Set to **False** to disable Sentry error logging in Marvin.  Default is **True**.

* **add_github_message**:
    Set to **False** to disable the Github Issue message on all Marvin Errors.  Default is **True**.

* **use_token**:
    Set this value to your valid API token.  This ensures proper API authentication across iPython sessions.

* **default_release**:
    Set to to the release you want to use by default when importing Marvin. If set to **null**, uses the latest available version for your access mode. Default is **null**.
