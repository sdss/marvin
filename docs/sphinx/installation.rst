
|

.. admonition:: Warning
    :class: warning

    Marvin does not work well with the system Python in OSX.
    Please, make sure you are using a supported Python installation before
    following these instructions. Good installations include
    `Anaconda <https://www.continuum.io/downloads>`_,
    `Miniconda <http://conda.pydata.org/miniconda.html>`_, or
    `homebrew <http://brew.sh/>`_. After installing one of these distribution,
    make sure you are actually using it by running ``which python`` and ``which pip``.

|


.. _marvin-installation:

Installation
============

.. _marvin-install-quick:

Quick Install
-------------

To quickly install Marvin, use::

  pip install sdss-marvin

If you are using an Anaconda distribution of Python, you may use the following available ``conda`` environment,
`here <https://anaconda.org/SDSS/marvin/files>`_. Once downloaded, set up the virtual environment with::

  conda env create -f marvin_2.5.0.yml

.. _marvin-install-dev:

Developers Installation
-----------------------

To develop for marvin, follow these instructions::

    git clone https://github.com/sdss/marvin
    cd marvin
    git submodule update --init --recursive
    pip install -r requirements_dev.txt
    python setup.py develop

This will checkout the repository, set up the git submodule dependencies, and install marvin into your python path.  You will only need to run this once.  Afterwards, you can start developing for Marvin.

.. _marvin-install-auth:

Access and Authentication
-------------------------

Public Access
^^^^^^^^^^^^^

By default Marvin is set up to run in `public` access mode, with the latest SDSS public data release, e.g. DR15.  You can check your access from within an ``iPython`` terminal by typing::

  from marvin import config
  config.access

A ``config.access`` of **public** means you are set up for public access only.  In this mode, you only have access to publically available data.  A ``config.access`` of **collab** indicates you are set up for SDSS collaboration proprietary data access.


.. _sdss-collaboration-access:

SDSS Collaboration Access
^^^^^^^^^^^^^^^^^^^^^^^^^

For SDSS collaboration members, authentication is required to access proprietary collaboration data, and Marvin must have ``config.access`` set to **collab**.  See :ref:`more here <marvin-access>`. To set up authentication for Marvin, you must perform the following

.. _setup-netrc:

Set up your netrc
~~~~~~~~~~~~~~~~~

SDSS uses ``.netrc`` authentication to access data content on many domains. To set this up, create and edit a file in your
home called ``.netrc`` an copy these lines inside::

    machine api.sdss.org
       login <username>
       password <password>

    machine data.sdss.org
       login <username>
       password <password>

and replace ``<username>`` and ``<password>`` with your login credentials. The default SDSS username and password is also
acceptable for anonymous access.  **Finally, run** ``chmod 600 ~/.netrc`` **to make the file only accessible to your user.**

.. _api-token-auth:

API Token Authentication
~~~~~~~~~~~~~~~~~~~~~~~~

Marvin requires token authentication to grant access and use of its API.  Marvin uses the standard `JSON Web Tokens <https://jwt.io/introduction/>`_ for token authentication.  To receive a valid token, you must :ref:`login <marvin-api-login>` with your valid SDSS credentials, via the ``.netrc``.  With your ``netrc`` access in place, you will receive a valid API token.  Tokens remain valid for 300 days.::

  # login to receive a token
  config.login()

  # see token
  config.token


.. _auto-login:

Automatically Logging In
~~~~~~~~~~~~~~~~~~~~~~~~

As the default mode of marvin is **public**, you will need to authenticate and change to **collab** access inside every new ``iPython`` session.  To simplify this process, marvin can be configured to automatically perform the access and authentication checks.  To configure marvin, you must set up a :ref:`custom marvin configuration file <marvin_custom_yaml>`.  Inside a ``~/.marvin/marvin.yml`` file, set the following lines::

  check_access: True
  use_token: [token]

You can replace **[token]** with your authenticated API JSON token (without any string quotes).  Upon import of marvin, Marvin will check for valid credentials and automatically set up your collaboration access.

.. _marvin-environment:

Marvin Environment
------------------

Marvin requires a certain environment structure to access and (optionally) download data.  By default,
marvin will look for data files in a directory structure that mirrors the
`Science Archive Server <https://data.sdss.org/sas>`_. :ref:`Data downloaded via marvin <marvin-download-objects>` will
also be stored according to that structure. The root of this directory structure is
defined by the environment variable  ``$SAS_BASE_DIR``. For example, if marvin needs
to use the ``drpall`` file for DR15, it will try to find it in
``$SAS_BASE_DIR/dr15/manga/spectro/redux/v2_4_3/drpall-v2_4_3.fits``.

The Marvin environment structure is as follows::

  ======================   ==============================================   ======
  Environment Variable     Default Path                                     Access
  ======================   ==============================================   ======
  SAS_BASE_DIR             $HOME/sas
  MANGA_SPECTRO_REDUX      $SAS_BASE_DIR/dr15/manga/spectro/redux           DR15
  MANGA_SPECTRO_ANALYSIS   $SAS_BASE_DIR/dr15/manga/spectro/analysis        DR15

  MANGA_SPECTRO_REDUX      $SAS_BASE_DIR/mangawork/manga/spectro/redux      collab
  MANGA_SPECTRO_ANALYSIS   $SAS_BASE_DIR/mangawork/manga/spectro/analysis   collab
  ======================   ==============================================   ======

Marvin will check for these environment variables in your local system.  If the above environment variables are
not already defined, Marvin will use the specifed default paths.  Otherwise Marvin will adopt your custom paths.
If you wish to define custom paths, you can update the environment variable paths in your
``.bashrc`` or ``.cshrc`` file.  As a general advice, if you are
not using other products that require setting those environment variables, you should only
define ``$SAS_BASE_DIR`` (or not define it and let Marvin configure itself).

.. _marvin-sdss-depends:

Dependencies on SDSS software
-----------------------------

Marvin depends on three pieces of SDSS-wide software:

* `marvin_brain <https://github.com/sdss/marvin_brain>`_: contains some core functionality, such as the API call framework, the basic web server, etc.
* `tree <https://github.com/sdss/tree>`_: defines the structure of the Science Archive Sever, relative paths to data products, etc.
* `sdss_access <https://github.com/sdss/sdss_access>`_: tools for efficiently accessing data files, rsyncing data, etc.

For convenience, marvin includes these products as external libraries. This means that
you most likely do not need to worry about any of these products. However, with the exception of the **tree** product,
if any of these libraries are already installed in your system (i.e., you have defined
``$MARVIN_BRAIN_DIR``, or ``$SDSS_ACCESS_DIR``), marvin will use the system
wide products instead of its own versions. This is useful for development but note that
it can also lead to confusions about what version marvin is using.


.. _marvin-install-issues:

Install and Runtime Issues
--------------------------

.. important::

    We can use your help to expand this section. If you have encountered an issue
    or have questions that should be addressed here, please
    `submit and issue <https://github.com/sdss/marvin/issues/new>`_.

Pip Failure with Python-Memcache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If pip fails while installing ``python-memcached``, make sure that you have the latest version of ``setuptools`` by running ``pip install -U setuptools``. Then, try running ``pip install sdss-marvin`` again.

.. _marvin-update:

How do I update marvin?
^^^^^^^^^^^^^^^^^^^^^^^

To upgrade an existing Marvin installation, run::

  pip install -U sdss-marvin

By default, ``pip`` will update any underlying package on which marvin depends. If you want to prevent that you can upgrade marvin with ``pip install -U --no-deps sdss-marvin``. This could, however, make marvin to not work correctly. Instead, you can try ``pip install -U --upgrade-strategy only-if-needed sdss-marvin``, which will upgrade a dependency only if needed.



Permissions Error
^^^^^^^^^^^^^^^^^
If your Marvin installation fails at any point during the pip install process with permissions problems,
try running ``sudo pip install sdss-marvin``.  Note that an Anaconda or Homebrew distribution will not require
permissions when pip installing things, so if you are receiving permissions errors, you may want to check that
you are not using the Mac OSX system version of Python.

If you receive a permissions error regarding `pip` attempting to install a package in a different directory other
than the Anaconda one, e.g. `/lib/python3.6`, try following the solution indicated in `Marvin Issue 373 <https://github.com/sdss/marvin/issues/373>`_


How to test that marvin has been installed correctly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Marvin is built to have you started with minimum configuration on your part. This means that
marvin is likely to import but maybe not all features will be available. Here are a few commands
you can try that will inform you if there are problems with your installation.

From a terminal window, type::

    check_marvin

This will perform a variety of checks with Marvin and output the results to the terminal.  We may ask you for this output when
diagnosing any installation issues.  After installing marvin, start a python/ipython session and run::

    import marvin
    print(marvin.config.urlmap)

If you get a dictionary with API routes, marvin is connecting correctly to the API server at
Utah and you can use the remote features. If you get ``None``, you may want to
check the steps in :ref:`setup-netrc`.  If you get an error message such as

::

    BrainError: Requests Timeout Error: HTTPSConnectionPool(host='api.sdss.org', port=443): Read timed out.
    Your request took longer than 5 minutes and timed out. Please try again or simplify your request.

this means the servers at Utah have timed out and may possibly be down.  Simply wait and try again later.

Marvin Remote Access Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the above test crashes, or you attempt to use a Marvin Tool remotely, and you see this error::

    AttributeError: 'Extensions' object has no attribute 'get_extension_for_class'

This is an issue with the Urllib and Requests python package.  See `this Issue <https://github.com/sdss/marvin/issues/102>`_ for an
ongoing discussion if this problem has been solved.

Lots of Warnings Upon import
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see lots of warnings upon import of marvin, from `/_bootstrap.py` and referencing `numpy.ufunc size changed,
may indicate binary incompatibility`, such as
::

    import marvin
    /anaconda3/envs/marvin_public/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    /anaconda3/envs/marvin_public/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)

this arises when a Python package that uses Cython is compiled against a different version of numpy than is
actually installed.  See
`this article <https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility>`_
for more information.  The consensus is that these warnings are fairly harmless and benign.

Matplotlib backend problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some users have reported that after installing marvin they get an error such as:

**Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if
Python is not installed as a framework.**

This problem is caused by matplotlib not being able to use the MacOS backend if you are using
Anaconda. You need to switch your matplolib backend to ``Agg`` or ``TkAgg``.  Follow `these instructions
<http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python>`_ to fix
the problem. If you do want to use the MacOS backend, consider installing Python using
`homebrew <http://brew.sh/>`_.

Web Browser Oddities
^^^^^^^^^^^^^^^^^^^^

If the MPL dropdown list in the top menu bar is blank, or other elements appear to disappear, this is an indication
your browser cache is creating conflicts.  The solution is to clear your browser cache, close and restart your browser from scratch.
You can also clear your browser cookies.

As a reminder, we recommend these browsers for the best Marvin web experience:

* Google Chrome 53+ or higher
* Mozilla Firefox 50+ or higher
* Safari 10+ or Safari Technology Preview

|

.. _marvin-install-ipython:

Using IPython
-------------

If you plan to work with Marvin interactively, from the Python terminal, we recommend you use
`IPython <https://ipython.org/>`_, which provides many nice features such as autocompletion,
between history, color coding, etc. It's also especially useful if you plan to use Matplotlib,
as IPython comes with default interactive plotting. If you installed Python via the Anaconda or Miniconda
distributions, then you already have IPython installed.  Just run ``ipython`` in your terminal.  If you
need to install it, do ``pip install jupyter``.

|

.. _marvin-install-windows:

Marvin on Windows
-----------------

Marvin was originally designed to work on Mac or Linux operating systems. However it is possible at the moment to get Marvin working on Windows machines. The following guidelines have been tested on a Windows 10 machine running Python 3.6.

* Install a `Python version for Windows <https://www.python.org/downloads/windows/>`_.  Make sure to check the box to include Python in your environment variable Paths.  If you are using `Anaconda <https://conda.io/docs/user-guide/install/windows.html>`_ to install Python, make sure to check both the "Add Anaconda to my PATH environment variable" and "Register Anaconda as my default Python 3.6"
* Marvin expects a HOME directory.  Add this snippet of code before any of use of Marvin.

::

    import os
    os.environ['HOME'] = '/path/you/want/as/marvin/home/directory'
    os.environ['SAS_BASE_DIR'] = os.path.join(os.getenv("HOME"), 'sas')

To add a permanent `HOME` path, follow these instructions.
    * open File Explorer, right click "This PC" on the left scroll bar and click Properties
    * on the left, click 'Advanced System Settings'.  You need Admin Privileges to do this.
    * on the bottom, there should be an 'Environment Variables' box.  Below the User Variables column, click New.
    * add a new HOME environment variable that points to /path/you/want/as/marvin/home/directory.

* Create the ``.netrc`` file and place it the directory you designated as `HOME`.  You will need to modify the permissons of this file to match the expected `chmod 600` permissions for Mac/Linux users.  When creating the file, you can name it as anything but can rename it to ``.netrc`` from the command prompt.

With this, you should be able to run Marvin in windows.  You can test it with `import marvin`.  Currently, Marvin cannot download files due to issues with forward slashes in `sdss-access` but this will be fixed soon.  We will continue to update these guidelines as we make further progress on a Windows-Marvin installation.

|
