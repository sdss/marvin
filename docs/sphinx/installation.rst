
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

**Painless Installation**::

    pip install sdss-marvin

.. admonition:: Attention
    :class: attention

    If pip fails while installing ``python-memcached``, make sure that you have the latest version of ``setuptools`` by running ``pip install -U setuptools``. Then, try running ``pip install sdss-marvin`` again.

**or to upgrade an existing Marvin installation**::

    pip install --upgrade sdss-marvin

.. admonition:: Hint
    :class: hint

    By default, ``pip`` will update any underlying package on which marvin depends. If you want to prevent that you can upgrade marvin with ``pip install -U --no-deps sdss-marvin``. This could, however, make marvin to not work correctly. Instead, you can try ``pip install -U --upgrade-strategy only-if-needed sdss-marvin``, which will upgrade a dependency only if needed.

**Developer Installation (Medium Pain)**::

    git clone https://github.com/sdss/marvin
    cd marvin
    git submodule init
    git submodule update
    python setup.py install

If you experience problem after the installation, check the :ref:`marvin-faq`.

|

.. _setup-netrc:

Set up your netrc
-----------------

While Marvin is now publicly available, not all MaNGA data is so. As a result,
you need to add some configuration to allow you to access proprietary data. To
do that, create and edit a file in your home called ``.netrc`` an copy
these lines inside::

    machine api.sdss.org
       login sdss
       password <password>

    machine data.sdss.org
       login sdss
       password <password>

and replace ``<password>`` with the default SDSS data password. Finally, run
``chmod 600 ~/.netrc`` to make the file only accessible to your user.

|

.. _marvin-sdss-depends:

Marvin dependencies on SDSS software
------------------------------------

Marvin depends on three pieces of SDSS-wide software:

* `marvin_brain <https://github.com/sdss/marvin_brain>`_: contains some core functionality, such as the API call framework, the basic web server, etc.
* `tree <https://github.com/sdss/tree>`_: defines the structure of the Science Archive Sever, relative paths to data products, etc.
* `sdss_access <https://github.com/sdss/sdss_access>`_: tools for efficiently accessing data files, rsyncing data, etc.

For convenience, marvin includes these products as external libraries. This means that
you most likely do not need to worry about any of these products. However, if any
of these libraries is already installed in your system (i.e., you have defined
``$MARVIN_BRAIN_DIR``, ``$TREE_DIR``, or ``$SDSS_ACCESS_DIR``), marvin will use the system
wide products instead of its own versions. This is useful for development but note that
it can also lead to confusions about what version marvin is using.

|

.. _marvin-sasdir:

Local SAS directory structure
-----------------------------

By default, marvin will look for data files in a directory structure that mirrors the
`Science Archive Server <https://data.sdss.org/sas>`_. Data downloaded via marvin will
also be stored according to that structure. The root of this directory structure is
defined by the environment variable  ``$SAS_BASE_DIR``. For example, if marvin needs
to use the ``drpall`` file for MPL-5, it will try to find it in
``$SAS_BASE_DIR/mangawork/manga/spectro/redux/v2_0_1/drpall-v2_0_1.fits``.

If ``$SAS_BASE_DIR`` is not defined, marvin will assume that the base directory is ``$HOME/sas``.
You can also define your custom ``$MANGA_SPECTRO_REDUX`` and ``$MANGA_SPECTRO_ANALYSIS`` to
point to the redux and analysis data directories, respectively. As a general advice, if you are
not using other products that require setting those environment variables, you only want to
define ``$SAS_BASE_DIR`` (or not define it and assume the data will be stored in ``$HOME/sas``).

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


.. _marvin-install-issues:

Install and Runtime Issues
==========================

.. important::

    We can use your help to expand this section. If you have encountered an issue
    or have questions that should be addressed here, please
    `submit and issue <https://github.com/sdss/marvin/issues/new>`_.

How do I update marvin?
-----------------------

Just do ``pip install --upgrade sdss-marvin``. Marvin will get updated to the latest
version, along with all the dependencies. If you want to update marvin but keep other
packages in their currrent versions, do
``pip install --upgrade --upgrade-strategy only-if-needed sdss-marvin``. This will only
update dependencies if marvin does need it.


Permissions Error
-----------------
If your Marvin installation fails at any point during the pip install process with permissions problems,
try running ``sudo pip install sdss-marvin``.  Note that an Anaconda or Homebrew distribution will not require
permissions when pip installing things, so if you are receiving permissions errors, you may want to check that
you are not using the Mac OSX system version of Python.

If you receive a permissions error regarding `pip` attempting to install a package in a different directory other
than the Anaconda one, e.g. `/lib/python3.6`, try following the solution indicated in `Marvin Issue 373 <https://github.com/sdss/marvin/issues/373>`_


How to test that marvin has been installed correctly
----------------------------------------------------

Marvin is built to have you started with minimum configuration on your part. This means that
marvin is likely to import but maybe not all features will be available. Here are a few commands
you can try that will inform you if there are problems with your installation.

After installing marvin, start a python/ipython session and run::

    import marvin
    print(marvin.config.urlmap)

If you get a dictionary with API routes, marvin is connecting correctly to the API server at
Utah and you can use the remote features. However, if you get ``None``, you may want to
check the steps in :ref:`setup-netrc`.

Marvin Remote Access Problems
-----------------------------

If the above test crashes, or you attempt to use a Marvin Tool remotely, and you see this error::

    AttributeError: 'Extensions' object has no attribute 'get_extension_for_class'

This is an issue with the Urllib and Requests python package.  See `this Issue <https://github.com/sdss/marvin/issues/102>`_ for an
ongoing discussion if this problem has been solved.


Matplotlib backend problems
---------------------------

Some users have reported that after installing marvin they get an error such as:

**Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if
Python is not installed as a framework.**

This problem is caused by matplotlib not being able to use the MacOS backend if you are using
Anaconda. You need to switch your matplolib backend to ``Agg`` or ``TkAgg``.  Follow `these instructions
<http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python>`_ to fix
the problem. If you do want to use the MacOS backend, consider installing Python using
`homebrew <http://brew.sh/>`_.

Web Browser Oddities
--------------------

If the MPL dropdown list in the top menu bar is blank, or other elements appear to disappear, this is an indication
your browser cache is creating conflicts.  The solution is to clear your browser cache, close and restart your browser from scratch.
You can also clear your browser cookies.

As a reminder, we recommend these browsers for the best Marvin web experience:

* Google Chrome 53+ or higher
* Mozilla Firefox 50+ or higher
* Safari 10+ or Safari Technology Preview

|

.. _marvin-install-windows:

Installation on Windows
=======================

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

