
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

    If pip fails while installing `python-memcached`, make sure that you have the latest version of `setuptools` by running `pip install -U setuptools`. Then, try running `pip install sdss-marvin` again.

**or to upgrade an existing Marvin installation**::

    pip install --upgrade sdss-marvin

.. admonition:: Hint
    :class: hint

    By default, `pip` will update any underlying package on which marvin depends. If you want to prevent that you can upgrade marvin with `pip install -U --no-deps sdss-marvin`. This could, however, make marvin to not work correctly. Instead, you can try `pip install -U --upgrade-strategy only-if-needed sdss-marvin`, which will upgrade a dependency only if needed.

**Developer Installation (Medium Pain)**::

    git clone https://github.com/sdss/marvin
    cd marvin
    git submodule init
    git submodule update
    python setup.py install

If you experience problem after the intallation, check the :ref:`marvin-faq`.

|

.. _setup-netrc:

Set up your netrc
-----------------

While Marvin is now publicly available, not all MaNGA data is so. As a result,
you need to add some configuration to allow you to access propietary data. To
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
