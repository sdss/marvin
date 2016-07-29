
Installation
============

Marvin requires installing some dependencies before you can start using it.
In the future we hope this process will be mostly automatic but, for now,
please follow this instructions to install Marvin.

The full list of dependencies includes:

* Python 2.7 (the final Marvin version will be Python 3-compatible)
* GNU Modules
* tree
* sdss_access
* sdss_python_module
* marvin_brain
* numpy
* astropy
* sqlalchemy
* networkx
* matplotlib (optional, needed for plotting)
* pillow
* requests
* wtforms
* SQLAlchemy-boolean-search (custom fork)
* wtforms-alchemy (custom fork)

.. flask_cors

|

Getting modules
---------------

`GNU Modules <http://modules.sourceforge.net>`_ is a powerful tool to control
what software versions are installed in your computer, and to setup their
environment variables. SDSS software is configured to be used with ``modules``.
For a full tutorial on ``modules`` and how to install it manually you can visit
`this <https://trac.sdss.org/wiki/Software/modules>`_ page.

To use `sdss4tools <https://trac.sdss.org/browser/repo/sdss/sdss4tools?order=name>`_
to install ``modules``, run the following command on a fresh terminal ::

    svn export https://svn.sdss.org/public/repo/sdss/sdss4tools/trunk/bin/sdss4_getmodules

Then run ::

    ./sdss4_getmodules -m <path-to-modules>

where ``<path-to-modules>`` must be the path to the empty or non-existent directory
where you want to install the ``modules`` package. ``sdss4_getmodules`` will take care
of compiling ``modules``. If everything works you will get a message ending in ::

    bash users will need to add: source <path-to-modules>/init/bash to the .bashrc file
    tcshrc users will need to add: source <path-to-modules>/init/csh to the .tcshrc file

You must add the corresponding ``source`` statement to your ``.bashrc``, ``.profile``, or
``.cshrc`` file (likely located in your ``HOME`` directory). If you open a new terminal and write ``module`` you should get the help
page for the command.

|

sdss4install
------------

``sdss4install`` is a tool that lives in ``sdss4tools`` that helps significantly
to download, configure, and make available SDSS software. To install ``sdss4tool``
you can use the ``bootstrap`` installation script. First, create a directory where you
want to store SDSS sofware and export that path ::

    mkdir -p ~/software/sdss
    export SDSS4_PRODUCT_ROOT=~/software/sdss

(note that you do not need to add SDSS4_PRODUCT_ROOT to your ``.bashrc``). Then
download and run the script ::

    svn export https://svn.sdss.org/public/repo/sdss/sdss4tools/trunk/bin/sdss4bootstrap
    ./sdss4bootstrap -l

Next, edit your ``.bashrc`` or ``.cshrc`` file to include ::

    module use $SDSS4_PRODUCT_ROOT/repo/modulefiles

If you now run ``module avail`` in a new terminal you will get a list containing
``sdss4tools``. Now you are ready to install more SDSS software by just
using the ``sdss4install`` command.

Before we continue, let's make sure SVN has the right permissions to access the private
repositories. For that, run ::

    module load sdss4tools
    sdss4auth

which will prompt you for your trac username and password. When it asks you to store the
credentials say yes and make sure to run ::

    chmod g-rwX,o-rwX ~/.subversion

to ensure that your subversion directory only accesible by you.

|

SDSS packages
-------------

Marvin depends on a few SDSS packages. First, create a directory that will act as
your local SAS. This directory will follow the same structure as the remote SAS and will
be used for local and downloaded data. For example ::

    mkdir -p ~/sdss/sas
    export SAS_BASE_DIR=~/sdss/sas

Let's load ``sdss4tools`` so that you can use ``sdss4install`` to install the dependencies ::

    module load sdss4tools

The following commands should install all the necessary dependencies in your selected
``SDSS4_PRODUCT_ROOT``. At some point you may be asked for your SDSS Trac username and
password ::

    sdss4install sdss/tree trunk
    sdss4install sdss/sdss_access trunk
    sdss4install sdss/sdss_python_module branches/marvin
    sdss4install manga/marvin_brain trunk
    sdss4install manga/marvin branches/marvin_alpha

The last line actually installs Marvin from the
`marvin_refactor <https://trac.sdss.org/browser/repo/manga/marvin/branches/marvin_refactor>`_
branch. If you now do a ``module avail`` you should get something like ::

    ------------------------------ /home/albireo/software/modulefiles ------------------------
    marvin/marvin_refactor    sdss_python_module/marvin tree/dr12       tree/dr9
    marvin_brain/trunk        tree/bosswork             tree/dr13       tree/sdsswork(default)
    sdss4tools/0.2.6(default) tree/dr10                 tree/dr7
    sdss_access/trunk         tree/dr11                 tree/dr8

Doing ``module load marvin`` will setup all the necessary environment variables
that Marvin needs to work. (Also, you can ignore the errors about 'dust' and
'inspection/trunk'.) However, you may not want to do that every time you want to
work with Marvin. To have ``modules`` load Marvin for each new terminal you can
create a file containing ``modules`` commands ::

    cat > ~/.modules <<EOL
    module load sdss4tools
    module load marvin
    EOL

and edit your ``.bashrc`` or ``.cshrc`` to source it ::

    source ~/.modules

The lines in ``.modules`` will load ``sdss4tools`` and ``marvin`` for each new terminal.

|

Python packages
---------------

In addition to SDSS software, Marvin depends on a few Python libraries. These can easily
be installed with `pip <https://pip.pypa.io/en/stable/>`_. If your system does not have
``pip``, you can install it following these
`instructions <https://pip.pypa.io/en/stable/installing/>`_. Most packages can also
be installed with `easy_install <https://pypi.python.org/pypi/setuptools>`_.

With ``pip`` run the following commands and make sure they finish without errors. You may need to use ``sudo`` to run these commands. Also, some modern versions of Mac OSX do
not allow to install these products even with ``sudo``. If that is your case, try using
``pip install --user <package>``::

    pip install numpy
    pip install astropy
    pip install sqlalchemy
    pip install networkx
    pip install matplotlib
    pip install requests
    pip install pillow
    pip install wtforms
    pip install Flask

Additionally, Marvin requires installing two forks of Python packages. Those forks will
eventually be merged into Marvin, but during active development they live in GitHub
repositories. To install ``SQLAlchemy-boolean-search`` do ::

    git clone https://github.com/havok2063/SQLAlchemy-boolean-search.git
    cd SQLAlchemy-boolean-search
    python setup.py install

You may need ``sudo`` for the last command. Once the library is installed you can
remove the ``SQLAlchemy-boolean-search`` directory. Similarly, for ``wtforms-alchemy`` do ::

    git clone https://github.com/havok2063/wtforms-alchemy.git
    cd wtforms-alchemy
    python setup.py install

You should now be ready to use Marvin!

|

Testing the installation
------------------------

Let's do a quick check to make sure Marvin is working. In a fresh terminal do ::

    python
    >>> import marvin
    >>> marvin.config.mode
    'auto'

You may get a few warnings and info messages after ``import marvin``. That's ok,
we'll deal with them later. Congratulations, you have finished the Marvin installation!
Now go on to :ref:`marvin-first-steps`!

|

Using IPython
-------------

If you plan to work with Marvin interactively, from the Python terminal, we recommend you use
`IPython <https://ipython.org/>`_, which provides many nice features such as autocompletion,
between history, colour coding, etc. It's also especialyl useful if you plan to use Matplotlib,
as IPython comes with default interactive plotting. To install it, follow the instructions in
the webpage, or simply do ::

    pip install jupyter

And just run ``ipython`` in your terminal.
