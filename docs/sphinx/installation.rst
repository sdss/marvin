
Installation
============

Marvin requires installing some dependencies before you can start using it.
In the future we hope this process will be mostly automatic but, for now,
please follow this instructions to install Marvin.


Installing modules
------------------

`GNU Modules <http://modules.sourceforge.net>`_ is a powerful tool to control
what software versions are installed in your computer, and to setup their
environment variables. SDSS software is configured to be used with ``modules``.
For a full tutorial on ``modules`` and how to install it manually you can visit
`this <https://trac.sdss.org/wiki/Software/modules>`_ page.

To use `sdss4tools <https://trac.sdss.org/browser/repo/sdss/sdss4tools?order=name>`_
to install ``modules``, run the following command on a fresh terminal ::

    svn export https://svn.sdss.org/public/repo/sdss/sdss4tools/trunk/bin/sdss4_getmodules

The run ::

    ./sdss4_getmodules -m <path-to-modules>

where ``<path-to-modules>`` must be the path to the empty or inexistent directory
where you want to install the ``modules`` package. ``sdss4_getmodules`` will take care
of compiling ``modules``. If everything works you must get a message ending in ::

    bash users will need to add: source <path-to-modules>/init/bash to the .bashrc file
    tcshrc users will need to add: source <path-to-modules>/init/csh to the .tcshrc file

You must add the corresponding ``source`` statement to your ``.bashrc``, ``.profile``, or
``.cshrc``. If you open a new terminal and write ``module`` you should get the help
page for the command.


Installing sdss4install
-----------------------

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

If you now run ``module avail`` you must get a list containing ``sdss4tools``. Now
your are ready to install more SDSS software by just using the ``sdss4install`` command.


Installing SDSS packages
------------------------

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
    sdss4install manga/marvin branches/marvin_refactor

The last line actually installs Marvin from the
`marvin_refactor <https://trac.sdss.org/browser/repo/manga/marvin/branches/marvin_refactor>`_
branch. If you now do a ``module avail`` you should get something like ::

    ------------------------------ /home/albireo/software/modulefiles ------------------------
    marvin/marvin_refactor    sdss_python_module/marvin tree/dr12       tree/dr9
    marvin_brain/trunk        tree/bosswork             tree/dr13       tree/sdsswork(default)
    sdss4tools/0.2.6(default) tree/dr10                 tree/dr7
    sdss_access/trunk         tree/dr11                 tree/dr8

Doing ``module load marvin`` will setup all the necessary environment variables that Marvin
needs to work. However, you may not want to do that every time you want to work with Marvin.
Let's see how to have ``modules`` load Marvin for each new terminal.

TBD.

Installing Python packages
--------------------------

TBD.

.. You are invited to participate in the closed-alpha testing of Marvin 2.0 at the SDSS-IV Collaboration meeting in Madison, on the afternoon of Thursday the Thirtieth day of June, in the year of our lord Two Thousand and Sixteen.  If you choose to participate, here are some start instructions that we would like you to have completed before Madison.
..
.. 1.  Modules is currently required.  If you already have Modules installed then you do not need to do anything.  If you need Modules, please check out, and run, the Modules install script .
..
.. To check out, svn export https://svn.sdss.org/public/repo/sdss/sdss4tools/trunk/bin/sdss4_getmodules
..
.. To run, type ./sdss4_getmodules -m /your/path/to/a/nonexistent/modules/directory (e.g.  /Users/Me/modules’ )  (this directory should not exist or be empty!!)
..
.. If everything proceeds according to plan, you shall see a line instructing you to source a file depending on your terminal shell
..
.. Installation complete!
.. bash users will need to add: source /Users/Me/modules/init/bash to the .bashrc file
.. tcshrc users will need to add: source /Users/Me/modules/init/csh to the .tcshrc file
..
.. Test by opening a new terminal window, and type module
..
.. 2. sdss4tools is required.  If you already have sdss4tools and sdss4install installed, then you do not need to do anything.  If you need sdss4tools, please check out, and run, the sdss4bootstrap install script.
..
.. Follow the instructions on this wiki page:
.. https://trac.sdss.org/wiki/Software/sdss4install#sdss4tools
..
.. By the end, you should be able to module load sdss4tools.  If that works, you are ready to use sdss4install.
..
.. Your first task, if you choose to accept it, is to time the installation process, and report any difficulties you had during this process.
..
.. Once completed, please await for further instructions.
..
.. We look forward to you seeing you in Madison!
..
.. Cheers,
.. The Marvin Dev. Team (Brian, José, Brett, and Joel)
..
.. ---------------------------------------
.. Brian Cherinka, Ph.D
.. Dept of Physics & Astronomy
.. Johns Hopkins University
.. Baltimore, MD, 21218
.. phone: 1-410-516-5624
.. email: bcherin1@jhu.edu
.. ----------------------------------------
