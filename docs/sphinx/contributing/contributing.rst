.. _marvin-contributing:

Contributing
============

Marvin is an open-source project and the Marvin team welcomes and encourages code contributions. This document describes the general guidelines for contributing to Marvin.


Reporting issues
----------------

The easiest (but critical) way to contribute to Marvin is by reporting issues, unexpected behaviours, and desired new features to the development team. To do so, go to the `Marvin GitHub page <https://github.com/sdss/marvin/>`__ and fill out a `new issue`_. As much as possible, please follow the new issue template. We will try to at least acknowledge the issue in 24-48 hours but if that is not the case feel free to ping us again!

If you have questions about how to use Marvin that are not symptomatic of a problem with the code, please use the `SDSS Help Desk <http://skyserver.sdss.org/contact/default.asp>`_ instead.


Contributing code
-----------------

Coding standards
^^^^^^^^^^^^^^^^

We adhere to the `SDSS coding standards <http://sdss-python-template.readthedocs.io/en/latest/standards.html>`_. Before contributing any code please make sure you have read the relevant sections and that you set your linter accordingly.

Development process
^^^^^^^^^^^^^^^^^^^

The Marvin team uses the well-established `git <https://git-scm.com/>`__ procedure of forking, branching, and pull requests. If you are not familiar with it, consider reading some introductory tutorial such as `this one <https://www.atlassian.com/git/tutorials/syncing>`__.

To develop code for Marvin you will first need a `GitHub account <https://github.com/>`__. Then go to the `Marvin GitHub page <https://github.com/sdss/marvin/>`__ and `fork it <https://help.github.com/articles/fork-a-repo/>`__. Develop your changes in a new branch and, when ready, open a `pull request <https://help.github.com/articles/about-pull-requests/>`__. Pleade, make sure the pull request describes all the changes the code introduces. Before it can be merged into master, yoir pull request needs to be approved by at least one of the repository owners.

Testing
^^^^^^^

All new code must be properly tested and contain enough unittests. No new code will be accepted that does not meet this requirement. Tests must run in, at least, Python 2.7 and 3.6.

All new pull requests will trigger a `Travis <https://travis-ci.org/>`__ build. Unfortunaly, Marvin's CI system is broken at the time and most Travis runs will fail (most likely due to timeouts). We hope to fix this soon but in the meantime it is still useful to check the logs of the build, since your tests probably did run.

For local testing, you will need to set up a PostgreSQL server with a test database called ``manga``, restored from this `dump file <https://sas.sdss.org/marvin/data/travis_mangadb.sql>`__. Then run the command ``run_marvin --debug``, which will create a local flask HTTP server. You can now go to the ``python/marvin/tests`` directory and run ``pytest``, which will run all tests (fair warning, it may take a while!) or ``pytests <your-file>``.

.. _marvin-contributing-code-documentation:

Documentation
^^^^^^^^^^^^^

If your changes introduce a new feature or change the behaviour of user-facing commands, you will need to provide Sphinx documentation on how to use it. Marvin's documentation can be found in the `docs/ directory <https://github.com/sdss/marvin/tree/master/docs>`__. Add your documentation where you think it is appropriate. As part of the pull request review, you may be asked to restructure or move the documentation.


Contributing documentation
--------------------------

Code is only useful if it is well documented! An excellent way of contributing to Marvin is by writing documentation. There are different forms of documentation that you can submit:

* Sphinx documentation of current features, as described in :ref:`marvin-controbuting-code-documentation`.
* Plain-text documentation. If you do not want to write Sphinx documentation, you can still send us the text and we will format it appropriately. Send your text in the form of a `new issue`_.
* Open a `new issue`_ pointing out parts of the documentation that are unclear, misleading, or deprecated.
* Write :ref:`marvin-tutorials <tutorials>`! Tutorials can be in Sphinx or ipyhton notebook format.
* If you have used Marvin in the classroom and developed activities based on it we would love to hear about it!


.. _marvin-contributing-vacs:

Contributing a VAC
------------------

`Value Added Catalogues (VAC) <http://www.sdss.org/dr15/data_access/value-added-catalogs/>`_ are an important part of how SDSS and MaNGA data is analysed and distributed. Following SDSS policy, Marvin does not directly support VACs, but it does supply a framework for VAC owners to implement access to their catalogues from Marvin.

At this time, only file-based access is supported (i.e., no querying or remote access is available) but the `~.VACMixIn` class provides a convenient way of matching targets with their VAC information and returning it to the user. Very little knowledge of how Marvin internally works is required! The directory `marvin/contrib/vacs <https://github.com/sdss/marvin/blob/master/python/marvin/contrib/vacs>`__ contains the base code and a list of already implemented VACs that you can use as a template.

Example: Galaxy Zoo: 3D
^^^^^^^^^^^^^^^^^^^^^^^

The `Galaxy Zoo: 3D project <https://www.zooniverse.org/projects/klmasters/galaxy-zoo-3d>`__ provides classifications of internal structures of MaNGA galaxies. For each observed MaNGA galaxy there is a Galaxy Zoo: 3D FITS file (you can find all the files `here <https://data.sdss.org/sas/mangawork/manga/sandbox/galaxyzoo3d/>`__) with the format ``<version>/<mangaid>-<ifusize>-<galzooid>.fits.gz``, where ``<version>`` is the version of the Galaxy Zoo VAC release, and ``<galzooid>`` is an internal identifier. The source code for this VAC's implementation can be found at  `marvin/contrib/vacs/galaxyzoo3d.py <https://github.com/sdss/marvin/blob/master/python/marvin/contrib/vacs/galaxyzoo3d.py>`__:

.. literalinclude:: ../../../python/marvin/contrib/vacs/galaxyzoo3d.py
   :language: python
   :linenos:

The file itself contains just a subclass of `~.VACMixIn`. In the docstring, we make sure to include the name of the VAC, a URL with a description of its contents, and a short description of what the VAC provides and what the class returns.

The global section of the `~marvin.contrib.vacs.galaxyzoo3d.GalaxyZoo3DVAC` class defines :

* The ``name`` of the VAC (this is the name that users will enter to access the VAC from Marvin). This is the only required attribute that we need to override from the parent class.
* A ``version`` dictionary that defines the relationship between Marvin releases (e.g., ``MPL-6``, ``DR15``) and internal Galaxy Zoo: 3D versions.
* An ``include`` attribute that contains a list of Marvin Tools classes to which this VAC must be added. In this particular case we only want the Galaxy Zoo VAC to show in `~marvin.tools.cube.Cube`, `~marvin.tools.maps.Maps`, and `~marvin.tools.modelcube.ModelCube`. If ``include`` is not defined the VAC will be added to all the Marvin Tools classes with the exception of `~marvin.tools.plate.Plate`.

`~.VACMixIn.get_data` is the only method that you need to override from `~.VACMixIn`. You will have noted that `~.VACMixIn.get_data` receives a single, ``parent_object`` argument, which is the object (e.g., a `~marvin.tools.maps.Maps` instance) that is trying to access the VAC information. You can use it and its attributes to do the matching with the VAC information.

We use `sdss_access <http://sdss-access.readthedocs.io/en/stable>`__ to download the necessary files, so we need to be sure that the paths to the VAC files are included in the `tree <http://sdss-tree.readthedocs.io/en/latest/>`_. For this particular filetype the entry in tree looks like::

    mangagalaxyzoo3d = $MANGA_SANDBOX/galaxyzoo3d/{gz3dver}/{mangaid}_{ifusize}_{zooid}.fits.gz

In addition to the tree path name (``mangagalaxyzoo3d``) we need to define a dictionary of path parameters. We use the ``parent_object`` to determine the release (and thus the Galaxy Zoo ``version``) and the ``mangaid``. Because for a given ``mangaid`` ``ifusize`` and ``zooid`` are fixed, we can replace them with wild cards.

First, we use `~marvin.contrib.vacs.VACMixIn.get_path` to determine whether the file is already present in the local SAS. If that is not the case, we use `~marvin.contrib.vacs.VACMixIn.download_vac` to retrieve it. It only rests to open the FITS file and return it. Easy!

Now that we have implemented the VAC, let's make sure it works:

.. code-block:: python

    >>> from marvin.tools.maps import Maps
    >>> my_map = Maps('8485-1901')
    >>> galaxyzoo3d_data = my_map.vacs.galaxyzoo3d
    >>> print(my_map.vacs.galaxyzoo3d.info())

    Filename: /Users/albireo/Documents/MaNGA/mangawork/manga/sandbox/galaxyzoo3d/v1_0_0/1-209232_19_5679839.fits.gz
    No.    Name      Ver    Type      Cards   Dimensions   Format
    0  PRIMARY       1 PrimaryHDU      23   (3, 525, 525)   uint8
    1                1 ImageHDU        23   (525, 525)   float64
    2                1 ImageHDU        23   (525, 525)   float64
    3                1 ImageHDU        23   (525, 525)   float64
    4                1 ImageHDU        23   (525, 525)   float64
    5                1 BinTableHDU     38   1R x 15C   [E, E, 11A, 19A, D, D, K, 90A, I, I, I, K, K, K, 70A]
    6                1 BinTableHDU     30   1R x 11C   [D, D, D, D, D, D, D, D, D, D, K]
    7                1 BinTableHDU     30   0R x 11C   [D, D, D, D, D, D, D, D, D, D, D]
    8                1 BinTableHDU     18   17R x 5C   [8A, 7A, 24A, 41A, 57A]
    9                1 BinTableHDU     16   0R x 4C   [D, D, D, D]
    10                1 BinTableHDU     16   0R x 4C   [D, D, D, D]

Writing your own VAC
^^^^^^^^^^^^^^^^^^^^

If you are a VAC owner by now you will have hopefully decided that you want to provide access to it with Marvin. How do you do that?

* First, make sure you read the :ref:`Coding standards` and :ref:`Development process` sections.
* For the Marvin repository and clone your fork. From the ``python/marvin/contrib/vacs`` directory select an example that looks similar to what you need. Within the ``vacs/`` directory. Duplicate that file and rename it to the name of your VAC.
* Modify the name of the class that inherits from `.VACMixIn` (important!) and change the ``name`` attribute to the name of your VAC. Update the docstring with information about your VAC. Most importantly, make sure the description clearly states what your VAC is returning.
* To be able to use `~marvin.contrib.vacs.VACMixIn.download_vac` the archetype of your VAC's file path must have been added to `sdss_paths.ini <https://github.com/sdss/tree/blob/master/data/sdss_paths.ini>`__. Refer to the `tree`_ documentation to learn how to do that.
* Test you implementation by accessing your VAC as ``object.vacs.<your-vac-name>``.
* As the VAC owner, you are responsible for making sure your class returns the correct information. We **strongly** suggest you implement unittests to make sure that is the case. You can see some examples `here <https://github.com/sdss/marvin/blob/master/python/marvin/tests/contrib/test_vacs.py>`__.
* Once you are happy with the result, open a new pull request to integrate your changes.
* If you have any doubt, contact the Marvin team, we will be happy to help you throughout the process.


Classes
^^^^^^^

.. autosummary::
    marvin.contrib.vacs.base.VACMixIn


Available VACs
^^^^^^^^^^^^^^

.. include:: ./available_vacs.rst


.. _new issue: https://github.com/sdss/marvin/issues/new
.. _sdss_access.path.Path.full: http://sdss-access.readthedocs.io/en/stable/api.html#sdss_access.path.path.BasePath.full
