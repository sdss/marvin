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


Contributing a VAC
------------------

`Value Added Catalogues (VAC) <http://www.sdss.org/dr14/data_access/value-added-catalogs/>`__ are an important part of how SDSS and MaNGA data is analysed and distributed. Following SDSS policy, Marvin does not provide direct support for VACs, but it does supply a framework for VAC owners to implement access to their catalogues from Marvin.

At this time, only file-based access is supported (i.e., no querying or remote access is available) but the `~.VACMixIn` class provides a convenient way of matching targets with their VAC information and returning it to the user. Very little knowledge of how Marvin internally works is required! The directory `marvin/contrib/vacs <https://github.com/sdss/marvin/tree/python/marvin/contrib/vacs>`__ contains the base code and a list of already implemented VACs that you can use as a template. Let up have a look at a couple of them.

Example 1: DAPall
^^^^^^^^^^^^^^^^^

The file `marvin/contrib/vacs/dapall.py <https://github.com/sdss/marvin/tree/python/marvin/contrib/vacs/dapall.py>`__ shows how we would implement the contents of the `DAPall file <https://www.sdss.org/dr15/manga/manga-data/catalogs/#DAPALLFile>`__ as a VAC. Note that the DAPall is **not** a VAC and it is implemented differently, but we use it here as an example. The code of the file is

.. literalinclude:: ../../python/marvin/contrib/vacs/dapall.py
   :language: python
   :linenos:

In this example we assume that the path(s) to the VAC file(s) (in this case the DAPall file) are included in the `tree <http://sdss-tree.readthedocs.io/en/latest/>`_ product so that they can be readily access with `sdss_access <http://sdss-access.readthedocs.io/en/latest/>`. If that is not the case you will need to create a pull request in the `tree repository <https://github.com/sdss/tree>`__ to include the new path.

The file itself contains just a class that must subclass from `~.VACMixIn`. In the docstring, make sure you include the name of your VAC, a URL with a description of its contents, and a short description of what the VAC provides and what the class returns.

We define two global attributes, the ``name`` (required) of the VAC, and ``path_params``, which is a dictionary with the keyword parameters that we need to pass to `sdss_access.path.Path.full`_ to obtain the full path of the VAC file. `~.VACMixIn.get_data` is the only method that you need to override from `~.VACMixIn`. You will have noted that `~.VACMixIn.get_data` receives a single, ``parent_object`` argument, which is the object (e.g., a `~marvin.tools.cube.Cube`, `~marvin.tools.maps.Maps`, or `~marvin.tools.modelcube.ModelCube` object) that is trying to access the VAC information. You can use it and its attributes to do the matching with your VAC information.

First, we use `~.VACMixIn.file_exists` to determine whether the VAC file exists in the local SAS. `~.VACMixIn.file_exists` uses the ``name`` we have defined for our VAC and ``path_params`` to make a call to `sdss_access.path.Path.full`_ and get the location of the file in the SAS. If `~.VACMixIn.file_exists` returns False, we use `~.VACMixIn.download_vac` in a similar way to download the file. Otherwise, we get the full path of the file in the local SAS using `~.VACMixIn.get_path`.

The rest of the method is just Python and there are no Marvin magic involved. We open the file using `astropy.table.Table` and then select the rows that match the ``parent_object`` plate-ifu. In the case of the DAPall file there are multiple rows, one for each bintype and template. We check if the ``parent_object`` contains information about the bintype and template; if it does not (e.g., if the ``parent object`` is a `~marvin.tools.cube.Cube`), we return all the rows; otherwise we use the extra information to select a single row to return.


Example 2: Galaxy Zoo: 3D
^^^^^^^^^^^^^^^^^^^^^^^^^

Let us have a look at a different VAC, this time one that is really implemented as such in Marvin. Instead of a single file such as the DAPall, the `Galaxy Zoo: 3D project <https://www.zooniverse.org/projects/klmasters/galaxy-zoo-3d>`__ provides one file per observed galaxy (you can find all the files `here <https://data.sdss.org/sas/mangawork/manga/sandbox/galaxyzoo3d/>`__). The source code for this VAC's implementation can be found at  `marvin/contrib/vacs/galaxyzoo3d.py <https://github.com/sdss/marvin/tree/python/marvin/contrib/vacs/galaxyzoo3d.py>`__:

.. literalinclude:: ../../python/marvin/contrib/vacs/galaxyzoo3d.py
   :language: python
   :linenos:

As in the previous example, we need to provide a name for the VAC. In this case we also define a ``version`` attribute globally (this is not necessary, but it is a good place to define it). The `~.VACMixIn.get_data` method is even simpler in this case. The main difference is that here we use the `~.VACMixIn.set_sandbox_path` method to define the path to the file, relative to the `MaNGA sandbox <https://data.sdss.org/sas/mangawork/manga/sandbox>`_. Note that the path contains the catalogue version as well as the mangaid of the target, but the last part is a wild card (*). This is because the last part of the filename is a unique identifier that we do not know how to determine (and we do not need to!) Now we can use `~.VACMixIn.get_path`, `~.VACMixIn.file_exists`, and `~.VACMixIn.download_vac` as in the previous example and they will use the path we just defined. Because of the wild card, if more than one file matching the patter were found (which should not happen for this VAC), only the first one would be returned.

Now that we have implemented the VAC, let's check if it works:

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
* For the Marvin repository and clone your fork. From the ``python/marvin/contrib/vacs`` directory select an example that looks similar to what you need. Within the ``vacs/`` directory. duplicate that file and rename it to the name of your VAC.
* Modify the name of the class that inherits from `.VACMixIn` (important!) and change the ``name`` attribute to the name of your VAC. Update the docstring with information about your VAC. Most importantly, make sure the description clearly states what your VAC is returning.
* If the path to your VAC is already in the `tree`_, make sure you understand the ``path_params`` you need to use to retrieve if. If it is not, consider adding it (a product included in the tree can be accessed easily not only by Marvin but by any other piece of SDSS software). Alternatively, and if your VAC lives in the `MaNGA sandbox`_, you can use the `~.VACMixIn.set_sandbox_path` method.
* Test you implementation by accessing your VAC as ``object.vacs.<your-vac-name>``.
* Once you are happy with the result, open a new pull request to integrate your changes.
* If you have any doubt, contact the Marvin team, we will be happy to help you throughout the process.


Classes
^^^^^^^

.. autosummary::
    marvin.contrib.vacs.base.VACMixIn


Available VACs
^^^^^^^^^^^^^^

.. autosummary:: marvin.contrib.vacs


.. _new issue: https://github.com/sdss/marvin/issues/new
.. _sdss_access.path.Path.full: http://sdss-access.readthedocs.io/en/stable/api.html#sdss_access.path.path.BasePath.full
