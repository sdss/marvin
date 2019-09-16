.. _marvin-contributing:

Contributing
============

Marvin is an open-source project and the Marvin team welcomes and encourages code contributions. This document 
describes the general guidelines for contributing to Marvin.


Reporting issues
----------------

The easiest (but critical) way to contribute to Marvin is by reporting issues, unexpected behaviours, and desired new 
features to the development team. To do so, go to the `Marvin GitHub page <https://github.com/sdss/marvin/>`__ and fill out 
a `new issue`_. As much as possible, please follow the new issue template. We will try to at least acknowledge the issue in 
24-48 hours but if that is not the case feel free to ping us again!

If you have questions about how to use Marvin that are not symptomatic of a problem with the code, please use the 
`SDSS Help Desk <http://skyserver.sdss.org/contact/default.asp>`_ instead.


Contributing code
-----------------

.. _marvin-code-standards:

Coding standards
^^^^^^^^^^^^^^^^

We adhere to the `SDSS coding standards <http://sdss-python-template.readthedocs.io/en/latest/standards.html>`_. Before 
contributing any code please make sure you have read the relevant sections and that you set your linter accordingly.

.. _marvin-code-dev:

Development process
^^^^^^^^^^^^^^^^^^^

The Marvin team uses the well-established `git <https://git-scm.com/>`__ procedure of forking, branching, and pull requests. 
If you are not familiar with it, consider reading some introductory tutorial such as 
`this one <https://www.atlassian.com/git/tutorials/syncing>`__.

To develop code for Marvin you will first need a `GitHub account <https://github.com/>`__. Then go to the 
`Marvin GitHub page <https://github.com/sdss/marvin/>`__ and `fork it <https://help.github.com/articles/fork-a-repo/>`__. 
Develop your changes in a new branch and, when ready, open a 
`pull request <https://help.github.com/articles/about-pull-requests/>`__. Please, make sure the pull request 
describes all the changes the code introduces. Before it can be merged into master, your pull request needs to 
be approved by at least one of the repository owners.

Testing
^^^^^^^

All new code must be properly tested and contain enough unittests. No new code will be accepted that does not 
meet this requirement. Tests must run in, at least, Python 2.7 and 3.6.  All new pull requests will trigger 
a `Travis <https://travis-ci.org/>`__ build. Unfortunaly, Marvin's CI system is broken at the time and most 
Travis runs will fail (most likely due to timeouts). We hope to fix this soon but in the meantime it is still 
useful to check the logs of the build, since your tests probably did run.

For local testing, you will need to set up a PostgreSQL server with a test database called ``manga``, 
restored from this `dump file <https://sas.sdss.org/marvin/data/travis_mangadb.sql>`__. Then run the 
command ``run_marvin --debug``, which will create a local flask HTTP server. You can now go to 
the ``python/marvin/tests`` directory and run ``pytest``, which will run all tests (fair warning, it may 
take a while!) or ``pytests <your-file>``.

.. _marvin-contributing-code-documentation:

Documentation
^^^^^^^^^^^^^

If your changes introduce a new feature or change the behaviour of user-facing commands, you will need to 
provide Sphinx documentation on how to use it. Marvin's documentation can be found in the 
`docs/ directory <https://github.com/sdss/marvin/tree/master/docs>`__. Add your documentation where you think 
it is appropriate. As part of the pull request review, you may be asked to restructure or move the documentation.


Contributing documentation
--------------------------

Code is only useful if it is well documented! An excellent way of contributing to Marvin is by writing documentation. 
There are different forms of documentation that you can submit:

* Sphinx documentation of current features, as described in :ref:`marvin-contributing-code-documentation`.
* Plain-text documentation. If you do not want to write Sphinx documentation, you can still send us the text and we will 
format it appropriately. Send your text in the form of a `new issue`_.
* Open a `new issue`_ pointing out parts of the documentation that are unclear, misleading, or deprecated.
* Write :ref:`marvin-tutorials <tutorials>`! Tutorials can be in Sphinx or ipyhton notebook format.
* If you have used Marvin in the classroom and developed activities based on it we would love to hear about it!


.. _marvin-contributing-vacs:

Contributing a VAC
------------------

`Value Added Catalogues (VAC) <http://www.sdss.org/dr15/data_access/value-added-catalogs/>`_ are an important part of 
how SDSS and MaNGA data is analysed and distributed. Following SDSS policy, Marvin does not directly support VACs, but it 
does supply a framework for VAC owners to implement access to their catalogues from Marvin.

At this time, only file-based access is supported (i.e., no querying or remote access is available) but 
the `~.VACMixIn` class provides a convenient way of matching targets with their VAC information and returning it 
to the user. Very little knowledge of how Marvin internally works is required! The directory 
`marvin/contrib/vacs <https://github.com/sdss/marvin/blob/master/python/marvin/contrib/vacs>`__ contains the 
base code and a list of already implemented VACs that you can use as a template.

Including your VAC into Marvin involves subclassing and custom tailoring the `~.VACMixIn` class to meet your needs.
The `~.VACMixIn` class has several methods available to aid the setup of your VAC.  

* set_summary_file - defines the full file path to the main summary VAC FITS file
* get_target - extracts the row data from the VAC for the specified target observation
* get_ancillary_file - defines and downloads any paths to ancillary VAC files

Creating a `~.VACMixIn` also automatically adds your VAC to the `~.marvin.tools.vacs.VACs` tool, which you can read more 
about here. For now, let's see an example of how to put it all together.  

Example: MaNGA Morphologies from Galaxy Zoo 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example adds the `MaNGA Galaxy Zoo <https://www.sdss.org/dr15/data_access/value-added-catalogs/?vac_id=manga-morphologies-from-galaxy-zoo>`_ 
Value-Added Catalog into Marvin.  This project contains a single summary FITS file, located `here <https://dr15.sdss.org/sas/dr15/manga/morphology/galaxyzoo/>`_,
which contains all of the measured morphologies from Galazy Zoo for MaNGA targets collected in one place.  The format of the summary 
file is ``MaNGA_gz-<version>.fits``, where ``<version>`` is the version of the GZ VAC release.  The source code for this VAC's 
implementation can be found at `marvin/contrib/vacs/galaxyzoo.py <https://github.com/sdss/marvin/blob/master/python/marvin/contrib/vacs/galaxyzoo.py>`_.   

.. literalinclude:: ../../../python/marvin/contrib/vacs/galaxyzoo.py
   :language: python
   :linenos:

The file itself contains a subclass of `~.VACMixIn`.  In the docstring, we make sure to include the name of the VAC, a URL 
with a description of its contents, and a short description of what the VAC provides and what the class returns. The global section 
of the `~marvin.contrib.vacs.galaxyzoo.GZVAC` class defines:

* The ``name`` of the VAC (this is the name that users will enter to access the VAC from Marvin).
* A short one-line ``description`` of what the VAC returns
* A ``version`` dictionary that defines the relationship between Marvin releases (e.g., ``MPL-7``, ``DR15``) and internal MaNGA Galaxy Zoo versions.
* An ``include`` attribute that contains a list of Marvin Tools classes to which this VAC must be added. In this particular case we only want the Galaxy Zoo VAC to show in `~marvin.tools.cube.Cube`, `~marvin.tools.maps.Maps`, and `~marvin.tools.modelcube.ModelCube`. If ``include`` is not defined the VAC will be added to all the Marvin Tools classes with the exception of `~marvin.tools.plate.Plate`.

There are two methods required to override from `~.VACMixIn`, `~.VACMixIn.set_summary_file` and `~.VACMixIn.get_target`.

`~.VACMixIn.set_summary_file` is a method needed to define the file path to the summary VAC FITS file.  This ``summary_file`` path can
be used throughout the rest of the class when you need to access the VAC file.  It also allows the `~marvin.tools.vacs.VACs` tool
to load your VAC file and make it available separately from any Marvin Tools.

The VAC file path is defined using the `~marvin.contrib.vacs.VACMixIn.get_path` method.  This method accepts a name and a 
dictionary of path parameters that is used to construct the full filepath to your VAC, which are predefined in the 
SDSS `tree <http://sdss-tree.readthedocs.io/en/latest/>`_ product.  As Marvin uses 
`sdss_access <http://sdss-access.readthedocs.io/en/stable>`__ to define and download all necessary files, we need 
to make sure that the path to the VAC file is included in the ``tree``.  See **Defining a path** for a guide on how 
to define filepaths in ``tree``.  For this particular file, the entry in ``tree`` looks like::

    mangagalaxyzoo = $MANGA_MORPHOLOGY/galaxyzoo/MaNGA_gz-{ver}.fits  

In addition to the tree path name ``mangagalaxyzoo`` we also define a dictionary of path parameters. We use the supplied 
``release`` argument to determine the Galazy Zoo version, ``ver``.  We use `~marvin.contrib.vacs.VACMixIn.get_path` to determine 
the file path and set it as the ``summary_file``.  This also checks whether the file is already present in the local SAS.

`~.VACMixIn.get_target` is the main method needed to extract row data from the VAC file for a given target observation.  
`~.VACMixIn.get_target` receives a single, ``parent_object`` argument, which is the object (e.g., a `~marvin.tools.maps.Maps` 
instance) that is trying to access the VAC information. You can use it and its attributes to do the matching with the VAC information.  

The Galaxy Zoo VAC uses the ``mangaid`` as its target identifier.  We can extract that attribute from the ``parent_object``.  In case
the VAC file is not already present in the local SAS, we use `~marvin.contrib.vacs.VACMixIn.download_vac` to retrieve it.

Once you have the file(s), all that is left is to return the data you want from the FITS files.  Here we open the FITS file with
`astropy.io.fits.open`, use the ``mangaid`` to select the target row of interest, and return the FITS record for the given row to
the user.  

Now that we have implemented the VAC, let's make sure it works:

.. code-block:: python

    >>> from marvin.tools import Cube
    >>> cube = Cube('8485-1901')
    >>> gz_data = cube.vacs.galaxyzoo
    >>> # access the data from the summary file
    >>> print(gz_data)

    FITS_rec([(221394, 'J153010.73+484124.8', 19., 48.69020093, 232.54470389, '1-209232', 6318., 48.69020093, 232.54470389, 'original', 44., 0.93617021, 0.95360705, 44., 0.95631384, 2., 0.04255319, 0.22727273, 2., 0.04346881, 1., 0.0212766, 0., 0.01, 0.00021734, 47., 46.01, 0., 0., 0., 0., 0., 2., 1., 1., 2., 1., 2., 2., 0., 0., 0., 0., 0., 2., 1., 1., 2., 1., 2., 2., 0., 0., 0., 0., 0., 2., 1., 1., 2., 1., 2., 2., 0., 0., 0., 0., 0., 1., 0.5, 0.46666667, 1., 0.5, 0., 0., 0., 0., 0., 1., 0.5, 0.28125, 1., 0.5, 2., 2., 1., 0.02173913, 0.0952381, 1., 0.02173913, 45., 0.97826087, 0.84090909, 45., 0.97826087, 46., 46., 21., 0.47727273, 0.48571429, 21., 0.47727273, 23., 0.52272727, 0.33333333, 23., 0.52272727, 0., 0., 0., 0., 0., 44., 44., 0., nan, 1., 0., nan, 0., nan, 0.11111111, 0., nan, 0., nan, 0.22580645, 0., nan, 0., 0., 0., nan, 0., 0., nan, 0., nan, 1., 0., nan, 0., nan, 0.96525097, 0., nan, 0., 0., 0., nan, 0.4, 0., nan, 0., nan, 0.73809524, 0., nan, 0., nan, 0.12121212, 0., nan, 0., nan, 0.04651163, 0., nan, 0., nan, 0.1, 0., nan, 0., nan, 0., 0., nan, 0., 0.)],
            dtype=(numpy.record, [('nsa_id', '>i8'), ('IAUNAME', 'S19'), ('IFUDESIGNSIZE', '>f8'), ('IFU_DEC', '>f8'), ('IFU_RA', '>f8'), ('MANGAID', 'S8'), ('MANGA_TILEID', '>f8'), ('OBJECT_DEC', '>f8'), ('OBJECT_RA', '>f8'), ('survey', 'S77'), ('t01_smooth_or_features_a01_smooth_count', '>f8'), ('t01_smooth_or_features_a01_smooth_count_fraction', '>f8'), ('t01_smooth_or_features_a01_smooth_debiased', '>f8'), ('t01_smooth_or_features_a01_smooth_weight', '>f8'), ('t01_smooth_or_features_a01_smooth_weight_fraction', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_count', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_count_fraction', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_debiased', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_weight', '>f8'), ('t01_smooth_or_features_a02_features_or_disk_weight_fraction', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_count', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_count_fraction', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_debiased', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_weight', '>f8'), ('t01_smooth_or_features_a03_star_or_artifact_weight_fraction', '>f8'), ('t01_smooth_or_features_count', '>f8'), ('t01_smooth_or_features_weight', '>f8'), ('t02_edgeon_a04_yes_count', '>f8'), ('t02_edgeon_a04_yes_count_fraction', '>f8'), ('t02_edgeon_a04_yes_debiased', '>f8'), ('t02_edgeon_a04_yes_weight', '>f8'), ('t02_edgeon_a04_yes_weight_fraction', '>f8'), ('t02_edgeon_a05_no_count', '>f8'), ('t02_edgeon_a05_no_count_fraction', '>f8'), ('t02_edgeon_a05_no_debiased', '>f8'), ('t02_edgeon_a05_no_weight', '>f8'), ('t02_edgeon_a05_no_weight_fraction', '>f8'), ('t02_edgeon_count', '>f8'), ('t02_edgeon_weight', '>f8'), ('t03_bar_a06_bar_count', '>f8'), ('t03_bar_a06_bar_count_fraction', '>f8'), ('t03_bar_a06_bar_debiased', '>f8'), ('t03_bar_a06_bar_weight', '>f8'), ('t03_bar_a06_bar_weight_fraction', '>f8'), ('t03_bar_a07_no_bar_count', '>f8'), ('t03_bar_a07_no_bar_count_fraction', '>f8'), ('t03_bar_a07_no_bar_debiased', '>f8'), ('t03_bar_a07_no_bar_weight', '>f8'), ('t03_bar_a07_no_bar_weight_fraction', '>f8'), ('t03_bar_count', '>f8'), ('t03_bar_weight', '>f8'), ('t04_spiral_a08_spiral_count', '>f8'), ('t04_spiral_a08_spiral_count_fraction', '>f8'), ('t04_spiral_a08_spiral_debiased', '>f8'), ('t04_spiral_a08_spiral_weight', '>f8'), ('t04_spiral_a08_spiral_weight_fraction', '>f8'), ('t04_spiral_a09_no_spiral_count', '>f8'), ('t04_spiral_a09_no_spiral_count_fraction', '>f8'), ('t04_spiral_a09_no_spiral_debiased', '>f8'), ('t04_spiral_a09_no_spiral_weight', '>f8'), ('t04_spiral_a09_no_spiral_weight_fraction', '>f8'), ('t04_spiral_count', '>f8'), ('t04_spiral_weight', '>f8'), ('t05_bulge_prominence_a10_no_bulge_count', '>f8'), ('t05_bulge_prominence_a10_no_bulge_count_fraction', '>f8'), ('t05_bulge_prominence_a10_no_bulge_debiased', '>f8'), ('t05_bulge_prominence_a10_no_bulge_weight', '>f8'), ('t05_bulge_prominence_a10_no_bulge_weight_fraction', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_count', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_count_fraction', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_debiased', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_weight', '>f8'), ('t05_bulge_prominence_a11_just_noticeable_weight_fraction', '>f8'), ('t05_bulge_prominence_a12_obvious_count', '>f8'), ('t05_bulge_prominence_a12_obvious_count_fraction', '>f8'), ('t05_bulge_prominence_a12_obvious_debiased', '>f8'), ('t05_bulge_prominence_a12_obvious_weight', '>f8'), ('t05_bulge_prominence_a12_obvious_weight_fraction', '>f8'), ('t05_bulge_prominence_a13_dominant_count', '>f8'), ('t05_bulge_prominence_a13_dominant_count_fraction', '>f8'), ('t05_bulge_prominence_a13_dominant_debiased', '>f8'), ('t05_bulge_prominence_a13_dominant_weight', '>f8'), ('t05_bulge_prominence_a13_dominant_weight_fraction', '>f8'), ('t05_bulge_prominence_count', '>f8'), ('t05_bulge_prominence_weight', '>f8'), ('t06_odd_a14_yes_count', '>f8'), ('t06_odd_a14_yes_count_fraction', '>f8'), ('t06_odd_a14_yes_debiased', '>f8'), ('t06_odd_a14_yes_weight', '>f8'), ('t06_odd_a14_yes_weight_fraction', '>f8'), ('t06_odd_a15_no_count', '>f8'), ('t06_odd_a15_no_count_fraction', '>f8'), ('t06_odd_a15_no_debiased', '>f8'), ('t06_odd_a15_no_weight', '>f8'), ('t06_odd_a15_no_weight_fraction', '>f8'), ('t06_odd_count', '>f8'), ('t06_odd_weight', '>f8'), ('t07_rounded_a16_completely_round_count', '>f8'), ('t07_rounded_a16_completely_round_count_fraction', '>f8'), ('t07_rounded_a16_completely_round_debiased', '>f8'), ('t07_rounded_a16_completely_round_weight', '>f8'), ('t07_rounded_a16_completely_round_weight_fraction', '>f8'), ('t07_rounded_a17_in_between_count', '>f8'), ('t07_rounded_a17_in_between_count_fraction', '>f8'), ('t07_rounded_a17_in_between_debiased', '>f8'), ('t07_rounded_a17_in_between_weight', '>f8'), ('t07_rounded_a17_in_between_weight_fraction', '>f8'), ('t07_rounded_a18_cigar_shaped_count', '>f8'), ('t07_rounded_a18_cigar_shaped_count_fraction', '>f8'), ('t07_rounded_a18_cigar_shaped_debiased', '>f8'), ('t07_rounded_a18_cigar_shaped_weight', '>f8'), ('t07_rounded_a18_cigar_shaped_weight_fraction', '>f8'), ('t07_rounded_count', '>f8'), ('t07_rounded_weight', '>f8'), ('t09_bulge_shape_a25_rounded_count', '>f8'), ('t09_bulge_shape_a25_rounded_count_fraction', '>f8'), ('t09_bulge_shape_a25_rounded_debiased', '>f8'), ('t09_bulge_shape_a25_rounded_weight', '>f8'), ('t09_bulge_shape_a25_rounded_weight_fraction', '>f8'), ('t09_bulge_shape_a26_boxy_count', '>f8'), ('t09_bulge_shape_a26_boxy_count_fraction', '>f8'), ('t09_bulge_shape_a26_boxy_debiased', '>f8'), ('t09_bulge_shape_a26_boxy_weight', '>f8'), ('t09_bulge_shape_a26_boxy_weight_fraction', '>f8'), ('t09_bulge_shape_a27_no_bulge_count', '>f8'), ('t09_bulge_shape_a27_no_bulge_count_fraction', '>f8'), ('t09_bulge_shape_a27_no_bulge_debiased', '>f8'), ('t09_bulge_shape_a27_no_bulge_weight', '>f8'), ('t09_bulge_shape_a27_no_bulge_weight_fraction', '>f8'), ('t09_bulge_shape_count', '>f8'), ('t09_bulge_shape_weight', '>f8'), ('t10_arms_winding_a28_tight_count', '>f8'), ('t10_arms_winding_a28_tight_count_fraction', '>f8'), ('t10_arms_winding_a28_tight_debiased', '>f8'), ('t10_arms_winding_a28_tight_weight', '>f8'), ('t10_arms_winding_a28_tight_weight_fraction', '>f8'), ('t10_arms_winding_a29_medium_count', '>f8'), ('t10_arms_winding_a29_medium_count_fraction', '>f8'), ('t10_arms_winding_a29_medium_debiased', '>f8'), ('t10_arms_winding_a29_medium_weight', '>f8'), ('t10_arms_winding_a29_medium_weight_fraction', '>f8'), ('t10_arms_winding_a30_loose_count', '>f8'), ('t10_arms_winding_a30_loose_count_fraction', '>f8'), ('t10_arms_winding_a30_loose_debiased', '>f8'), ('t10_arms_winding_a30_loose_weight', '>f8'), ('t10_arms_winding_a30_loose_weight_fraction', '>f8'), ('t10_arms_winding_count', '>f8'), ('t10_arms_winding_weight', '>f8'), ('t11_arms_number_a31_1_count', '>f8'), ('t11_arms_number_a31_1_count_fraction', '>f8'), ('t11_arms_number_a31_1_debiased', '>f8'), ('t11_arms_number_a31_1_weight', '>f8'), ('t11_arms_number_a31_1_weight_fraction', '>f8'), ('t11_arms_number_a32_2_count', '>f8'), ('t11_arms_number_a32_2_count_fraction', '>f8'), ('t11_arms_number_a32_2_debiased', '>f8'), ('t11_arms_number_a32_2_weight', '>f8'), ('t11_arms_number_a32_2_weight_fraction', '>f8'), ('t11_arms_number_a33_3_count', '>f8'), ('t11_arms_number_a33_3_count_fraction', '>f8'), ('t11_arms_number_a33_3_debiased', '>f8'), ('t11_arms_number_a33_3_weight', '>f8'), ('t11_arms_number_a33_3_weight_fraction', '>f8'), ('t11_arms_number_a34_4_count', '>f8'), ('t11_arms_number_a34_4_count_fraction', '>f8'), ('t11_arms_number_a34_4_debiased', '>f8'), ('t11_arms_number_a34_4_weight', '>f8'), ('t11_arms_number_a34_4_weight_fraction', '>f8'), ('t11_arms_number_a36_more_than_4_count', '>f8'), ('t11_arms_number_a36_more_than_4_count_fraction', '>f8'), ('t11_arms_number_a36_more_than_4_debiased', '>f8'), ('t11_arms_number_a36_more_than_4_weight', '>f8'), ('t11_arms_number_a36_more_than_4_weight_fraction', '>f8'), ('t11_arms_number_a37_cant_tell_count', '>f8'), ('t11_arms_number_a37_cant_tell_count_fraction', '>f8'), ('t11_arms_number_a37_cant_tell_debiased', '>f8'), ('t11_arms_number_a37_cant_tell_weight', '>f8'), ('t11_arms_number_a37_cant_tell_weight_fraction', '>f8'), ('t11_arms_number_count', '>f8'), ('t11_arms_number_weight', '>f8')]))

.. _marvin-advanced-vacs:

Advanced Data Access in VACs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most VACs will only have data in a single summary FITS file.  Some VACs also have ancillary data for targets, such as spectra 
or imaging data, that you may also want to return, in which case a simple dictionary or FITS row record may not be sufficient.
For these cases, `~.VACTarget` is a customizable class that provides a framework for returning more complicated data. The
following VACs are examples where more complicated data are returned:

* `Firefly VAC <https://github.com/sdss/marvin/blob/master/python/marvin/contrib/vacs/firefly.py>`_ - returns target data from the summary FITS file as well as methods to display 2-d analysis maps
* `HI VAC <https://github.com/sdss/marvin/blob/master/python/marvin/contrib/vacs/hi.py>`_ - returns target data from the summary FITS file as well as methods to plot ancillary spectral data
* `Visual Morphology VAC <https://github.com/sdss/marvin/blob/master/python/marvin/contrib/vacs/visual_morph.py>`_ - returns target data from the summary FITS file as well as methods to display mosaic images

In each case, they subclass the `~.VACTarget` class, customize it to access and return ancillary data, and provide methods
to plot or display said ancillary data.  This class is then instantiated and returned inside the `~.VACMixIn.get_target` method. 

.. _marvin-add-methods:

Custom Added Convenience Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within your VAC module file, one may define convenience methods that become available to your main VAC catalog, when it is ingested
into the Marvin `~.VACs` Tool.  These can provide any functionality or custom plots you wish users' to be able to see/have access
to.  See the `HI VAC <https://github.com/sdss/marvin/blob/master/python/marvin/contrib/vacs/hi.py>`_ for an example.  Inside is 
defined a convenience method, ``plot_mass_fraction`` for plotting the HI mass fraction of the complete MaNGA-HI catalog.  

.. literalinclude:: ../../../python/marvin/contrib/vacs/hi.py
   :language: python
   :linenos:
   :lines: 132-160

These convenience methods accept as input a single ``vacdata_object`` argument.  This argument represents the VAC object 
defined automatically by `~.VACDataClass`.  This object has access to the complete VAC catalog using the ``data`` 
property of the `~.VACDataClass`.  In the global section, of the `~.HIVAC`, there is defined a new global attribute, 
``add_methods``, which lists any defined convenience methods to add to your VAC inside the `~.VACs` tool.

.. literalinclude:: ../../../python/marvin/contrib/vacs/hi.py
   :language: python
   :linenos:
   :lines: 45-46

Included is the name of the ``plot_mass_fraction`` method in the list of ``add_methods``.  Once defined, it becomes 
available in the `~.VACs` tools.  These defined convenience methods are meant to provide conveniences on the main VAC 
catalog, and should be **not** be target specific.  
::

    from marvin.tools.vacs import VACs
    v = VACs()
    v.HI.plot_mass_fraction()

.. _marvin-write-vacs:

Writing your own VAC
^^^^^^^^^^^^^^^^^^^^
If you are a VAC owner by now you will have hopefully decided that you want to provide access to it with Marvin. How do you do that?

#. First, make sure you read the :ref:`marvin-code-standards` and :ref:`marvin-code-dev` sections.

Setup the Github Repos:

#. You must have an account on Github.  If you do not already have one, you can sign up `here <https://github.com/join>`_.
#. Once you have a Github account set up, fork the Marvin repo.  Follow the instructions in `Forking a Repo <https://help.github.com/en/articles/fork-a-repo>`_ for help forking a repo and `keeping it in sync <https://help.github.com/en/articles/syncing-a-fork>`_ with the original source.  
#. Setup Marvin with the :ref:`marvin-install-dev`.     
    
Requirements of the VAC:

#. Each MaNGA VAC data file must be a valid astronomy FITS file format.
#. Each VAC must have a version number associated with its release, e.g. `v1.0.0`, or `1.1.0`, indicated either in the full path or name of the file.  For example, data for the DR15 MaNGA-HI VAC has the assigned version, `v1_0_1`, and points to, https://dr15.sdss.org/sas/dr15/manga/HI/v1_0_1/.  **Note:** It is acceptable if your version maps to the versions associated with a MaNGA MPL or DR release.   

Creating a VAC Path Template:

#. To enable Marvin to locate and download your VAC, a template of your VAC's file path must have first been added to `sdss_paths.ini <https://github.com/sdss/tree/blob/master/data/sdss_paths.ini>`__. Refer to the SDSS `tree <https://sdss-tree.readthedocs.io/en/latest/paths.html>`_ documentation to learn how to do that.

Writing a new VAC script:

#. Create a new file within the ``python/marvin/contrib/vacs`` directory.  You can either start fresh or select an existing example that looks similar to what you need. Within the ``vacs/`` directory, duplicate that file and rename it to the name of your VAC.
#. Modify the name of the class that inherits from `.VACMixIn` (important!) and change the ``name`` attribute to the name of your VAC. Update the docstring with information about your VAC. Most importantly, make sure the description clearly states what your VAC is returning.
#. Customize the class to suit your needs and return the data from your VAC you want users to have access to.  See the previous :ref:`marvin-contributing-vacs` section for details on what the `~.VACMixIn` is and how to use it.

Test your VAC in marvin:

#. Test your implementation by accessing your VAC as ``object.vacs.<your-vac-name>``.  See :ref:`How to Use VACs<marvin-vacs>` on how to access your VAC object from within a Marvin Tools object.
#. Test your VAC also shows up in the Marvin `VACs` tool.  
#. As the VAC owner, you are responsible for making sure your class returns the correct information. We **strongly** suggest you implement unittests to make sure that is the case. You can see some examples `here <https://github.com/sdss/marvin/blob/master/python/marvin/tests/contrib/test_vacs.py>`_.

Add Documentation for your VAC:

#. Add your VAC entry in ``docs/sphinx/contributing/available_vacs.rst``.  Copy and paste an existing entry and change the existing VAC path to point to your new VAC class.
#. Add your VAC entry in ``docs/sphinx/reference/vacs.rst``.  Under ``Available VACs``, copy and paste an existing entry.  Update the VAC heading and path to the VAC file with your new VAC information.

Finally:

#. Once you are happy with the result, open a new `Pull Request <https://help.github.com/en/articles/creating-a-pull-request-from-a-fork>`_ to integrate your changes.
#. If you have any doubt or questions, please contact the Marvin team, we will be happy to help you throughout the process.



Classes
^^^^^^^

.. autosummary::
    marvin.contrib.vacs.base.VACMixIn

.. _marvin-available-vacs:

Available VACs
^^^^^^^^^^^^^^

.. include:: ./available_vacs.rst


.. _new issue: https://github.com/sdss/marvin/issues/new
.. _sdss_access.path.Path.full: http://sdss-access.readthedocs.io/en/stable/api.html#sdss_access.path.path.BasePath.full
