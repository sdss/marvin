
.. _marvin-known-issues:

Known Issues in Marvin
======================

If you have find an issue or bug not listed here, please let us know or `submit a new Github Issue <https://github.com/sdss/marvin/issues/new>`_. For a `full list of Issues, Feature Requests, and Documentation Requests <https://github.com/sdss/marvin/issues>`_ please see the `Marvin Github repo <https://github.com/sdss/marvin>`_.

FYIs
----

Tools
^^^^^

* **MaStar Products** - Since the MaStar datamodel changed with MPL-5, Marvin does not currently handle any MaStar data products.  MaStar products **may** be accessible in MPL-4, but this has not been thoroughly vetted and tested.

* **Coordinates** - Although we have made significant effort to test them, spaxel selection from coordinates (both indices and spherical) should not be considered science-grade yet. +/- 1 pixel offsets around the real position position are not unexpected, and the behaviour of :func:`~marvin.tools.cube.Cube.getSpaxel()` may not be consistent in all access mode (file vs. API). This is especially true in case of spaxel indices measured from the centre of the cube. When doing science analysis on the data please be careful and, if possible, double check the retrieved values independently. If you find any discrepancy between the values retreived and the ones you expect, please `submit a new Github Issue <https://github.com/sdss/marvin/issues/new>`_.

* **Ivar propagation in ratio maps** - Inverse variation propagation in the ratio maps is set to ``None``.

* **Queries** - Marvin Queries are currently synchronous.  This means that within one iPython session, you can submit only one query at a time, and it will block your terminal until it responds or times out.

* **getAperture** - :func:`~marvin.tools.cube.Cube.getAperture()` is currently broken due to a change in ``photutils``. This will be fixed and improved in a future version.

* **Model Flux** - The ``model_flux`` attribute of :ref:`marvin-tools-spaxel` and :ref:`marvin-tools-bin` is the (binned) observed spectrum that the DAP fit. The ``model`` attribute is the fitted DAP spectrum.

Web
^^^

* **Point-and-Click Model Fits** - On the individual galaxy page, the modelfits shown in the point-and-click display is from the unbinned MODELCUBE FITS files, i.e. SPX-MILESHC.

* **Dynamic DAP Maps** - For the DAP map display on the individual galaxy page, you can only choose one binning-template option for all the selected maps.

* **MPL-3 and below** - Marvin web does not yet fully support loading of Cubes from MPL-3 and below.

.. _known-browser:

* **Browser compatibility** - Marvin is not fully compatible with Safari. This is mostly due to the current
  stable version of Safari not being compliant with the latest HTML standards. In the future we will try to
  make Marvin more stable in Safari, as long as that does not mean sacrificing functionality. In the meantime, please use Chrome or Firefox. Alternatively, you can try the
  `beta version of Safari <https://developer.apple.com/safari/technology-preview/>`_, which is significantly
  more HTML compliant.


Bugs
----

Here are a list of known bugs:

Tools
^^^^^

* When a Cube is instantiated from a file, the Maps object derived from could be instantiated remotely even if the Maps file is present locally. See `this issue <https://github.com/sdss/marvin/issues/40>`_.

Web
^^^

* **Autocomplete Galaxy ID list** - Upon intitial page load, this may initially crash and fail to load a list.  A list of possible ids should appear after navigating to a new page.

