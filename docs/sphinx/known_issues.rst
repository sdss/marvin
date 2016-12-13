
.. _marvin-known-issues:

Known Issues in Marvin
======================

Here are a list of known issues in the latest release of Marvin.  If you have find an issue or bug not listed here, please let us know.

FYIs
----

Tools
^^^^^

* **Python 3 compatibility** - Marvin should be generally compatible with Python 3 but beware that at this point we have not run systematic tests to confirm it. It is our intention, however, to make Marvin totally Python 3 compatible as soon as possible and we will appreciate users reporting issues and problems that they found while using Marvin with Python 3 (in this case, please remember to specify the version of Python 3 you were using).

* **Coordinates** - Although we have made significant effort to test them, spaxel selection from coordinates (both indices and spherical) should not be considered science-grade yet. +/- 1 pixel offsets around the real position position are not unexpected, and the behaviour of :func:`~marvin.tools.cube.Cube.getSpaxel()` may not be consistent in all access mode (file vs API). This is especially true in case of spaxel indices measured from the centre of the cube. When doing science analysis on the data please be careful and, if possible, double check the retrieved values independently. If you find any discrepancy between the values retreived and the ones you expect, `file and issue <https://github.com/sdss/marvin/issues>`_.

* **Ivar propagation in ratio maps** - Inverse variation propagation in the ratio maps may be incorrect and should be checked carefully before using them for any science-grade purpose.

* **Queries** - Marvin Queries are currently synchronous.  This means that within one iPython session, you can submit only one query at a time, and it will block your terminal until it responds or times out.

* **getAperture** - :func:`~marvin.tools.cube.Cube.getAperture()` is currently broken due to a change in ``photutils``. This will be fixed and improved in a future version.

Web
^^^

* **Point-and-Click Model Fits** - On the individual galaxy page, the modelfits shown in the point-and-click display is from the unbinned MODELCUBE FITS files, i.e. SPX-MILESHC.

* **Dynamic DAP Maps** - For the DAP map display on the individual galaxy page, you can only choose one binning-template option for all the selected maps.

* **MPL-3 and below** - Marvin web does not yet fully support loading of Cubes from MPL-3 and below.

.. _known-browser:

* **Browser compatibility** - Marvin is not fully compatible with Safari. This is mostly due to the current
  stable version of Safari not being compliant with the latest HTML standards. In the future we will try to
  make Marvin more stable in Safari, as long as that does not mean sacrificing functionality. In the meantime,
  please use Chrome or Firefox. Alternatively, you can try the
  `beta version of Safari <https://developer.apple.com/safari/technology-preview/>`_, which is significantly
  more HTML compliant.


Bugs
----

Here are a list of known bugs:

Tools
^^^^^

* When a Cube is instantiated from a file, the Maps object derived from could be instantiated remotely even if the Maps file is present locally. See `this issue <https://github.com/sdss/marvin/issues/40>`_.

* The elliptical Petrosian colours that can be used for querying (e.g., ``petroth50_el_g_r``)
  are incorrect as they are calculated using ``petroth50_el``, which is the half-light radius and not the flux (!).
  This will be fixed in marvin 2.0.10, but beware if you are using a previous version.

Web
^^^

* **Autocomplete Galaxy ID list** - Upon intitial page load, this may initially crash and fail to load a list.  A list of possible ids should appear after navigating to a new page.

Additionally, see a list of Marvin Issues on the Github repo: `Issues <https://github.com/sdss/marvin/issues>`_
