
.. _marvin-known-issues:

Known Issues in Marvin
======================

Here are a list of known issues in the latest release of Marvin.  If you have find an issue or bug not listed here, please let us know.

FYIs
----

Tools
^^^^^

* **Python 3 compatibility** - Marvin should be generally compatible with Python 3 but beware that at this point we have not run systematic tests to confirm it. It is our intention, however, to make Marvin totally Python 3 compatible as soon as possible and we will appreciate users reporting issues and problems that they found while using Marvin with Python 3 (in this case, please remember to specify the version of Python 3 you were using).

* **Coordinates** - Although we have made significant effort to test them, spaxel selection from coordinates (both indices and spherical) should not be considered science-grade yet. +/- 1 pixel offsets around the real position position are not unexpected, and the behaviour of ``getSpaxel`` may not be consistent in all access mode (file vs API). This is especially true in case of spaxel indices measured from the centre of the cube. When doing science analysis on the data please be careful and, if possible, double check the retrieved values independently. If you find any discrepancy between the values retreived and the ones you expect, `file and issue <https://github.com/sdss/marvin/issues>`_.

* **Queries** - Marvin Queries are currently synchronous.  This means that within one iPython session, you can submit only one query at a time, and it will block your terminal until it responds or times out.

API
^^^

Web
^^^

* **Point-and-Click Model Fits** - On the individual galaxy page, the modelfits shown in the point-and-click display is from the unbinned MODELCUBE FITS files, i.e. SPX-MILESHC.
* **Dynamic DAP Maps** - For the DAP map display on the individual galaxy page, you can only choose one binning-template option for all the selected maps.


Bugs
----

Here are a list of known bugs:

Tools
^^^^^

* When a Cube is instantiated from a file, the Maps object derived from could be instantiated remotely even if the Maps file is present locally. See `this issue <https://github.com/sdss/marvin/issues/40>`_.

API
^^^

Web
^^^

Additionally, see a list of Marvin Issues on the Github repo: `Issues <https://github.com/sdss/marvin/issues>`_
