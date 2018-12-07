
.. _marvin-known-issues:

Known Issues in Marvin
======================

|report new issue|_
-------------------

.. |report new issue| replace:: **Report New Issue**
.. _report new issue: https://github.com/sdss/marvin/issues/new


* **Sigma Corrections**:
  For MPL-6, we now raise an explicit error when attempting to apply the correction to `stellar_sigma`, using the `inst_sigma_correction` method.  The error message now suggests to upgrade to MPL-7 data.  In the web display of maps, when selecting the ``stellar_sigma`` or ``emline_sigma`` maps, we automatically apply the relevant sigma correction.  A corrected map is indicated via the **Corrected: [name]** map title.  Uncorrected maps, for example, in MPL-6, retain the original title name.

* **Marvin 2.2.1 MPL-6 Maps** - All H-alpha extensions in the Marvin MAPS, using MPL-6, map to NII_6585 extensions instead.  Additionally, the Marvin Maps for SPX_ELLCOO and BIN_LWELLCOO do not include the new channel R/Reff.  It is advisable to upgrade to Marvin 2.2.2, where these bugs have been fixed.


FYIs
````

Tools
:::::

* **MaStar Products** - Since the MaStar datamodel changed with MPL-5, Marvin does not currently handle any MaStar data products.  MaStar products **may** be accessible in MPL-4, but this has not been thoroughly vetted and tested.

* **Coordinates** - Although we have made significant effort to test them, spaxel selection from coordinates (both indices and spherical) should not be considered science-grade yet. +/- 1 pixel offsets around the real position position are not unexpected, and the behaviour of :func:`~marvin.tools.cube.Cube.getSpaxel()` may not be consistent in all access mode (file vs. API). This is especially true in case of spaxel indices measured from the centre of the cube. When doing science analysis on the data please be careful and, if possible, double check the retrieved values independently. If you find any discrepancy between the values retreived and the ones you expect, please `submit a new Github Issue <https://github.com/sdss/marvin/issues/new>`_.

* **Queries** - Marvin Queries are currently synchronous.  This means that within one iPython session, you can submit only one query at a time, and it will block your terminal until it responds or times out.

* **Query Timing** - Queries work, but timing is important.  You should craft your queries carefully so they will not crash or timeout.  See :ref:`marvin-query-practice` for best practices regarding large queries.

* **getAperture** - :func:`~marvin.tools.cube.Cube.getAperture()` is currently broken due to a change in ``photutils``. This will be fixed and improved in a future version.

* **Model Flux** - The ``model_flux`` attribute of :ref:`marvin-tools-spaxel` and :ref:`marvin-tools-bin` is the (binned) observed spectrum that the DAP fit. The ``model`` attribute is the fitted DAP spectrum.

* In auto or local more, if a tools is instantiated from a plate-ifu or mangaid, Marvin will first try to find the appropriate file in the user's local SAS. Note that any modification to the file path in the local SAS will make Marvin fail when trying to find the file. This include un-gzipping the file.

Web
:::

* **Point-and-Click Model Fits** - On the individual galaxy page, the model fits shown in the point-and-click display is from the HYB10 MODELCUBE FITS files (i.e., HYB10-GAU-MILESHC).

* **Dynamic DAP Maps** - For the DAP map display on the individual galaxy page, you can only choose one binning-template option for all the selected maps.

* **MPL-3 and below** - Marvin web does not fully support loading of Cubes from MPL-3 and below.

.. _known-browser:

* **Browser compatibility** - Marvin is not fully compatible with Safari. This is mostly due to the current
  stable version of Safari not being compliant with the latest HTML standards. In the future we will try to
  make Marvin more stable in Safari, as long as that does not mean sacrificing functionality. In the meantime, please use Chrome or Firefox. Alternatively, you can try the
  `beta version of Safari <https://developer.apple.com/safari/technology-preview/>`_, which is significantly
  more HTML compliant.


Bugs
````

Here are a list of known bugs:

Tools
:::::

* **Marvin 2.2.1 MPL-6 Maps** - All H-alpha extensions in the Marvin MAPS, using MPL-6, map to NII_6585 extensions instead.  Additionally, the Marvin Maps for SPX_ELLCOO and BIN_LWELLCOO do not include the new channel R/Reff.  It is advisable to upgrade to Marvin 2.2.2, where these bugs have been fixed.

* When a Cube is instantiated from a file, the Maps object derived from could be instantiated remotely even if the Maps file is present locally. See `this issue <https://github.com/sdss/marvin/issues/40>`_.

* **Queries** - Marvin Queries work!, but they are sometimes intermittent.  You sometimes may receive this error ``MarvinError: API Query call failed: Requests Http Status Error: 404 Client Error: Not Found for url: https://api.sdss.org/test/marvin/api/query/cubes/.``  If you do, then just wait a moment, and try your query again.  Sometimes the query succeeds on the server-side and caches your results, but fails when sending it back to you.  We don't yet know why this happens, but we are currently trying to understand this problem!


Web
:::

* **Autocomplete Galaxy ID list** - Upon initial page load, this may initially crash and fail to load a list.  A list of possible ids should appear after navigating to a new page.


Still having problems?
``````````````````````

Marvin
::::::

* `Full list of Issues, Feature Requests, and Documentation Requests <https://github.com/sdss/marvin/issues>`_
* `Source code <https://github.com/sdss/marvin>`_

DRP and DAP Known Issues
::::::::::::::::::::::::

Technical Reference Manual
''''''''''''''''''''''''''

(SDSS Collaboration access only)

* `MPL-7 <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-7/knownissues>`_
* `MPL-6 <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-6/knownissues>`_
* `MPL-5 <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/knownissues>`_
* `MPL-4 <https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-4/knownissues>`_

Specific Measurements
'''''''''''''''''''''

(SDSS Collaboration access only)

* `How much should I trust the DAP measurements? <https://trac.sdss.org/wiki/MANGA/TRM/TRM_ActiveDev/dap/GettingStarted#ProductCertifications>`_
* `Velocity Dispersion Measurements <https://trac.sdss.org/wiki/MANGA/TRM/TRM_ActiveDev/knownissues#Velocitydispersionmeasurements>`_
* `Flagging <https://trac.sdss.org/wiki/MANGA/TRM/TRM_ActiveDev/knownissues#Flagging>`_

MaNGA Technical Publications
::::::::::::::::::::::::::::

.. TODO link to DAP paper

* `Bundy et al. (2015): MaNGA Overview <http://adsabs.harvard.edu/abs/2015ApJ...798....7B>`_
* `Law et al. (2016): DRP <http://adsabs.harvard.edu/abs/2016AJ....152...83L>`_
* `Full list of MaNGA technical publications <http://www.sdss.org/science/technical_publications/#sdss-iv-manga>`_

|
