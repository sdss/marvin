
.. _marvin-query-parameters:

Query Parameters
================

Marvin provides the ability to query on or return a number of different parameters.  Currently we have vetted and provided a small set.  See the :ref:`marvin_parameter_list` for the currently available list.  The naming conventions here are the same for the filter parameter names.  There are more parameters.  If you wish to query on a parameter that you do not see here, please let us know, and we will make it available.  Or if you are adventurous, you can peruse the full unvetted list using ``query.get_available_params('all')``.

.. _marvin-queryparam-getstart:

Getting Started
^^^^^^^^^^^^^^^

All Manga query parameters are grouped by parameter type.  We offer six groups of parameters, and have provided a toolset designed to help you search for and locate what parameters are available, and to easily input them into Queries.

* Metadata
* Spaxel Metadata
* Emission Lines
* Kinematics
* Spectral Indices
* NSA Catalog `(NASA Sloan-Atlas Catalog) <http://www.sdss.org/dr13/manga/manga-target-selection/nsa/>`_

From within Marvin, to list which groups are available import the **query_params** object.  The **query_params** object is a Python list object.

::

    # import the query_params
    from marvin.utils.datamodel.query.base import query_params

    # display the list of parameter groups
    query_params
    [<ParameterGroup name=Metadata, paramcount=7>,
     <ParameterGroup name=Spaxel Metadata, paramcount=3>,
     <ParameterGroup name=Emission Lines, paramcount=13>,
     <ParameterGroup name=Kinematics, paramcount=6>,
     <ParameterGroup name=Spectral Indices, paramcount=1>,
     <ParameterGroup name=NSA Catalog, paramcount=11>]

Shown is a list of all available query groups, with the number of parameters within each group. You can access individual groups, and list their parameters.

::

    # access a group
    query_params['metadata']
    <ParameterGroup name=Metadata, paramcount=7>

    # list parameters
     query_params['metadata'].list_params()
    [<QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, display=Plate-IFU>,
     <QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.ra, name=ra, short=ra, display=RA>,
     <QueryParameter full=cube.dec, name=dec, short=dec, display=Dec>,
     <QueryParameter full=cube.plate, name=plate, short=plate, display=Plate>,
     <QueryParameter full=bintype.name, name=name, short=bin, display=Bintype>,
     <QueryParameter full=template.name, name=name, short=template, display=Template>]

Access individual parameters with the same list indexing technique for groups.

::

    # grab ra and dec
    query_params['metadata']['ra']
     <QueryParameter full=cube.ra, name=ra, short=ra, display=RA>

    query_params['metadata']['dec']
    <QueryParameter full=cube.dec, name=dec, short=dec, display=Dec>

    # slice it like a list
    query_params['metadata'][0:3]
    [<QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, display=Plate-IFU>,
     <QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.ra, name=ra, short=ra, display=RA>]

To generate a list of names that are formatted as ready-input into Marvin Queries, use the **full** keyword.

::

    # format the RA query parameter
    query_params['metadata']['ra'].full
    'cube.ra'

    # format the entire list of metadata parameters
    query_params['metadata'].list_params(full=True)
    ['cube.plateifu',
     'cube.mangaid',
     'cube.ra',
     'cube.dec',
     'cube.plate',
     'bintype.name',
     'template.name']

You can combine lists.  Make a list of the galaxy RA, Dec, NSA redshift, and g-r color parameters.

::

    # make a custom list of parameters
    meta = query_params['metadata']
    nsa = query_params['nsa']
    myparams = meta.list_params(['ra','dec'], full=True) + nsa.list_params(['z', 'absmag_g_r'], full=True)

    myparams
    ['cube.ra', 'cube.dec', 'nsa.z', 'nsa.elpetro_absmag_g_r']

    # input into a Marvin Query
    from marvin.tools.query import Query
    query = Query(search_filter='nsa.z < 0.1', return_params=myparams)

If you want all parameters from all groups, use the **query_params.list_params** method.

::

    # return all parameters from all groups
    query_params.list_params()
    ['cube.plateifu',
     'cube.mangaid',
     'cube.ra',
     'cube.dec',
     'cube.plate',
     'bintype.name',
      ...
      ...
     'nsa.z',
     'nsa.elpetro_ba',
     'nsa.elpetro_mag_g_r',
     'nsa.elpetro_absmag_g_r',
     'nsa.elpetro_logmass',
     'nsa.elpetro_th50_r',
     'nsa.sersic_logmass',
     'nsa.sersic_ba']

You can also select the parameters from individual groups. Let's return all the NSA and Kinematic parameters.

::

    myparams = query_params.list_params(['nsa', 'kin'])
    myparams
    ['nsa.iauname',
     'nsa.ra',
     'nsa.dec',
     'nsa.z',
     'nsa.elpetro_ba',
     'nsa.elpetro_mag_g_r',
     'nsa.elpetro_absmag_g_r',
     'nsa.elpetro_logmass',
     'nsa.elpetro_th50_r',
     'nsa.sersic_logmass',
     'nsa.sersic_ba',
     'spaxelprop.emline_gvel_ha_6564',
     'spaxelprop.emline_gvel_oiii_5008',
     'spaxelprop.emline_gsigma_ha_6564',
     'spaxelprop.emline_gsigma_oiii_5008',
     'spaxelprop.stellar_vel',
     'spaxelprop.stellar_sigma']

We can input these directly into a Marvin Query.  Note that returning lots of parameters or a mix of spaxel and galaxy parameters may result in long query times or a large result set.

::

    from marvin.tools.query import Query
    query = Query(search_filter='nsa.z < 0.1', return_params=myparams)
    results = query.run()

    print(results.columns)
    print(results.results[0])

    [u'cube.mangaid', u'cube.plate', u'cube.plateifu', u'ifu.name', 'nsa.iauname', 'nsa.ra', 'nsa.dec', 'nsa.z', 'nsa.elpetro_ba', 'nsa.elpetro_mag_g_r', 'nsa.elpetro_absmag_g_r', 'nsa.elpetro_logmass', 'nsa.elpetro_th50_r', 'nsa.sersic_logmass', 'nsa.sersic_ba', 'spaxelprop.emline_gvel_ha_6564', 'spaxelprop.emline_gvel_oiii_5008', 'spaxelprop.emline_gsigma_ha_6564', 'spaxelprop.emline_gsigma_oiii_5008', 'spaxelprop.stellar_vel', 'spaxelprop.stellar_sigma', u'spaxelprop.x', u'spaxelprop.y']

    (u'1-209232', 8485, u'8485-1901', u'1901', u'J153010.73+484124.8', 232.544703894, 48.6902009334, 0.0407447, 0.87454, 0.646084027458681, 1.16559028625488, 9.56547591284382, 1.33067, 9.62935046578146, 0.773047, 4.95878, 0.674934, 110.361, 128.882, 32.2628, 95.9309, 6, 15)


Using Query Params
^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   Accessing Groups <tools/query/queryparams_groups>

.. toctree::
   :maxdepth: 2

   Accessing Parameters <tools/query/queryparams_params>


.. _marvin_queryparam_api

Reference/API
^^^^^^^^^^^^^

.. rubric:: Class

.. autosummary:: marvin.utils.datamodel.query.base.ParameterGroupList
.. autosummary:: marvin.utils.datamodel.query.base.ParameterGroup
.. autosummary:: marvin.utils.datamodel.query.base.QueryParameter

.. rubric:: Methods

.. autosummary::

    marvin.utils.datamodel.query.base.ParameterGroupList.list_groups
    marvin.utils.datamodel.query.base.ParameterGroupList.list_params
    marvin.utils.datamodel.query.base.ParameterGroup.list_params

.. _marvin_parameter_list:

Parameter List
^^^^^^^^^^^^^^

Metadata
--------
* **cube.plateifu**: **(default)** The plate+ifudesign name for this object
* **cube.mangaid**: **(default)** The mangaid for this object
* **cube.ra**: OBJRA - Right ascension of the science object in J2000
* **cube.dec**: OBJDEC - Declination of the science object in J2000
* **cube.plate**: **(default)** The plateid
* **bintype.name**: The type of binning used in DAP maps
* **template.name**: The stellar libary template used in DAP maps

Spaxel Metadata
---------------
* **spaxelprop.x**: The spaxel x position
* **spaxelprop.y**: The spaxel y position
* **spaxelprop.spx_snr**: The spaxel r-band signal-to-noise ratio

Emission Lines
--------------
* **spaxelprop.emline_gflux_ha_6564**: Gaussian profile integrated flux for Ha emission line
* **spaxelprop.emline_gflux_hb_4862**: Gaussian profile integrated flux for Hb emission line
* **spaxelprop.emline_gflux_nii_6549**: Gaussian profile integrated flux for NII emission line
* **spaxelprop.emline_gflux_nii_6585**: Gaussian profile integrated flux for NII emission line
* **spaxelprop.emline_gflux_oiid_3728**: Gaussian profile integrated flux for OIId emission line
* **spaxelprop.emline_gflux_oiii_4960**: Gaussian profile integrated flux for OIII emission line
* **spaxelprop.emline_gflux_oiii_5008**: Gaussian profile integrated flux for OIII emission line
* **spaxelprop.emline_gflux_sii_6718**: Gaussian profile integrated flux for SII emission line
* **spaxelprop.emline_gflux_sii_6732**: Gaussian profile integrated flux for SII emission line
* **spaxelprop.nii_to_ha**: The NII/Ha ratio computed from emline_gflux
* **spaxelprop.oiii_to_hb**: The OIII/Hb ratio computed from emline_gflux
* **spaxelprop.sii_to_ha**: The SII/Ha ratio computed from emline_gflux
* **spaxelprop.ha_to_hb**: The Ha/Hb ratio computed from emline_gflux

Kinematics
----------
* **spaxelprop.emline_gvel_ha_6564**: Gaussian profile velocity for Ha emission line
* **spaxelprop.emline_gvel_oiii_5008**: Gaussian profile velocity for OIII emission line
* **spaxelprop.emline_gsigma_ha_6564**: Gaussian profile velocity dispersion for Ha emission line; must be corrected using EMLINE_INSTSIGMA
* **spaxelprop.emline_gsigma_oiii_5008**: Gaussian profile velocity dispersion for OIII emission line; must be corrected using EMLINE_INSTSIGMA
* **spaxelprop.stellar_vel**: Stellar velocity relative to NSA redshift
* **spaxelprop.stellar_sigma**: Stellar velocity dispersion (must be corrected using STELLAR_SIGMACORR)

Spectral Indices
----------------
* **spaxelprop.specindex_d4000**: Measurements of spectral indices

NSA Catalog
-----------
* **nsa.iauname**: The accepted IAU name
* **nsa.ra**: Right ascension of the galaxy
* **nsa.dec**: Declination of the galaxy
* **nsa.z**: The heliocentric redshift
* **nsa.elpetro_ba**: Axis ratio b/a from elliptical petrosian fit.
* **nsa.elpetro_mag_g_r**: g-r color computed from the Azimuthally-averaged SDSS-style Petrosian flux in FNugriz
* **nsa.elpetro_logmass**: Log of the stellar mass from K-correction fit in h-2 solar masses to elliptical petrosian magnitudes.
* **nsa.elpetro_th50_r**: Elliptical petrosian 50% light radius (derived from r band), in arcsec.
* **nsa.sersic_logmass**: Log of the stellar mass from 2D Sersic fit
* **nsa.sersic_ba**: Axis ratio b/a from 2D Sersic fit.

|
