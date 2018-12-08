
.. _query-dm:

Query Datamodel
===============

This page contains a general guide to the query datamodel and its usage.  Marvin provides the ability to query on or return a number of different parameters.  These parameteres are described in the datamodel, along with their relevant syntax and groupings.  The datamodel lives in `marvin.utils.datamodel.query`.  The full datamodel for any release is available :ref:`here <marvin-datamodels>`.

The following guide showcases the datamodel for DR15.  Other releases behave similarly, but may contain a different number of parameters.

Accessing
---------

You can access the query datamodel for a given release from within a built `Marvin Query` object::

    >>> query = Query(search_filter='nsa.z < 0.1')
    >>> qdm = query.datamodel
    >>> print(qdm)
    <QueryDataModel release='DR15', n_groups=5, n_parameters=987, n_total=0>

Or from importing the `datamodel` directly.  Importing the datamodel directly provides a list of available datamodels to choose from. ::

    >>> # import the list of datamodels
    >>> from marvin.utils.datamodel.query import datamodel
    >>> print(datamodel)
    [<QueryDataModel release='MPL-4', n_groups=4, n_parameters=309, n_total=0>,
     <QueryDataModel release='MPL-5', n_groups=4, n_parameters=301, n_total=0>,
     <QueryDataModel release='MPL-6', n_groups=5, n_parameters=987, n_total=0>,
     <QueryDataModel release='MPL-7', n_groups=5, n_parameters=987, n_total=0>,
     <QueryDataModel release='DR15', n_groups=5, n_parameters=987, n_total=0>]

Let's select the DR15 query datamodel::

    >>> qdm = datamodel['DR15']
    >>> print(qdm)
    <QueryDataModel release='DR15', n_groups=5, n_parameters=987, n_total=0>

.. _marvin_qdm_groups:

Groups
------

All Manga query parameters are grouped by parameter type.  We offer six groups of parameters, and have provided a toolset designed to help you search for and locate what parameters are available, and to easily input them into Queries.

* **Metadata** - metadata associated with the galaxy
* **Spaxel metadata** - metadata associated with an individual spaxel **(currently unavailable)**
* **Emission Lines** -  DAP emission-line measurements **(currently unavailable)**
* **Kinematics** - DAP kinematic measurements **(currently unavailable)**
* **Spectral Indices** - DAP spectral index measurements **(currently unavailable)**
* **NSA Catalog** `(NASA Sloan-Atlas Catalog) <http://www.sdss.org/dr13/manga/manga-target-selection/nsa/>`_ - global galaxy properties
* **ObsInfo** - parameters relevant to the observations of individual exposures
* **DAPall Summary** - measurements from the DAPall summary file
* **Other** - all other non-classified parameters

To access the list of groups, use the `groups` attribute::

    >>> qdm.groups
    [<ParameterGroup name=Metadata, n_parameters=11>,
     <ParameterGroup name=NSA Catalog, n_parameters=158>,
     <ParameterGroup name=ObsInfo, n_parameters=64>,
     <ParameterGroup name=DAPall Summary, n_parameters=620>,
     <ParameterGroup name=Other, n_parameters=134>]

Shown is a list of all available query groups, with the number of parameters within each group. You can access individual groups, and list their parameters using either the `parameters` attribute, or the `list_params` method.

::

    >>> # access a group
    >>> meta = qdm.groups['metadata']
    <ParameterGroup name=Metadata, n_parameters=11>

    >>> # list parameters in the metadata group
    >>> meta.parameters
    [<QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, remote=plateifu, display=Plate-IFU>,
     <QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, remote=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.ra, name=ra, short=ra, remote=ra, display=RA>,
     <QueryParameter full=cube.dec, name=dec, short=dec, remote=dec, display=Dec>,
     <QueryParameter full=cube.plate, name=plate, short=plate, remote=plate, display=Plate>,
     <QueryParameter full=bintype.name, name=bintype_name, short=bin, remote=bintype_name, display=Bintype>,
     <QueryParameter full=template.name, name=template_name, short=template, remote=template_name, display=Template>,
     <QueryParameter full=cube_header_keyword.label, name=label, short=label, remote=label, display=Label>,
     <QueryParameter full=cube_header_value.value, name=value, short=value, remote=value, display=Value>,
     <QueryParameter full=maps_header_keyword.name, name=maps_header_keyword_name, short=maps_header_keyword_name,   remote=maps_header_keyword_name, display=Name>,
     <QueryParameter full=maps_header_value.value, name=value, short=value, remote=value, display=Value>]

Alternatively with the `list_params` method::

    >>> meta.list_params()

Access individual parameters with the same list indexing technique for groups.

::

    >>> # grab ra and dec
    >>> meta['ra']
    <QueryParameter full=cube.ra, name=ra, short=ra, remote=ra, display=RA>

    >>> meta['dec']
    <QueryParameter full=cube.dec, name=dec, short=dec, remote=dec, display=Dec>

    >>> # slice it like a list
    >>> meta[0:3]
    [<QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, remote=plateifu, display=Plate-IFU>,
     <QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, remote=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.ra, name=ra, short=ra, remote=ra, display=RA>]

.. _marvin_qdm_params:

Parameters
----------

All queryable parameters are `QueryParameter` objects.  These provide a variety of formats for the naming of the paramter, the most important being the `full` attribute.  `full` represents the **unique** parameter name guranteed to be queryable.  Additional name formats of interest might be `short`, which provides a shortcut name to the paramter, and `display`, which provides a display name useful in plots.

To access a parameter's full name::

    >>> # RA parameter
    >>> ra = meta['ra']

    >>> ra.full
    'cube.ra'

Each `QueryParameter` also has a `property` attribute which references the DAP DataModel Property if it exists::

    >>> stvel = qdm.groups['kin'].stellar_vel
    >>> print(stvel)
    <QueryParameter full=spaxelprop.stellar_vel, name=stellar_vel, short=stvel, remote=stellar_vel, display=Stellar Velocity>

    >>> stvel.property
    <Property 'stellar_vel', channel='None', release='2.0.2', unit=u'km / s'>

.. _marvin_qdm_queryuse:

Using within Marvin Queries
---------------------------

The parameter syntax Marvin prefers for all input into **search_filter** and **return_params** is the **full** attribute on the `QueryParameter`.

To generate a list of names that are formatted as ready-input into Marvin Queries, use the **full** keyword.

::

    >>> # format the RA query parameter
    >>> meta['ra'].full
    'cube.ra'

    # format the entire list of metadata parameters
    >>> meta.list_params('full')
    ['cube.plateifu',
     'cube.mangaid',
     'cube.ra',
     'cube.dec',
     'cube.plate',
     'bintype.name',
     'template.name',
     'cube_header_keyword.label',
     'cube_header_value.value',
     'maps_header_keyword.name',
     'maps_header_value.value']

You can combine lists.  Make a list of the galaxy RA, Dec, NSA redshift, and g-r color parameters.

::

    >>> # make a custom list of parameters
    >>> meta = qdm['metadata']
    >>> nsa = qdm['nsa']
    >>> myparams = meta.list_params('full', subset=['ra','dec']) + nsa.list_params('full', subset=['nsa.z', 'absmag_g_r'])

    >>> myparams
    ['cube.ra', 'cube.dec', 'nsa.z', 'nsa.elpetro_absmag_g_r']

    >>> # input into a Marvin Query
    >>> from marvin.tools.query import Query
    >>> query = Query(search_filter='nsa.z < 0.1', return_params=myparams)

If you want all parameters from all groups, use the `groups.list_params` method with the `full` keyword.

::

    >>> # return all parameters from all groups
    >>> qdm.groups.list_params('full')
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

You can also select the parameters from individual groups. Let's return all the Kinematic parameters.

::

    >>> myparams = query_params.list_params('full', groups=[kin'])
    >>> myparams
    [ ...
     'spaxelprop.emline_gvel_ha_6564',
     'spaxelprop.emline_gvel_oiii_5008',
     'spaxelprop.emline_gsigma_ha_6564',
     'spaxelprop.emline_gsigma_oiii_5008',
     'spaxelprop.stellar_vel',
     'spaxelprop.stellar_sigma']

We can input these directly into a Marvin Query.  Note that returning lots of parameters or a mix of spaxel and galaxy parameters may result in long query times or a large result set.

::

    >>> from marvin.tools.query import Query
    >>> query = Query(search_filter='nsa.z < 0.1', return_params=myparams)
    >>> results = query.run()

    >>> results.results[0]
    ResultRow(mangaid=u'1-209232', plate=8485, plateifu=u'8485-1901', ifu_name=u'1901', emline_gvel_ha_6564=4.95878, emline_gvel_oiii_5008=0.674934, emline_gsigma_ha_6564=110.361, emline_gsigma_oiii_5008=128.882, stellar_vel=32.2628, stellar_sigma=95.9309, stellar_cont_fresid_68th_percentile=0.0358072, stellar_cont_fresid_99th_percentile=0.162992, stellar_cont_rchi2=2.03099, stellar_sigma_ivar=0.0733379, stellar_sigma_mask=0, stellar_sigmacorr=62.5568, stellar_vel_ivar=0.117477, stellar_vel_mask=0, z=0.0407447, x=6, y=15, bintype_name=u'ALL', template_name=u'GAU-MILESHC')

.. _marvin_qdm_best:

Best
----

We provide a small subset of most common parameters that have also been tested and vetted.  We call these parameters **best**.  You can access these within the datamodel::

    >>> # get a list of groups containing the best parameters
    >>> qdm.best_groups
    [<ParameterGroup name=Metadata, n_parameters=7>,
     <ParameterGroup name=NSA Catalog, n_parameters=11>]

    >>> # get a full list of best parameters
    >>> qdm.best
    [<QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, remote=plateifu, display=Plate-IFU>,
     <QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, remote=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.ra, name=ra, short=ra, remote=ra, display=RA>,
     <QueryParameter full=cube.dec, name=dec, short=dec, remote=dec, display=Dec>,
     <QueryParameter full=cube.plate, name=plate, short=plate, remote=plate, display=Plate>,
     <QueryParameter full=bintype.name, name=bintype_name, short=bin, remote=bintype_name, display=Bintype>,
     <QueryParameter full=template.name, name=template_name, short=template, remote=template_name, display=Template>,
     <QueryParameter full=nsa.iauname, name=iauname, short=iauname, remote=iauname, display=IAU Name>,
     <QueryParameter full=nsa.ra, name=ra, short=ra, remote=ra, display=RA>,
     <QueryParameter full=nsa.dec, name=dec, short=dec, remote=dec, display=Dec>,
     <QueryParameter full=nsa.z, name=z, short=z, remote=z, display=Redshift>,
     <QueryParameter full=nsa.elpetro_ba, name=elpetro_ba, short=axisratio, remote=elpetro_ba, display=Elpetro axis ratio>,
     <QueryParameter full=nsa.elpetro_mag_g_r, name=elpetro_mag_g_r, short=g_r, remote=elpetro_mag_g_r, display=g-r>,
     <QueryParameter full=nsa.elpetro_absmag_g_r, name=elpetro_absmag_g_r, short=absmag_g_r, remote=elpetro_absmag_g_r, display=Absmag g-r>,
     <QueryParameter full=nsa.elpetro_logmass, name=elpetro_logmass, short=logmass, remote=elpetro_logmass, display=Elpetro Stellar Mass>,
     <QueryParameter full=nsa.elpetro_th50_r, name=elpetro_th50_r, short=th50_r, remote=elpetro_th50_r, display=r-band half-light radius>,
     <QueryParameter full=nsa.sersic_logmass, name=sersic_logmass, short=sersic_logmass, remote=sersic_logmass, display=Sersic Stellar Mass>,
     <QueryParameter full=nsa.sersic_ba, name=sersic_ba, short=sersic_ba, remote=sersic_ba, display=Sersic axis ratio>]

.. _marvin_querydm_api

Reference/API
-------------

.. rubric:: Class Inheritance Diagram

.. inheritance-diagram:: marvin.utils.datamodel.query.base

.. rubric:: Class

.. autosummary:: marvin.utils.datamodel.query.base.DataModel
.. autosummary:: marvin.utils.datamodel.query.base.ParameterGroupList
.. autosummary:: marvin.utils.datamodel.query.base.ParameterGroup
.. autosummary:: marvin.utils.datamodel.query.base.QueryParameter

.. rubric:: Methods

.. autosummary::

    marvin.utils.datamodel.query.base.ParameterGroupList.list_groups
    marvin.utils.datamodel.query.base.ParameterGroupList.list_params
    marvin.utils.datamodel.query.base.ParameterGroup.list_params

.. _marvin_best_parameter_list:

Best Parameter List
-------------------
The list of available "best" parameters and the group they belong in.  For a full list of parameters, see the Query Datamodel.

Metadata
""""""""
* **cube.plateifu**: **(default)** The plate+ifudesign name for this object
* **cube.mangaid**: **(default)** The mangaid for this object
* **cube.ra**: OBJRA - Right ascension of the science object in J2000
* **cube.dec**: OBJDEC - Declination of the science object in J2000
* **cube.plate**: **(default)** The plateid
* **bintype.name**: The type of binning used in DAP maps
* **template.name**: The stellar libary template used in DAP maps

.. Spaxel Metadata
.. """""""""""""""
.. * **spaxelprop.x**: The spaxel x position
.. * **spaxelprop.y**: The spaxel y position
.. * **spaxelprop.spx_snr**: The spaxel r-band signal-to-noise ratio

.. Emission Lines
.. """"""""""""""
.. * **spaxelprop.emline_gflux_ha_6564**: Gaussian profile integrated flux for Ha emission line
.. * **spaxelprop.emline_gflux_hb_4862**: Gaussian profile integrated flux for Hb emission line
.. * **spaxelprop.emline_gflux_nii_6549**: Gaussian profile integrated flux for NII emission line
.. * **spaxelprop.emline_gflux_nii_6585**: Gaussian profile integrated flux for NII emission line
.. * **spaxelprop.emline_gflux_oiid_3728**: Gaussian profile integrated flux for OIId emission line
.. * **spaxelprop.emline_gflux_oiii_4960**: Gaussian profile integrated flux for OIII emission line
.. * **spaxelprop.emline_gflux_oiii_5008**: Gaussian profile integrated flux for OIII emission line
.. * **spaxelprop.emline_gflux_sii_6718**: Gaussian profile integrated flux for SII emission line
.. * **spaxelprop.emline_gflux_sii_6732**: Gaussian profile integrated flux for SII emission line
.. * **spaxelprop.nii_to_ha**: The NII/Ha ratio computed from emline_gflux
.. * **spaxelprop.oiii_to_hb**: The OIII/Hb ratio computed from emline_gflux
.. * **spaxelprop.sii_to_ha**: The SII/Ha ratio computed from emline_gflux
.. * **spaxelprop.ha_to_hb**: The Ha/Hb ratio computed from emline_gflux

.. Kinematics
.. """"""""""
.. * **spaxelprop.emline_gvel_ha_6564**: Gaussian profile velocity for Ha emission line
.. * **spaxelprop.emline_gvel_oiii_5008**: Gaussian profile velocity for OIII emission line
.. * **spaxelprop.emline_gsigma_ha_6564**: Gaussian profile velocity dispersion for Ha emission line; must be corrected using EMLINE_INSTSIGMA
.. * **spaxelprop.emline_gsigma_oiii_5008**: Gaussian profile velocity dispersion for OIII emission line; must be corrected using EMLINE_INSTSIGMA
.. * **spaxelprop.stellar_vel**: Stellar velocity relative to NSA redshift
.. * **spaxelprop.stellar_sigma**: Stellar velocity dispersion (must be corrected using STELLAR_SIGMACORR)

.. Spectral Indices
.. """"""""""""""""
.. * **spaxelprop.specindex_d4000**: Measurements of spectral indices

NSA Catalog
"""""""""""
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
