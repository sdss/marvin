
.. _marvin-queryparams_groups:

Accessing Groups
================

All query parameters are partioned into groups, to make interacting with parameters easier.  All parameter groups are accessible in the top level **query_param** object, which is importable from **marvin.tools.query.query_utils**.

Listing the Groups
------------------

You can list the available group objects straight from **query_params**.  **query_params** is a Marvin **ParameterGroupList** object and behaves like a Python list object.  Each group is a **ParameterGroup** object, which displays the **name** of the group and the number of parameters, **paramcount**, within it.  The Marvin **ParameterGroup** object also behaves as a Python list object.

::

    from marvin.utils.datamodel.query.base import query_params

    query_params
    [<ParameterGroup name=Metadata, paramcount=7>,
     <ParameterGroup name=Spaxel Metadata, paramcount=3>,
     <ParameterGroup name=Emission Lines, paramcount=13>,
     <ParameterGroup name=Kinematics, paramcount=6>,
     <ParameterGroup name=Spectral Indices, paramcount=1>,
     <ParameterGroup name=NSA Catalog, paramcount=11>]

You can also get a list of all the groups by name.

::

    query_params.list_groups()
    ['Metadata',
     'Spaxel Metadata',
     'Emission Lines',
     'Kinematics',
     'Spectral Indices',
     'NSA Catalog']


Name Indexing
-------------

You can access an item in the list by using string name indexing.  The name indexing aims to be somewhat flexible to naming and spelling convention, as long as you are close.  For example, all of the following equivalently access the NSA Catalog parameter group.

::

    query_params['NSA Catalog']
    <ParameterGroup name=NSA Catalog, paramcount=11>

    query_params['nsacatalog']

    query_params['nsa']

    query_params['catalognsa']

    query_params['nsa--cat!logue']

    query_params['sancatlogye']

If you type something that is too ambiguous, you will receive an error, and request to be more specific.

::

    query_params['meta']

    KeyError: "meta is too ambiguous.  Did you mean one of ['Metadata', 'Spaxel Metadata']?"


Slicing
-------

Since **query_params** is a Python list object, it can be sliced (i.e. indexed) like a list.

::

    query_params[2:4]

    [<ParameterGroup name=Emission Lines, paramcount=13>,
     <ParameterGroup name=Kinematics, paramcount=6>]


Listing Parameters
------------------

**query_params** also provides a method to list the individual parameters among any or all of the provided groups.  Use the **list_params** method on the main object.  If you wish to access individual parameters from individual groups, it is recommended to instead use the **list_params** access method on individual groups for grabbing specific parameters.

All
^^^

To list all parameters across all groups.

::

    query_params.list_params()

    ['cube.plateifu',
     'cube.mangaid',
     'cube.ra',
     'cube.dec',
     'cube.plate',
     'bintype.name',
     'template.name',
     'spaxelprop.x',
     'spaxelprop.y',
     'spaxelprop.spx_snr',
     'spaxelprop.emline_gflux_ha_6564',
     'spaxelprop.emline_gflux_hb_4862',
     'spaxelprop.emline_gflux_nii_6549',
     'spaxelprop.emline_gflux_nii_6585',
     'spaxelprop.emline_gflux_oiid_3728',
     'spaxelprop.emline_gflux_oiii_4960',
     'spaxelprop.emline_gflux_oiii_5008',
     'spaxelprop.emline_gflux_sii_6718',
     'spaxelprop.emline_gflux_sii_6732',
     'spaxelprop.nii_to_ha',
     'spaxelprop.oiii_to_hb',
     'spaxelprop.sii_to_ha',
     'spaxelprop.ha_to_hb',
     'spaxelprop.emline_gvel_ha_6564',
     'spaxelprop.emline_gvel_oiii_5008',
     'spaxelprop.emline_gsigma_ha_6564',
     'spaxelprop.emline_gsigma_oiii_5008',
     'spaxelprop.stellar_vel',
     'spaxelprop.stellar_sigma',
     'spaxelprop.specindex_d4000',
     'nsa.iauname',
     'nsa.ra',
     'nsa.dec',
     'nsa.z',
     'nsa.elpetro_ba',
     'nsa.elpetro_mag_g_r',
     'nsa.elpetro_absmag_g_r',
     'nsa.elpetro_logmass',
     'nsa.elpetro_th50_r',
     'nsa.sersic_logmass',
     'nsa.sersic_ba']

By Group
^^^^^^^^

To list all parameters for a given group, specify the group name in the **list_params** method.

::

    query_params.list_params('kin')

    ['spaxelprop.emline_gvel_ha_6564',
     'spaxelprop.emline_gvel_oiii_5008',
     'spaxelprop.emline_gsigma_ha_6564',
     'spaxelprop.emline_gsigma_oiii_5008',
     'spaxelprop.stellar_vel',
     'spaxelprop.stellar_sigma']

Multiple Groups
^^^^^^^^^^^^^^^

To list all parameters for a subset of groups, specify a list of group names.

::

    query_params.list_params(['kin, emission'])

    ['spaxelprop.emline_gvel_ha_6564',
     'spaxelprop.emline_gvel_oiii_5008',
     'spaxelprop.emline_gsigma_ha_6564',
     'spaxelprop.emline_gsigma_oiii_5008',
     'spaxelprop.stellar_vel',
     'spaxelprop.stellar_sigma',
     'spaxelprop.emline_gflux_ha_6564',
     'spaxelprop.emline_gflux_hb_4862',
     'spaxelprop.emline_gflux_nii_6549',
     'spaxelprop.emline_gflux_nii_6585',
     'spaxelprop.emline_gflux_oiid_3728',
     'spaxelprop.emline_gflux_oiii_4960',
     'spaxelprop.emline_gflux_oiii_5008',
     'spaxelprop.emline_gflux_sii_6718',
     'spaxelprop.emline_gflux_sii_6732',
     'spaxelprop.nii_to_ha',
     'spaxelprop.oiii_to_hb',
     'spaxelprop.sii_to_ha',
     'spaxelprop.ha_to_hb']

Input into Queries
------------------

Using **query_params**, it is easy to grab a set of paramters to return in a Marvin Query

::

    from marvin.tools.query import Query

    # grab the kinematic parameters
    kinparams = query_params.list_params('kin')

    # run a query and returning the kinematic parameters
    query = Query('haflux > 25', return_params=kinparams)
    results = query.run()

    print(results.columns)
    print(results.results[0])

    [u'cube.mangaid', u'cube.plate', u'cube.plateifu', u'ifu.name', 'spaxelprop.emline_gvel_ha_6564', 'spaxelprop.emline_gvel_oiii_5008', 'spaxelprop.emline_gsigma_ha_6564', 'spaxelprop.emline_gsigma_oiii_5008', 'spaxelprop.stellar_vel', 'spaxelprop.stellar_sigma', u'emline_gflux_ha_6564', u'spaxelprop.x', u'spaxelprop.y']

    (u'1-209232', 8485, u'8485-1901', u'1901', -22.0634, -10.3607, 102.44, 118.802, 19.9859, 87.1722, 26.3961, 16, 16)





|
