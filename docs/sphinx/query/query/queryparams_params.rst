
.. _marvin-queryparams_params:

Accessing Parameters
====================

This describes how to interact with individual parameters from a single **ParameterGroup** list object.

Listing the Params
------------------

Use the **list_params** access method.  This method provides a list of **QueryParameter** objects.

::

    meta = query_params['metadata']
    print(meta)
    <ParameterGroup name=Metadata, paramcount=7>

    meta.list_params()

    [<QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, display=Plate-IFU>,
     <QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.ra, name=ra, short=ra, display=RA>,
     <QueryParameter full=cube.dec, name=dec, short=dec, display=Dec>,
     <QueryParameter full=cube.plate, name=plate, short=plate, display=Plate>,
     <QueryParameter full=bintype.name, name=name, short=bin, display=Bintype>,
     <QueryParameter full=template.name, name=name, short=template, display=Template>]

Understanding the **QueryParameter**
------------------------------------

Each parameter is a Marvin **QueryParameter** object.  This object provides access to several naming conventions for a given parameter.

* **full**: This is the full parameter name with syntax **database_table_name.parameter_name**.  Use this naming convention for a unique and robust parameter input into Marvin Queries.

* **name**: The real name of the parameter.

* **short**: A short-hand version of the parameter name.  Eventually this will be usable as input into Marvin Queries.

* **display**: A display name usable for plots and the web display.

You can format the lists of parameters into any of these naming conventions.

::

    # format the list to the full names
    meta.list_params(full=True)
    ['cube.plateifu',
     'cube.mangaid',
     'cube.ra',
     'cube.dec',
     'cube.plate',
     'bintype.name',
     'template.name']

    # format the list to the display names
    meta.list_params(display=True)
    ['Plate-IFU', 'Manga-ID', 'RA', 'Dec', 'Plate', 'Bintype', 'Template']

Name Indexing
-------------

As with the group list, parameters can be accessed via fuzzy string names.

::

    meta['bintype.name']
    <QueryParameter full=bintype.name, name=name, short=bin, display=Bintype>

    meta['bintype']
    <QueryParameter full=bintype.name, name=name, short=bin, display=Bintype>

    meta['bin']
    <QueryParameter full=bintype.name, name=name, short=bin, display=Bintype>

    meta['mangdi']
    <QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, display=Manga-ID>

::

Slicing
-------

As with the group list, you can slice via indexing.

::

    meta[2:4]

    [<QueryParameter full=cube.ra, name=ra, short=ra, display=RA>,
     <QueryParameter full=cube.dec, name=dec, short=dec, display=Dec>]


Selecting Individual Parameters
-------------------------------

To select individual parameters, access it from the list.  To get its parameter name, use the **full** attribute.

::

    meta['mangaid']
    <QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, display=Manga-ID>

    meta['mangaid'].full
    'cube.mangaid'

To select a subset of parameters, use **list_params** and input a list of string names.  These names can also be fuzzy.

::

    # get a custom subset
    meta.list_params(['ra', 'dec', 'temp'])
    [<QueryParameter full=cube.ra, name=ra, short=ra, display=RA>,
     <QueryParameter full=cube.dec, name=dec, short=dec, display=Dec>,
     <QueryParameter full=template.name, name=name, short=template, display=Template>]

    # get as a list of usable names
    meta.list_params(['ra', 'dec', 'temp'], full=True)
    ['cube.ra', 'cube.dec', 'template.name']

You can also make a custom list of specific parameters from multiple groups.

::

    # make a list containing RA, Dec, NSA redshift, and absolute magnitude g-r color
    nsa = query_params['nsa']
    myparams = meta.list_params(['ra', 'dec'], full=True) + nsa.list_params(['z', 'absmag_g_r'], full=True)

    myparams
    ['cube.ra', 'cube.dec', 'template.name', 'nsa.z', 'nsa.elpetro_absmag_g_r']

Input into Queries
------------------

As with the groups, you can pass your custom list into Marvin Queries

::

    # build and run a query and return your custom parameter set
    from marvin.tools.query import Query
    query = Query(search_filter='nsa.z < 0.1', returnparams=myparams)
    results = query.run()

    print(results.columns)
    print(results.results[0])

    [u'cube.mangaid', u'cube.plate', u'cube.plateifu', u'ifu.name', 'cube.ra', 'cube.dec', 'nsa.z', 'nsa.elpetro_absmag_g_r']

    (u'1-209232', 8485, u'8485-1901', u'1901', 232.544703894, 48.6902009334, 0.0407447, 1.16559028625488)

|
