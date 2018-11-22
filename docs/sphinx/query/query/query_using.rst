.. _marvin-query_using:

Using a Query
=============

This page describes how to interact with a Marvin Query.

Applying a Filter
-----------------

The **search_filter** keyword is a pseudo-natural language string dictating the filter conditions on your query.  It is most similar to an SQL where clause.  It's pseudo-natural language because you can type the condition almost as you would say it.  The pseudo-SQL syntax of the condition takes the form of **"parameter operand value"**.

Filters can be simple, using a single parameter: :code:`nsa.z < 0.1`.  Or complex, using a combination of many parameters: :code:`(nsa.sersic_logmass < 10 or nsa.sersic_n < 2) and (haflux > 25 or emline_sew_ha_6564 > 6)`.

See the :ref:`marvin-sqlboolean` tutorial on how to design search filters.  See the :ref:`marvin-query-examples` for examples of how to write MaNGA specific filter strings.  When you want to perform a new query or update an old query, currently, you must create a new query, or run `q.reset()`.

Functional Filters
^^^^^^^^^^^^^^^^^^

Marvin can also accept filters in functional form.  We currently only have one available, **npergood**.  This function returns the number of galaxies that have spaxels satisfying a given condition above some percentage.  For example, to **return galaxies that have H-alpha flux > 25 in more than 20% of their spaxels**, the filter is

::

    myfilter = 'npergood(haflux > 25) > 20'

Handling Return Parameters
--------------------------

Queries will always return a set of default parameters: the galaxy **mangaid**, **plateifu**, **plate id**, and **ifu design name**.  You can return addtional parameters beyond the defaults and those used in the filter with the **return_params** keyword.  This takes a list of paramater names.  All of the available parameters are contained in the Query `datamodel` attribute.  To learn more about query parameters, see :ref:`the Query Datamodel <query-dm>`.

::

    query = Query(search_filter='nsa.z < 0.1', return_params=['cube.ra', 'cube.dec'])
    results = query.run()

    print(results.columns)
    <ParameterGroup name=Columns, n_parameters=7>
     [<QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, remote=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.plate, name=plate, short=plate, remote=plate, display=Plate>,
     <QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, remote=plateifu, display=Plate-IFU>,
     <QueryParameter full=ifu.name, name=ifu_name, short=ifu_name, remote=ifu_name, display=Name>,
     <QueryParameter full=cube.ra, name=ra, short=ra, remote=ra, display=RA>,
     <QueryParameter full=cube.dec, name=dec, short=dec, remote=dec, display=Dec>,
     <QueryParameter full=nsa.z, name=z, short=z, remote=z, display=Redshift>]

    print(results.results[0])
    ResultRow(mangaid=u'1-209232', plate=8485, plateifu=u'8485-1901', ifu_name=u'1901', ra=232.544703894, dec=48.6902009334, z=0.0407447)

Alternatively you can set them after you create the query but before you run it.

::

    query = Query(search_filter='nsa.z < 0.1')
    query.set_return_params(['cube.ra', 'cube.dec'])
    results = query.run()

    print(results.columns)
    <ParameterGroup name=Columns, n_parameters=7>
     [<QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, remote=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.plate, name=plate, short=plate, remote=plate, display=Plate>,
     <QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, remote=plateifu, display=Plate-IFU>,
     <QueryParameter full=ifu.name, name=ifu_name, short=ifu_name, remote=ifu_name, display=Name>,
     <QueryParameter full=cube.ra, name=ra, short=ra, remote=ra, display=RA>,
     <QueryParameter full=cube.dec, name=dec, short=dec, remote=dec, display=Dec>,
     <QueryParameter full=nsa.z, name=z, short=z, remote=z, display=Redshift>]

We provide a subset of common, "vetted" parameters we call **best**.  You can quickly access these parameters using the `get_available_params` method.  This returns the **best** subset of the query datamodel structure by default (see :ref:`Best Parameters<marvin_qdm_best>` and :ref:`the Query Datamodel <query-dm>`).

::

    # get the vetted query_params structure
    query = Query()
    params = query.get_available_params()

    # number of vetted parameters
    len(params.list_params())
    41

    params
    [<ParameterGroup name=Metadata, paramcount=7>,
     <ParameterGroup name=Spaxel Metadata, paramcount=3>,
     <ParameterGroup name=Emission Lines, paramcount=13>,
     <ParameterGroup name=Kinematics, paramcount=6>,
     <ParameterGroup name=Spectral Indices, paramcount=1>,
     <ParameterGroup name=NSA Catalog, paramcount=11>]

You can also retrieve a list of all available parameters in Marvin.  A lot of these parameters have not yet been vetted and may not work, so use at your own risk.  Inform us if you want any of these parameters included in the vetted list.

::

    # get all the query parameters
    query = Query()
    params = query.get_available_params('all')

    # number of parameters (for MPL-5)
    len(params)
    753

    print(params)
    [u'maskbit.bit',
     u'maskbit.description',
     u'maskbit.flag',
     u'maskbit.label',
     u'maskbit_labels.flag',
     u'maskbit_labels.labels',
     u'maskbit_labels.maskbit',
     u'binid.id',
     u'spaxelprop.spaxelprops',
     u'spaxelprop.spaxelprops5',
     u'binmode.name',
     u'binmode.structures',
     u'bintype.name',
     ...
     ...
      u'nsa.xcen',
     u'nsa.xpos',
     u'nsa.ycen',
     u'nsa.ypos',
     u'nsa.z',
     u'nsa.zdist',
     u'nsa.zsdssline',
     u'nsa.zsrc']

Sorting the Query Results
-------------------------

You can return your results pre-sorted by some parameter using the **sort** keyword.

::

    query = Query(search_filter='nsa.z < 0.1', sort='nsa.z')
    results = query.run()

    print(results.results)
    <ResultSet(set=1/1, index=0:2, count_in_set=2, total=2)>
    [ResultRow(mangaid=u'12-98126', plate=7443, plateifu=u'7443-12701', ifu_name=u'12701', ra=230.50746239, dec=43.53234133, z=0.020478),
     ResultRow(mangaid=u'1-209232', plate=8485, plateifu=u'8485-1901', ifu_name=u'1901', ra=232.544703894, dec=48.6902009334, z=0.0407447)]


Changing the Result Limit
-------------------------

For queries that contain less than 1000 results, Marvin will return the entire result set.  For results above 1000 rows, Marvin will paginate the results and return only the first **100** rows.  You can change this number with the `limit` keyword.

::

    # return the first 10,000 rows
    query = Query(search_filter='haflux > 25', limit=10000)
    results = query.run()
    Results contain of a total of 67186, only returning the first 10000 results

    print(results.count, results.totalcount)
    10000, 67186


One-Step Querying
-----------------

You can create and run a query in a single step using the `doQuery` convienence function.  `doQuery` accepts all the same arguments and keywords as `Query`.

::

    # import it
    from marvin.tools.query import doQuery

    # run the query and retrieve the results in one step
    query, results = doQuery(search_filter='nsa.z < 0.1')


Showing a Query
---------------

You can see the SQL-constructed query using the **show** method.  Note that this will only work if you happen to have a local MaNGA database and are running your queries locally.  Normally this will return a warning message that you cannot see this remotely.  To see a remote query, use the **showQuery** method on your **Results**.

::

    # show the SQL constructed query
    query = Query(search_filter='nsa.z < 0.1')
    query.show()


Query Timing
------------
Query requests have a default timeout of 5 minutes.  Most queries should finish within this time.  However, for time-consuming queries, you may wish to follow these guidelines: :ref:`marvin-query-practice`.
