.. _marvin-query_using:

Using a Query
=============

This page describes how to interact with a Marvin Query.

Applying a Filter
-----------------

The **searchfilter** is a pseudo-natural language string dictating the filter conditions on your query.  It is most similar to an SQL where clause.  It's pseudo-natural language because you can type the condition almost as you would say it.  The pseudo-SQL syntax of the condition takes the form of **parameter operand value**.

Filters can be simple, using a single parameter: :code:`nsa.z < 0.1`.  Or complex, using a combination of many parameters: :code:`(nsa.sersic_logmass < 10 or nsa.sersic_n < 2) and (haflux > 25 or emline_sew_ha_6564 > 6)`.

Functional Filters
^^^^^^^^^^^^^^^^^^

Marvin can also accept filters in functional form.  We currently only have one available, **npergood**.  This function returns the number of galaxies that have spaxels satisfying a given condition above some percentage.  For example, to **return galaxies that have H-alpha flux > 25 in more than 20% of their spaxels**, the filter is

::

    myfilter = 'npergood(haflux > 25) > 20'

See the :ref:`marvin-sqlboolean` tutorial on how to design search filters.  See the :ref:`marvin-query-examples` for examples of how to write MaNGA specific filter strings.  When you want to perform a new query or update an old query, currently, you must create a new query, or run `q.reset()`.


Handling Return Parameters
--------------------------

Queries will always return a set of default parameters: the galaxy **mangaid**, **plateifu**, **plate id**, and **ifu design name**.  You can return addtional parameters beyond the defaults and those used in the filter with the **returnparams** keyword.  This takes a list of paramater names.  To learn more about query parameters, see :ref:`marvin-query-parameters`.

::

    query = Query(searchfilter='nsa.z < 0.1', returnparams=['cube.ra', 'cube.dec'])
    results = query.run()

    print(results.columns)
    [u'cube.mangaid', u'cube.plate', u'cube.plateifu', u'ifu.name', 'cube.ra', 'cube.dec', 'nsa.z']

    print(results.results[0])
    (u'1-209232', 8485, u'8485-1901', u'1901', 232.544703894, 48.6902009334, 0.0407447)

Alternatively you can set them after you create the query but before you run it.

::

    query = Query(searchfilter='nsa.z < 0.1')
    query.set_returnparams(['cube.ra', 'cube.dec'])
    results = query.run()

    print(results.columns)
    [u'cube.mangaid', u'cube.plate', u'cube.plateifu', u'ifu.name', 'cube.ra', 'cube.dec', 'nsa.z']


Sorting the Query Results
-------------------------

You can return your results pre-sorted by some parameter using the **sort** keyword.

::

    query = Query(searchfilter='nsa.z < 0.1', sort='nsa.z')
    results = query.run()

    print(results.results)
    [NamedTuple(mangaid=u'1-209151', plate=8485, plateifu=u'8485-12702', name=u'12702', z=0.0185246),
     NamedTuple(mangaid=u'1-209191', plate=8485, plateifu=u'8485-12701', name=u'12701', z=0.0234253),
     NamedTuple(mangaid=u'1-209113', plate=8485, plateifu=u'8485-1902', name=u'1902', z=0.0378877),
     NamedTuple(mangaid=u'1-209232', plate=8485, plateifu=u'8485-1901', name=u'1901', z=0.0407447),
     ...,
     ...]


Changing the Result Limit
-------------------------

For queries that contain less than 1000 results, Marvin will return the entire result set.  For results above 1000 rows, Marvin will paginate the results and return only the first **100** rows.  You can change this number with the **limit** keyword.

::

    # return the first 10,000 rows
    query = Query(searchfilter='haflux > 25', limit=10000)
    results = query.run()
    Results contain of a total of 67186, only returning the first 10000 results

    print(results.count, results.totalcount)
    10000, 67186


One-Step Querying
-----------------

You can create and run a query in a single step using the **doQuery** convienence function.  doQuery accepts all the same arguments and keywords as Query.

::

    # import it
    from marvin.tools.query import doQuery

    # run the query and retrieve the results in one step
    query, results = doQuery(searchfilter='nsa.z < 0.1')


Showing a Query
---------------

You can see the SQL-constructed query using the **show** method.  Note that this will only work if you happen to have a local MaNGA database and are running your queries locally.  Normally this will return a warning message that you cannot see this remotely.  To see a remote query, use the **showQuery** method on your **Results**.

::

    # show the SQL constructed query
    query = Query(searchfilter='nsa.z < 0.1')
    query.show()


Query Timing
------------
Query requests have a default timeout of 5 minutes.  Most queries should finish within this time.  However, for time-consuming queries, you may wish to follow these guidelines: :ref:`marvin-query-practice`.
