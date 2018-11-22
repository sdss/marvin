
.. _marvin-query-practice:

Query Best Practices
====================

At this time, Marvin Queries are synchronous.  This means you can only submit one at a time, and when you do, it blocks your Python terminal until the query completes.  In addition, there is an implicit timeout of 5 minutes.  This means if your request to the server does not receive a response within 5 minutes, it will timeout and close the connection.  Large queries may be potentially problematic.

Until we can update this process, when submitting queries, here are some things to consider to improve your querying experience.

* If you want specific properties, return them as query parameters.

* Change the query limit to return a larger set of results at a time.

* Pickle your query and results.

* For spaxel queries, explicitly filter on a specific binning scheme.

Return Specific Parameters
--------------------------

If you already know which properties you'd like to look at, return them as explicit parameters in the Query.  Rather than returning a generic result, converting it to an object, and extracting properties from there.  Converting results to Spaxel Objects can be time-consuming, depending on how you create the Spaxel Object.

Good Idea
^^^^^^^^^

.. code-block:: python

    from marvin.tools.query import Query
    filter = 'nsa.z < 0.1 and emline_gflux_ha_6564 > 25'
    params = ['emline_gflux_hb_4862', 'stellar_vel', 'nii_to_ha']
    q = Query(search_filter=filter, return_params=params)
    r = q.run()

    # get list of stellar velocities
    stvel = r.getListOf('stellar_vel')
    # or
    stvel = [res.stellar_vel for res in r.results]


Bad Idea
^^^^^^^^

.. code-block:: python

    from marvin.tools.query import Query
    filter = 'nsa.z < 0.1 and emline_gflux_ha_6564 > 25'
    q = Query(search_filter=filter)
    r = q.run()
    r.convertToTool('spaxel')

    # get list of stellar velocities
    stvel=[]
    for spaxel in r.objects:
        stvel.append(spaxel.properties.stellar_vel.value)


Change the Query Limit
----------------------
For queries returning over 1000 results, the queries become paginated returning, by default, 100 results at a time.  You can change the limit that is returned in a page with the **limit** keyword option into Query.

Good Idea
^^^^^^^^^

.. code-block:: python

    from marvin.tools.query import Query
    filter = 'nsa.z < 0.1 and emline_gflux_ha_6564 > 25'
    params = ['emline_gflux_hb_4862', 'stellar_vel', 'nii_to_ha']
    q = Query(search_filter=filter, return_params=params, limit=10000)
    r = q.run()
    Results contain of a total of 62065, only returning the first 10000 results

    # I want to collect all the results.  I can do fewer loops.
    uberresults = []
    uberresults.extend(r.results)
    for chunk in xrange(0, r.totalcount, r.limit):
        r.getNext()
        uberresults.extend(r.results)
        if r.end == r.totalcount: break

Bad Idea
^^^^^^^^

.. code-block:: python

    from marvin.tools.query import Query
    filter = 'nsa.z < 0.1 and emline_gflux_ha_6564 > 25'
    params = ['emline_gflux_hb_4862', 'stellar_vel', 'nii_to_ha']
    q = Query(search_filter=filter, return_params=params)
    r = q.run()
    Results contain of a total of 62065, only returning the first 100 results

    # I don't want to loop over 62,000 results in chunks of 100

Pickle your Query and Results
-----------------------------
For queries that take a long time, or output lots of results, it can be beneficial to pickle your results.  This saves the entire Marvin Results object as is and lets you restore it later, locally.  One trick can be to loop through your result pages, saving each set as a new pickle file.

Good Idea
^^^^^^^^^

.. code-block:: python

    from marvin.tools.query import Query
    filter = 'nsa.z < 0.1 and emline_gflux_ha_6564 > 25'
    params = ['emline_gflux_hb_4862', 'stellar_vel', 'nii_to_ha']
    q = Query(search_filter=filter, return_params=params, limit=10000)
    r = q.run()
    f='results_{0}_chunk{1}_to_{2}.mpf'.format(p.replace(' ','_'), r.start, r.end)
    r.save(f)

    # loop over and pickle each set of results
    for chunk in xrange(0, r.totalcount, r.limit):
        r.getNext()
        f='results_{0}_chunk{1}_to_{2}.mpf'.format(p.replace(' ','_'), r.start, r.end)
        r.save(f)
        if r.end == r.totalcount: break

Bad Idea
^^^^^^^^

.. code-block:: python

    Not pickling.  Or having to redo your whole query each time.

Filter your Spaxels by Bin Type
-------------------------------
The query we have been using in these examples produces 62,000 results.  These are a list of spaxels that satisfy the imposed conditions.  However by default spaxel queries will return any and all bin types, which may not be what you want.  If you want to cut down on the number of results, try filtering on a given bintype.

Good Idea
^^^^^^^^^

.. code-block:: python

    from marvin.tools.query import Query

    # Let's only get unbinned spaxels (i.e. bintype.name == 'SPX')
    filter = 'nsa.z < 0.1 and emline_gflux_ha_6564 > 25 and bintype.name==SPX'

    params = ['emline_gflux_hb_4862', 'stellar_vel', 'nii_to_ha', 'bintype.name', 'template.name']
    q = Query(search_filter=filter, return_params=params, limit=10000)
    r = q.run()
    Results contain of a total of 22054, only returning the first 10000 results

    # Now our results only contain a total of 22000. Much more manageable.

Bad Idea
^^^^^^^^

.. code-block:: python

    from marvin.tools.query import Query
    filter = 'nsa.z < 0.1 and emline_gflux_ha_6564 > 25'
    params = ['emline_gflux_hb_4862', 'stellar_vel', 'nii_to_ha', 'bintype.name', 'template.name']
    q = Query(search_filter=filter, return_params=params, limit=10000)
    r = q.run()
    Results contain of a total of 62065, only returning the first 10000 results

    # This list contains spaxels from all four bintypes.
    set(r.getListOf('bintype.name'))
    {u'ALL', u'NRE', u'SPX', u'VOR10'}


If you find other tips and tricks to improve querying, let us know and we shall include it for all to see.
