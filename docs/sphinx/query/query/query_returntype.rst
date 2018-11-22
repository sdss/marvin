.. _marvin-query_returntype:

Returning Marvin Objects
========================

By default, Marvin Queries return a list of tuples of parameters.  You can convert your query results directly into Marvin objects using the **returntype** keyword in the Query.  The return type can be

* **cube** - returns a :ref:`marvin-tools-cube` object
* **maps** - returns a :ref:`marvin-tools-maps` object
* **spaxel** - returns a :ref:`marvin-tools-spaxel` object
* **modelcube** - returns a :ref:`marvin-tools-modelcube` object

**NOTE**: This is time intensive.  Depending on the size of your results, this conversion may take awhile.  Be wary of doing this on the initial query as this will also perform the conversion on the server-side, delaying your results.

::

    # return Marvin Cube objects

    query = Query(search_filter='nsa.z < 0.1', returntype='cube')
    results = query.run()
    Converting results to Marvin Cube objects

    results.objects[0]
    <Marvin Cube (plateifu=u'8485-1901', mode='remote', data_origin='api')>

When you convert your results into Marvin objects, they are stored in the **objects** attribute in your **Results**.


|
