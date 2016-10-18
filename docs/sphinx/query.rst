
.. _marvin-query:

Query
=====

Query is a new Marvin tool that allows you to programmtically search the MaNGA dataset in a pseudo-natural language format.
Queries work with just three main input keyword arguments:

* **searchfilter** - a pseudo-natural language string dictating the filter conditions on your query
* **returnparams** - a list of parameters you want the query to return (:ref:`marvin-query-parameters`)
* **returntype** - a string name indicating the type of Marvin object you wish to return

Queries take your inputs, parse the filter, run the Query, and return the results as a Marvin :ref:`marvin-results` object.
When in local mode, queries will assume you have a database to query on.  You probably don't have a database.  This means for you, queries
primarily work in remote mode.  Querying while in remote mode will trigger a Marvin-API request to the Marvin at Utah where it performs your
query and returns the results.  To see how to handle your results, go to :ref:`marvin-results`.

|

Filters
-------

See the :ref:`marvin-sqlboolean` tutorial on how to design search filters.  See the :ref:`marvin-query-examples` for examples of how to write MaNGA specific filter strings.

|

Return Parameters
-----------------
Queries will return a set of default parameters no matter what.  If you want to return additional parameters, input them here as a string list.  See :ref:`marvin-query-parameters` for a list of available parameters to return.

.. code-block:: python

    # To see the parameters returned in you query
    print q.params
    ['cube.mangaid', 'cube.plate', 'ifu.name', 'nsa.z']

|

Return Type
-----------
The results of your Query by default might not be in the format you desire.  Instead you may want to return a list of Marvin Tool objects such as Cubes, Spaxels, or Maps.  The return type can be

* **cube** - returns a :ref:`marvin-tools-cube` object

**NOTE**: This is time intensive.  Depending on the size of your results, this conversion may take awhile.  Be wary.

|

Simple Query
------------

Simple Query from initialization

.. code-block:: python

    from marvin.tools.query import Query
    q = Query(searchfilter='nsa.z < 0.1')
    q.run()

or in steps

.. code-block:: python

    searchfilter = 'nsa.z < 0.1'
    q = Query()
    q.set_filter(searchfilter=searchfilter)
    q._create_query_modelclasses()
    q._join_tables()
    q.add_condition()
    q.run()

Get Results

.. code-block:: python

    r = q.run()
    r.results

Returns

.. code-block:: python

    [(u'1-24099', 7991, u'1902', u'1902', 0.0281657855957747),
     (u'1-38103', 8082, u'1901', u'1901', 0.0285587850958109),
     (u'1-38157', 8083, u'1901', u'1901', 0.037575539201498),
     (u'1-38347', 8083, u'1902', u'1902', 0.036589004099369),
     (u'1-43214', 8135, u'1902', u'1902', 0.117997065186501),
     (u'1-43629', 8143, u'1901', u'1901', 0.031805731356144),
     (u'1-43663', 8140, u'1902', u'1902', 0.0407325178384781),
     (u'1-43679', 8140, u'1901', u'1901', 0.0286782365292311),
     (u'1-43717', 8137, u'1902', u'1902', 0.0314487814903259),
     (u'1-44047', 8143, u'1902', u'1902', 0.04137859120965)]

Do it all at once

.. code-block:: python

    from marvin.tools.query import doQuery
    q, r = doQuery(searchfilter='nsa.z < 0.1')
    r.results

See :ref:`marvin-query-examples` for examples of different types of queries.  When you want to perform a new query or update an old query, currently, you must start a fresh query, or run ```q.reset()```.

|

Show Query
----------
In **local mode**, you can see your query before you submit it.  When operating in **remote mode**, you cannot see your query before you submit, however you can examine your query after you run it.

From the Results object
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # do a query
   q = Query(searchfilter='nsa.z < 0.1')
   r = q.run()

   # show the Query
   r.showQuery()
    'SELECT mangadatadb.cube.mangaid, mangadatadb.cube.plate, mangadatadb.ifudesign.name, mangasampledb.nsa.z \nFROM mangadatadb.cube JOIN mangadatadb.ifudesign ON mangadatadb.ifudesign.pk = mangadatadb.cube.ifudesign_pk JOIN mangasampledb.manga_target ON mangasampledb.manga_target.pk = mangadatadb.cube.manga_target_pk JOIN mangasampledb.manga_target_to_nsa ON mangasampledb.manga_target.pk = mangasampledb.manga_target_to_nsa.manga_target_pk JOIN mangasampledb.nsa ON mangasampledb.nsa.pk = mangasampledb.manga_target_to_nsa.nsa_pk JOIN mangadatadb.pipeline_info AS drpalias ON drpalias.pk = mangadatadb.cube.pipeline_info_pk \nWHERE mangasampledb.nsa.z < 0.1 AND drpalias.pk = 21'

From the Query object (if in local mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # show the entire SQL query
    q.show()
    SELECT mangadatadb.cube.mangaid, mangadatadb.cube.plate, mangadatadb.ifudesign.name, mangasampledb.nsa.z
    FROM mangadatadb.cube JOIN mangadatadb.ifudesign ON mangadatadb.ifudesign.pk = mangadatadb.cube.ifudesign_pk JOIN mangasampledb.manga_target ON mangasampledb.manga_target.pk = mangadatadb.cube.manga_target_pk JOIN mangasampledb.manga_target_to_nsa ON mangasampledb.manga_target.pk = mangasampledb.manga_target_to_nsa.manga_target_pk JOIN mangasampledb.nsa ON mangasampledb.nsa.pk = mangasampledb.manga_target_to_nsa.nsa_pk JOIN mangadatadb.pipeline_info AS drpalias ON drpalias.pk = mangadatadb.cube.pipeline_info_pk

    # show only the filter condition
    q.show('filter')
    mangasampledb.nsa.z < 0.1 AND drpalias.pk = 21

    # show only the tables you have joined to
    q.show('joins') or q.show('tables')
    ['ifudesign', 'manga_target', 'manga_target_to_nsa', 'nsa']

See :ref:`marvin-query-examples` for examples of different types of queries.

Queries produce results.  Go to :ref:`marvin-results` to see how to handle your query results.
