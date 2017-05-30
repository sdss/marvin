
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


Filters
-------

See the :ref:`marvin-sqlboolean` tutorial on how to design search filters.  See the :ref:`marvin-query-examples` for examples of how to write MaNGA specific filter strings.


Return Parameters
-----------------
Queries will return a set of default parameters no matter what.  If you want to return additional parameters, input them here as a string list.  See :ref:`marvin-query-parameters` for a list of available parameters to return.

.. code-block:: python

    # To see the parameters returned in you query
    print q.params
    ['cube.mangaid', 'cube.plate', 'ifu.name', 'nsa.z']


Return Type
-----------
The results of your Query by default might not be in the format you desire.  Instead you may want to return a list of Marvin Tool objects such as Cubes, Spaxels, or Maps.  The return type can be

* **cube** - returns a :ref:`marvin-tools-cube` object

**NOTE**: This is time intensive.  Depending on the size of your results, this conversion may take awhile.  Be wary.


Query Timing
------------
Query requests have an default timeout of 5 minutes.  Most queries should finish within this time.  However, for time-consuming queries, you may wish to follow these guidelines: :ref:`marvin-query-practice`.


Simple Query
------------

Simple Query from initialization

.. code-block:: python

    # import the query class
    from marvin.tools.query import Query

    # make a new query that searches for all galaxies with NSA z < 0.1
    q = Query(searchfilter='nsa.z < 0.1')

    # to see the parameters your query will return
    q.params
    [u'cube.mangaid', u'cube.plate', u'cube.plateifu', u'ifu.name', u'nsa.z']

    # run the query
    q.run()

    # let's also return the RA and Dec of each cube
    returnparams = ['cube.ra', 'cube.dec']
    q = Query(seachfilter='nsa.z < 0.1', returnparams=returnparams)

    q.params
    [u'cube.mangaid', u'cube.plate', u'cube.plateifu', u'ifu.name', u'cube.ra', u'cube.dec', u'nsa.z']


Get the :ref:`marvin-results` from a query.

.. code-block:: python

    # run your query and return a Marvin Results object
    r = q.run()

    # the actual results are stored in r.results as a list of NamedTuples
    r.results

which returns a list of `NamedTuples <https://docs.python.org/2/library/collections.html#collections.namedtuple>`_.

.. code-block:: python

    [NamedTuple(mangaid=u'1-22286', plate=7992, plateifu=u'7992-12704', name=u'12704', z=0.099954180419445),
     NamedTuple(mangaid=u'1-22298', plate=7992, plateifu=u'7992-12702', name=u'12702', z=0.0614774264395237),
     NamedTuple(mangaid=u'1-22333', plate=7992, plateifu=u'7992-3704', name=u'3704', z=0.0366250574588776),
     NamedTuple(mangaid=u'1-22347', plate=7992, plateifu=u'7992-3701', name=u'3701', z=0.0437936186790466),
     NamedTuple(mangaid=u'1-22383', plate=7992, plateifu=u'7992-3702', name=u'3702', z=0.0542150922119617),
     NamedTuple(mangaid=u'1-22412', plate=7992, plateifu=u'7992-9101', name=u'9101', z=0.0190997123718262),
     NamedTuple(mangaid=u'1-22414', plate=7992, plateifu=u'7992-6103', name=u'6103', z=0.0922721400856972),
     NamedTuple(mangaid=u'1-22438', plate=7992, plateifu=u'7992-1901', name=u'1901', z=0.016383046284318),
     NamedTuple(mangaid=u'1-22662', plate=7992, plateifu=u'7992-6104', name=u'6104', z=0.027131162583828),
     NamedTuple(mangaid=u'1-22970', plate=7992, plateifu=u'7992-3703', name=u'3703', z=0.0564263463020325)]

Do it all at once using the doQuery method.  doQuery accepts all the same arguments and keywords as Query.

.. code-block:: python

    # import it
    from marvin.tools.query import doQuery

    # run the query and retrieve the results in one step
    q, r = doQuery(searchfilter='nsa.z < 0.1')

    # look at results
    r.results

See :ref:`marvin-query-examples` for examples of different types of queries.  When you want to perform a new query or update an old query, currently, you must start a fresh query, or run ```q.reset()```.


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


Saving and Restoring Your Queries
---------------------------------

Using `Python pickling <https://docs.python.org/2/library/pickle.html>`_, Marvin can save your queries locally, and restore them later for use again.

Saving
^^^^^^

.. code-block:: python

    # make a query
    f = 'nsa.sersic_logmass < 11 and nsa.z < 0.1'
    q = Query(searchfilter=f)
    print(q)
    Marvin Query(mode='remote', limit=100, sort=None, order='asc')

    # save it for later
    q.save('myquery')
    '/Users/Brian/myquery.mpf'

Restoring
^^^^^^^^^

Restoring is a Marvin Query class method.  That means you run it from the class itself after import.

.. code-block:: python

    # import the Query class
    from marvin.tools.query import Query

    # Load a saved query from a pickle file
    newq = Query.restore('/Users/Brian/myquery.mpf')

    # Your query is now loaded
    print(newq)
    Marvin Query(mode='remote', limit=100, sort=None, order='asc')
    newq.searchfilter
    'sersic_logmass >= 9.5 and sersic_logmass < 11 and sersic_n < 2'

.. _marvin-query_getstart:

Getting Started
^^^^^^^^^^^^^^^

The basic usage of searching the MaNGA dataset with Marvin Queries is shown below.  Queries allow you to perform searches filtering the sample on specific parameter conditions, as well return additonal desired parameters.  Queries accept two basic keywords, **searchfilter** and **returnparams**.

You search the MaNGA dataset by constructing a string filter condition in a pseudo-SQL syntax of **parameter operand value**.  You only need to care about constructing your filter or **where clause**, and Marvin will do the rest.

* **Condition I Want**: find all galaxies with a redshift less than 0.1.
* **Construction**: **Parameter**: 'redshift (z or nsa.z)' + **Operand**: less then (<) + **Value**: 0.1
* **Marvin Filter Syntax**: 'nsa.z < 0.1'

::

    from marvin.tools.query import Query

    # search for galaxies with an NSA redshift < 0.1
    myfilter = 'nsa.z < 0.1'

    # create a query
    query = Query(searchfilter=myfilter)

You can optionally return parameters using the **returnparams** keyword, specified as a list of strings.

::

    # return the galaxy RA and Dec as well
    myfilter = 'nsa.z < 0.1'
    myparams = ['cube.ra', 'cube.dec']

    query = Query(searchfilter=myfilter, returnparams=myparams)

To see what parameters are available for returning and searching on, see the :ref:`marvin_parameter_list` on the :ref:`marvin-query-parameters` page.

::

Finally, to run the query

::

    results = query.run()

Queries will always return a set of default parameters: the galaxy **mangaid**, **plateifu**, **plate id**, and **ifu design name**.  Additionally, queries will always return any parameters used in your filter condition, plus any requested return parameters.

::

    # see the returned columns
    print(results.columns)
    [u'cube.mangaid', u'cube.plate', u'cube.plateifu', u'ifu.name', 'cube.ra', 'cube.dec', 'nsa.z']

    # look at the first row result
    print(results.results[0])
    (u'1-209232', 8485, u'8485-1901', u'1901', 232.544703894, 48.6902009334, 0.0407447)

.. _marvin_query_using

Using Query
^^^^^^^^^^^

* Applying a Filter
* Manipulating Queries
* Handling Return Parameters
* Returning Marvin objects
* Saving Queries

.. .. toctree::
..    :maxdepth: 2

..    Accessing Groups <tools/query/queryparams_groups>

.. .. toctree::
..    :maxdepth: 2

..    Accessing Parameters <tools/query/queryparams_params>


.. _marvin_query_api

Reference/API
^^^^^^^^^^^^^

.. rubric:: Class

.. autosummary:: marvin.tools.query.query.Query

.. rubric:: Methods

.. autosummary::

    marvin.tools.query.query.Query.reset
    marvin.tools.query.query.Query.run
    marvin.tools.query.query.Query.show
    marvin.tools.query.query.Query.save
    marvin.tools.query.query.Query.restore


|


