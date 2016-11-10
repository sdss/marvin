
.. _marvin-results:

Results
=======

When you perform a :ref:`marvin-query` in Marvin, you get a Results object.  This page describes some basic manipulation of query Results.  See the :ref:`marvin-results-ref` Reference for a full description of all the methods available.

To perform a simple Query

.. code-block:: python

    from marvin.tools.query import Query
    q = Query(searchfilter='nsa.z < 0.1')

and get the Results

.. code-block:: python

    r = q.run()

    # number of results
    r.totalcount
    1219

    # print results.
    r.results
    [NamedTuple(mangaid=u'1-22286', plate=7992, name=u'12704', z=0.099954180419445),
     NamedTuple(mangaid=u'1-22298', plate=7992, name=u'12702', z=0.0614774264395237),
     NamedTuple(mangaid=u'1-22333', plate=7992, name=u'3704', z=0.0366250574588776),
     NamedTuple(mangaid=u'1-22347', plate=7992, name=u'3701', z=0.0437936186790466),
     NamedTuple(mangaid=u'1-22383', plate=7992, name=u'3702', z=0.0542150922119617),
     NamedTuple(mangaid=u'1-22412', plate=7992, name=u'9101', z=0.0190997123718262),
     NamedTuple(mangaid=u'1-22414', plate=7992, name=u'6103', z=0.0922721400856972),
     NamedTuple(mangaid=u'1-22438', plate=7992, name=u'1901', z=0.016383046284318),
     NamedTuple(mangaid=u'1-22662', plate=7992, name=u'6104', z=0.027131162583828),
     NamedTuple(mangaid=u'1-22970', plate=7992, name=u'3703', z=0.0564263463020325)]


.. _marvin-results-singlestep:

Alternatively, you can perform it in a single step

.. code-block:: python

    from marvin.tools.query import doQuery
    q, r = doQuery(searchfilter='nsa.z < 0.1')


.. _marvin-results-view:

Viewing Results
---------------
The query results are stored in the r.results attribute.  This returns a list of `NamedTuples <https://docs.python.org/2/library/collections.html#collections.namedtuple>`_.

.. code-block:: python

    r.results
    [NamedTuple(mangaid=u'1-22286', plate=7992, name=u'12704', z=0.099954180419445),
     NamedTuple(mangaid=u'1-22298', plate=7992, name=u'12702', z=0.0614774264395237),
     NamedTuple(mangaid=u'1-22333', plate=7992, name=u'3704', z=0.0366250574588776),
     NamedTuple(mangaid=u'1-22347', plate=7992, name=u'3701', z=0.0437936186790466),
     NamedTuple(mangaid=u'1-22383', plate=7992, name=u'3702', z=0.0542150922119617),
     NamedTuple(mangaid=u'1-22412', plate=7992, name=u'9101', z=0.0190997123718262),
     NamedTuple(mangaid=u'1-22414', plate=7992, name=u'6103', z=0.0922721400856972),
     NamedTuple(mangaid=u'1-22438', plate=7992, name=u'1901', z=0.016383046284318),
     NamedTuple(mangaid=u'1-22662', plate=7992, name=u'6104', z=0.027131162583828),
     NamedTuple(mangaid=u'1-22970', plate=7992, name=u'3703', z=0.0564263463020325)]

For an introduction to NamedTuples see `here
<http://stackoverflow.com/questions/2970608/what-are-named-tuples-in-python>`_.


To View the Column Names

.. code-block:: python

    columns = r.getColumns()
    print(columns)
    [u'mangaid', u'plate', u'name', u'z']

or to view the Full Column Names

.. code-block:: python

    fullnames = r.mapColumnsToParams()
    print(fullnames)
    ['cube.mangaid', 'cube.plate', 'ifu.name', 'nsa.z']

Query Time
^^^^^^^^^^

The total runtime of your query that produced the results is saved as a Python `datetime.timedelta <https://docs.python.org/2/library/datetime.html#timedelta-objects>`_ object.

.. code-block:: python

    # see the runtime as a timedelta object broken into [days, seconds, microseconds]
    print(r.query_runtime)
    datetime.timedelta(0, 0, 19873)

    # see the total runtime in seconds
    print(r.query_runtime.total_seconds())
    0.019873

.. _marvin-results-pages:

Get Next/Previous Chunks in List
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For results over 1000 objects, Marvin automatically paginates results in groups
of 100. If you have the returntype attribute specified, then you will also
generate the new Marvin Tools for the new chunk.  You can view the next or
previous chunk with

.. code-block:: python

    r.getNext()
    r.getPrevious()

You can also specify a chunk value

.. code-block:: python

    # get next 5 entries
    r.getNext(5)
    INFO: Retrieving next 5, from 10 to 15
    [NamedTuple(mangaid=u'1-23023', plate=7992, name=u'1902', z=0.0270670596510172),
     NamedTuple(mangaid=u'1-23877', plate=7990, name=u'12702', z=0.0283643137663603),
     NamedTuple(mangaid=u'1-23891', plate=7991, name=u'3704', z=0.0274681802839041),
     NamedTuple(mangaid=u'1-23894', plate=7990, name=u'3701', z=0.0304149892181158),
     NamedTuple(mangaid=u'1-23914', plate=7990, name=u'9101', z=0.028008446097374)]

    # get previous 5 entries
    r.getPrevious(5)
    INFO: Retrieving previous 5, from 5 to 10
    [NamedTuple(mangaid=u'1-22412', plate=7992, name=u'9101', z=0.0190997123718262),
     NamedTuple(mangaid=u'1-22414', plate=7992, name=u'6103', z=0.0922721400856972),
     NamedTuple(mangaid=u'1-22438', plate=7992, name=u'1901', z=0.016383046284318),
     NamedTuple(mangaid=u'1-22662', plate=7992, name=u'6104', z=0.027131162583828),
     NamedTuple(mangaid=u'1-22970', plate=7992, name=u'3703', z=0.0564263463020325)]

.. _marvin-results-subset:

Get Subset
^^^^^^^^^^
To retrieve a subset of the results, use ``r.getSubset``.  getSubset works by specifying the starting index to grab from and a limit
on the number to grab (default is 10).  Having the returntype specified will also generate the corresponding Marvin Tools for the new
subset

.. code-block:: python

    # Get the count of objects in results
    r.totalcount
    1219

    # Get a subet of 10 objects starting at index 100
    r.getSubset(100)
    [NamedTuple(mangaid=u'1-44117', plate=8141, name=u'12705', z=0.0477223694324493),
     NamedTuple(mangaid=u'1-44141', plate=8141, name=u'3704', z=0.0473998412489891),
     NamedTuple(mangaid=u'1-44163', plate=8141, name=u'6102', z=0.031343836337328),
     NamedTuple(mangaid=u'1-44172', plate=8141, name=u'12704', z=0.0482183173298836),
     NamedTuple(mangaid=u'1-44180', plate=8141, name=u'3701', z=0.0315594673156738),
     NamedTuple(mangaid=u'1-44183', plate=8138, name=u'3704', z=0.0262834001332521),
     NamedTuple(mangaid=u'1-44216', plate=8138, name=u'3701', z=0.0495306216180325),
     NamedTuple(mangaid=u'1-44219', plate=8138, name=u'9102', z=0.0633076727390289),
     NamedTuple(mangaid=u'1-44418', plate=8143, name=u'3704', z=0.0315773263573647),
     NamedTuple(mangaid=u'1-44436', plate=8143, name=u'6103', z=0.0435708276927471)]

    # Get a subset of 5 objects starting at index 25
    r.getSubset(25, limit=5)
    [NamedTuple(mangaid=u'1-24390', plate=7990, name=u'3702', z=0.0296944621950388),
     NamedTuple(mangaid=u'1-24476', plate=7990, name=u'12705', z=0.0295156575739384),
     NamedTuple(mangaid=u'1-25554', plate=7990, name=u'12704', z=0.0268193148076534),
     NamedTuple(mangaid=u'1-25593', plate=7990, name=u'6104', z=0.0261989794671535),
     NamedTuple(mangaid=u'1-25609', plate=7990, name=u'9102', z=0.0291846375912428)]

.. _marvin-results-downlaod:

Downloading Results
-------------------

Download the results of your query.  The downloaded object (FITS file) is
determined by the returntype parameter, which defaults to cube if not specified.

.. code-block:: python

    r.download()


.. _marvin-results-sort:

Sorting Results
---------------
You can sort the results on specific columns

.. code-block:: python

    r.getColumns()
    [u'mangaid', u'plate', u'name', u'z']

    r.results
    [NamedTuple(mangaid=u'1-22286', plate=7992, name=u'12704', z=0.099954180419445),
     NamedTuple(mangaid=u'1-22298', plate=7992, name=u'12702', z=0.0614774264395237),
     NamedTuple(mangaid=u'1-22333', plate=7992, name=u'3704', z=0.0366250574588776),
     NamedTuple(mangaid=u'1-22347', plate=7992, name=u'3701', z=0.0437936186790466),
     NamedTuple(mangaid=u'1-22383', plate=7992, name=u'3702', z=0.0542150922119617),
     NamedTuple(mangaid=u'1-22412', plate=7992, name=u'9101', z=0.0190997123718262),
     NamedTuple(mangaid=u'1-22414', plate=7992, name=u'6103', z=0.0922721400856972),
     NamedTuple(mangaid=u'1-22438', plate=7992, name=u'1901', z=0.016383046284318),
     NamedTuple(mangaid=u'1-22662', plate=7992, name=u'6104', z=0.027131162583828),
     NamedTuple(mangaid=u'1-22970', plate=7992, name=u'3703', z=0.0564263463020325)]

    # Sort the results by mangaid
    r.sort('mangaid')
    [NamedTuple(mangaid=u'1-22286', plate=7992, name=u'12704', z=0.099954180419445),
     NamedTuple(mangaid=u'1-22298', plate=7992, name=u'12702', z=0.0614774264395237),
     NamedTuple(mangaid=u'1-22333', plate=7992, name=u'3704', z=0.0366250574588776),
     NamedTuple(mangaid=u'1-22347', plate=7992, name=u'3701', z=0.0437936186790466),
     NamedTuple(mangaid=u'1-22383', plate=7992, name=u'3702', z=0.0542150922119617),
     NamedTuple(mangaid=u'1-22412', plate=7992, name=u'9101', z=0.0190997123718262),
     NamedTuple(mangaid=u'1-22414', plate=7992, name=u'6103', z=0.0922721400856972),
     NamedTuple(mangaid=u'1-22438', plate=7992, name=u'1901', z=0.016383046284318),
     NamedTuple(mangaid=u'1-22662', plate=7992, name=u'6104', z=0.027131162583828),
     NamedTuple(mangaid=u'1-22970', plate=7992, name=u'3703', z=0.0564263463020325)]

    # Sort the results by IFU name in descending order
    r.sort('ifu.name', order='desc')
    [NamedTuple(mangaid=u'1-22412', plate=7992, name=u'9101', z=0.0190997123718262),
     NamedTuple(mangaid=u'1-22662', plate=7992, name=u'6104', z=0.027131162583828),
     NamedTuple(mangaid=u'1-22414', plate=7992, name=u'6103', z=0.0922721400856972),
     NamedTuple(mangaid=u'1-22333', plate=7992, name=u'3704', z=0.0366250574588776),
     NamedTuple(mangaid=u'1-22970', plate=7992, name=u'3703', z=0.0564263463020325),
     NamedTuple(mangaid=u'1-22383', plate=7992, name=u'3702', z=0.0542150922119617),
     NamedTuple(mangaid=u'1-22347', plate=7992, name=u'3701', z=0.0437936186790466),
     NamedTuple(mangaid=u'1-22438', plate=7992, name=u'1901', z=0.016383046284318),
     NamedTuple(mangaid=u'1-22286', plate=7992, name=u'12704', z=0.099954180419445),
     NamedTuple(mangaid=u'1-22298', plate=7992, name=u'12702', z=0.0614774264395237)]


|

.. _marvin-results-extract:

Extracting Results
------------------
You can extract columns from the results and format them in specific ways.

Get List Of
^^^^^^^^^^^
Extract a column and return it as a single list

.. code-block:: python

    r.getListOf('mangaid')
    [u'1-22286', u'1-22298', u'1-22333', u'1-22347', u'1-22383', u'1-22412', u'1-22414', u'1-22438',
     u'1-22662', u'1-22970']

Get Dict Of
^^^^^^^^^^^
Return the results either as a list of dictionaries or a dictionary of lists

.. code-block:: python

    # Get a list of dictionaries
    r.getDictOf(format_type='listdict')
    [{'cube.mangaid': u'1-22286',
      'cube.plate': 7992,
      'ifu.name': u'12704',
      'nsa.z': 0.099954180419445}, ...]

    # Get a dictionary of lists
    r.getDictOf(format_type='dictlist')
    {'cube.mangaid': [u'1-22286', u'1-22298', u'1-22333', u'1-22347', u'1-22383', u'1-22412',
                      u'1-22414', u'1-22438', u'1-22662', u'1-22970'],
     'cube.plate': [7992, 7992, 7992, 7992, 7992, 7992, 7992, 7992, 7992, 7992],
     'ifu.name': [u'12704', u'12702', u'3704', u'3701', u'3702', u'9101', u'6103', u'1901', u'6104', u'3703'],
     'nsa.z': [0.099954180419445, 0.0614774264395237, 0.0366250574588776, 0.0437936186790466,
               0.0542150922119617, 0.0190997123718262, 0.0922721400856972, 0.016383046284318,
               0.027131162583828, 0.0564263463020325]}

    # Get a dictionary of only one parameter
    r.getDictOf('mangaid')
    [{'cube.mangaid': u'1-22286'},
     {'cube.mangaid': u'1-22298'},
     {'cube.mangaid': u'1-22333'},
     {'cube.mangaid': u'1-22347'},
     {'cube.mangaid': u'1-22383'},
     {'cube.mangaid': u'1-22412'},
     {'cube.mangaid': u'1-22414'},
     {'cube.mangaid': u'1-22438'},
     {'cube.mangaid': u'1-22662'},
     {'cube.mangaid': u'1-22970'}]

|

.. _marvin-results-convert:

Converting Your Results
-----------------------
You can convert your results to a variety of forms.

To Marvin Tool
^^^^^^^^^^^^^^
You can convert directly to Marvin Tools objects.  Available objects are Cube, Spaxel, RSS, Maps, ModelCube.  To successfully convert to
a particular Marvin object, the results must contain the minimum default information needed to uniquely create that object.  The new
Tools are stored in a separate Results attribute called **objects**.

Conversion names: 'cube', 'maps', 'spaxel', 'rss', 'modelcube'

Minimum Default Parameters:
 * Cube and RSS objects: needs at least a mangaid.
 * Spaxel object: needs a mangaID, and a X and Y position.
 * Maps and ModelCube objects: need a mangaid, a bintype, and a template

.. code-block:: python

    r = q.run()
    r.results
    [NamedTuple(mangaid=u'14-12', name=u'1901', nsa.z=-9999.0),
     NamedTuple(mangaid=u'14-13', name=u'1902', nsa.z=-9999.0),
     NamedTuple(mangaid=u'27-134', name=u'1901', nsa.z=-9999.0),
     NamedTuple(mangaid=u'27-100', name=u'1902', nsa.z=-9999.0),
     NamedTuple(mangaid=u'27-762', name=u'1901', nsa.z=-9999.0)]

    # convert results to Marvin Cube tools
    r.convertToTool('cube')
    r.objects
    [<Marvin Cube (plateifu='7444-1901', mode='remote', data_origin='api')>,
     <Marvin Cube (plateifu='7444-1902', mode='remote', data_origin='api')>,
     <Marvin Cube (plateifu='7995-1901', mode='remote', data_origin='api')>,
     <Marvin Cube (plateifu='7995-1902', mode='remote', data_origin='api')>,
     <Marvin Cube (plateifu='8000-1901', mode='remote', data_origin='api')>]


To Astropy Table
^^^^^^^^^^^^^^^^

.. code-block:: python

    r.toTable()
    <Table length=10>
    mangaid  plate   name          z
    unicode7 int64 unicode5     float64
    -------- ----- -------- ---------------
     1-22286  7992    12704 0.0999541804194
     1-22298  7992    12702 0.0614774264395
     1-22333  7992     3704 0.0366250574589
     1-22347  7992     3701  0.043793618679
     1-22383  7992     3702  0.054215092212
     1-22412  7992     9101 0.0190997123718
     1-22414  7992     6103 0.0922721400857
     1-22438  7992     1901 0.0163830462843
     1-22662  7992     6104 0.0271311625838
     1-22970  7992     3703  0.056426346302


To JSON object
^^^^^^^^^^^^^^

.. code-block:: python

    r.toJson()
    '[["1-22286", 7992, "12704", 0.099954180419445], ["1-22298", 7992, "12702", 0.0614774264395237], ["1-22333", 7992, "3704", 0.0366250574588776], ["1-22347", 7992, "3701", 0.0437936186790466], ["1-22383", 7992, "3702", 0.0542150922119617], ["1-22412", 7992, "9101", 0.0190997123718262], ["1-22414", 7992, "6103", 0.0922721400856972], ["1-22438", 7992, "1901", 0.016383046284318], ["1-22662", 7992, "6104", 0.027131162583828], ["1-22970", 7992, "3703", 0.0564263463020325]]'


|
