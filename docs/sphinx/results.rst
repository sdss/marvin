
.. _marvin-results:

Results
=======

When you perform a :ref:`marvin-query` in Marvin, you get a Results object.  This page describes some basic manipulation of query Results.  See the :ref:`marvin-results-ref` Reference for a full description of all the methods available.

To perform a simple Query::

    from marvin.tools.query import Query
    q = Query(searchfilter='nsa.z < 0.1')

and get the Results::

    r = q.run()

    # number of results
    r.count
    72

    # print results
    r.results
    [(u'1-24099', 7991, u'1902', u'1902', 0.0281657855957747),
     (u'1-38103', 8082, u'1901', u'1901', 0.0285587850958109),
     (u'1-38157', 8083, u'1901', u'1901', 0.037575539201498),
     (u'1-38347', 8083, u'1902', u'1902', 0.036589004099369),
     (u'1-43214', 8135, u'1902', u'1902', 0.117997065186501),
     ...
     ...

.. _marvin-results-singlestep

Alternatively, you can perform it in a single step::

    from marvin.tools.query import doQuery
    q, r = doQuery(searchfilter='nsa.z < 0.1')

|

Viewing Results
---------------
Basics::

    r.results
    [(u'1-24099', 7991, u'1902', u'1902', 0.0281657855957747),
     (u'1-38103', 8082, u'1901', u'1901', 0.0285587850958109),
     (u'1-38157', 8083, u'1901', u'1901', 0.037575539201498),
     (u'1-38347', 8083, u'1902', u'1902', 0.036589004099369),
     (u'1-43214', 8135, u'1902', u'1902', 0.117997065186501),
     ...
     ...

To View the Column Names::

    columns = r.getColumns()
    print columns
    [u'mangaid', u'plate', u'name', u'z']

or to view the Full Column Names::

    fullnames = r.mapColumnsToParams()
    print fullnames
    ['cube.mangaid', 'cube.plate', 'ifu.name', 'nsa.z']

Get Next/Previous Chunks in List
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For results over 150 objects, Marvin automatically paginates results in groups of 10. Currently, in local mode only, you can view the next or previous chunk with::

    r.getNext()
    r.getPrevious()

You can also specify a chunk value::

    r = q.run()
    # get next 5 entries
    r.getNext(5)
    Retrieving next 5, from 35 to 40
    [(u'4-4231', u'1902', -9999.0),
     (u'4-14340', u'1901', -9999.0),
     (u'4-14510', u'1902', -9999.0),
     (u'4-13634', u'1901', -9999.0),
     (u'4-13538', u'1902', -9999.0)]

    # get previous 5 entries
    r.getPrevious(5)
    Retrieving previous 5, from 30 to 35
    [(u'4-3988', u'1901', -9999.0),
     (u'4-3862', u'1902', -9999.0),
     (u'4-3293', u'1901', -9999.0),
     (u'4-3602', u'1902', -9999.0),
     (u'4-4602', u'1901', -9999.0)]

Get Subset
^^^^^^^^^^
To retrieve a subset of the results, use ```r.getSubset```.  getSubset works by specifying the starting index to grab from and a limit
on the number to grab (default is 10)::

    # Get the count of objects in results
    r.count
    1219L

    # Get a subet of 10 objects starting at index 100
    r.getSubset(100)
    [(u'1-44117', 8141, u'12705', 0.0477223694324493),
     (u'1-44141', 8141, u'3704', 0.0473998412489891),
     (u'1-44163', 8141, u'6102', 0.031343836337328),
     (u'1-44172', 8141, u'12704', 0.0482183173298836),
     (u'1-44180', 8141, u'3701', 0.0315594673156738),
     (u'1-44183', 8138, u'3704', 0.0262834001332521),
     (u'1-44216', 8138, u'3701', 0.0495306216180325),
     (u'1-44219', 8138, u'9102', 0.0633076727390289),
     (u'1-44418', 8143, u'3704', 0.0315773263573647),
     (u'1-44436', 8143, u'6103', 0.0435708276927471)]

    # Get a subset of 5 objects starting at index 25
    r.getSubset(25, limit=5)
    [(u'1-24390', 7990, u'3702', 0.0296944621950388),
     (u'1-24476', 7990, u'12705', 0.0295156575739384),
     (u'1-25554', 7990, u'12704', 0.0268193148076534),
     (u'1-25593', 7990, u'6104', 0.0261989794671535),
     (u'1-25609', 7990, u'9102', 0.0291846375912428)]

Get All
^^^^^^^
You get all of the results with::

    r.getAll()

When operating Marvin in remote mode, all of the results are always returned.

|

.. _marvin-results-downlaod:

Downloading Results
-------------------

Download the results of your query.  The downloaded object (FITS file) is determined by the returntype parameter, which defaults to cube if not specified.

::

    r.download()

|

.. _marvin-results-sort:

Sorting Results
---------------
You can sort the results on specific columns::

    r = q.run()
    r.getColumns()
    [u'mangaid', u'name', u'nsa.z']
    r.results
    [(u'4-3988', u'1901', -9999.0),
     (u'4-3862', u'1902', -9999.0),
     (u'4-3293', u'1901', -9999.0),
     (u'4-3602', u'1902', -9999.0),
     (u'4-4602', u'1901', -9999.0)]

    # Sort the results by mangaid
    r.sort('mangaid')
    [(u'4-3293', u'1901', -9999.0),
     (u'4-3602', u'1902', -9999.0),
     (u'4-3862', u'1902', -9999.0),
     (u'4-3988', u'1901', -9999.0),
     (u'4-4602', u'1901', -9999.0)]

    # Sort the results by IFU name in descending order
    r.sort('ifu.name', order='desc')
    [(u'4-3602', u'1902', -9999.0),
     (u'4-3862', u'1902', -9999.0),
     (u'4-3293', u'1901', -9999.0),
     (u'4-3988', u'1901', -9999.0),
     (u'4-4602', u'1901', -9999.0)]


|

.. _marvin-results-extract:

Extracting Results
------------------
You can extract columns from the results and format them in specific ways.

Get List Of
^^^^^^^^^^^
Extract a column and return it as a single list::

    r = q.run()
    r.getListOf('mangaid')
    [u'4-3988', u'4-3862', u'4-3293', u'4-3602', u'4-4602']

Get Dict Of
^^^^^^^^^^^
Return the results either as a list of dictionaries or a dictionary of lists::

    r = q.run()
    # Get a list of dictionaries
    r.getDictOf(format_type='listdict')
    [{'cube.mangaid': u'4-3988', 'ifu.name': u'1901', 'nsa.z': -9999.0},
     {'cube.mangaid': u'4-3862', 'ifu.name': u'1902', 'nsa.z': -9999.0},
     {'cube.mangaid': u'4-3293', 'ifu.name': u'1901', 'nsa.z': -9999.0},
     {'cube.mangaid': u'4-3602', 'ifu.name': u'1902', 'nsa.z': -9999.0},
     {'cube.mangaid': u'4-4602', 'ifu.name': u'1901', 'nsa.z': -9999.0}]

    # Get a dictionary of lists
    r.getDictOf(format_type='dictlist')
    {'cube.mangaid': [u'4-3988', u'4-3862', u'4-3293', u'4-3602', u'4-4602'],
     'ifu.name': [u'1901', u'1902', u'1901', u'1902', u'1901'],
     'nsa.z': [-9999.0, -9999.0, -9999.0, -9999.0, -9999.0]}

    # Get a dictionary of only one parameter
    r.getDictOf('mangaid')
    [{'cube.mangaid': u'4-3988'},
     {'cube.mangaid': u'4-3862'},
     {'cube.mangaid': u'4-3293'},
     {'cube.mangaid': u'4-3602'},
     {'cube.mangaid': u'4-4602'}]

|

.. _marvin-results-convert:

Converting Your Results
-----------------------
You can convert your results to a variety of forms.

To Marvin Tool
^^^^^^^^^^^^^^
You can convert directly to Marvin Tools objects.  Available objects are Cube, Spaxel, RSS, and MAPS.  To successfully convert to
a particular Marvin object, the results must contain the minimum default information needed to uniquely create that object.

For example, a Cube object needs at least a plate-IFU, or manga-id.  A Spaxel needs a plate-IFU or manga-ID, and a X and Y position.

::

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
::

    r = q.run()
    r.toTable()
    <Table length=5>
    mangaid    name   nsa.z
    unicode6 unicode4   float64
    -------- -------- ------------
      4-3602     1902      -9999.0
      4-3862     1902      -9999.0
      4-3293     1901      -9999.0
      4-3988     1901      -9999.0
      4-4602     1901      -9999.0


To JSON object
^^^^^^^^^^^^^^
::

    r = q.run()
    r.toJson()
    '[["4-3602", "1902", -9999.0], ["4-3862", "1902", -9999.0], ["4-3293", "1901", -9999.0],
      ["4-3988", "1901", -9999.0], ["4-4602", "1901", -9999.0]]'


|
