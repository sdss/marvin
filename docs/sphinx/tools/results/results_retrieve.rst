
.. _marvin-results_retrieve:

Retrieving Your Results
-----------------------

By default, Marvin will paginate large query results to limit strains on bandwidth.  For results with over 1000 objects,
Marvin automatically paginates results in groups of `Query.limit`, which defaults to 100.  This page describes various methods
of retrieving your query results.

.. _marvin-results-all:

Getting All the Results
^^^^^^^^^^^^^^^^^^^^^^^

If you're feeling rebellious, we provide a method of retrieving all the results in a single request, with `getAll`::

    r.count, r.totalcount
    (100, 1282)

    r.getAll()
    Returned all 1282 results

    r.results
    <ResultSet(set=1/1, index=0:1281, count_in_set=1282, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.26038932800293, z=0.0361073),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.48788070678711, z=0.0699044),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_absmag_g_r=0.543312072753906, z=0.0408897),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_absmag_g_r=0.757579803466797, z=0.028215),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', elpetro_absmag_g_r=1.09770011901855, z=0.0171611),
     ...]

This method has a built-in limit.  It will not return all the results for results with over 500,000 rows and/or 50 columns of data.  You may
override this limit with the `force` keyword argument.::

    r.getAll(force=True)

Depending on the number of results, the amount of data being returned, and the bandwidth of your internet connection,
the `getAll` method may be very slow, or simply not work entirely.  Therefore, we also provide a number of methods to page through your
results and access them in piecemeal.  The remaining sections describe these alternative methods.

.. _marvin-results-extend:

Extending the Set
^^^^^^^^^^^^^^^^^

You can extend your current set of results with the next set using `extendSet`.  This retrieves the next page of results and
adds it to your current page::

    # count in current set
    r.count
    100

    # the ResultSet
    r.results
    <ResultSet(set=1/13, index=0:100, count_in_set=100, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.26038932800293, z=0.0361073),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.48788070678711, z=0.0699044),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_absmag_g_r=0.543312072753906, z=0.0408897),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_absmag_g_r=0.757579803466797, z=0.028215),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', elpetro_absmag_g_r=1.09770011901855, z=0.0171611),
     ResultRow(mangaid=u'1-113403', plate=7815, plateifu=u'7815-12703', ifu_name=u'12703', elpetro_absmag_g_r=0.745466232299805, z=0.0715126
     ...]

     # extend the set with the next chunk of data
     r.extendSet()
     INFO: Retrieving next 100, from 100 to 200

     r.count
     200

     r.results
    <ResultSet(set=1/7, index=0:200, count_in_set=200, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.26038932800293, z=0.0361073),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.48788070678711, z=0.0699044),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_absmag_g_r=0.543312072753906, z=0.0408897),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_absmag_g_r=0.757579803466797, z=0.028215),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', elpetro_absmag_g_r=1.09770011901855, z=0.0171611),
     ResultRow(mangaid=u'1-113403', plate=7815, plateifu=u'7815-12703', ifu_name=u'12703', elpetro_absmag_g_r=0.745466232299805, z=0.0715126),
     ResultRow(mangaid=u'1-113418', plate=7815, plateifu=u'7815-12704', ifu_name=u'12704', elpetro_absmag_g_r=1.44098854064941, z=0.0430806),
     ...]

You can grow your set by a larger number than the default with the `chunk` keyword argument.::

    # extend your set of results by
    r.extendSet(chunk=500)

.. _marvin-results-loop:

Loop over the Data
^^^^^^^^^^^^^^^^^^

You can attempt to loop over all the results, automatically calling `extendSet`, to grow your results to the total count.  `loop` also accepts a `chunk` keyword argument to vary the amount of rows returned in each iteration.

::

    r.loop()
    Retrieving next 100, from 100 to 200
    Retrieving next 100, from 200 to 300
    Retrieving next 100, from 300 to 400
    Retrieving next 100, from 400 to 500
    Retrieving next 100, from 500 to 600
    Retrieving next 100, from 600 to 700
    Retrieving next 100, from 700 to 800
    Retrieving next 100, from 800 to 900
    Retrieving next 100, from 900 to 1000
    Retrieving next 100, from 1000 to 1100
    Retrieving next 100, from 1100 to 1200
    WARNING: You have reached the end.
    Retrieving next 100, from 1200 to 1282

    r.count
    1282

Both `extendSet` and `loop` will grow the existing `ResultSet` to encompass all of the results.  Use these if you want to get all the results
locally.  The following methods work only on specific pages at a time, and replace your existing `ResultSet`.

.. _marvin-results-pages:

Get Next/Previous Chunks in List
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can view, and grab, the next or previous chunk with

.. code-block:: python

    r.getNext()
    r.getPrevious()

You can also specify a chunk value

.. code-block:: python

    # get next 5 entries
    r.getNext(chunk=5)
    Retrieving next 5, from 100 to 105
    <ResultSet(set=21/257, index=100:105, count_in_set=5, total=1282)>
    [ResultRow(mangaid=u'1-135548', plate=8601, plateifu=u'8601-12702', ifu_name=u'12702', elpetro_absmag_g_r=1.05030250549316, z=0.030559),
     ResultRow(mangaid=u'1-135568', plate=8601, plateifu=u'8601-12701', ifu_name=u'12701', elpetro_absmag_g_r=0.790615081787109, z=0.0938565),
     ResultRow(mangaid=u'1-135641', plate=8588, plateifu=u'8588-12704', ifu_name=u'12704', elpetro_absmag_g_r=1.44169998168945, z=0.030363),
     ResultRow(mangaid=u'1-135657', plate=8588, plateifu=u'8588-1901', ifu_name=u'1901', elpetro_absmag_g_r=1.22106170654297, z=0.0364618),
     ResultRow(mangaid=u'1-135679', plate=8588, plateifu=u'8588-6103', ifu_name=u'6103', elpetro_absmag_g_r=1.4596061706543, z=0.0331057)]

    # get previous 5 entries
    r.getPrevious(chunk=5)
    Retrieving previous 5, from 95 to 100
    <ResultSet(set=20/257, index=95:100, count_in_set=5, total=1282)>
    [ResultRow(mangaid=u'1-135512', plate=8601, plateifu=u'8601-6102', ifu_name=u'6102', elpetro_absmag_g_r=0.778741836547852, z=0.0279629),
     ResultRow(mangaid=u'1-135516', plate=8550, plateifu=u'8550-6104', ifu_name=u'6104', elpetro_absmag_g_r=1.33112716674805, z=0.0314747),
     ResultRow(mangaid=u'1-135517', plate=8588, plateifu=u'8588-6101', ifu_name=u'6101', elpetro_absmag_g_r=1.17428970336914, z=0.0317611),
     ResultRow(mangaid=u'1-135530', plate=8550, plateifu=u'8550-9101', ifu_name=u'9101', elpetro_absmag_g_r=1.7724609375, z=0.0283296),
     ResultRow(mangaid=u'1-135545', plate=8601, plateifu=u'8601-6103', ifu_name=u'6103', elpetro_absmag_g_r=1.43307685852051, z=0.0301334)]

.. _marvin-results-subset:

Get Subset
^^^^^^^^^^
To retrieve a subset of the results, use ``r.getSubset``.  getSubset works by specifying the starting index to grab from and a limit
on the number to grab (default is 10).  Having the returntype specified will also generate the corresponding Marvin Tools for the new
subset

.. code-block:: python

    # Get a subet of objects starting at index 300 (note the chunk is now 5, due to the above code example)
    r.getSubset(start=300)
    <ResultSet(set=61/257, index=300:305, count_in_set=5, total=1282)>
    [ResultRow(mangaid=u'1-211227', plate=8603, plateifu=u'8603-12702', ifu_name=u'12702', elpetro_absmag_g_r=1.62772369384766, z=0.0276667),
     ResultRow(mangaid=u'1-211239', plate=8550, plateifu=u'8550-6102', ifu_name=u'6102', elpetro_absmag_g_r=2.10557651519775, z=0.0265494),
     ResultRow(mangaid=u'1-211311', plate=8550, plateifu=u'8550-3704', ifu_name=u'3704', elpetro_absmag_g_r=1.36248779296875, z=0.0298414),
     ResultRow(mangaid=u'1-216520', plate=8440, plateifu=u'8440-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.60188102722168, z=0.0239988),
     ResultRow(mangaid=u'1-216540', plate=8440, plateifu=u'8440-12701', ifu_name=u'12701', elpetro_absmag_g_r=1.17086791992188, z=0.0289061)]


    # Get a subset of 10 objects starting at index 500
    r.getSubset(start=500, limit=10)
    <ResultSet(set=51/129, index=500:510, count_in_set=10, total=1282)>
    [ResultRow(mangaid=u'1-256465', plate=8451, plateifu=u'8451-6104', ifu_name=u'6104', elpetro_absmag_g_r=1.55233573913574, z=0.0574997),
     ResultRow(mangaid=u'1-256496', plate=8258, plateifu=u'8258-3704', ifu_name=u'3704', elpetro_absmag_g_r=0.784034729003906, z=0.0584715),
     ResultRow(mangaid=u'1-256506', plate=8258, plateifu=u'8258-3703', ifu_name=u'3703', elpetro_absmag_g_r=1.70734596252441, z=0.0587373),
     ResultRow(mangaid=u'1-256546', plate=8258, plateifu=u'8258-12702', ifu_name=u'12702', elpetro_absmag_g_r=0.911626815795898, z=0.0212478),
     ResultRow(mangaid=u'1-256574', plate=8258, plateifu=u'8258-12703', ifu_name=u'12703', elpetro_absmag_g_r=1.10363006591797, z=0.0660807),
     ResultRow(mangaid=u'1-256647', plate=8466, plateifu=u'8466-6102', ifu_name=u'6102', elpetro_absmag_g_r=1.71620178222656, z=0.0487626),
     ResultRow(mangaid=u'1-256819', plate=8466, plateifu=u'8466-1902', ifu_name=u'1902', elpetro_absmag_g_r=1.69637107849121, z=0.0662541),
     ResultRow(mangaid=u'1-256860', plate=8466, plateifu=u'8466-9101', ifu_name=u'9101', elpetro_absmag_g_r=1.00818252563477, z=0.0245857),
     ResultRow(mangaid=u'1-25688', plate=7990, plateifu=u'7990-6103', ifu_name=u'6103', elpetro_absmag_g_r=1.4294376373291, z=0.0292359),
     ResultRow(mangaid=u'1-257100', plate=8466, plateifu=u'8466-12705', ifu_name=u'12705', elpetro_absmag_g_r=1.02963256835938, z=0.045258)]

.. _marvin-results-downlaod:

Downloading Results
^^^^^^^^^^^^^^^^^^^

Download the results of your query.  This downloads the MaNGA FITS file data products for all the objects in your current `ResultSet`.
The downloaded object (FITS file) is determined by the returntype parameter, which defaults to cube if not specified.

.. code-block:: python

    # downloads all the data using sdss_acccess
    r.download()

`download` also accepts a `limit` keyword, to limit the number of downloaded objects::

    r.download(limit=5)


