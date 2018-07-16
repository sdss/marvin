
.. _marvin-results_convert:

Converting Your Results
^^^^^^^^^^^^^^^^^^^^^^^
You can convert your results to a variety of forms.

To Marvin Objects
"""""""""""""""""

You can convert directly to Marvin Tools objects with ``convertToTool``.  ``convertToTool`` accepts as its main argument **tooltype**, a string name of the tool you wish to convert tool.

Available objects are Cube, Spaxel, RSS, Maps, ModelCube.  To successfully convert to
a particular Marvin object, the results must contain the minimum default information needed to uniquely create that object.  The new
Tools are stored in a separate Results attribute called **objects**.

Conversion names: 'cube', 'maps', 'spaxel', 'rss', 'modelcube'

Minimum Default Parameters:
 * Cube and RSS objects: needs at least a mangaid or plateifu.
 * Spaxel object: needs a plateifu/mangaID, and a X and Y position.
 * Maps and ModelCube objects: need a plateifu/mangaid, a bintype, and a template

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

``convertToTool`` also accepts optional keyword **mode** if you wish to instantiante Marvin Objects differently that your Queries and Results.  Let's say you run a query and retrieve a list of results.  By default, this will naturally occur in remote mode.  But now you want to save your query/results, download all the objects in your results, and convert the list of results into local file-based Marvin Cubes.  Just pass convertToTool the **mode** keyword as **auto**, and let Marvin figure it all out for you.

.. code-block:: python

    # Save the query and results as Marvin Pickle files
    q.save('myquery.mpf')
    r.save('myresults.mpf')

    # download the results into my local SAS
    r.download()

    # convert the tools but do so locally
    print(r.mode)
    u'remote'
    r.convertToTool('cube', mode='auto')
    r.objects
    [<Marvin Cube (plateifu='7444-1901', mode='local', data_origin='file')>,
     <Marvin Cube (plateifu='7444-1902', mode='local', data_origin='file')>,
     <Marvin Cube (plateifu='7995-1901', mode='local', data_origin='file')>,
     <Marvin Cube (plateifu='7995-1902', mode='local', data_origin='file')>,
     <Marvin Cube (plateifu='8000-1901', mode='local', data_origin='file')>]

.. _marvin-results_todf:

To Pandas Dataframe
"""""""""""""""""""

.. code-block:: python

    r.toDF()  # r.toDataFrame()
         mangaid  plate     plateifu ifu_name         z
    0   1-488712   8449    8449-3703     3703  0.042059
    1   1-245458   8591    8591-3701     3701  0.041524
    2   1-605884   8439   8439-12703    12703  0.025108
    3   1-379741   8712   8712-12705    12705  0.032869

.. _marvin-results_totable:

To Astropy Tables
"""""""""""""""""

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
""""""""""""""

.. code-block:: python

    r.toJson()
    '[["1-22286", 7992, "12704", 0.099954180419445], ["1-22298", 7992, "12702", 0.0614774264395237], ["1-22333", 7992, "3704", 0.0366250574588776], ["1-22347", 7992, "3701", 0.0437936186790466], ["1-22383", 7992, "3702", 0.0542150922119617], ["1-22412", 7992, "9101", 0.0190997123718262], ["1-22414", 7992, "6103", 0.0922721400856972], ["1-22438", 7992, "1901", 0.016383046284318], ["1-22662", 7992, "6104", 0.027131162583828], ["1-22970", 7992, "3703", 0.0564263463020325]]'


To FITS
"""""""

.. code-block:: python

    r.toFits(filename='myresults.fits')
    Writing new FITS file myresults.fits


To CSV
""""""

.. code-block:: python

    r.toCSV(filename='myresults.csv')
    Writing new FITS file myresults.csv


