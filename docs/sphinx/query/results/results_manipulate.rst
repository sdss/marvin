
.. _marvin-results_manipulate:

Manipulating Your Results
-------------------------

This page describes ways of manipulating the Marvin `Results` objects and extracting columns.

.. _marvin-results-add:

Adding Results
^^^^^^^^^^^^^^

Returning lots of parameters with your query may sometimes result in slower query responses, due to the volume of data
returned.  With Marvin `Results` you can perform two similar queries returning different parameters, and then combine them
into a single `Result` object.::

    # perform a query returning redshift, and absolute magnitude g-r color
    q1 = Query(search_filter='nsa.z < 0.1', returnparams=['absmag_g_r'])
    r1 = q1.run()

    # perform a second query returning some NSA parameters and RA, Dec coordinates
    q2 = Query(search_filter='nsa.z < 0.1', returnparams=['nsa.elpetro_ba', 'nsa.sersic_logmass', 'cube.ra', 'cube.dec'])
    r2 = q2.run()

    # combine the results
    new_r = r1 + r2

This returns a new `ResultSet` that combines all columns from both queries and builds a new Marvin `Result` that you can interact with.::

    # show the columns of data in the new Results
    new_r.columns
    <ParameterGroup name=Columns, n_parameters=10>
     [<QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, remote=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.plate, name=plate, short=plate, remote=plate, display=Plate>,
     <QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, remote=plateifu, display=Plate-IFU>,
     <QueryParameter full=ifu.name, name=ifu_name, short=ifu_name, remote=ifu_name, display=Name>,
     <QueryParameter full=nsa.elpetro_absmag_g_r, name=elpetro_absmag_g_r, short=absmag_g_r, remote=elpetro_absmag_g_r, display=Absmag g-r>,
     <QueryParameter full=nsa.z, name=z, short=z, remote=z, display=Redshift>,
     <QueryParameter full=nsa.elpetro_ba, name=elpetro_ba, short=axisratio, remote=elpetro_ba, display=Elpetro axis ratio>,
     <QueryParameter full=nsa.sersic_logmass, name=sersic_logmass, short=sersic_logmass, remote=sersic_logmass, display=Sersic Stellar Mass>,
     <QueryParameter full=cube.ra, name=ra, short=ra, remote=ra, display=RA>,
     <QueryParameter full=cube.dec, name=dec, short=dec, remote=dec, display=Dec>]

    # show new results
    new_r.results
    <ResultSet(set=1/13, index=0:100, count_in_set=100, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.26038932800293, z=0.0361073, elpetro_ba=0.42712, sersic_logmass=10.3649916322316, ra=50.179936141, dec=-1.0022917898),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.48788070678711, z=0.0699044, elpetro_ba=0.752286, sersic_logmass=10.7910706881067, ra=317.504479435, dec=9.86822191739),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_absmag_g_r=0.543312072753906, z=0.0408897, elpetro_ba=0.517058, sersic_logmass=9.37199275559893, ra=317.374745914, dec=10.0519434342),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_absmag_g_r=0.757579803466797, z=0.028215, elpetro_ba=0.570455, sersic_logmass=9.82192731931789, ra=316.639658795, dec=10.7512221884),

.. _marvin-results-sort:

Sorting Results
^^^^^^^^^^^^^^^
You can sort the results on specific columns, using the `sort` method.

.. code-block:: python

    # Sort the results by redshift
    r.sort('z')
    <ResultSet(set=1/13, index=0:100, count_in_set=100, total=1282)>
    [ResultRow(mangaid=u'1-619066', plate=8554, plateifu=u'8554-12704', ifu_name=u'12704', elpetro_absmag_g_r=0.869690895080566, z=0.00054371),
     ResultRow(mangaid=u'1-575771', plate=8332, plateifu=u'8332-1902', ifu_name=u'1902', elpetro_absmag_g_r=1.25316619873047, z=0.00814191),
     ResultRow(mangaid=u'1-43148', plate=8135, plateifu=u'8135-6101', ifu_name=u'6101', elpetro_absmag_g_r=0.984879493713379, z=0.0108501),
     ResultRow(mangaid=u'1-25517', plate=7990, plateifu=u'7990-12703', ifu_name=u'12703', elpetro_absmag_g_r=1.00057220458984, z=0.0113986),
     ResultRow(mangaid=u'1-286805', plate=8329, plateifu=u'8329-12702', ifu_name=u'12702', elpetro_absmag_g_r=0.741434097290039, z=0.0128534),
     ResultRow(mangaid=u'1-217256', plate=8247, plateifu=u'8247-12701', ifu_name=u'12701', elpetro_absmag_g_r=0.719453811645508, z=0.0141216),
     ResultRow(mangaid=u'1-137912', plate=8250, plateifu=u'8250-12703', ifu_name=u'12703', elpetro_absmag_g_r=0.227899551391602, z=0.014213),
     ResultRow(mangaid=u'1-44565', plate=8143, plateifu=u'8143-12703', ifu_name=u'12703', elpetro_absmag_g_r=1.05989074707031, z=0.0152769),
     ResultRow(mangaid=u'1-235136', plate=8325, plateifu=u'8325-3702', ifu_name=u'3702', elpetro_absmag_g_r=0.938411712646484, z=0.0153338),
     ...]

Or change the `order` of the sort to either ascending (`asc`), or descending (`desc`)::

    r.sort('z', order='desc')

.. _marvin-results-extract:

Extracting Columns
^^^^^^^^^^^^^^^^^^
You can extract columns from the results and format them in specific ways.  You can index the results on a specific column to extract a list
of that parameter for **only** the current `ResultSet`.

::

    # extract the redshift column of the current set of results
    redshift = r.results['z']
    len(redshift)
    100

You can also extract a list of a single parameter with `getListOf`::

    redshift = r.getListOf('z')

Or return the entire column of data using the `return_all` keyword::

    redshift = r.getListOf('z', return_all=True)

You can optionally return a Numpy ndarray instead of a list, using the `to_ndarray` keyword argument.::

    r.getListOf('z', to_ndarray=True)

You can convert or extract data into a Python dictionary format as well, with `getDictOf`.  The default return format is a list of dictonaries::

    values = r.getDictOf()
    [{'elpetro_absmag_g_r': 1.26038932800293,
      u'ifu_name': u'9102',
      'mangaid': u'1-109394',
      'plate': 8082,
      'plateifu': u'8082-9102',
      'z': 0.0361073},
     {'elpetro_absmag_g_r': 1.48788070678711,
      u'ifu_name': u'3701',
      'mangaid': u'1-113208',
      'plate': 8618,
      'plateifu': u'8618-3701',
      'z': 0.0699044},
     {'elpetro_absmag_g_r': 0.543312072753906,
      u'ifu_name': u'9102',
      'mangaid': u'1-113219',
      'plate': 7815,
      'plateifu': u'7815-9102',
      'z': 0.0408897},
      ...
      ]

You can also just return a specific column of data.::

    redshift = r.getDictOf('z')
    [{'z': 0.00054371},
     {'z': 0.00814191},
     {'z': 0.0108501},
     {'z': 0.0113986},
     {'z': 0.0128534}, ...]

The `format_type` keyword can either be `dictlist`, which returns a dictionary of lists, or `listdict`, which returns a list of dictionaries.  The default is `listdict`.::

    redshift = r.getDictOf('z', format_type='dictlist')
    {'z': [0.0361073,
      0.0699044,
      0.0408897,
      0.028215,
      0.0171611,
      0.0715126, ..
      ]

`getDictOf` also allows to optionally return all of the data with the `return_all` keyword argument.::

    r.getDictOf('z', return_all=True)


.. _marvin-results-save:

Saving Results to Pickle
^^^^^^^^^^^^^^^^^^^^^^^^

You can save the Marvin `Result` object for later use as a Python pickle object with the `save` method.::

    r.save('myresults.mpf')

.. _marvin-results-restore:

Restoring Results from Pickle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To restore a Marvin `Results` pickle file object, use the `restore` class method on `Results`::

    #import the results class
    from marvin.tools.results import Results

    # load a Results pickle file
    my_results = Results.restore('myresults.mpf')
