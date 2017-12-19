
.. _marvin-results_set:

The ResultSet Object
--------------------

Your query results come as a Marvin :class:`marvin.tools.results.ResultSet` object.  As `ResultSet` is sub-classed from a Python list, it behaves exactly as a Python list object.  `ResultSet` contains a list of query results, where each item in the list is a Marvin `ResultRow` object.  The Marvin `ResultRow` behaves exactly as a python `NamedTuple <https://docs.python.org/2/library/collections.html#collections.namedtuple>`_ object.

The representation of the `ResultSet` indicates some metadata like the total result count, the current set (page) of the total, the current count in the current set, and the array indices out of the total set.  For example, ``<ResultSet(set=1/257, index=0:5, count_in_set=5, total=1282)>``

::

    # let's get a sample ResultSet
    q = Query(searchfilter='nsa.z < 0.1')
    r = q.run()
    res = r.results

To see what columns are available, you can access them via the `columns` attribute.  This returns a Query `ParameterGroup` object containing only those parameters contained in your query.

::

    cols = res.columns
    <ParameterGroup name=Columns, n_parameters=6>
     [<QueryParameter full=cube.mangaid, name=mangaid, short=mangaid, remote=mangaid, display=Manga-ID>,
     <QueryParameter full=cube.plate, name=plate, short=plate, remote=plate, display=Plate>,
     <QueryParameter full=cube.plateifu, name=plateifu, short=plateifu, remote=plateifu, display=Plate-IFU>,
     <QueryParameter full=ifu.name, name=ifu_name, short=ifu_name, remote=ifu_name, display=Name>,
     <QueryParameter full=nsa.elpetro_absmag_g_r, name=elpetro_absmag_g_r, short=absmag_g_r, remote=elpetro_absmag_g_r, display=Absmag g-r>,
     <QueryParameter full=nsa.z, name=z, short=z, remote=z, display=Redshift>]

To convert to a normal Python list, use the `to_list` method::

    res_list = res.to_list()

To convert to a list of Python dictionaries, use the `to_dict` method::

    res_dict = res.to_dict()

To in-place sort the set, use the `sort` method::

    res.sort('z')

You can slice a `ResultSet` to return a subset of data which is a new `ResultSet`::

    subset = res[0:5]
    print(subset)
    <ResultSet(set=1/257, index=0:5, count_in_set=5, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.26038932800293, z=0.0361073),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.48788070678711, z=0.0699044),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_absmag_g_r=0.543312072753906, z=0.0408897),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_absmag_g_r=0.757579803466797, z=0.028215),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', elpetro_absmag_g_r=1.09770011901855, z=0.0171611)]


If you have more than one group of `ResultSets`, you can merge them together using the ``+`` operand::

    # get and run a query 1
    q = Query(searchfilter='nsa.z < 0.1', returnparams=['absmag_g_r'])
    r = q.run()

Using **numpy**, you can handle the `ResultSet` and extract a subset of elements that satisfy some condition.  Slicing a `ResultSet` with Numpy array of indices will return a standard Numpy array.  For fancier manipulation, consider converting the results into an :ref:`Astropy Table <marvin-results_totable>` or :ref:`Pandas dataframe <marvin-results_todf>`::

    # extract from the set those rows with redshift < 0.07 and g-r color > 1.5
    sub = np.where((np.array(res['z']) < 0.07) & (np.array(res['g_r']) > 1.5))[0]

    # return a Numpy array subset
    subset = res[sub]

    len(subset)
    24

    print(subset)
    array([[u'1-113520', u'7815', u'7815-1901', u'1901', u'1.75103473663',
            u'0.0167652'],
           [u'1-113525', u'8618', u'8618-6103', u'6103', u'1.57906627655',
            u'0.0169457'],
           [u'1-113525', u'7815', u'7815-1902', u'1902', u'1.57906627655',
            u'0.0169457'],
           [u'1-113663', u'8618', u'8618-3703', u'3703', u'2.80322933197',
            u'0.0316328'],
            ...
            ],
          dtype='<U14')






