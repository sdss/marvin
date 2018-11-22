
.. _marvin-results_set:

The ResultSet Object
--------------------

Your query results come as a Marvin :class:`marvin.tools.results.ResultSet` object.  As `ResultSet` is sub-classed from a Python list, it behaves exactly as a Python list object.  `ResultSet` contains a list of query results, where each item in the list is a Marvin `ResultRow` object.  The Marvin `ResultRow` behaves exactly as a python `NamedTuple <https://docs.python.org/2/library/collections.html#collections.namedtuple>`_ object.

Basics
^^^^^^

The representation of the `ResultSet` indicates some metadata like the total result count, the current set (page) of the total, the current count in the current set, and the array indices out of the total set.  For example, ``<ResultSet(set=1/257, index=0:5, count_in_set=5, total=1282)>``

::

    # let's get a sample ResultSet
    q = Query(search_filter='nsa.z < 0.1')
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

Combining
^^^^^^^^^
If you have more than one group of `ResultSets`, you can merge them together using the ``+`` operand.  Adding works in either the row-wise or
column-wise direction, depending on the data one is adding.  Marvin will only add sets together that come from queries using the **same release** of data,
as well as the **same search_filter**.  These two parameters uniquely identify a query+results.  You cannot add results coming from different queries.

**Note:** This describes the underlying functionality of adding sets together.  However, in practice it is recommended to add Marvin Results together
at the top level only.  See :ref:`marvin-results-add`.

row-wise
""""""""
If the indices of all rows in the two `ResultSets` are the same, Marvin will add them row-wise.  This is useful when you want to combine different return parameters for the same query into a single set::

    # get and run a query 1
    q = Query(search_filter='nsa.z < 0.1', return_params=['absmag_g_r'])
    r = q.run()

    e = r.results[0:5]
    print(e)
    <ResultSet(set=1/257, index=0:5, count_in_set=5, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.26038932800293, z=0.0361073),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.48788070678711, z=0.0699044),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_absmag_g_r=0.543312072753906, z=0.0408897),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_absmag_g_r=0.757579803466797, z=0.028215),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', elpetro_absmag_g_r=1.09770011901855, z=0.0171611)]

    # get and run a query 1 with additional return parameters
    q2 = Query(search_filter='nsa.z < 0.1', return_params=['nsa.elpetro_ba', 'nsa.sersic_logmass', 'cube.ra', 'cube.dec'])
    r2 = q2.run()

    e2 = r2.results[0:5]
    print(e2)
    <ResultSet(set=1/257, index=0:5, count_in_set=5, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_ba=0.42712, sersic_logmass=10.3649916322316, ra=50.179936141, dec=-1.0022917898, z=0.0361073),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_ba=0.752286, sersic_logmass=10.7910706881067, ra=317.504479435, dec=9.86822191739, z=0.0699044),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_ba=0.517058, sersic_logmass=9.37199275559893, ra=317.374745914, dec=10.0519434342, z=0.0408897),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_ba=0.570455, sersic_logmass=9.82192731931789, ra=316.639658795, dec=10.7512221884, z=0.028215),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', elpetro_ba=0.373641, sersic_logmass=8.72936001627318, ra=316.541566803, dec=10.3454195236, z=0.0171611)]

     # add them together
     new_set = e + e2
     print(new_set)
    <ResultSet(set=1/257, index=0:5, count_in_set=5, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.26038932800293, z=0.0361073, elpetro_ba=0.42712, sersic_logmass=10.3649916322316, ra=50.179936141, dec=-1.0022917898),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.48788070678711, z=0.0699044, elpetro_ba=0.752286, sersic_logmass=10.7910706881067, ra=317.504479435, dec=9.86822191739),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_absmag_g_r=0.543312072753906, z=0.0408897, elpetro_ba=0.517058, sersic_logmass=9.37199275559893, ra=317.374745914, dec=10.0519434342),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_absmag_g_r=0.757579803466797, z=0.028215, elpetro_ba=0.570455, sersic_logmass=9.82192731931789, ra=316.639658795, dec=10.7512221884),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', elpetro_absmag_g_r=1.09770011901855, z=0.0171611, elpetro_ba=0.373641, sersic_logmass=8.72936001627318, ra=316.541566803, dec=10.3454195236)]

column-wise
"""""""""""
If the indices of the rows in the two `ResultSets` do not match, then Marvin will simply append them together into a new list.  This is useful when you simply want to construct a custom list of objects.::

    # grab the first set of 5 from query 1
    e = r.results[0:5]

    # grab some middle chunk of 10 from query 1
    e2 = r.results[50:60]

    new_set = e + e2
    print(new_set)
    <ResultSet(set=1/86, index=0:15, count_in_set=15, total=1282)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.26038932800293, z=0.0361073),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.48788070678711, z=0.0699044),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_absmag_g_r=0.543312072753906, z=0.0408897),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_absmag_g_r=0.757579803466797, z=0.028215),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', elpetro_absmag_g_r=1.09770011901855, z=0.0171611),
     ResultRow(mangaid=u'1-115162', plate=7977, plateifu=u'7977-12703', ifu_name=u'12703', elpetro_absmag_g_r=1.13131713867188, z=0.0738627),
     ResultRow(mangaid=u'1-115320', plate=7977, plateifu=u'7977-3703', ifu_name=u'3703', elpetro_absmag_g_r=0.99519157409668, z=0.0275274),
     ResultRow(mangaid=u'1-124604', plate=8439, plateifu=u'8439-6103', ifu_name=u'6103', elpetro_absmag_g_r=1.38611221313477, z=0.0253001),
     ResultRow(mangaid=u'1-133922', plate=8486, plateifu=u'8486-6104', ifu_name=u'6104', elpetro_absmag_g_r=1.51949119567871, z=0.0174718),
     ResultRow(mangaid=u'1-133941', plate=8486, plateifu=u'8486-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.04214859008789, z=0.0189045),
     ResultRow(mangaid=u'1-133945', plate=8486, plateifu=u'8486-3703', ifu_name=u'3703', elpetro_absmag_g_r=1.70501899719238, z=0.0183248),
     ResultRow(mangaid=u'1-133948', plate=8486, plateifu=u'8486-6103', ifu_name=u'6103', elpetro_absmag_g_r=1.62374401092529, z=0.0195194),
     ResultRow(mangaid=u'1-133976', plate=8486, plateifu=u'8486-9101', ifu_name=u'9101', elpetro_absmag_g_r=1.26091766357422, z=0.0182938),
     ResultRow(mangaid=u'1-133987', plate=8486, plateifu=u'8486-1902', ifu_name=u'1902', elpetro_absmag_g_r=1.73217391967773, z=0.0195435),
     ResultRow(mangaid=u'1-134004', plate=8486, plateifu=u'8486-1901', ifu_name=u'1901', elpetro_absmag_g_r=1.27153015136719, z=0.0185601)]


Subsets
^^^^^^^
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
