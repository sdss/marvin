

.. _marvin-results:

Results
=======

.. _marvin-results_getstart:

Getting Started
---------------

When you perform a :ref:`marvin-query` in Marvin, you get a Results object.  This page describes the basic usage of Marvin Results.  See the :ref:`marvin-results-ref` Reference for a full description of all the methods available.

To perform a simple Query

.. code-block:: python

    from marvin.tools.query import Query
    q = Query(search_filter='nsa.z < 0.1', return_params=['absmag_g_r'])

and to get the Results, use the `run` method on your ``Query`` object.  Once your query is run you can access the list of query results from the
`results` attribute.

.. code-block:: python

    r = q.run()

    # number of results
    r.totalcount
    2560

    # view the results
    r.results
    <ResultSet(set=1/26, index=0:100, count_in_set=100, total=2560)>
    [ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', elpetro_absmag_g_r=1.26038932800293, z=0.0361073),
     ResultRow(mangaid=u'1-113208', plate=8618, plateifu=u'8618-3701', ifu_name=u'3701', elpetro_absmag_g_r=1.48788070678711, z=0.0699044),
     ResultRow(mangaid=u'1-113219', plate=7815, plateifu=u'7815-9102', ifu_name=u'9102', elpetro_absmag_g_r=0.543312072753906, z=0.0408897),
     ResultRow(mangaid=u'1-113375', plate=7815, plateifu=u'7815-9101', ifu_name=u'9101', elpetro_absmag_g_r=0.757579803466797, z=0.028215),
     ResultRow(mangaid=u'1-113379', plate=7815, plateifu=u'7815-6101', ifu_name=u'6101', elpetro_absmag_g_r=1.09770011901855, z=0.0171611),
     ResultRow(mangaid=u'1-113403', plate=7815, plateifu=u'7815-12703', ifu_name=u'12703', elpetro_absmag_g_r=0.745466232299805, z=0.0715126),
     ResultRow(mangaid=u'1-113418', plate=7815, plateifu=u'7815-12704', ifu_name=u'12704', elpetro_absmag_g_r=1.44098854064941, z=0.0430806),
     ...]

The returned object is a `ResultSet`, which is a fancy list of Python tuple objects containing the results of your query.  The representation of the
`ResultSet` indicates some metadata like the total result count, the current set (page) of the total, the current count in the current set, and the array
indices out of the total set.

The `Results` object also contains the `Query Datamodel` which you can explore.  ::

    r.datamodel
    <QueryDataModel release='MPL-4', n_groups=7, n_parameters=565, n_total=0>

.. _marvin-results-singlestep:

Alternatively, you can perform it in a single step

.. code-block:: python

    from marvin.tools.query import doQuery
    q, r = doQuery(search_filter='nsa.z < 0.1')


.. _marvin_results_using

Using Results
-------------

.. toctree::
   :maxdepth: 2

   The ResultSet <tools/results/results_set>

.. toctree::
   :maxdepth: 2

   Retrieving Results <tools/results/results_retrieve>

.. toctree::
   :maxdepth: 2

   Manipulating Results <tools/results/results_manipulate>

.. toctree::
   :maxdepth: 2

   Plotting Results <tools/results/results_plot>

.. toctree::
   :maxdepth: 2

   Converting your Results <tools/results/results_convert>

.. _marvin_results_api

Reference/API
-------------

.. rubric:: Class Inheritance Diagram

.. inheritance-diagram:: marvin.tools.results

.. rubric:: Class

.. autosummary:: marvin.tools.results.Results
.. autosummary:: marvin.tools.results.ResultSet

.. rubric:: Methods

.. autosummary::

    marvin.tools.results.Results.extendSet
    marvin.tools.results.Results.loop
    marvin.tools.results.Results.getNext
    marvin.tools.results.Results.getPrevious
    marvin.tools.results.Results.getSubset
    marvin.tools.results.Results.getAll
    marvin.tools.results.Results.plot
    marvin.tools.results.Results.hist
    marvin.tools.results.Results.save
    marvin.tools.results.Results.restore

|
