

.. _marvin-results:

Results
=======

.. _marvin-results_getstart:

Getting Started
---------------

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
    2560

    # print results.
    r.results
    <ResultSet(set=1/26, index=0:100, count_in_set=100, total=2560)>
    [ResultRow(mangaid=u'1-488712', plate=8449, plateifu=u'8449-3703', ifu_name=u'3703', z=0.0420595),
     ResultRow(mangaid=u'1-245458', plate=8591, plateifu=u'8591-3701', ifu_name=u'3701', z=0.0415244),
     ResultRow(mangaid=u'1-605884', plate=8439, plateifu=u'8439-12703', ifu_name=u'12703', z=0.0251076),
     ResultRow(mangaid=u'1-379741', plate=8712, plateifu=u'8712-12705', ifu_name=u'12705', z=0.0328685),
     ResultRow(mangaid=u'1-92502', plate=8548, plateifu=u'8548-6103', ifu_name=u'6103', z=0.0202585),...]


.. _marvin-results-singlestep:

Alternatively, you can perform it in a single step

.. code-block:: python

    from marvin.tools.query import doQuery
    q, r = doQuery(searchfilter='nsa.z < 0.1')


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

   Converting your Results <tools/results/results_convert>

.. _marvin_results_api

Reference/API
-------------

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
    marvin.tools.results.Results.save
    marvin.tools.results.Results.restore

|
