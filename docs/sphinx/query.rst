
.. _marvin-queries:

Query
=====

See the :ref:`marvin-sqlboolean` tutorial on how to design search filters.

Simple Query
------------

Simple Query from initialization::

    from marvin.tools.query import Query
    q = Query(searchfilter='nsa.z < 0.1')
    q.run()

or in steps::

    searchfilter = 'nsa.z < 0.1'
    q = Query()
    q.set_filter(searchfilter=searchfilter)
    q._create_query_modelclasses()
    q._join_tables()
    q.add_condition()
    q.run()

Get Results::

    r = q.run()
    r.results

Returns::

    [(u'1-24099', 7991, u'1902', u'1902', 0.0281657855957747),
     (u'1-38103', 8082, u'1901', u'1901', 0.0285587850958109),
     (u'1-38157', 8083, u'1901', u'1901', 0.037575539201498),
     (u'1-38347', 8083, u'1902', u'1902', 0.036589004099369),
     (u'1-43214', 8135, u'1902', u'1902', 0.117997065186501),
     (u'1-43629', 8143, u'1901', u'1901', 0.031805731356144),
     (u'1-43663', 8140, u'1902', u'1902', 0.0407325178384781),
     (u'1-43679', 8140, u'1901', u'1901', 0.0286782365292311),
     (u'1-43717', 8137, u'1902', u'1902', 0.0314487814903259),
     (u'1-44047', 8143, u'1902', u'1902', 0.04137859120965)]

Do it all at once::

    from marvin.tools.query import doQuery
    q, r = doQuery(searchfilter='nsa.z < 0.1')
    r.results

Show Query
----------


See :ref:`marvin-query-examples` for examples of different types of queries.

Go to :ref:`marvin-results` to see how to handle query results
