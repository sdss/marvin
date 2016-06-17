Results
=======

Simple Query::

    from marvin.tools.query import Query
    q = Query(searchfilter='nsa.z < 0.1')

Get Results::

    r = q.run()
    r.results

.