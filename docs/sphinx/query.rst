Query
=====

How to design search filters: :doc:`boolean-search-tutorial`

:doc:`query-examples`

Simple Query::

    from marvin.tools.query import Query
<<<<<<< HEAD:docs/sphinx/query.rst
    q = Query(searchfilter='nsa.z < 0.1')
=======
    q = Query()
    q.set_filter(searchfilter='nsa.z < 0.1')

or

    searchfilter = 'nsa.z < 0.1'
    q = Query(searchfilter=searchfilter)
>>>>>>> 10c0c2af019eb1910ebbaf5bf221d671dd293fce:docs/sphinx/queries.rst

Get Results::

    r = q.run()
    r.results


Returns::

    [(u'1-209232', u'SFLUX', u'1-209232', 8485, u'1901', u'Ha', 8485, 13),
     (u'1-209232', u'EW', u'1-209232', 8485, u'1901', u'Ha', 8485, 13),
     (u'1-209232', u'INSTSIGMA', u'1-209232', 8485, u'1901', u'Ha', 8485, 13),
     (u'1-209232', u'GSIGMA', u'1-209232', 8485, u'1901', u'Ha', 8485, 13),
     (u'1-209232', u'GVEL', u'1-209232', 8485, u'1901', u'Ha', 8485, 13),
     (u'1-209232', u'GFLUX', u'1-209232', 8485, u'1901', u'Ha', 8485, 13),
     (u'1-209232', u'SFLUX', u'1-209232', 8485, u'1901', u'Ha', 8485, 13),
     (u'1-209232', u'EW', u'1-209232', 8485, u'1901', u'Ha', 8485, 13),
     (u'1-209232', u'INSTSIGMA', u'1-209232', 8485, u'1901', u'Ha', 8485, 13),
     (u'1-209232', u'GSIGMA', u'1-209232', 8485, u'1901', u'Ha', 8485, 13)]

Do it all at once::

    from marvin.tools.query import doQuery
    q, r = doQuery(searchfilter='nsa.z < 0.1')
    r.results

<<<<<<< HEAD:docs/sphinx/query.rst
.
=======
voila
>>>>>>> 10c0c2af019eb1910ebbaf5bf221d671dd293fce:docs/sphinx/queries.rst
