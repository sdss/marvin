Results
=======

Simple Query::

    from marvin.tools.query import Query
    q = Query(searchfilter='nsa.z < 0.1')

Get Results::

    r = q.run()
    r.results

Alternatively::
    
    from marvin.tools.query import doQuery
    q, r = doQuery(searchfilter='nsa.z < 0.1')


Viewing Results
---------------

Basics::

    r.results

View Columns::

    columns = r.getColumns()

Marvin automatically paginates results in groups of 10. In local mode only, you can view the next or previous chunk with::

    r.getNext()
    r.getPrevious()


You get all of the results with::
    
    r.getAll()


Downloading Results
-------------------

Download::
    
    r.download()

.