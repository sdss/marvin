
.. _marvin-query_saving:

Saving Queries
==============

For large or many queries, you may not want to recreate them every single session.  Using `Python pickling <https://docs.python.org/2/library/pickle.html>`_, Marvin can save your queries locally, and restore them later for use again.  Marvin will save the query locally as a binary pickle object, which can be restored later.

Saving
^^^^^^

To save your query, use the **save** method on your query.  **save** accepts as argument a filename.  If you do not specify a directory, the file will be saved in your user directory by default.  If you do not specify a filename, Marvin will generate one for you.

.. code-block:: python

    # make a query
    query = Query(search_filter='nsa.z < 0.1')
    print(q)
    Marvin Query(filter=nsa.z < 0.1, mode=u'remote', limit=100, sort=None, order=u'asc')

    # save it for later
    q.save('myquery')
    '/Users/Brian/myquery.mpf'

Restoring
^^^^^^^^^

To restore a previously saved query, use the **restore** method.  Restoring is a Marvin Query class method.  That means you run it from the class itself after import, instead of from your instance object.

.. code-block:: python

    # import the Query class
    from marvin.tools.query import Query

    # Restore a saved query from a pickle file
    newq = Query.restore('/Users/Brian/myquery.mpf')

    # Your query is now loaded
    print(newq)
    Marvin Query(filter=nsa.z < 0.1, mode='remote', limit=100, sort=None, order='asc')
    newq.search_filter
    'nsa.z < 0.1'

|
