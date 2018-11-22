
.. _marvin-query:

Query
=====

.. _marvin-query_getstart:

Getting Started
^^^^^^^^^^^^^^^

The basic usage of searching the MaNGA dataset with Marvin Queries is shown below.  Queries allow you to perform searches filtering the sample on specific parameter conditions, as well as return additional desired parameters.  Queries accept two basic keywords, **search_filter** and **return_params**.

You search the MaNGA dataset by constructing a string filter condition in a pseudo-SQL syntax of **parameter operand value**.  You only need to care about constructing your filter or **where clause**, and Marvin will do the rest.

* **Condition I Want**: find all galaxies with a redshift less than 0.1.
* **Constructor**: **Parameter**: 'redshift (z or nsa.z)' + **Operand**: less than (<) + **Value**: 0.1
* **Marvin Filter Syntax**: 'nsa.z < 0.1'

::

    from marvin.tools.query import Query

    # search for galaxies with an NSA redshift < 0.1
    myfilter = 'nsa.z < 0.1'

    # create a query
    query = Query(search_filter=myfilter)

You can optionally return parameters using the **return_params** keyword, specified as a list of strings.

::

    # return the galaxy RA and Dec as well
    myfilter = 'nsa.z < 0.1'
    myparams = ['cube.ra', 'cube.dec']

    query = Query(search_filter=myfilter, return_params=myparams)

Queries contain a datamodel.  You can access the datamodel used for a given query and data release with the `datamodel` attribute::

  query.datamodel
  <QueryDataModel release='MPL-5', n_groups=7, n_parameters=697, n_total=0>

To explore the query datamodel and see what parameters are available for returning and searching on, see the datamodel :ref:`How To Guide <query-dm>`

Finally, you can run the query with **run**.  Queries produce results.  Go to :ref:`marvin-results` to see how to manage your query results.

::

    results = query.run()

Queries will always return a set of default parameters: the galaxy **mangaid**, **plateifu**, **plate id**, and **ifu design name**.  Additionally, queries will always return any parameters used in your filter condition, plus any requested return parameters.

::

    # see the returned columns
    print(results.columns)
    [u'cube.mangaid', u'cube.plate', u'cube.plateifu', u'ifu.name', 'cube.ra', 'cube.dec', 'nsa.z']

    # look at the first row result
    print(results.results[0])
    ResultRow(mangaid=u'1-109394', plate=8082, plateifu=u'8082-9102', ifu_name=u'9102', ra=50.179936141, dec=-1.0022917898, z=0.0361073)

.. _marvin_query_using

Using Query
^^^^^^^^^^^
.. toctree::
   :maxdepth: 2

   Using the Query <tools/query/query_using>

.. toctree::
   :maxdepth: 2

   The Query Datamodel <datamodel/query_dm>

.. toctree::
   :maxdepth: 2

   Returning Marvin objects <tools/query/query_returntype>

.. toctree::
   :maxdepth: 2

   Saving Queries <tools/query/query_saving>

.. _marvin_query_api

Reference/API
^^^^^^^^^^^^^

.. rubric:: Class Inheritance Diagram

.. inheritance-diagram:: marvin.tools.query.Query

.. rubric:: Class

.. autosummary:: marvin.tools.query.Query

.. rubric:: Methods

.. autosummary::

    marvin.tools.query.Query.run
    marvin.tools.query.Query.reset
    marvin.tools.query.Query.show
    marvin.tools.query.Query.save
    marvin.tools.query.Query.restore


|
