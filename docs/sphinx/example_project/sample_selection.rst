.. _example-project-sample-selection:

.. ipython:: python
   :suppress:

   import matplotlib
   matplotlib.style.use('seaborn-darkgrid')
   from marvin import config
   config.mode = 'remote'


Marvin Example Project: Sample Selection
========================================

You can select a sample of galaxies by running a query on the MaNGA database either via:

- `Marvin-Web <https://sas.sdss.org/marvin2/search/>`_
- Marvin-Tools (continue below or see the `Jupyter Notebook <https://github.com/sdss/marvin/blob/master/docs/sphinx/jupyter/example_project_sample_selection.ipynb>`_).


Query Syntax
------------

Marvin uses a simplified query syntax (for both the `Web <https://sas.sdss.org/marvin2/search/>`_ and Tools) that understands the MaNGA database schema, so you don't have to write complicated SQL queries.


Example: find galaxies with stellar mass between :math:`10^{10}` and :math:`10^{11}`.
`````````````````````````````````````````````````````````````````````````````````````

Create the query and run it (limit to only 5 results for demo purposes):

.. ipython:: python

    from marvin.tools.query import doQuery
    
    q, r = doQuery(searchfilter='nsa.sersic_logmass >= 10 and nsa.sersic_logmass <= 11', limit=5)

**Tip** see :ref:`Example Queries <marvin-query-examples>` and :ref:`Marvin Query Syntax Tutorial <marvin-sqlboolean>` for help with designing search filters.

View the results:

.. ipython:: python

    df = r.toDF()
    df

Convert to :ref:`marvin-maps`:

.. ipython:: python

    r.convertToTool('maps')
    r.objects

|
