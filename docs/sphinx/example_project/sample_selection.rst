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

Create the Query and then run it:

.. ipython:: python

    from marvin.tools.query import Query
    
    q = Query(searchfilter='nsa.sersic_logmass >= 10 and nsa.sersic_logmass <= 11', limit=5)

    r = q.run()

View the results:

.. ipython:: python

    r.results


See the :ref:`marvin-sqlboolean` tutorial on how to design search filters.  See the :ref:`marvin-query-examples` for examples of how to write MaNGA specific filter strings.
