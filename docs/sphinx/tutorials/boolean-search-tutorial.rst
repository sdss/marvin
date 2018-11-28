
.. _marvin-sqlboolean:

Boolean Search Syntax
=====================

Marvin Queries use a pseudo-natural language SQL syntax.  This is a short form version of SQL statements used to query
databases.  SQL statements generally are of the form **select** ``parameters`` **from** ``table`` **join** ``other tables`` **where** ``filter conditions``.  This syntax is designed to simplify how much SQL you need to write.  Rather than submitting the full SQL statement, you submit only a simplified **where** clause and an optional list of properties to return.  This eliminates the need to have detailed knowledge of the MaNGA database schema, table design, available columns, and the keys needed to join the tables.

String Construction
-------------------

The syntax for Marvin Query **where** clauses is a boolean search string.  Boolean search strings consist of a **parameter**-**operand**-**value** combination (e.g., :code:`a > 5`), where

* **parameter** is the variable name,

* **operand** must be  :code:`==`, :code:`=`, :code:`!=`, :code:`<`,
  :code:`<=`, :code:`>=`, or :code:`>`, and

  * :code:`==` finds exact matches whereas :code:`=` finds elements that contain
    the value.

* **value** can be a float, integer, or string.

  * Strings with spaces must be enclosed in quotes.

  * :code:`*` acts a wildcard.

These **parameter**-**operand**-**value** combinations can be joined with the
boolean operands (in order of descending precedence):

1. :code:`not`
2. :code:`and`
3. :code:`or`

and grouped with parentheses :code:`()`. For example,::

    a = 5 or b = 7 and not c = 7

is equivalent to::

    a = 5 or (b = 7 and (not c = 7))

Special Operands
----------------

In addition to the standard operands, there are special operands as well.

 * **between**: selects within a range, e.g. `a between 1 and 2`, which is equivalent to `a >= 1 and a <= 2`
 * **&**: a bitwise operator to perform bitwise and selections, e.g. `a & 256`, which selects rows where the maskbit value `a` includes 256.
 * **& ~**: a not bitwise operator, e.g. `a & ~256`, which selects rows where the maskbit value `a` does not include 256.

Marvin provides two special function syntax strings to provide specific queries, the **radial** function, and the **npergood** function.


Parameter Names
---------------

In MaNGA, parameter names have hierarchical dotted field names, and are structured as **schema.table.parameter**, e.g. :code:`mangadatadb.cube.plateifu`.  If a parameter name only exists in one column in one database table, it is considered unique.

::

    # The NSA Sersic log stellar mass (sersic_logmass) is a unique parameter in the NSA table
    my_filter  = 'sersic_logmass < 1'

Many parameters are not unique.  In this case, you must go one level up and specify the table name as well.  It is best practice however to adopt the syntax of **table.parameter** to ensure a fully unique parameter selection.

::

    # RA, Dec are not unique parameter names
    my_filter = 'ra > 180'

    query = Query(search_filter=my_filter)
    MarvinError: Could not set parameters. Multiple entries found for key.  Be more specific: 'ra matches multiple parameters in the lookup table: mangasampledb.nsa.ra, mangadatadb.cube.ra'.

    # Correct filter
    my_filter = 'cube.ra > 180'

.. _marvin-filter-examples:

Example SQL Constructions
-------------------------
::

    # Filter string
    my_filter = "nsa.z < 0.02 and ifu.name = 19*"

    # Converts to
    and_(nsa.z<0.02, ifu.name=19*)

    # SQL syntax
    mangasampledb.nsa.z < 0.02 AND lower(mangadatadb.ifudesign.name) LIKE lower('19%')

::

    # Filter string
    my_filter = 'cube.plate < 8000 and ifu.name = 19 or not (nsa.z > 0.1 or not cube.ra > 225.)'

    # Converts to
    or_(and_(cube.plate<8000, ifu.name=19), not_(or_(nsa.z>0.1, not_(cube.ra>225.))))

    # SQL syntax
    mangadatadb.cube.plate < 8000 AND lower(mangadatadb.ifudesign.name) LIKE lower(('%' || '19' || '%'))
    OR NOT (mangasampledb.nsa.z > 0.1 OR mangadatadb.cube.ra <= 225.0)


::

    # Filter string
    my_filter = 'nsa.z < 0.1 or (nsa.sersic_logmass between 9.5 and 11)'

    # Converts to
    or_(nsa.z<0.1, nsa.sersic_logmassbetween9.5and11)

    # SQL syntax
    (mangasampledb.nsa.z < 0.1 OR CAST(CASE WHEN (mangasampledb.nsa.sersic_mass > 0.0) THEN log(mangasampledb.nsa.sersic_mass) WHEN (mangasampledb.nsa.sersic_mass = 0.0) THEN 0.0 END AS FLOAT) BETWEEN 9.5 AND 11.0)


For more details on boolean search string syntax see the
` original SQLAlchemy-boolean-search documentation
<http://sqlalchemy-boolean-search.readthedocs.io/en/latest/>`_.
