
.. _marvin-sqlboolean:

Boolean Search Tutorial
=======================

Boolean search strings consist of a **parameter**-**operand**-**value** combination
(e.g., :code:`a > 5`), where

* **parameter** is the variable name,

* **operand** must be  :code:`==`, :code:`=`, :code:`!=`, :code:`<`,
  :code:`<=`, :code:`>=`, or :code:`>`, and

  * :code:`==` finds exact matches whereas :code:`=` finds elements that contain
    the value.

* **value** can be a float, integer, or string.

  * Strings with spaces must be enclosed in quotes.

  * :code:`*` acts a wildcard.

These **paramter**-**operand**-**value** combinations can be joined with the
boolean operands (in order of descending precedence):

1. :code:`not`
2. :code:`and`
3. :code:`or`

and grouped with parentheses :code:`()`. For example,::

    a = 5 or b = 7 and not c = 7

is equivalent to::

    a = 5 or (b = 7 and (not c = 7))

In MaNGA, parameter names have hierarchical dotted field names, and are structured as **schema.table.parameter**, e.g. :code:`mangadatadb.cube.plateifu`.  If a parameter name only exists in one column in one database table, it is considered unique.  Most parameters are unique can be specified using only their parameter name.

::

    # Redshift (z) is a unique parameter name
    filter  = 'z < 0.1'

    # Plateifu (plateifu) is unique
    filter = 'plateifu == 8485-1901'

Some parameters are not unique.  In this case, you must go one level up and specify the table name as well.

::

    # RA, Dec are not unique parameter names
    filter = 'ra > 180'

    query = Query(search_filter=filter)
    MarvinError: Could not set parameters. Multiple entries found for key.  Be more specific: 'ra matches multiple parameters in the lookup table: mangasampledb.nsa.ra, mangadatadb.cube.ra'.

    # Correct filter
    filter = 'cube.ra > 180'

.. _marvin-filter-examples:

MaNGA Examples
--------------
::

    # Filter string
    filter = "nsa.z < 0.012 and ifu.name = 19*"
    # Converts to
    and_(nsa.z<0.012, ifu.name=19*)
    # SQL syntax
    mangasampledb.nsa.z < 0.012 AND lower(mangadatadb.ifudesign.name) LIKE lower('19%')

    # Filter string
    filter = 'cube.plate < 8000 and ifu.name = 19 or not (nsa.z > 0.1 or not cube.ra > 225.)'
    # Converts to
    or_(and_(cube.plate<8000, ifu.name=19), not_(or_(nsa.z>0.1, not_(cube.ra>225.))))
    # SQL syntax
    mangadatadb.cube.plate < 8000 AND lower(mangadatadb.ifudesign.name) LIKE lower(('%' || '19' || '%'))
    OR NOT (mangasampledb.nsa.z > 0.1 OR mangadatadb.cube.ra <= 225.0)


For more details on boolean search string syntax see the
`SQLAlchemy-boolean-search documentation
<http://sqlalchemy-boolean-search.readthedocs.io/en/latest/>`_.
