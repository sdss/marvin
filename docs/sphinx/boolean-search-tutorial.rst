Boolean Search Tutorial
=======================

Boolean search strings consist of a **name**-**operator**-**value** combination
(e.g., :code:`a > 5`), where 

* **name** is the variable name,

* **operator** must be  :code:`==`, :code:`=`, :code:`!=`, :code:`<`,
  :code:`<=`, :code:`>=`, or :code:`>`, and
  
  * :code:`==` finds exact matches whereas :code:`=` finds elements that contain
    the value.

* **value** can be a float, integer, or string.

  * Strings with spaces must be enclosed in quotes.

  * :code:`*` acts a wildcard.

These **name**-**operator**-**value** combinations can be joined with the
boolean operands (in order of descending precedence):

1. :code:`not`
2. :code:`and`
3. :code:`or` 

and grouped with parentheses :code:`()`. For example,::
    
    a = 5 or b = 7 and not c = 7

is equivalent to::
    
    a = 5 or (b = 7 and (not c = 7))

Variable names can have hierarchical dotted field names, such as
:code:`cube.plateifu`.



For more details on boolean search string syntax see the
`SQLAlchemy-boolean-search documentation
<http://sqlalchemy-boolean-search.readthedocs.io/en/latest/>`_
