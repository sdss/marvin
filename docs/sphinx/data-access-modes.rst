Data Access Modes 
=================

.. _local-mode:

Local Mode
----------

In Local mode, :doc:`marvin_tools` will access data stored locally on your
machine. If you specify a file name, then it will open that file. Alternatively,
you can provide a MaNGA-ID or a plate-IFU identifier. Marvin will check to see
if you have a database (e.g., running :doc:`marvin_tools` at Utah) and use that
if possible. More likely, you will not have access to the database, so it will
look for a FITS file. If neither of those options is successful, then it will
download the requested data if automatic downloading is enabled.

.. _remote-mode:

Remote Mode
-----------

In Remote mode, :doc:`marvin_tools` will retrieve the data remotely via the
:doc:`api` if a MaNGA-ID or a plate-IFU identifier is provided.


.. _auto-mode:

Auto Mode
---------

In Auto mode, Marvin first tries `Local Mode`_, but if that attempt fails, it
automatically switches to `Remote Mode`_ .


Mode Decision Tree
------------------

|

.. image:: ../Mode_Decision_Tree.png
    :width: 800px
    :align: center
    :alt: Mode decision tree

|
