API
===

.. note::
    
    An API, or application programmatic interface, is a piece of software to
    send and receive data over the web in a programmatic manner.

Marvin-API has classes that mirror the Marvin-tools data organization classes
(e.g., :ref:`marvin-tools-spectrum`, :ref:`marvin-tools-spaxel`,
:ref:`marvin-tools-rss`, :ref:`marvin-tools-cube`, and
:ref:`marvin-tools-plate`) to allow you to retrieve data and meta-data.

For instance, you can retrieve the spectrum of a particular spaxel or the
spectra of all spaxels or fibers of a galaxy. You can also get basic information
about a cube or a plate like RA and DEC. Or convert MaNGA ID to plate-IFU.

The most ground-breaking feature is that you can run queries using Marvin-tools,
which will trigger a Marvin-API request and return the query results. Marvin-API
returns data in JSON format and the Marvin-tools classes translate it to more
user-friendly python data types (e.g. numpy arrays).

If you are bold, then it is possible to directly interact with Marvin-API using
the :ref:`marvin-api-interaction` class::
    
    from marvin.api import Interaction
    Interaction()

.. note::
    
    Check the Interaction, curl, and url syntax

Or you can use curl::
    
    curl -X POST --data "searchfilter=nsa_redshift<0.1"
    http://sas.sdss.org/marvin2/api/query/cubes/

Or simply go to::

    http://sas.sdss.org/marvin2/api/



.. notes::
    
    Describe what it is and how to use API. Describe how to use the Interaction
    class. and the config.urlmap


