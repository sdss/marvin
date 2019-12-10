
.. _datamodel-mpl7:

MPL-7
=====

This datamodel corresponds to the MPL-7 (v2_4_3, 2.2.1) release of MaNGA data.


DataModels
----------

* :ref:`DRP<mpl7drp>`
* :ref:`DAP<mpl7dap>`
* :ref:`Query<mpl7query>`
* :ref:`Maskbits<mpl7masks>`
* :ref:`VACs<mpl7vacs>`

.. _mpl7drp:

.. datamodel:: marvin.utils.datamodel.drp.MPL:MPL7
   :prog: DRP DataModel
   :title: MPL-7 DataModel
   :spectra:
   :datacubes:
   :rss:
   :description: Here describes the DRP datamodels for spectra, datacubes, and rss properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl7dap:

.. datamodel:: marvin.utils.datamodel.dap:MPL7
   :prog: DAP DataModel
   :title: MPL-7 DataModel
   :bintypes:
   :templates:
   :models:
   :properties:
   :description: Here describes the DAP datamodels for bintypes, templates, models, and map properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl7query:

.. datamodel:: marvin.utils.datamodel.query:MPL7
   :prog: Query DataModel
   :title: MPL-7 DataModel
   :parameters:
   :description: Here describes the datamodel for all queryable parameters.  Each table displays relevant information such as the full query name and the group it belongs to. The "full query name" is what is input in all query search filters and return parameters.  The table can be scrolled horizonally for additional info.

.. _mpl7masks:

Maskbits
--------

.. datamodel:: marvin.utils.datamodel.dap:MPL7
   :prog: DRP Maskbits
   :bitmasks:
   :bittype: DRP

.. datamodel:: marvin.utils.datamodel.dap:MPL7
   :prog: DAP Maskbits
   :bitmasks:
   :bittype: DAP

.. datamodel:: marvin.utils.datamodel.dap:MPL7
   :prog: Targeting Maskbits
   :bitmasks:
   :bittype: Target

.. _mpl7vacs:

VACs
----

.. datamodel:: marvin.utils.datamodel.vacs:datamodel
   :prog: Available VACs
   :vac: MPL-7
   :description: A list of the contributed VACs available in this MPL.

