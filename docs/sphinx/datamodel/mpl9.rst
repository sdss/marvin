
.. _datamodel-mpl9:

MPL-9
=====

This datamodel corresponds to the MPL-9 (v2_7_1, 2.4.1) release of MaNGA data.


DataModels
----------

* :ref:`DRP<mpl9drp>`
* :ref:`DAP<mpl9dap>`
* :ref:`Query<mpl9query>`
* :ref:`Maskbits<mpl9masks>`
* :ref:`VACs<mpl9vacs>`

.. _mpl9drp:

.. datamodel:: marvin.utils.datamodel.drp.MPL:MPL9
   :prog: DRP DataModel
   :title: MPL-9 DataModel
   :spectra:
   :datacubes:
   :rss:
   :description: Here describes the DRP datamodels for spectra, datacubes, and rss properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl9dap:

.. datamodel:: marvin.utils.datamodel.dap:MPL9
   :prog: DAP DataModel
   :title: MPL-9 DataModel
   :bintypes:
   :templates:
   :models:
   :properties:
   :description: Here describes the DAP datamodels for bintypes, templates, models, and map properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl9query:

.. datamodel:: marvin.utils.datamodel.query:MPL9
   :prog: Query DataModel
   :title: MPL-9 DataModel
   :parameters:
   :description: Here describes the datamodel for all queryable parameters.  Each table displays relevant information such as the full query name and the group it belongs to. The "full query name" is what is input in all query search filters and return parameters.  The table can be scrolled horizonally for additional info.

.. _mpl9masks:

Maskbits
--------

.. datamodel:: marvin.utils.datamodel.dap:MPL9
   :prog: DRP Maskbits
   :bitmasks:
   :bittype: DRP

.. datamodel:: marvin.utils.datamodel.dap:MPL9
   :prog: DAP Maskbits
   :bitmasks:
   :bittype: DAP

.. datamodel:: marvin.utils.datamodel.dap:MPL9
   :prog: Targeting Maskbits
   :bitmasks:
   :bittype: Target

.. _mpl9vacs:

VACs
----

.. datamodel:: marvin.utils.datamodel.vacs:datamodel
   :prog: Available VACs
   :vac: MPL-9
   :description: A list of the contributed VACs available in this MPL.

