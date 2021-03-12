
.. _datamodel-mpl11:

MPL-11
======

This datamodel corresponds to the MPL-11 (v3_1_0, 3.1.0) release of MaNGA data.


DataModels
----------

* :ref:`DRP<mpl11drp>`
* :ref:`DAP<mpl11dap>`
* :ref:`Query<mpl11query>`
* :ref:`Maskbits<mpl11masks>`
* :ref:`VACs<mpl11vacs>`

.. _mpl11drp:

.. datamodel:: marvin.utils.datamodel.drp.MPL:MPL11
   :prog: DRP DataModel
   :title: MPL-11 DataModel
   :spectra:
   :datacubes:
   :rss:
   :description: Here describes the DRP datamodels for spectra, datacubes, and rss properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl11dap:

.. datamodel:: marvin.utils.datamodel.dap:MPL11
   :prog: DAP DataModel
   :title: MPL-11 DataModel
   :bintypes:
   :templates:
   :models:
   :properties:
   :description: Here describes the DAP datamodels for bintypes, templates, models, and map properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl11query:

.. datamodel:: marvin.utils.datamodel.query:MPL11
   :prog: Query DataModel
   :title: MPL-11 DataModel
   :parameters:
   :description: Here describes the datamodel for all queryable parameters.  Each table displays relevant information such as the full query name and the group it belongs to. The "full query name" is what is input in all query search filters and return parameters.  The table can be scrolled horizonally for additional info.

.. _mpl11masks:

Maskbits
--------

.. datamodel:: marvin.utils.datamodel.dap:MPL11
   :prog: DRP Maskbits
   :bitmasks:
   :bittype: DRP

.. datamodel:: marvin.utils.datamodel.dap:MPL11
   :prog: DAP Maskbits
   :bitmasks:
   :bittype: DAP

.. datamodel:: marvin.utils.datamodel.dap:MPL11
   :prog: Targeting Maskbits
   :bitmasks:
   :bittype: Target

.. _mpl11vacs:

VACs
----

.. datamodel:: marvin.utils.datamodel.vacs:datamodel
   :prog: Available VACs
   :vac: MPL-11
   :description: A list of the contributed VACs available in this MPL.

