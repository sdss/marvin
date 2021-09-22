
.. _datamodel-mpl8:

MPL-8
=====

This datamodel corresponds to the MPL-8 (v2_5_3, 2.3.0) release of MaNGA data.


DataModels
----------

* :ref:`DRP<mpl8drp>`
* :ref:`DAP<mpl8dap>`
* :ref:`Query<mpl8query>`
* :ref:`Maskbits<mpl8masks>`
* :ref:`VACs<mpl8vacs>`

.. _mpl8drp:

.. datamodel:: marvin.utils.datamodel.drp.MPL:MPL8
   :prog: DRP DataModel
   :title: MPL-8 DataModel
   :spectra:
   :datacubes:
   :rss:
   :description: Here describes the DRP datamodels for spectra, datacubes, and rss properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl8dap:

.. datamodel:: marvin.utils.datamodel.dap:MPL8
   :prog: DAP DataModel
   :title: MPL-8 DataModel
   :bintypes:
   :templates:
   :models:
   :properties:
   :description: Here describes the DAP datamodels for bintypes, templates, models, and map properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl8query:

.. datamodel:: marvin.utils.datamodel.query:MPL8
   :prog: Query DataModel
   :title: MPL-8 DataModel
   :parameters:
   :description: Here describes the datamodel for all queryable parameters.  Each table displays relevant information such as the full query name and the group it belongs to. The "full query name" is what is input in all query search filters and return parameters.  The table can be scrolled horizonally for additional info.

.. _mpl8masks:

Maskbits
--------

.. datamodel:: marvin.utils.datamodel.dap:MPL8
   :prog: DRP Maskbits
   :bitmasks:
   :bittype: DRP

.. datamodel:: marvin.utils.datamodel.dap:MPL8
   :prog: DAP Maskbits
   :bitmasks:
   :bittype: DAP

.. datamodel:: marvin.utils.datamodel.dap:MPL8
   :prog: Targeting Maskbits
   :bitmasks:
   :bittype: Target

.. _mpl8vacs:

VACs
----

.. datamodel:: marvin.utils.datamodel.vacs:datamodel
   :prog: Available VACs
   :vac: MPL-8
   :description: A list of the contributed VACs available in this MPL.

