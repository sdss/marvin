
.. _datamodel-mpl10:

MPL-10
======

This datamodel corresponds to the MPL-10 (v3_0_1, 3.0.1) release of MaNGA data.


DataModels
----------

* :ref:`DRP<mpl10drp>`
* :ref:`DAP<mpl10dap>`
* :ref:`Query<mpl10query>`
* :ref:`Maskbits<mpl10masks>`
* :ref:`VACs<mpl10vacs>`

.. _mpl10drp:

.. datamodel:: marvin.utils.datamodel.drp.MPL:MPL10
   :prog: DRP DataModel
   :title: MPL-10 DataModel
   :spectra:
   :datacubes:
   :rss:
   :description: Here describes the DRP datamodels for spectra, datacubes, and rss properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl10dap:

.. datamodel:: marvin.utils.datamodel.dap:MPL10
   :prog: DAP DataModel
   :title: MPL-10 DataModel
   :bintypes:
   :templates:
   :models:
   :properties:
   :description: Here describes the DAP datamodels for bintypes, templates, models, and map properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl10query:

.. datamodel:: marvin.utils.datamodel.query:MPL10
   :prog: Query DataModel
   :title: MPL-10 DataModel
   :parameters:
   :description: Here describes the datamodel for all queryable parameters.  Each table displays relevant information such as the full query name and the group it belongs to. The "full query name" is what is input in all query search filters and return parameters.  The table can be scrolled horizonally for additional info.

.. _mpl10masks:

Maskbits
--------

.. datamodel:: marvin.utils.datamodel.dap:MPL10
   :prog: DRP Maskbits
   :bitmasks:
   :bittype: DRP

.. datamodel:: marvin.utils.datamodel.dap:MPL10
   :prog: DAP Maskbits
   :bitmasks:
   :bittype: DAP

.. datamodel:: marvin.utils.datamodel.dap:MPL10
   :prog: Targeting Maskbits
   :bitmasks:
   :bittype: Target

.. _mpl10vacs:

VACs
----

.. datamodel:: marvin.utils.datamodel.vacs:datamodel
   :prog: Available VACs
   :vac: MPL-10
   :description: A list of the contributed VACs available in this MPL.

