
.. _datamodel-mpl4:

MPL-4
=====

This datamodel corresponds to the MPL-4 (v1_5_1, 1.0.0) release of MaNGA data.


DataModels
----------

* :ref:`DRP<mpl4drp>`
* :ref:`DAP<mpl4dap>`
* :ref:`Query<mpl4query>`
* :ref:`Maskbits<mpl4masks>`

.. _mpl4drp:

.. datamodel:: marvin.utils.datamodel.drp.MPL:MPL4
   :prog: DRP DataModel
   :title: MPL-4 DataModel
   :spectra:
   :datacubes:
   :rss:
   :description: Here describes the DRP datamodels for spectra, datacubes, and rss properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl4dap:

.. datamodel:: marvin.utils.datamodel.dap:MPL4
   :prog: DAP DataModel
   :title: MPL-4 DataModel
   :bintypes:
   :templates:
   :properties:
   :description: Here describes the DAP datamodels for bintypes, templates, and map properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl4query:

.. datamodel:: marvin.utils.datamodel.query:MPL4
   :prog: Query DataModel
   :title: MPL-4 DataModel
   :parameters:
   :description: Here describes the datamodel for all queryable parameters.  Each table displays relevant information such as the full query name and the group it belongs to. The "full query name" is what is input in all query search filters and return parameters.  The table can be scrolled horizonally for additional info.

.. _mpl4masks:

Maskbits
--------

.. datamodel:: marvin.utils.datamodel.dap:MPL4
   :prog: DRP Maskbits
   :bitmasks:
   :bittype: DRP

.. datamodel:: marvin.utils.datamodel.dap:MPL4
   :prog: DAP Maskbits
   :bitmasks:
   :bittype: DAP

.. datamodel:: marvin.utils.datamodel.dap:MPL4
   :prog: Targeting Maskbits
   :bitmasks:
   :bittype: Target


