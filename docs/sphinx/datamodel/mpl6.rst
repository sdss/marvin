
.. _datamodel-mpl6:

MPL-6
======

This datamodel corresponds to the MPL-6 (v2_2_1, 2.1.3) release of MaNGA data.


DataModels
----------

* :ref:`DRP<mpl6drp>`
* :ref:`DAP<mpl6dap>`
* :ref:`Query<mpl6query>`
* :ref:`Maskbits<mpl6masks>`

.. _mpl6drp:

.. datamodel:: marvin.utils.datamodel.drp.MPL:MPL6
   :prog: DRP DataModel
   :title: MPL-6 DataModel
   :spectra:
   :datacubes:
   :rss:
   :description: Here describes the DRP datamodels for spectra, datacubes, and rss properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl6dap:

.. datamodel:: marvin.utils.datamodel.dap:MPL6
   :prog: DAP DataModel
   :title: MPL-6 DataModel
   :bintypes:
   :templates:
   :models:
   :properties:
   :description: Here describes the DAP datamodels for bintypes, templates, models, and map properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl6query:

.. datamodel:: marvin.utils.datamodel.query:MPL6
   :prog: Query DataModel
   :title: MPL-6 DataModel
   :parameters:
   :description: Here describes the datamodel for all queryable parameters.  Each table displays relevant information such as the full query name and the group it belongs to. The "full query name" is what is input in all query search filters and return parameters.  The table can be scrolled horizonally for additional info.

.. _mpl6masks:

Maskbits
--------

.. datamodel:: marvin.utils.datamodel.dap:MPL6
   :prog: DRP Maskbits
   :bitmasks:
   :bittype: DRP

.. datamodel:: marvin.utils.datamodel.dap:MPL6
   :prog: DAP Maskbits
   :bitmasks:
   :bittype: DAP

.. datamodel:: marvin.utils.datamodel.dap:MPL6
   :prog: Targeting Maskbits
   :bitmasks:
   :bittype: Target


