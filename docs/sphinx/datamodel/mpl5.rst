
.. _datamodel-mpl5:

MPL-5
=====

This datamodel corresponds to the MPL-5 (v2_0_1, 1.1.1) release of MaNGA data.


DataModels
----------

* :ref:`DRP<mpl5drp>`
* :ref:`DAP<mpl5dap>`
* :ref:`Query<mpl5query>`
* :ref:`Maskbits<mpl5masks>`

.. _mpl5drp:

.. datamodel:: marvin.utils.datamodel.drp.MPL:MPL5
   :prog: DRP DataModel
   :title: MPL-5 DataModel
   :spectra:
   :datacubes:
   :rss:
   :description: Here describes the DRP datamodels for spectra, datacubes, and rss properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl5dap:

.. datamodel:: marvin.utils.datamodel.dap:MPL5
   :prog: DAP DataModel
   :title: MPL-5 DataModel
   :bintypes:
   :templates:
   :models:
   :properties:
   :description: Here describes the DAP datamodels for bintypes, templates, models, and map properties.  Each table displays relevant information such as property name, a description and units, as well as which FITS extension the property corresponds to.  Each table can be scrolled horizonally for additional info.

.. _mpl5query:

.. datamodel:: marvin.utils.datamodel.query:MPL5
   :prog: Query DataModel
   :title: MPL-5 DataModel
   :parameters:
   :description: Here describes the datamodel for all queryable parameters.  Each table displays relevant information such as the full query name and the group it belongs to. The "full query name" is what is input in all query search filters and return parameters.  The table can be scrolled horizonally for additional info.

.. _mpl5masks:

Maskbits
--------

.. datamodel:: marvin.utils.datamodel.dap:MPL5
   :prog: DRP Maskbits
   :bitmasks:
   :bittype: DRP

.. datamodel:: marvin.utils.datamodel.dap:MPL5
   :prog: DAP Maskbits
   :bitmasks:
   :bittype: DAP

.. datamodel:: marvin.utils.datamodel.dap:MPL5
   :prog: Targeting Maskbits
   :bitmasks:
   :bittype: Target

