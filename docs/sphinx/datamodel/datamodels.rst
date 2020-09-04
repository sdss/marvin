
.. _marvin-datamodels:

==========
DataModels
==========

Marvin contains datamodels for each of the releases of MaNGA dataset.  The links below contain full descriptions of what
is available for each release.  Each release contains a variety of datamodels.

* **DRP** datamodel: describes the output of the MaNGA Data Reduction Pipeline.
* **DAP** datamodel: describes the output of the MaNGA Data Analysis Pipeline.
* :ref:`Query datamodel <query-dm>`: describes all of the available parameters one can query on using Marvin.
* **Maskbit** datamodel: describes the designated targeting and quality bits and flags used by MaNGA
* **VACs** datamodel: describes which MaNGA VACs have been contributed to Marvin

Generally all datamodels provide a `parameter/property name`, `description`, `units`, its corresponding name in its respective FITS file,
and corresponding database schema, table, and column name.  All datamodels can be accessed within in the ``marvin.utils.datamodel`` module.

By Release
----------

* :ref:`datamodel-mpl10`
* :ref:`datamodel-mpl9`
* :ref:`datamodel-dr16`
* :ref:`datamodel-mpl8`
* :ref:`datamodel-dr15`
* :ref:`datamodel-mpl7`
* :ref:`datamodel-mpl6`
* :ref:`datamodel-mpl5`
* :ref:`datamodel-mpl4`

DRP Datamodel
-------------

The DRP datamodel contains three descriptors of MaNGA data, **spectra**, **datacubes**, and **rss**.  **spectra** are 1-d arrays of data,
**rss** are 2d-arrays of data, while **datacubes** are 3-d arrays of data.  Access the DRP datamodels for cubes with
``from marvin.utils.datamodel.drp import datamodel`` and the datamodels for rss files with ``from marvin.utils.datamodel.drp import datamodel_rss``.

DAP Datamodel
-------------

The DAP datamodel contains four descriptors of ManGA data, **bintypes**, **templates**, **models**, and **properties**.
**Bintypes** and **templates** describe the stellar template library and binning scheme used by the MaNGA DAP.  **Properties**
represent the available 2-d outputs of the DAP, (e.g. MAPS), while **models** represent the model-fitting used by the DAP, (e.g. MODELCUBES).
Access the DAP datamodels with ``from marvin.utils.datamodel.dap import datamodel``.

Query Datamodel
---------------

The Query datamodel contains a list of all available queryable parameters for a given release.  The important columns being
the **Group** and **full name**.  See the :ref:`How to Guide<query-dm>` for a full description of how to use the query datamodel.  Access
the Query datamodels with ``from marvin.utils.datamodel.query import datamodel``.

Maskbit Datamodel
-----------------

The Maskbit datamodel contains a description of targeting and quality flags used in the DRP, DAP, and by the Targeting Catalogs.

VACs Datamodel
--------------

The VACs datamodel contains a list and description of all VACs in a given release that have been made accessible in Marvin.  See the individual
"By Release" datamodel pages for available VACs in that release. Access the VAC datamodels with ``from marvin.utils.datamodel.vacs import datamodel``.
