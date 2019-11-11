
.. _marvin-datamodels:

==========
DataModels
==========

Marvin contains datamodels for each of the releases of MaNGA dataset.  The links below contain full descriptions of what is available for each release.  Each release contains a variety of datamodels.

* **DRP** datamodel: describes the output of the MaNGA Data Reduction Pipeline.
* **DAP** datamodel: describes the output of the MaNGA Data Analysis Pipeline.
* :ref:`Query datamodel <query-dm>`: describes all of the available parameters one can query on using Marvin.
* **Maskbit** datamodel: describes the designated targeting and quality bits and flags used by MaNGA

Generally all datamodels provide a `parameter/property name`, `description`, `units`, its corresponding name in its respective FITS file, and corresponding database schema, table, and column name.

By Release
----------

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

The DRP datamodel contains two descriptors of MaNGA data, **spectra** and **datacubes**.  **spectra** are 1-d arrays of data, while **datacubes** are 3-d arrays of data.

DAP Datamodel
-------------

The DAP datamodel contains four descriptors of ManGA data, **bintypes**, **templates**, **models**, and **properties**.  **Bintypes** and **templates** describe the stellar template library and binning scheme used by the MaNGA DAP.  **Properties** represent the available 2-d outputs of the DAP, (e.g. MAPS), while **models** represent the model-fitting used by the DAP, (e.g. MODELCUBES)

Query Datamodel
---------------

The Query datamodel contains a list of all available queryable parameters for a given release.  The important columns being the **Group** and **full name**.  See the :ref:`How to Guide<query-dm>` for a full description of how to use the query datamodel.

Maskbit Datamodel
-----------------

The Maskbit datamodel contains a description of targeting and quality flags used in the DRP, DAP, and by the Targeting Catalogs.
