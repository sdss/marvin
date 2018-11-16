
.. role:: green
.. role:: orange
.. role:: red
.. role:: purple

.. py:currentmodule:: marvin.tools

.. _galaxy-tools:

Galaxy Tools
============

Introduction
------------

Marvin Tools provide the core functionality accessing MaNGA data with Marvin. At their lowest level they are class wrappers around data products or elements (DRP datacubes, DAP maps, spaxels). Their goals is to provide a more natural way to interact with the data, unconstrained by specific data storage architectures such as files or databases. The tools are built on top of the :ref:`data access modes <marvin-dma>` which abstracts the data access regardless of their actual origin.

Marvin Tools provide:

.. todo:: Add links in this list once all the documentation is in place.

- Access DRP :ref:`Cubes <marvin-cube>` and their spectra.
- Access :ref:`Row-stacked Spectra <marvin-rss>` files.
- Access DAP :ref:`Maps <marvin-maps>` and :ref:`ModelCubes <marvin-modelcube>`.
- Convenient access to all the DRP and DAP properties for a given :ref:`Spaxel or Bin <marvin-subregion-tools>`.
- The data is delivered as :ref:`quantities <marvin-quantities>` with attached variance and mask, as well as associated properties.
- Easy handling of maskbits and labels.
- DAP :ref:`Map arithmetic <marvin-enhanced-map>`.
- Autocompletion of properties and channels (powered by a datamodel).
- Extract all spaxels within a region.
- Access to NSA and DRPall data.
- Easy data download.


Using the tools
---------------

Data access modes
^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    ../core/data-access-modes

Storing data
^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    downloads
    pickling

Accessing catalogue data
^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    catalogues

Defining apertures
^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    aperture


Maskbits
^^^^^^^^

.. toctree::
    :maxdepth: 2

    utils/maskbit

Datamodels
^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    datamodel

Advanced use of Galaxy Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    cube
    rss
    dap_tools
    plate

Plotting
^^^^^^^^

.. toctree::
    :maxdepth: 2

    utils/plotting
    bpt

Image utilities
^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 2

    image


.. _visual-guide:

Visual guide
------------

All **object-** and **search-based** tools in Marvin are linked together. To better understand the flow amongst the various Tools, here is a visual guide.

.. image:: ../../Marvin_Visual_Guide.png
    :width: 800px
    :align: center
    :alt: marvin visual guide

* The :red:`red squares` and :green:`green squares` indicate the set of Marvin Tools available.
* The :orange:`orange circles` highlight how each Tool links together via a method or an attribute. In each transition link, a lowercase Tool name represents an instantiation of that tool, e.g. ``cube = Cube()``. To go from a Marvin ``Cube`` to a Marvin ``Spaxel``, you can use the ``cube.getSpaxel`` method or the ``cube[x,y]`` notation. Conversely, to go from a ``Spaxel`` to a ``Cube``, you would use the ``spaxel.cube`` attribute. Single or bidirectional arrows tell you which directions you can flow to and from the various tools.
* :purple:`Purple circles` represent display endpoints. If you want to display something, this shows you how which tool the plotting command is connected to, and how to navigate there.


Reference
---------

Tools
^^^^^

.. autosummary::

   marvin.tools.cube.Cube
   marvin.tools.rss.RSS
   marvin.tools.maps.Maps
   marvin.tools.modelcube.ModelCube

Quantities
^^^^^^^^^^

.. autosummary::

    marvin.tools.quantities.analysis_props.AnalysisProperty
    marvin.tools.quantities.spectrum.Spectrum
    marvin.tools.quantities.map.Map
    marvin.tools.rss.RSSFiber
    marvin.tools.quantities.datacube.DataCube

MixIns
^^^^^^

.. autosummary::

    marvin.tools.mixins.nsa.NSAMixIn
    marvin.tools.mixins.dapall.DAPallMixIn
    marvin.tools.mixins.aperture.GetApertureMixIn
