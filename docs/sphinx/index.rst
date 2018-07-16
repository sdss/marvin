.. Marvin documentation master file, created by
   sphinx-quickstart on Sun Apr 10 08:50:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: Marvin documentation

.. warning:: Marvin documentation is in the process of being restructured and improved. Some content can be missing while this task continue. In the meantime, please use the `stable version <http://sdss-marvin.readthedocs.io/en/stable/>`__ of the documentation. If you have suggestions to improve Marvin documentation, please `open an issue <https://github.com/sdss/marvin/issues/new/choose>`__.

|

.. image:: _static/logo5_lores.jpg
    :width: 600px
    :align: center
    :alt: MaNGA logo

|

Marvin Documentation
====================

Marvin is a tool specifically designed to visualise and analyse `MaNGA <https://www.sdss.org/manga>`_ data. It is developed and maintained by the MaNGA team. Marvin allows you to:

* Access reduced MaNGA datacubes local, remotely, or via a web interface.
* Access and visualise data analysis products.
* Perform powerful queries on data and metadata.
* Abstract the datamodel and forget where the data actually lives.
* Make good visualisation and scientific decisions by preventing common mistakes when accessing the data.

Marvin's code is publicly available in our `Github <https://github.com/sdss/marvin>`__ page. If you are using Marvin in any way (Web, API, or Tools) to do your science, please remember to :ref:`acknowledge and cite<marvin-citation>` us in your paper!

To install marvin simply run ``pip install sdss-marvin``. More details and known problem with installation can be found :ref:`here <marvin-installation>`.

.. note:: This documentation assumes that you are familiar with MaNGA data and its acronyms. If that is not your case, consider reading `this <http://www.apage.com>`_.

.. warning:: Marvin 2.x is the last version that will support Python 2.
  Marvin 3 and following will require Python 3.6+.


Getting Started
---------------

If you are new to Marvin check the following links before diving into the full documentation:

* The :ref:`lean tutorial <marvin-lean-tutorial>` is your quick start guide to Marvin.
* :ref:`What's new in Marvin? <whats-new>`, :ref:`changelog <marvin-changelog>`, and :ref:`known issues <marvin-known-issues>`.
* Marvin uses `quantities <http://docs.astropy.org/en/stable/units/quantity.html>`_ to represent data (spectra, data cubes, etc). Here is a quick :ref:`introduction <marvin-quantities>`.
* More :ref:`tutorials <marvin-tutorials>` and :ref:`frequently asked questions <marvin-faq>`.


.. toctree::
   :maxdepth: 2
   :glob:
   :caption: Marvin at a Glance

   installation
   cheatsheet

.. Quick Tools
.. ===========

.. .. toctree::
..    :maxdepth: 2
..    :glob:

..    tools

.. Components
.. ==========

.. :doc:`Core <core>`
.. ------------------

.. * :doc:`core/config`
.. * :doc:`core/data-access-modes`
.. * :doc:`core/downloads`
.. * :doc:`tools/quantities`

.. :doc:`Tools <tools>`
.. --------------------

.. * :doc:`query`
.. * :doc:`results`
.. * :doc:`tools/plate`
.. * :doc:`tools/cube`
.. * :doc:`tools/modelcube`
.. * :doc:`tools/maps`

..   * :doc:`tools/bpt`

.. * :doc:`tools/map`

..   * :doc:`tools/enhanced-map`

.. * :doc:`tools/spaxel`
.. * :doc:`tools/bin`

.. * :doc:`utils`

..   * :doc:`Image <utils/images>`
..   * :doc:`Map Plotting <utils/plot-map>`
..   * :doc:`Scatter Plotting <utils/plot-scatter>`
..   * :doc:`Histogram Plotting <utils/plot-hist>`
..   * :doc:`Maskbit <utils/maskbit>`

.. :doc:`API <api>`
.. ----------------

.. toctree::
   :maxdepth: 3
   :caption: User Docs

   tools/index


.. :doc:`Web <web>`
.. ----------------

.. :doc:`Datamodels <datamodel/datamodels>`
.. ----------------------------------------

.. * :doc:`datamodel/mpl4`
.. * :doc:`datamodel/mpl5`
.. * :doc:`datamodel/mpl6`


.. toctree::
   :maxdepth: 1
   :caption: API/Code Reference

   api/index


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`

.. * :ref:`search`
