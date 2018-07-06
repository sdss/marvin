.. Marvin documentation master file, created by
   sphinx-quickstart on Sun Apr 10 08:50:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: header_no_toc
  :class: class_header_no_toc

.. title:: Marvin documentation

|

.. image:: _static/logo5_lores.jpg
    :width: 600px
    :align: center
    :alt: MaNGA logo

Marvin
======

.. warning:: Marvin 2.x is the last version that will support Python 2.
  Marvin 3 and following will require Python 3.6+.

Marvin is a tool specifically designed to visualise and analyse `MaNGA <https://www.sdss.org/manga>`_ data. It is
developed and maintained by the MaNGA team. Marvin allows you to:

* Access reduced MaNGA datacubes local, remotely, or via a web interface.
* Access and visualise data analysis products.
* Perform powerful queries on data and metadata.
* Abstract the datamodel and forget where the data actually lives.
* Make good visualisation and scientific decisions by preventing common mistakes when accessing the data.

Marvin's code is publicly available in our `Github <https://github.com/sdss/marvin>`__ page. If you are using Marvin in any way (Web, API, or Tools) to do your science, please
remember to :ref:`acknowledge and cite<marvin-citation>` us in your paper!

To install marvin simply run ``pip install sdss-marvin``. More details and known problem with installation can be found :ref:`here <marvin-installation>`.

.. note:: This documentation assumes that you are familiar with MaNGA data and its acronyms. If that is not your case, consider reading `this <http://www.apage.com>`_.


Getting Started:
----------------

If you are new to Marvin check the following links before diving into the full documentation:

* The :ref:`lean tutorial <marvin-lean-tutorial>` is your quick start guide to Marvin.
* :ref:`What's new in Marvin? <whats-new>`, :ref:`changelog <marvin-changelog>`, and :ref:`known issues <marvin-known-issues>`.
* Marvin uses `quantities <http://docs.astropy.org/en/stable/units/quantity.html>`_ to represent data (spectra, data cubes, etc). Here is a quick :ref:`introduction <marvin-quantities>`.
* More :ref:`tutorials <marvin-tutorials>` and :ref:`frequently asked questions <marvin-faq>`.

Contents:
---------

.. toctree::
   :maxdepth: 2
   :glob:

   installation
   cheatsheet
   tools
   core
   api
   web
   apiref

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

.. :doc:`Web <web>`
.. ----------------

.. :doc:`Datamodels <datamodel/datamodels>`
.. ----------------------------------------

.. * :doc:`datamodel/mpl4`
.. * :doc:`datamodel/mpl5`
.. * :doc:`datamodel/mpl6`


.. API/Code Reference
.. ==================

.. .. toctree::
..    :maxdepth: 4
..    :glob:

..    api/general
..    api/api
..    api/tools
..    api/quantities
..    api/queries
..    api/utils
..    api/web
..    api/db
..    api/brain/api
..    api/brain/utils

|

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`

.. * :ref:`search`
