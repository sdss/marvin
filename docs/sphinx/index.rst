.. Marvin documentation master file, created by
   sphinx-quickstart on Sun Apr 10 08:50:42 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: Marvin documentation


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

.. To install marvin simply run ``pip install sdss-marvin``. More details and known problem with installation can be found :ref:`here <marvin-installation>`.

.. note:: This documentation assumes that you are familiar with MaNGA data and its acronyms. If that is not your case, consider reading `this <https://www.sdss.org/manga/getting-started/>`_.

.. warning:: Marvin 2.x is the last version that will support Python 2.
  Marvin 3 and following will require Python 3.6+.


At a Glance
-----------

If you are new to Marvin check the following links before diving into the full documentation:

* :ref:`Installation <marvin-installation>`
* :ref:`What's new in Marvin? <whats-new>`
* The :ref:`Getting Started <marvin-getting_started>` is your quick start guide to Marvin.
* For quick reference, check out the :ref:`Cheatsheet <marvin-cheatsheet>`.
* For more detailed examples, see the :ref:`Tutorials <marvin-tutorials>`.
* Here is an :ref:`Introduction <marvin-quantities>` to the `Astropy quantity <http://docs.astropy.org/en/stable/units/quantity.html>`_ class that Marvin uses to represent data (spectra, data cubes, etc).
* We welcome :ref:`Contributions <marvin-contributing>` to Marvin!
* If you find Marvin useful for your research, please :ref:`Cite Marvin <marvin-citation>` to help others find it, too.


.. toctree::
   :maxdepth: 2
   :caption: At a Glance
   :hidden:

   installation
   whats-new
   getting-started
   cheatsheet
   tutorials/index
   contributing/contributing
   citation

.. toctree::
   :maxdepth: 2
   :caption: User Docs

   getting-started
   tools/index
   query/index
   web
   api


.. toctree::
   :maxdepth: 3
   :caption: Datamodels

   datamodel/datamodels

.. toctree::
   :maxdepth: 2
   :caption: API/Code Reference

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Project Details

   Changelog <changelog>
   Known Issues <known-issues>
   faq


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

* :ref:`search`
