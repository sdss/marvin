.. role:: green
.. role:: orange
.. role:: red
.. role:: purple


.. _visual-guide:

Visual Guide to Marvin Tools
----------------------------

All **object-** and **search-based** tools in Marvin are seamlessly linked together.  To better understand the flow amongst the various Tools, here is a visual guide.

|

.. image:: ../../Marvin_Visual_Guide.png
    :width: 800px
    :align: center
    :alt: marvin visual guide

|

* The :red:`red squares` and :green:`green squares` indicate the set of Marvin Tools available.
* The :orange:`orange circles` highlight how each Tool links together via a method or an attribute.  In each transition link, a ``lowercase`` Tool name represents an instantiation of that tool, e.g. ``cube = Cube()``.  To go from a ``Marvin Cube`` to a ``Marvin Spaxel``, you can use the ``cube.getSpaxel`` method or the ``cube[x,y]`` notation.  Conversely, to go from a ``Spaxel`` to a ``Cube``, you would use the ``spaxel.cube`` attribute.  Single- or Bi- directional arrows tell you which directions you can flow to and from the various tools.
* :purple:`Purple circles` represent display endpoints.  If you want to display something, this shows you how which tool the plotting command is connected to, and how to navigate there.
