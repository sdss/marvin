
.. _marvin-plate:

Plate
=====

.. _marvin-plate_getstart:

Getting Started
---------------

The Marvin `Plate` object is a representation of a MaNGA plate, and contains all available Marvin `Cubes` associated with the targets observed on this plate.  The easiest way to instantiate it is with the `plate` keyword.

::

    from marvin.tools.plate import Plate

    plate = Plate(plate=8485)
    print(plate)
     <Marvin Plate (plate=8485, n_cubes=4, mode='local', data_origin='db')>

     # check the number of available cubes
     print(len(plate))
     4

`Plate` acts as a Python list object.  To access the `Cubes` for the plate, you can use list indexing

::

    plate[0]
    <Marvin Cube (plateifu='8485-1902', mode='local', data_origin='db')>

or fuzzy string indexing

::

    plate['1901']
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='db')>

    plate['8485-1901']
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='db')>

or, alternatively, the `cubeXXXX` attributes.  Each `Plate` maps its cubes onto attributes with the designation **cube[IFUNAME]**.

::

    plate.cube12701
    <Marvin Cube (plateifu='8485-12701', mode='local', data_origin='db')>

.. _marvin-plate-using:

Using Plate
-----------

The Marvin `Plate` object is subclassed from both `Marvin Tools` and a fuzzy Python list.  Thus it behaves as both a Marvin object and a Python list object.  The `Plate` is a list of Marvin `Cube` objects associated with the targets observed for this plate id.

To instantiate a `Plate`, specify the plate id.

::

    from marvin.tools.plate import Plate

    plate = Plate(plate=8485)
    print(plate)
     <Marvin Plate (plate=8485, n_cubes=4, mode='local', data_origin='db')>

Marvin will attempt to load all available cubes for the given plate, using the multi-modal data access system.

.. _marvin-plate_basic:

Basic Attributes
^^^^^^^^^^^^^^^^

Some basic parameters for the `Plate` are made available as attributes, e.g. `cartid`, `designid`, `ra`, `dec`, `dateobs`, etc.

::

    # cart id
    plate.cartid
    '3'

    # design id
    plate.designid
    8980

    # plate RA, Dec center
    plate.ra, plate.dec
    (234.06426, 48.589874)

    # date of observation
    plate.dateobs
    u'2015-04-20'

    # survey mode
    plate.surveymode
    u'MaNGA dither'

.. _marvin-plate_access:

Accessing Cubes
^^^^^^^^^^^^^^^

You can access cubes for the `Plate` in one of three ways, either via Python list indexing

::

    plate[0]
    <Marvin Cube (plateifu='8485-1902', mode='local', data_origin='db')>

or fuzzy string indexing

::

    plate['1901']
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='db')>

    plate['8485-1901']
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='db')>

or, alternatively, the `cubeXXXX` attributes.  Each `Plate` maps its cubes onto attributes with the designation **cube[IFUNAME]**.

::

    plate.cube12701
    <Marvin Cube (plateifu='8485-12701', mode='local', data_origin='db')>

.. _marvin-plate_save:

Saving and Restoring
^^^^^^^^^^^^^^^^^^^^

Like other Marvin Tools, you can save a `Plate` locally as a Python pickle object, using the `save` method.

::

    plate.save('myplate.mpf')

as well as restore a Plate pickle object using the `restore` class method

::

    from marvin.tools.plate import Plate

    plate = Plate.restore('myplate.mpf')

.. _marvin-plate-api:

Reference/API
-------------

.. rubric:: Class Inheritance Diagram

.. inheritance-diagram:: marvin.tools.plate.Plate

.. rubric:: Class

.. autosummary:: marvin.tools.plate.Plate

.. rubric:: Methods

.. autosummary::

    marvin.tools.plate.Plate.save
    marvin.tools.plate.Plate.restore


