.. _marvin-cube:

Cube
====

:ref:`marvin-tools-cube` is a class to interact with a fully reduced DRP data cube for a galaxy. First specify the galaxy that you want by creating a :ref:`marvin-tools-cube` object. If you have the DRP cube file on your machine and in the expected directory (see :ref:`marvin-sasdir`), then you're good to go! If not, don't worry because Marvin will simply retrieve the data from the :ref:`marvin-databases` at Utah. Let's grab a cube and plot the spectrum of its central spaxel.


.. If remote, fetches data on request: getSpaxel()
   getWavelength doesn't work via API
   AttributeError: 'Cube' object has no attribute '_useDB'

::

    from marvin.tools.cube import Cube
    cube = Cube(mangaid='1-209232')
    cube[17, 17].spectrum.plot()

.. image:: ../_static/spec_8485-1901_17-17.png


Here ``cube[17, 17]`` is a shorthand for ``cube.getSpaxel(x=17, y=17, xyorig='lower')``. This returns a :ref:`marvin-tools-spaxel` object. We then use the ``spectrum`` attribute of the :ref:`marvin-tools-spaxel` object to get a :ref:`marvin-tools-spectrum` object that has a ``plot`` method to show the spectrum.

If we want the model spectrum, we must explicitly ask for it by setting ``modelcube=True`` when we initialize the :ref:`marvin-tools-spaxel` object.

::

        spax = cube.getSpaxel(x=17, y=17, xyorig='lower', modelcube=True)
        spax.model.plot()

.. image:: ../_static/modelspec_8485-1901_17-17.png

Then we just access the model spectrum with the ``model`` attribute, which returns a :ref:`marvin-tools-spectrum` object, and call its ``plot`` method again.



If all of this jumping between Marvin objects is making your head spin, please consult this handy diagram: :ref:`visual-guide`.

|
