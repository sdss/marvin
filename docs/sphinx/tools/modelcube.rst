.. _marvin-modelcube:

ModelCube
=========

:ref:`marvin-tools-modelcube` is a class to interact with a DAP model data cube for a galaxy. First specify the galaxy that you want by creating a :ref:`marvin-tools-cube` object. If you have the DAP model cube file on your machine and in the expected directory (see ```:ref:sasdir-setup```), then you're good to go! If not, don't worry because Marvin will simply retrieve the data from the :ref:`marvin-databases` at Utah. Let's grab a model cube and plot the spectrum of its central spaxel.


.. everything below here is unchanged from cube.rst


::
    
    from marvin.tools.cube import Cube
    cube = Cube(mangaid='1-209232')
    cube[17, 17].spectrum.plot()

.. image:: ../_static/spec_8485-1901_17-17.png


Here ``cube[17, 17]`` is a shorthand for ``cube.getSpaxel(x=17, x=17, xyorig='lower')``. This returns a :ref:`marvin-tools-spaxel` object. We then use the ``spectrum`` attribute of the :ref:`marvin-tools-spaxel` object to get a :ref:`marvin-tools-spectrum` object that has a ``plot`` method to show the spectrum.

If we want the model spectrum, we must explicitly ask for it by setting ``modelcube=True`` when we initialize the :ref:`marvin-tools-spaxel` object.

::

        spax = cube.getSpaxel(x=17, y=17, xyorig='lower', modelcube=True)
        spax.model.plot()

.. image:: ../_static/modelspec_8485-1901_17-17.png

Then we just access the model spectrum with the ``model`` attribute, which returns a :ref:`marvin-tools-spectrum` object, and call its ``plot`` method again.



If all of this jumping between Marvin objects is making your head spin, please consult this handy diagram: ```:ref:marvin-road-map```.

|