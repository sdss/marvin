.. _marvin-cube:

Cube
====

For the docstring see :ref:`marvin-tools-cube`.

A class to interface with MaNGA DRP data cubes.

This class represents a fully reduced DRP data cube, initialised either from a
file, a database, or remotely via the Marvin API.

filename, plate-IFU, mangaID

If remote, fetches data on request: getSpaxel()

getWavelength doesn't work via API

AttributeError: 'Cube' object has no attribute '_useDB'

::
    
    from marvin.tools.cube import Cube
    cc = Cube(mangaid='1-209232')
    cc.download()
    cc.getSpaxel()



|