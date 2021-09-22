
.. _marvin-context:

Tools as Context-managers
=========================

Many marvin tools, e.g. `~marvin.tools.cube.Cube`, or `~marvin.tools.maps.Maps`, act as a 
means of accessing an underlying data-file.  When `data_origin == 'file'`, unless you delete 
the specific tool instance, the underlying file-handler will remain active.  Therefore, 
if you expect to be creating many tool instances (by looping through a list of galaxies, 
for instance), you may create more open files than your system can handle.  By using the tool 
as a context manager, the underlying data file is closed precisely when it is no longer relevant.  

Any subclass of ``MarvinToolsClass`` can be used as a context manager.  This includes the following
tools:

 - `~marvin.tools.modelcube.ModelCube`
 - `~marvin.tools.maps.Maps`
 - `~marvin.tools.cube.Cube`
 - `~marvin.tools.plate.Plate`
 - `~marvin.tools.rss.RSS`

The following example shows how to open a Cube as a context manager, but the same is true for any 
of the other tools.

.. code-block:: python
    
    >>> with Cube(filename='/Users/Brian/Work/Manga/redux/v2_3_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz') as cube:
    >>>     # can access contents of file inside context
    >>>     cube.data.info()
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU      75   ()      
      1  FLUX          1 ImageHDU       100   (34, 34, 4563)   float32   
    ...   
     22  ICORREL       1 BinTableHDU     32   21213R x 5C   [J, J, J, J, D]   
     23  ZCORREL       1 BinTableHDU     32   21458R x 5C   [J, J, J, J, D]

    >>> # once context ceases to be active, the file is closed, and data are inaccessible
    >>> cube.spectral_resolution
    ValueError: I/O operation on closed file

While this can be inconvenient when acting on a single Cube, it can save you from having to manually 
close files when working with many Cubes in sequence, such as when running the same analysis on many 
Cubes in a loop.

.. code-block:: python
    
    >>> for plateifu in plateifus:
    >>>     with Cube(plateifu) as cube:
    >>>         cube_fit_stars_gas(l=cube.wave, f=cube.flux, w=cube.ivar)

