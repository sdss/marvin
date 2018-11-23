
.. _marvin-getting_started_old:

Getting Started
===============

If you have not yet installed Marvin, please follow the :ref:`marvin-installation` instructions before proceeding.  In particular, make sure you have a **netrc** file in your local home directory.  This will enable Marvin to access data remotely, and download files.

From your terminal, type ipython.  Ipython is an Interactive Python shell terminal.  It is recommended to always use ipython instead of python.::

    > ipython

To jump right in, try the :ref:`marvin-lean-tutorial`.

This page explains how to do the following:

* :ref:`marvin-getstart_accessremote`
* :ref:`marvin-getstart_displayplots`
* :ref:`marvin-getstart_accesslocally`
* :ref:`marvin-getstart_querysample`
* :ref:`marvin-getstart_downloadbulk`
* :ref:`marvin-getstart_converttools`
* :ref:`marvin-getstart_lookimages`

.. _marvin-getstart_accessremote:

Accessing Objects Remotely
--------------------------

Marvin has a variety of Tools designed to help you access the various MaNGA data products.  Let's start with the basic MaNGA data product, the 3d datacube output by the Data Reduction Pipeline (DRP).  The Marvin Cube class is designed to aid your interaction with MaNGA's datacubes.

::

    # import the Marvin Cube tool
    from marvin.tools.cube import Cube

Once the tool is imported, you can instantiate a particular target object with the following

::

    # instantiate a Marvin Cube for the MaNGA object with plate-ifu 8485-1901
    cube = Cube(plateifu='8485-1901')

    # display a string representation of your cube
    print(cube)
    <Marvin Cube (plateifu='8485-1901', mode='remote', data_origin='api')>

You have just created a Marvin Cube for plate-ifu **8485-1901**.  You will also see the **mode** and **data_origin** keywords.  These keywords inform you of how your cube was accessed.  You can see that your cube was opened remotely via the Marvin API.  (If you already have this cube locally, you may see a different indicator for mode and data_origin.  That's ok!). For all remote access, Marvin loads things lazily, meaning it loads the bare minimum it needs to in order to provide you a working cube quickly.  Additional information is loaded on demand via the Marvin API.  For a remote Cube, you have access to the full header of your datacube and a few select properties of the Cube.

::

    # view the datacube primary header for 8485-1901
    print(cube.header)

    # some select properties, e.g. RA, Dec, are added as Cube attributes
    cube.ra
    cube.dec

Marvin has many Tools available that all behave in a similar way to the Marvin Cube.  See :ref:`marvin-tools` for a description of all the Tools currently available.

.. _marvin-getstart_displayplots:

Displaying Plots
----------------

Marvin makes it really easy to very quickly plot a spectrum from a spaxel, or a 2-d Map output by MaNGA's Data Analysis Pipeline (DAP).

Accessing and Displaying Spectra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Marvin Cubes give you access to the MaNGA datacube.  Datacubes are a 2-d array of spaxels, with a third spectral dimension.  A spaxel is a small (0.5 x 0.5 arcsec) spatial region on the sky with an associated spectrum.  A Spectrum has associated with it a flux, inverse variance, mask, and wavelength array.  To access spaxels from Marvin Cubes, you can index them directly like you would a normal 2d-array in Python or IDL.  In this manner, the default indexing is from the lower left corner of the array.

::

    # access the spaxel from the lower left corner of the Cube for 8485-1901
    spaxel = cube[0,0]

    # represent the spaxel
    print(spaxel)
    <Marvin Spaxel (x=0, y=0; x_cen=-17, y_cen=-17>

Notice the **x, y** attributes.  These are the indices using the lower left corner of the array as the 0-point.  **x_cen, y_cen** displays the corresponding indices at the center of the datacube.

Alternatively, you can use the **getSpaxel** method on the Cube object.  By default, the **getSpaxel** method will index spaxels relative to the center of the datacube.  Index 0,0 is the center of the 2d-array, rather than the lower left corner.

::

    # access the spaxel relative to the center of the Cube
    spaxel = cube.getSpaxel(0,0)

    print(spaxel)
    <Marvin Spaxel (x=17, y=17; x_cen=0, y_cen=0>

Notice how the coordinate reference changes between the two spaxel examples.

To plot the spectrum for this Marvin Spaxel, you must first access the Marvin Spectrum object for this Spaxel using the **spectrum** attribute on each spaxel. Once you have the spectrum, you can access its data with the **flux**, **ivar**, **mask**, **wavelength** keywords, or plot it with the **plot** method.

::

    # get the spectrum
    flux = spaxel.flux

    # plot the spectrum
    flux.plot()

    # access the data as Numpy arrays
    flux.value
    array([ 0.47127277,  0.41220659,  0.47146896, ...,  0.        ,
            0.        ,  0.        ], dtype=float32)

    # the ivar array
    flux.ivar
    array([ 0.47127277,  0.41220659,  0.47146896, ...,  0.        ,
            0.        ,  0.        ], dtype=float32)

    # the mask array
    flux.mask
    array([   0,    0,    0, ..., 1026, 1026, 1026], dtype=int32)

    # the wavelength array
    flux.wavelength
    array([  3621.59598486,   3622.42998417,   3623.26417553, ...,
            10349.03843826,  10351.42166679,  10353.80544415])

Accessing and Displaying Maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Marvin has the ability to quickly access and display any of DAP Maps available in a given MPL.

::

    # from our previous cube, let's access the default Maps associated with 8485-1901
    maps = cube.getMaps()

    # display the string representation of the your maps object
    print(maps)
    maps = <Marvin Maps (plateifu='8485-1901', mode='remote', data_origin='api', bintype=SPX, template_kin=GAU-MILESHC)>

The default Maps object created is the unbinned maps DAP object.  You can request a map with a different bintype or stellar template model using the **bintype** and **template_kin** keywords.  To access individial maps, you can do so either via array indexing, or using the **getMap** method on Marvin Maps.  Individual maps are uniquely identified by **property** name and **channel**.  This is the same syntax used by DAP data model for MaNGA MAPS objects.

With the array-indexing mode, you specify the full **property+channel**, as a lowercase, underscore-spaced string.  When using the **getMap** method, you specify property and channel individual via keywords.

::

    # grab the H-alpha emission line map by array indexing
    hamap = maps['emline_gflux_ha_6564']

    # alternatively, use getMap
    hamap = maps.getMap('emline_gflux', channel='ha_6564')

    # display the Map object
    print(hamap)
    <Marvin Map (plateifu='8485-1901', property='emline_gflux', channel='ha_6564')>

You have now accessed an individual Marvin Map.  The **property** **channel** keywords indicate whichs DAP property and channel (if any) you have accessed.  The raw arrays for the data, inverse variance, and mask are stored in the attributes **value**, **ivar**, **mask** on each map object.

::

    # access the 2-d H-alpha flux data values
    data = hamap.value

    print(type(data))
    <type 'numpy.ndarray'>

    print(data)
    array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           ...,
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.]])

    # 2-d inverse variance array
    hamap.ivar

    # DAP mask array
    hamap.mask

You can plot any map simply by using the **plot** method on your Map object.

::

    # plot the H-alpha flux map
    hamap.plot()

You should see a pop-up window containing the H-alpha emission line flux map for 8485-1901.  Marvin uses the Python package Matplotlib for all default plotting.  Many matplotlib plotting options are available in Marvin's **plot** method.  To see a full list of available options, use **plot?**, or see the :ref:`Maps page<marvin-tools-maps>`.  Help for all Marvin Tools and methods can be displayed by appending a **?** to the end of the name, excluding the parantheses.

::

    # see the help for the plot command
    hamap.plot?

    # change some default plot options. Let's change the S/N cutoff using in the plot, and change the default color map used.
    hamap.plot(snr_min=5, cmap='inferno')

.. _marvin-getstart_download:

Downloading Your Object
-----------------------

In the previous steps you have been accessing the MaNGA data for **8485-1901** remotely with Marvin.  But now you want to get your hands dirty with the real data file.  You can easily download MaNGA data products with Marvin.  There are many ways to download data with Marvin.  To download individual data file for the objects you are working with, use the **download** method attached to your object.  You can only download objects that have associated MaNGA data product files.

.. code-block:: python

    # download the DRP datacube file for 8485-1901
    cube.download()

    # download the DAP unbinned MAPS file for 8485-1901
    maps.download()

    # You cannot download individual maps because there is no associated DAP data product, so this will fail:
    hamap.download()
    # AttributeError: 'Map' object has no attribute 'download'

This describes a method for manual download of individual files.  There are other ways to download MaNGA files.  See :ref:`marvin-download-objects` for a full description of how to download data.


.. _marvin-getting-started-sas-base-dir:

MaNGA File Directory Organization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The files are stored in your local **SAS (Science Archive Server)** as set up by Marvin.  This local **SAS** is a direct mimic of the real **SAS**, used at Utah by MaNGA in SDSS-IV.  Marvin creates and uses an environment variable called **SAS_BASE_DIR**.  Unless you have this already set up, Marvin creates this in your local home directory.  To see where your **SAS_BASE_DIR** is located, use the Python **os** package.

::

    import os
    print(os.environ['SAS_BASE_DIR'])
    '/Users/Brian/Work/sdss/sas'

You should see a directory path printed. If you get an error of the sort **KeyError: 'SAS_BASE_DIR'**, then you are missing this environment variable.  Something has gone wrong with your Marvin set up and configuration.  Please contact the developers.

.. _marvin-getstart_accesslocally:

Accessing Objects Locally
-------------------------

In the previous section, you downloaded the data files for 8485-1901 directly to your computer.  Now let's access this file.  The beauty of Marvin is that you do not have to do anything different once you have downloaded a file to access it locally.  Simply call your object the same way as before, and Marvin's Smart Multi-Modal Data Access System will do the rest.

::

    # instantiate a Marvin Cube for plate-ifu 8485-1901
    cube = Cube(plateifu='8485-1901')

    # display the cube
    print(cube)
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='file')>

Notice that the **mode** is now **local**, and the **data_origin** is now set to **file**.  You are now accessing the full FITS file for the 3d datacube for 8485-1901.  Marvin uses the **Astropy io.fits** package for all FITS handling.  Please see the Astropy documentation for a full description of `FITS handling <http://docs.astropy.org/en/stable/io/fits/>`_.

::

    # print the full file name and path to your data file
    print(cube.filename)
    '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz'

    # access the FITS header
    cube.header

    # retrieve a list of file HDUs
    hdus = cube.data
    hdus.info()

    Filename: /Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz
    No.    Name         Type      Cards   Dimensions   Format
      0  PRIMARY     PrimaryHDU      74   ()
      1  FLUX        ImageHDU        99   (34, 34, 4563)   float32
      2  IVAR        ImageHDU        17   (34, 34, 4563)   float32
      3  MASK        ImageHDU        17   (34, 34, 4563)   int32
      4  WAVE        ImageHDU         9   (4563,)   float64
      5  SPECRES     ImageHDU         9   (4563,)   float64
      6  SPECRESD    ImageHDU         9   (4563,)   float64
      7  OBSINFO     BinTableHDU    144   9R x 63C   [25A, 17A, 5A, J, I, 8A, E, E, E, E, E, E, J, J, I, J, E, 12A, J, 8A, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, 13A, E, E, E, E, D, D, D, D, E, E, J, J, J, E, E, E, E, J, J, E, E, E, E]
      8  GIMG        ImageHDU        28   (34, 34)   float32
      9  RIMG        ImageHDU        28   (34, 34)   float32
     10  IIMG        ImageHDU        28   (34, 34)   float32
     11  ZIMG        ImageHDU        28   (34, 34)   float32
     12  GPSF        ImageHDU        28   (34, 34)   float32
     13  RPSF        ImageHDU        28   (34, 34)   float32
     14  IPSF        ImageHDU        28   (34, 34)   float32
     15  ZPSF        ImageHDU        28   (34, 34)   float32
     16  GCORREL     BinTableHDU     32   20155R x 5C   [J, J, J, J, D]
     17  RCORREL     BinTableHDU     32   21023R x 5C   [J, J, J, J, D]
     18  ICORREL     BinTableHDU     32   21718R x 5C   [J, J, J, J, D]
     19  ZCORREL     BinTableHDU     32   21983R x 5C   [J, J, J, J, D]

When you open a Marvin Cube in local mode, Marvin provides convenient quick access to the first 5 extensions of your file.  In your Marvin Cube, you have the **header**, **flux**, **ivar**, **mask**, and **wavelength** attributes.  The extension for spectral resolution is stored in Marvin Spaxels under **specres**.

::

    # access the 3-d array of flux values
    cube.flux
    array([[[ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            ...,
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.],
            [ 0.,  0.,  0., ...,  0.,  0.,  0.]]], dtype=float32)

    # see the dimensions as (z, y, x) or (spectral, y spatial, x spatial)
    cube.flux.shape
    (4563, 34, 34)

    # access the inverse and mask arrays
    cube.flux.ivar

    cube.flux.mask

We just loaded this Cube locally using the identifier **plateifu**.  You can also use **mangaid** as a valid identifier.  When using these keywords, Marvin will look for the file in your local **SAS** directory system.  Alternatively you can specify a full filename and path using the **filename** keyword.  This keyword is for loading explicit files stored anywhere and named anything.

::

    # Here I am specifying an explicit file on my hard drive
    myfile = '/Users/Brian/Work/mybestcube.fits'

    # load this cube
    cube = Cube(filename=myfile)

    # display it
    print(cube)
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='file')>

.. _marvin-getstart_querysample:

Querying the Sample
-------------------

Previously, you have been dealing with individual objects, on a case by case basis.  But what if you want to perform a query on the MaNGA sample and retrieve a subset of data.  You can do this using the Marvin Query tool.

::

    # import the Marvin Query Tool
    from marvin.tools.query import Query

    # create a filter condition using a pseudo natural language SQL syntax

    # let's look for low-mass galaxies (< 1e9) at redshifts less than 0.2.  These parameters come from the NSA catalog.  You don't need to
    # specify the nsa table, but we recommend keeping the syntax
    myfilter = 'nsa.z < 0.2 and nsa.sersic_mass < 1e9'

    # create the Marvin Query
    myquery = Query(search_filter=myfilter)

    # run your query
    myresults = myquery.run()

    # your results are stored in a Marvin Results Tool
    print(myresults)
    Marvin Results(results=..., query=u'SELECT ...', count=1, mode=remote)

You can do much more with Queries and Results.  See what else at the :ref:`marvin-query` and :ref:`marvin-results` pages.

.. _marvin-getstart_downloadbulk:

Download Objects in Bulk
------------------------

Marvin Queries return a subset of results based on your query and filter parameters.  This is all remote data.  If you want to download the MaNGA FITS files associated with your subset of results, just use the **download** method from your results.  The files are stored in their respective locations in your local **SAS**.

::

    # download the results
    results.download()

This downloads your results subset.  You can also download in bulk using a list of plate-ifus or manga-ids using **downloadList**.  See :ref:`marvin-download-objects` for more.

.. _marvin-getstart_converttools:

Converting to Marvin Objects
----------------------------

Marvin Queries return a paginated list of results as tuples of data values.  This is useful for quickly seeing data results, but what if you want to use the other Marvin Tools to interact with these results.  You can convert your list of results into a list of Marvin objects using the **convertToTool** method on Marvin Results.

::

    # convert my list of results into a list of Marvin Cube objects
    r.convertToTool('cube')

    # they are stored in the objects attribute.
    cubes = r.objects
    print(cubes)

    # access the first Marvin Cube
    cube1 = cubes[0]
    print(cube1)

Now you have the full power of the Marvin Tools at your disposal with your list of Query results.

.. _marvin-getstart_lookimages:

Looking at Images
-----------------

Sometimes it can helpful to see the optical SDSS image for the MaNGA target of interest.  You can easily do this right now with a Marvin Image utility function called **showImage**.  This function will display the PNG image of your target, from your local system if you have it, or remotely, if you do not.

::

    # import the utility function
    from marvin.utils.general.images import showImage

    # display the optical image for 8485-1901
    image = showImage(plateifu='8485-1901')

    # for a local image, see the image file name and path
    image.filename
    '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/1901.png'

This creates and returns a `Python Image Library object <https://pillow.readthedocs.io/en/latest/>`_, which you can manipulate as you see fit.  These images contain full WCS information in the **info** attribute, if you need to overlay things.  **info** returns a standard Python dictionary.  If you wish to convert to

::

    print(image)
    <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=562x562 at 0x11696FD10>

    # access the WCS information directly
    wcs_info = image.info

    # extract and convert to a full Astropy WCS object
    from marvin.utils.general.general import getWCSFromPng
    wcs = getWCSFromPng(image.filename)
    print(wcs)

    WCS Keywords

    Number of WCS axes: 2
    CTYPE : 'RA---TAN'  'DEC--TAN'
    CRVAL : 232.54470000000001  48.690201000000002
    CRPIX : 281.0  281.0
    PC1_1 PC1_2  : -2.47222222222e-05  0.0
    PC2_1 PC2_2  : 0.0  2.47222222222e-05
    CDELT : 1.0  1.0
    NAXIS : 0  0

|
