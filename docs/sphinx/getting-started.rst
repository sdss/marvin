
.. _marvin-getting_started:

Getting Started
===============

If you have not yet installed Marvin, please follow the :ref:`marvin-installation` instructions before proceeding.  In particular, make sure you have a **netrc** file in your local home directory.  This will enable Marvin to access data remotely, and download files.

Accessing Objects Remotely
--------------------------

From your terminal, type ipython.  Ipython is an Interactive Python shell terminal.  It is recommended to always use ipython instead of python.::

    > ipython

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

You have just created a Marvin Cube for plate-ifu **8485-1901**.  You will also see **mode** and **data_origin** keywords.  These keywords inform you of how your cube was accessed.  You can see that your cube was opened remotely via the Marvin API.  (If you already have this cube locally, you may see a different indicator for mode and data_origin.  That's ok!). With a cube




Marvin has many Tools available that all behave in a similar way to the Marvin Cube.  See :ref:`marvin-tools` for a description of all the Tools currently available.

Displaying Plots
----------------

Marvin makes it really easy to very quickly plot a spectrum from a spaxel, or a 2-d Map output by MaNGA's Derived Analysis Pipeline (DAP).

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
    spectrum = spaxel.spectrum

    # plot the spectrum
    spectrum.plot()

    # access the data as Numpy arrays
    spectrum.flux

    # ivar
    spectrum.ivar

    # mask
    spectrum.mask

    # wavelength
    spectrum.wavelength

Accessing and Displaying Maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Marvin has the ability to quickly access and display DAP Maps

::

    # from our previous cube, let's access the default Maps associated with 8485-1901
    maps = cube.getMaps()

    # display the string representation of the your maps object
    print(maps)
    maps = <Marvin Maps (plateifu='8485-1901', mode='remote', data_origin='api', bintype=SPX, template_kin=GAU-MILESHC)>

The default Maps object created is the unbinned maps DAP object.  You can request a map with a different bintype or stellar template model using the **bintype** and **template_kin** keywords.  To access individial maps, you can

::

    # grab the H-alpha emission line map
    hamap = maps['emline_gflux_ha_6564']

    # alternatively, use getMap
    hamap = maps.getMap('emline_glux', channel='ha_6564')

    # display the Map object
    print(hamap)
    <Marvin Map (plateifu='8485-1901', property='emline_gflux', channel='ha_6564')>

You have now accessed an individual Marvin Map.  The **property** **channel** keywords indicate whichs DAP property and channel (if any) you have accessed.  The raw arrays for the data, inverse variance, and mask are stored in the attributes **value**, **ivar**, **mask** on each map object.

::

    # access the 2-d H-alpha flux data values
    data = hamap.value

    print(type(data))

    print(data)

    # 2-d inverse variance array
    hamap.ivar

    # DAP mask array
    hamap.mask

You can plot any map simply by using the **plot** method on your Map object.  Marvin uses the Python packacke Matplotlib for all default plotting.  Many matplotlib plotting options are usable in Marvin's **plot** method.

::

    # plot the H-alpha flux map
    hamap.plot()

    # change some default plot values
    hamap.plot(options=..)


Downloading Your Object
-----------------------

In the previous steps you have been accessing the MaNGA data for **8485-1901** remotely with Marvin.  But now you want to get your hands dirty with the real data file.  You can easily download MaNGA data products with Marvin.  There are many ways to download data with Marvin.  To download individual data file for the objects you are working with, use the **download** method attached to your object.  You can only download objects that have associated MaNGA data product files.

::

    # download the DRP datacube file for 8485-1901
    cube.download()

    # download the DAP unbinned MAPS file for 8485-1901
    maps.download()

    # you cannot download individual maps because there is no associated DAP data product.  This will fail.
    hamap.download()

This describes a method for manual download of individual files.  There are other ways to download MaNGA files.  See :ref:`marvin-download-objects` for a full description of how to download data.

The files are stored in your local **SAS (Science Archive Server)** as set up by Marvin.  This local **SAS** is a direct mimic of the real **SAS**, used at Utah by MaNGA in SDSS-IV.  Marvin creates and uses an environment variable called **SAS_BASE_DIR**.  Unless you have this already set up, Marvin creates this in your local home directory.  To see where your **SAS_BASE_DIR** is located, us the Python **os** package.

::

    import os
    print(os.environ['SAS_BASE_DIR'])
    '/Users/Brian/Work/sdss/sas'

You should see a directory path printed. If you get an error of the sort **KeyError: 'SAS_BASE_DIR'**, then you are missing this environment variable.  Something has gone wrong with your Marvin set up and configuration.  Please contact the developers.

Accessing Objects Locally
-------------------------

In the previous section, you downloaded the data files for 8485-1901 directly to your computer.  Now let's access this file.  The beauty of Marvin is that you do not have to do anything different once you have downloaded a file to access it locally.  Simply call your object the same way as before, and Marvin's Smart Multi-Modal Data Access System will do the rest.

::

    # instantiate a Marvin Cube for plate-ifu 8485-1901
    cube = Cube(plateifu='8485-1901')

    # display the cube
    print(cube)
    <Marvin Cube (plateifu='8485-1901', mode='local', data_origin='file')>

Notice that the **mode** is now **local**, and the **data_origin** is now set to **file**.  You are now accessing the full FITS file for the 3d datacube for 8485-1901.  Marvin uses the **Astropy io.fits** package for all FITS handling.

::

    # print the full file name and path to your data file
    print(cube.filename)
    '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/manga-8485-1901-LOGCUBE.fits.gz'

    # access the FITS header
    cube.header

    # retrieve a list of file HDUs
    hdus = cube.data
    print(hdus)

When you open a Marvin Cube in local mode, you immediately get access to the data with the **flux**, **ivar**, and **mask** attributes.

::

    # access the 3-d array of flux values
    cube.flux

    # see the dimensions as (z, y, x) or (spectral, y spatial, x spatial)
    cube.flux.shape
    (4563, 34, 34)

    # access the inverse and mask arrays
    cube.ivar

    cube.mask


Querying the Sample
-------------------

Download Objects in Bulk
------------------------

Converting to Marvin Objects
----------------------------

Looking at Images
-----------------




