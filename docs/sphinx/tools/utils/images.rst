
.. _marvin-images:

Image Utilities
===============

.. admonition:: Warning
    :class: warning

    .. deprecated:: 2.3.0
        These utility functions have been deprecated.  Use :ref:`image-utils` instead.


If you want to grab the postage stamp PNG cutout images of the MaNGA galaxies, Marvin currently provides a few ways of doing so:

* **By PlateID**: Returns a list of images of galaxies observed on a given plate.

* **By Target List**: Returns a list of images from an input list of targets.

* **By Random Chance**: Returns a random set of images within a given MPL.

All image utilities behave in the same way.  Each function can be used in one of three ways:

* To navigate and retrieve paths to images in your **local** SAS.

* To retrieve URL paths to image locations on the **Utah** SAS.

* To download the images from the **Utah** SAS to your **local** SAS.

Each function accepts three optional keyword arguments which determine what it returns.  All of the Marvin Image utility functions use sdss_access under the hood, and build paths using a combination of the environment variables **SAS_BASE_DIR**, **MANGA_SPECTRO_REDUX**, and an internal rsync **REMOTE_BASE**.  These keywords simply toggle how to construct those paths and/or download.

* **mode**:
    The Marvin config mode being used.  Defaults to the ``marvin.config.mode``.  When in **local** mode, Marvin navigates paths/images in your local SAS filesystem.  When in **remote** mode, Marvin calls Utah to retrieve image lists there.  When in **auto** mode, the functions default to **remote** mode.  Use of **local** mode must be explicitly set.
* **as_url**:
    A boolean that, when set to **True**, converts the paths into URL paths.  Default is False.  When in **local** mode, paths get converted to the SAS url **https://data.sdss.org/sas/**.  When in **remote** mode, paths get converted into an rsync path **https://sdss@dtn01.sdss.org/sas/**.  When **False**, the functions generate paths based on your **MANGA_SPECTRO_REDUX**.

* **download**:
    A boolean that, when set to **True**, downloads all the images into your local SAS.  Only works in **remote** mode.  Attempting to download in **local** mode will result in a stern warning!

See :ref:`marvin-utils-general-images` for the reference to the basic utility functions we provide.

A secret fourth way of downloading images is via **downloadList**. See **Via Explicit Call** in :ref:`marvin-download-objects` and :meth:`~marvin.utils.general.general.downloadList`

Once you have downloaded images into your local SAS, you can easily display them using the ``showImage`` utility function, described below.  Or if you'd rather not download them, ``showImage`` also works remotely.  See :ref:`marvin-image-show`.

Common Usage
------------
The two most common uses will be to download images from Utah to your local system, and to get paths to your local images.  See the sections below for full, and specific, examples of all uses.

* **syntax to download**: ``image_utility_name(input, mode='remote', download=True)``
* **syntax to search locally**: ``image_utility_name(input, mode='local')``

By Target List
--------------
**getImagesByList** returns a list of image paths from a given input list of ids.  Ids can be either plateifus, or manga-ids.
::

    from marvin.utils.general.images import getImagesByList

    # make a list of targets, can be plateifus or manga-ids
    plateifus = ['8485-1901', '7443-12701']

    # download the images for the targets in my list from Utah into my local SAS
    images = getImagesByList(plateifus, mode='remote', download=True)

    # search my local SAS filesystem for images in the input list
    images = getImagesByList(plate, mode='local')
    print(images)
    ['/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/1901.png',
     '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/7443/stack/images/12701.png']

    # convert my local file image paths into the SAS URL paths
    images = getImagesByList(plateifus, mode='local', as_url=True)
    print(images)
    ['https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/1901.png',
     'https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/7443/stack/images/12701.png']


By PlateID
----------
**getImagesByPlate** returns a list of image paths from a given plateid
::

    from marvin.utils.general.images import getImagesByPlate

    plate = 8485

    # download the images for plate 8485 from Utah into my local SAS
    images = getImagesByPlate(plate, mode='remote', download=True)

    # search my local SAS filesystem for images connected to plate 8485
    # these are my local images
    images = getImagesByPlate(plate, mode='local')
    print(images)
    ['/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/12701.png',
      ....
     '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/9102.png']

    # convert my local file image paths into the SAS URL paths
    images = getImagesByPlate(plate, mode='local', as_url=True)
    print(images)
    ['https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/12701.png',
      ....
     'https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/9102.png']

    # generate rsync paths for the image files (located on Utah SAS) for plate 8485
    # these are images located at Utah but generated with my local SAS_BASE_DIR (notice the thumbnails)
    images = getImagesByPlate(plate, mode='remote', as_url=True)
    print(images)
    ['/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/12701.png',
     '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/12701_thumb.png',
      ....
     '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/9102.png',
     '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/9102_thumb.png']

    # generate rsync paths for the image files (located on Utah SAS) for plate 8485
    images = getImagesByPlate(plate, mode='remote', as_url=True)
    print(images)
    ['https://sdss@dtn01.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/12701.png',
     'https://sdss@dtn01.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/12701_thumb.png'
      ....
     'https://sdss@dtn01.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/9102.png',
     'https://sdss@dtn01.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/9102_thumb.png']

By Random Chance
----------------
**getRandomImages** returns a list of random images for a given MPL.  The default number returned is 10.
::

    from marvin.utils.general.images import getRandomImages

    # download 10 random images from Utah into my local SAS
    images = getRandomImages(mode='remote', download=True)

    # return 3 random images from  my local SAS filesystem
    images = getRandomImages(num=3, mode='local')
    print(images)
    ['/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/9101.png',
     '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/7443/stack/images/1902.png',
     '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/7443/stack/images/3702.png']

    # get the URLs for 5 random images
    images = getRandomImages(num=5, mode='local', as_url=True)
    print(images)
    ['https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/12704.png',
     'https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/3701.png',
     'https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/7443/stack/images/6101.png',
     'https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/12701.png',
     'https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/7443/stack/images/6103.png']


.. _marvin-image-show:

Displaying Images
-----------------

Once you have downloaded IFU png images into your local SAS using any of the above utility functions, you may display them using the ``showImage`` utility function.  This function quickly and coarsely opens and displays an image as a PIL Image Object (using the `Python Image Library <http://pillow.readthedocs.io/en/3.1.x/index.html>`_ python package.)  The image will be displayed and the image object is also returned.  Once the image object is returned, you can manipulate the image as you see fit.

When acting in ``mode=local``, ``showImage`` will attempt to locate the image file from your local SAS.  In ``mode=remote``, ``showImage`` will attempt to grab the requested image file from the Utah SAS.  In ``mode=auto``, local mode is tried first, then remote mode.

See :meth:`~marvin.utils.general.images.showImage` for the API reference.

::

    from marvin.utils.general.images import showImage

    # let's open the image for plateifu 8485-1901
    image = showImage(plateifu='8485-1901')
    print(image)
    <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=562x562 at 0x1142CE390>

    # this file was opened locally via mode = auto
    print(image.filename)
    /Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/1901.png

    # open an image remotely via mode = auto
    image = showImage(plateifu='7495-1902')
    WARNING: Local mode failed.  Trying remote.
    print(image.filename)
    https://data.sdss.org/sas/mangawork/manga/spectro/redux/v2_0_1/7495/stack/images/1902.png

    # open an image via a path

    # first get some paths to some local images
    imagepaths = getRandomImages(num=3, mode='local')
    print(images)
    ['/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/8485/stack/images/9101.png',
     '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/7443/stack/images/1902.png',
     '/Users/Brian/Work/sdss/sas/mangawork/manga/spectro/redux/v2_0_1/7443/stack/images/3702.png']

     # showImage only acts on one path a time
     image = showImage(path=imagepaths[0])

     # retrieve the image object so I can manipulate it, but don't show the image
     image = showImage(plateifu='8485-1901', show_image=False)

     # show the image without returning the image object
     image = showImage(plateifu='8485-1901', return_image=False)
     print(image)
     None

End of line!
